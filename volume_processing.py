import numpy as np
from PIL import Image
from pathlib import Path
from scipy import ndimage

# VERSÃO RESUMIDA
class VolumeManager:
    def __init__(self):
        self.volume = None
        self.z_thicknesses = None

        self.dims = (0, 0, 0)
        self.is_grayscale = False
        
        self.z_thicknesses = None 
        self.z_starts = None
        self.z_cumulative = None 
        self.total_z_height = 0

        self.interpolation_mode = 'none'  # Modo padrão

    def load_stack(self, folder):
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}


        path = Path(folder)
        files = sorted([f for f in path.iterdir() if f.suffix.lower() in valid_exts])

        if not files: return False

        # Carrega tudo em uma linha usando list comprehension
        temp_imgs = []
        for f in files:
            img = Image.open(f)
            # transformamos em array imediatamente para não manter objeto PIL aberto
            temp_imgs.append(np.array(img)) 
            
        # Empilha tudo num array NumPy 3D ou 4D
        try:
            self.volume = np.array(temp_imgs, dtype=np.uint8)
        except ValueError:
            print("Erro: As imagens possuem tamanhos diferentes!")
            return False
        
        shape = self.volume.shape # (D, H, W, C)

        if len(shape) == 3:
            # (Depth, Height, Width) -> Grayscale
            self.dims = shape
            self.is_grayscale = True
        elif len(shape) == 4:
            # (Depth, Height, Width, Channels) -> RGB
            self.dims = shape[:3] # Ignora o canal de cor na tupla de dimensões
            self.is_grayscale = False
        
        self.z_thicknesses = np.ones(self.dims[0], dtype=int)
        self.z_starts = np.arange(self.dims[0], dtype=int)
        self.update_geometry()
        
        print(f"Volume carregado. Shape: {self.volume.shape}. Grayscale: {self.is_grayscale}")
        return True
    
    def update_geometry(self):
        """Recalcula a altura física total e a régua de posições"""
        self.total_z_height = np.sum(self.z_thicknesses)
        
        # Cria a 'régua' acumulada.
        # Ex: [10, 20] -> [10, 30]. O pixel 15 cai no indice 1 (menor que 30, maior que 10)
        self.z_cumulative = np.cumsum(self.z_thicknesses)

    def set_layer_data(self, data):
        if self.volume is None: 
            return

        d = self.dims[0]
        count = min(len(data), d)
        
        # Extrai starts e sizes
        new_starts = [int(item[0]) for item in data[:count]]
        new_thicks = [int(item[1]) for item in data[:count]]
        
        # Completa se necessário
        if count < d:
            diff = d - count
            last_end = new_starts[-1] + new_thicks[-1] if count > 0 else 0
            for i in range(diff):
                new_starts.append(last_end + i)
                new_thicks.append(1)
        
        # Armazena AMBOS
        self.z_starts = np.array(new_starts, dtype=int)
        self.z_thicknesses = np.array(new_thicks, dtype=int)
        
        self.update_geometry()

    def get_slice_texture(self, axis, pct):
        """
        Retorna fatia 2D do volume com interpolação opcional.
        
        Args:
            axis: 'x', 'y' ou 'z'
            pct: Percentual da posição (0-100)
        
        Returns:
            Array 2D (grayscale) ou 3D (RGB) da fatia
        """
        d, h, w = self.dims[:3]
        
        # ─────────────────────────────────────────────────────────────
        # EIXO Z (Topo) - Sem interpolação necessária
        # ─────────────────────────────────────────────────────────────
        if axis == 'z':
            target_pixel_z = int((pct/100) * self.total_z_height - 1)
            idx = np.searchsorted(self.z_cumulative, target_pixel_z, side='right')
            idx = min(idx, d-1)
            return self.volume[idx, :, :]
        
        # ─────────────────────────────────────────────────────────────
        # EIXOS LATERAIS (X ou Y) - Aplicar interpolação
        # ─────────────────────────────────────────────────────────────
        
        # Extrai fatia lateral
        if axis == 'x':
            idx = int(pct/100 * (w-1))
            if self.is_grayscale:
                raw = self.volume[:, :, idx]
            else:
                raw = self.volume[:, :, idx, :]
        else:  # axis == 'y'
            idx = int(pct/100 * (h-1))
            if self.is_grayscale:
                raw = self.volume[:, idx, :]
            else:
                raw = self.volume[:, idx, :, :]
        
        # ─────────────────────────────────────────────────────────────
        # Aplica interpolação baseada no modo selecionado
        # ─────────────────────────────────────────────────────────────
        
        if self.interpolation_mode == 'none':
            # Original: repete cada camada por sua espessura
            return np.repeat(raw, self.z_thicknesses, axis=0)
        
        elif self.interpolation_mode == 'zoom':
            # RECOMENDADO: scipy.ndimage.zoom (rápido + boa qualidade)
            zoom_factor = self.total_z_height / d
            
            if self.is_grayscale:
                result = ndimage.zoom(raw, (zoom_factor, 1.0), order=1)
            else:
                result = ndimage.zoom(raw, (zoom_factor, 1.0, 1.0), order=1)
            
            return np.clip(result, 0, 255).astype(np.uint8)
        
        elif self.interpolation_mode == 'adaptive':
            # Máxima precisão: respeita espessuras variáveis
            return self._interpolate_adaptive(raw)
        
        elif self.interpolation_mode == 'cubic':
            # Mais suave: interpolação cúbica
            return self._interpolate_cubic(raw)
        
        else:
            # Fallback para 'none' se modo inválido
            return np.repeat(raw, self.z_thicknesses, axis=0)


    # ─────────────────────────────────────────────────────────────────
    # MÉTODO AUXILIAR 1: Seletor de modo de interpolação
    # ─────────────────────────────────────────────────────────────────

    def set_interpolation(self, mode):
        """
        Define o modo de interpolação para fatias laterais.
        
        Args:
            mode (str): Modo de interpolação
                - 'none': Sem interpolação (np.repeat) - RÁPIDO
                - 'zoom': scipy.ndimage.zoom - RECOMENDADO
                - 'adaptive': Respeita z_thicknesses exatamente - PRECISO
                - 'cubic': Interpolação cúbica - SUAVE
        
        Exemplo:
            >>> vm.set_interpolation('zoom')  # Recomendado para maioria
            >>> vm.set_interpolation('adaptive')  # Para espessuras irregulares
        """
        valid_modes = ['none', 'zoom', 'adaptive', 'cubic']
        
        if mode not in valid_modes:
            print(f"⚠️  Modo '{mode}' inválido. Modos válidos: {valid_modes}")
            print("   Usando 'zoom' (padrão)")
            mode = 'zoom'
        
        self.interpolation_mode = mode
        
        # Feedback visual
        descriptions = {
            'none': '⚡ Sem interpolação (mais rápido)',
            'zoom': '⭐ Zoom scipy (recomendado)',
            'adaptive': '🎯 Adaptativo (máxima precisão)',
            'cubic': '🎨 Cúbico (mais suave)'
        }
        
        print(f"✓ Interpolação: {descriptions.get(mode, mode)}")


    # ─────────────────────────────────────────────────────────────────
    # MÉTODO AUXILIAR 2: Interpolação adaptativa
    # ─────────────────────────────────────────────────────────────────

    def _interpolate_adaptive(self, raw):
        """
        Interpolação que respeita exatamente as espessuras das camadas.
        
        Para cada pixel Z de saída:
        1. Identifica as 2 camadas vizinhas
        2. Calcula peso baseado na distância
        3. Faz blend linear (LERP)
        
        Args:
            raw: Array 2D ou 3D da fatia lateral
        
        Returns:
            Array interpolado com altura = total_z_height
        """
        d = self.dims[0]
        
        # Calcula posição central de cada camada original
        layer_centers = np.zeros(d)
        for i in range(d):
            if i == 0:
                layer_centers[i] = self.z_thicknesses[i] / 2.0
            else:
                layer_centers[i] = self.z_cumulative[i-1] + self.z_thicknesses[i] / 2.0
        
        # Prepara array de saída
        if self.is_grayscale:
            result = np.zeros((self.total_z_height, raw.shape[1]), dtype=np.float32)
        else:
            result = np.zeros((self.total_z_height, raw.shape[1], raw.shape[2]), dtype=np.float32)
        
        # Interpola cada posição Z
        for z_out in range(self.total_z_height):
            # Encontra camada posterior mais próxima
            idx_after = np.searchsorted(layer_centers, z_out, side='left')
            
            if idx_after == 0:
                # Antes da primeira camada
                result[z_out] = raw[0]
            
            elif idx_after >= d:
                # Depois da última camada
                result[z_out] = raw[d-1]
            
            else:
                # Entre duas camadas - INTERPOLAÇÃO
                idx_before = idx_after - 1
                
                z_before = layer_centers[idx_before]
                z_after = layer_centers[idx_after]
                
                # Calcula peso (0.0 = 100% before, 1.0 = 100% after)
                weight = (z_out - z_before) / (z_after - z_before + 1e-10)
                weight = np.clip(weight, 0.0, 1.0)
                
                # Linear interpolation (LERP)
                result[z_out] = (1 - weight) * raw[idx_before] + weight * raw[idx_after]
        
        return result.astype(np.uint8)


    # ─────────────────────────────────────────────────────────────────
    # MÉTODO AUXILIAR 3: Interpolação cúbica
    # ─────────────────────────────────────────────────────────────────

    def _interpolate_cubic(self, raw):
        """
        Interpolação cúbica usando splines (mais suave que linear).
        
        Requer scipy. Se não disponível, faz fallback para adaptativa.
        
        Args:
            raw: Array 2D ou 3D da fatia lateral
        
        Returns:
            Array interpolado com altura = total_z_height
        """
        try:
            from scipy import interpolate
        except ImportError:
            print("⚠️  scipy não instalado. Usando interpolação adaptativa.")
            return self._interpolate_adaptive(raw)
        
        d = self.dims[0]
        
        # Posições centrais das camadas
        layer_centers = np.zeros(d)
        for i in range(d):
            if i == 0:
                layer_centers[i] = self.z_thicknesses[i] / 2.0
            else:
                layer_centers[i] = self.z_cumulative[i-1] + self.z_thicknesses[i] / 2.0
        
        # Posições de saída
        interpolated_z = np.arange(self.total_z_height, dtype=float)
        
        if self.is_grayscale:
            height_or_width = raw.shape[1]
            result = np.zeros((self.total_z_height, height_or_width), dtype=np.uint8)
            
            for col in range(height_or_width):
                # Cria interpolador cúbico
                if d >= 4:
                    f = interpolate.interp1d(
                        layer_centers,
                        raw[:, col],
                        kind='cubic',
                        fill_value='extrapolate'
                    )
                else:
                    # Fallback para linear se < 4 camadas
                    f = interpolate.interp1d(
                        layer_centers,
                        raw[:, col],
                        kind='linear',
                        fill_value='extrapolate'
                    )
                
                result[:, col] = np.clip(f(interpolated_z), 0, 255).astype(np.uint8)
        
        else:
            height_or_width = raw.shape[1]
            channels = raw.shape[2]
            result = np.zeros((self.total_z_height, height_or_width, channels), dtype=np.uint8)
            
            for col in range(height_or_width):
                for c in range(channels):
                    if d >= 4:
                        f = interpolate.interp1d(
                            layer_centers,
                            raw[:, col, c],
                            kind='cubic',
                            fill_value='extrapolate'
                        )
                    else:
                        f = interpolate.interp1d(
                            layer_centers,
                            raw[:, col, c],
                            kind='linear',
                            fill_value='extrapolate'
                        )
                    
                    result[:, col, c] = np.clip(f(interpolated_z), 0, 255).astype(np.uint8)
        
        return result
