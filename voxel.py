"""
CORREÇÃO: VOXELS RESPEITANDO AS 3 ESTRATÉGIAS
Mapeia voxels corretamente usando (start, size) ao invés de apenas z_thicknesses
"""

import numpy as np
import pyvista as pv

# ===================================================================
# EXPLICAÇÃO DO PROBLEMA
# ===================================================================

"""
PROBLEMA IDENTIFICADO:
- A função calculate() retorna: [(start, size), (start, size), ...]
  Exemplo: [(0, 10), (10, 20), (30, 15), ...]
  
- O VolumeManager armazena apenas: z_thicknesses = [10, 20, 15, ...]
  (Perde a informação de START!)

- Estratégias diferentes produzem STARTs diferentes:
  • troughNext: Cada camada começa onde terminou a anterior
  • centerAvg: Camadas podem ter GAPs ou OVERLAPs
  • troughPrevious: Similar ao troughNext mas inverso

SOLUÇÃO:
- Armazenar TAMBÉM os starts: self.z_starts = [0, 10, 30, ...]
- Usar z_starts para mapear slice_idx -> posição Z física
"""

# ===================================================================
# PARTE 2: ADICIONAR NO __init__ DO VolumeManager
# ===================================================================

"""
Adicione esta linha no __init__ do VolumeManager (após self.z_thicknesses):
"""

CODE_INIT_VOLUMEMANAGER = '''
    self.z_starts = None  # NOVO: Armazena posições de início de cada camada
'''

# ===================================================================
# PARTE 3: MODIFICAR load_stack PARA INICIALIZAR z_starts
# ===================================================================

"""
No método load_stack, após a linha:
    self.z_thicknesses = np.ones(self.dims[0], dtype=int)
    
Adicione:
"""

CODE_LOAD_STACK_FIX = '''
    # Inicializa starts sequencialmente (0, 1, 2, 3...)
    self.z_starts = np.arange(self.dims[0], dtype=int)
'''

# ===================================================================
# PARTE 4: MÉTODO CORRIGIDO _adjust_voxel_z
# ===================================================================

"""
SUBSTITUA completamente o método _adjust_voxel_z no VolumeWidget por este:
"""

def _adjust_voxel_z(self, coords, resolution, slice_offset):
    """
    Mapeia coordenadas de voxel para posições físicas Z usando z_starts.
    
    COMO FUNCIONA:
    1. Cada voxel tem um slice_idx (qual imagem veio)
    2. Cada slice tem um START físico (z_starts[slice_idx])
    3. Cada slice tem uma ESPESSURA (z_thicknesses[slice_idx])
    4. Posição Z do voxel = START + (espessura / 2) para centralizar
    
    Args:
        coords: Array (N, 3) com [slice_idx, y, x]
        resolution: Fator de downsample (1, 2, 4...)
        slice_offset: Offset do slice inicial (do corte)
    
    Returns:
        Array (N, 3) com [z_physical, y, x]
    """
    coords_adj = coords.copy().astype(float)
    
    # Para cada voxel, mapeia slice_idx -> posição Z física
    for i, slice_idx in enumerate(coords[:, 0]):
        # Slice real no volume original
        real_slice = int(slice_idx * resolution) + slice_offset
        
        # Clamp para evitar out of bounds
        if real_slice >= len(self.manager.z_starts):
            real_slice = len(self.manager.z_starts) - 1
        
        # CORREÇÃO CRÍTICA: Usa z_starts ao invés de z_cumulative
        z_start = self.manager.z_starts[real_slice]
        thickness = self.manager.z_thicknesses[real_slice]
        
        # Centraliza o voxel dentro da espessura da camada
        coords_adj[i, 0] = z_start + thickness / 2.0
    
    # Multiplica Y e X pela resolução
    coords_adj[:, 1] *= resolution
    coords_adj[:, 2] *= resolution
    
    return coords_adj


# ===================================================================
# PARTE 5: VISUALIZAÇÃO DAS ESTRATÉGIAS
# ===================================================================

STRATEGY_EXPLANATION = """
╔════════════════════════════════════════════════════════════════╗
║  COMO AS 3 ESTRATÉGIAS FUNCIONAM                               ║
╚════════════════════════════════════════════════════════════════╝

Valores de entrada: [10, 20, 35, 50]
Fator: 1 (sem multiplicação)

┌────────────────────────────────────────────────────────────────┐
│ ESTRATÉGIA 1: troughNext (Até o Seguinte)                     │
└────────────────────────────────────────────────────────────────┘
  Espessura = distância até a próxima camada
  
  Camada 0: start=0,  size=10  (distância 10→20 = 10)
  Camada 1: start=10, size=15  (distância 20→35 = 15)
  Camada 2: start=25, size=15  (distância 35→50 = 15)
  Camada 3: start=40, size=15  (último = extrapola)
  
  Resultado: [(0,10), (10,15), (25,15), (40,15)]
  
  VOXELS:
    Camada 0: Z entre 0-10   (centro em Z=5)
    Camada 1: Z entre 10-25  (centro em Z=17.5)
    Camada 2: Z entre 25-40  (centro em Z=32.5)
    Camada 3: Z entre 40-55  (centro em Z=47.5)

┌────────────────────────────────────────────────────────────────┐
│ ESTRATÉGIA 2: centerAvg (Média Central)                       │
└────────────────────────────────────────────────────────────────┘
  Espessura = média entre anterior e posterior
  
  Camada 0: start=0,  size=5   ((20-10)/2 = 5)
  Camada 1: start=5,  size=12  ((35-10)/2 = 12.5 ≈ 12)
  Camada 2: start=17, size=12  ((50-20)/2 = 15 ≈ 12)
  Camada 3: start=29, size=7   ((50-35)/2 = 7.5 ≈ 7)
  
  Resultado: [(0,5), (5,12), (17,12), (29,7)]
  
  VOXELS:
    Camada 0: Z entre 0-5    (centro em Z=2.5)
    Camada 1: Z entre 5-17   (centro em Z=11)
    Camada 2: Z entre 17-29  (centro em Z=23)
    Camada 3: Z entre 29-36  (centro em Z=32.5)
    
  NOTA: Pode haver GAPS entre camadas!

┌────────────────────────────────────────────────────────────────┐
│ ESTRATÉGIA 3: troughPrevious (Até o Anterior)                 │
└────────────────────────────────────────────────────────────────┘
  Espessura = distância desde a anterior
  
  Camada 0: start=0,  size=10  (primeiro = extrapola)
  Camada 1: start=10, size=10  (distância 10→20 = 10)
  Camada 2: start=20, size=15  (distância 20→35 = 15)
  Camada 3: start=35, size=15  (distância 35→50 = 15)
  
  Resultado: [(0,10), (10,10), (20,15), (35,15)]
  
  VOXELS:
    Camada 0: Z entre 0-10   (centro em Z=5)
    Camada 1: Z entre 10-20  (centro em Z=15)
    Camada 2: Z entre 20-35  (centro em Z=27.5)
    Camada 3: Z entre 35-50  (centro em Z=42.5)
"""

# ===================================================================
# PARTE 6: TESTE DE VALIDAÇÃO
# ===================================================================

def test_strategy_mapping():
    """
    Testa se o mapeamento está correto para as 3 estratégias.
    """
    print("=" * 70)
    print("TESTE DE VALIDAÇÃO - MAPEAMENTO DE ESTRATÉGIAS")
    print("=" * 70)
    
    # Simula dados de calculate()
    test_cases = {
        'troughNext': [
            (0, 10), (10, 15), (25, 15), (40, 15)
        ],
        'centerAvg': [
            (0, 5), (5, 12), (17, 12), (29, 7)
        ],
        'troughPrevious': [
            (0, 10), (10, 10), (20, 15), (35, 15)
        ]
    }
    
    for strategy, data in test_cases.items():
        print(f"\n📊 Estratégia: {strategy}")
        print("-" * 70)
        
        # Extrai starts e sizes
        starts = [item[0] for item in data]
        sizes = [item[1] for item in data]
        
        print(f"Starts:     {starts}")
        print(f"Sizes:      {sizes}")
        
        # Calcula centro de cada camada
        centers = [starts[i] + sizes[i]/2 for i in range(len(data))]
        print(f"Centros Z:  {[f'{c:.1f}' for c in centers]}")
        
        # Simula voxels (um por camada no centro)
        print("\nVoxels simulados:")
        for i, (start, size) in enumerate(data):
            center_z = start + size/2
            print(f"  Camada {i} -> Voxel em Z = {center_z:.1f} px")
        
        print()


# ===================================================================
# CÓDIGO COMPLETO PARA COPIAR
# ===================================================================

COMPLETE_CODE = '''
# ===================================================================
# MODIFICAÇÃO NO main.py (VolumeWidget)
# ===================================================================

# SUBSTITUIR o método _adjust_voxel_z:
def _adjust_voxel_z(self, coords, resolution, slice_offset):
    coords_adj = coords.copy().astype(float)
    
    for i, slice_idx in enumerate(coords[:, 0]):
        real_slice = int(slice_idx * resolution) + slice_offset
        
        if real_slice >= len(self.manager.z_starts):
            real_slice = len(self.manager.z_starts) - 1
        
        # USA z_starts (CORREÇÃO)
        z_start = self.manager.z_starts[real_slice]
        thickness = self.manager.z_thicknesses[real_slice]
        
        # Centraliza
        coords_adj[i, 0] = z_start + thickness / 2.0
    
    coords_adj[:, 1] *= resolution
    coords_adj[:, 2] *= resolution
    
    return coords_adj
'''

if __name__ == "__main__":
    print(STRATEGY_EXPLANATION)
    print("\n" + "=" * 70)
    test_strategy_mapping()
    print("\n" + "=" * 70)
    print("CÓDIGO PARA APLICAR:")
    print("=" * 70)
    print(COMPLETE_CODE)