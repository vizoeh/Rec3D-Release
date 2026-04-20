import cv2
import numpy as np
from pathlib import Path
from PIL import Image

### Método de Operador Reinhard

def get_dim(image_path):
    """
    Usa PIL para ler apenas o cabeçalho da imagem (rápido).
    Retorna: (width, height)
    """
    try:
        # O Image.open é "preguiçoso", não carrega a imagem na RAM
        with Image.open(str(image_path)) as img:
            return img.size # Retorna tupla (width, height)
    except Exception as e:
        print(f"Erro ao ler dimensões PIL: {e}")
        return 0, 0

def transfer_color_stats(source_img, target_img):
    """
    CLAUDE Versão otimizada usando cv2.meanStdDev (2-3x mais rápido).
    """
    
    src_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype("float32")
    tgt_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype("float32")

    # cv2.meanStdDev é MUITO mais rápido que .mean() e .std() separados
    (l_s, a_s, b_s) = cv2.split(src_lab)
    (l_t, a_t, b_t) = cv2.split(tgt_lab)

    # Calcula ambos de uma vez
    mean_s, std_s = cv2.meanStdDev(src_lab)
    mean_t, std_t = cv2.meanStdDev(tgt_lab)

    # Reshape para broadcasting
    mean_s = mean_s.ravel()
    std_s = std_s.ravel()
    mean_t = mean_t.ravel()
    std_t = std_t.ravel()

    eps = 1e-5
    
    # Operação vetorizada (mais rápida)
    for i, (canal_t, mean_s_i, std_s_i, mean_t_i, std_t_i) in enumerate(
        zip([l_t, a_t, b_t], mean_s, std_s, mean_t, std_t)
    ):
        canal_t[:] = ((canal_t - mean_t_i) * (std_s_i / (std_t_i + eps))) + mean_s_i

    merged = cv2.merge([l_t, a_t, b_t])
    merged = np.clip(merged, 0, 255).astype("uint8")
    
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
def run_stack_equalization(stack_path, reference_image_path):
    """
    Processa todas as imagens na pasta stack_path usando a referência.
    Sobrescreve as imagens existentes na pasta Rec3D (já que são cópias).
    """
    ref_path = Path(reference_image_path)
    folder = Path(stack_path)
    
    ### Carrega a imagem de referência
    ref_img = cv2.imread(str(ref_path))
    
    if ref_img is None:
        raise ValueError(f"Não foi possível ler a referência: {ref_path}")

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    
    ### Iterar transfer_color_stats para cada imagem na pasta, menos a referência
    # sorted é crucial para manter a ordem
    files = sorted([f for f in folder.iterdir() if f.suffix.lower() in valid_extensions])
    
    count = 0
    for file_path in files:
        # Pula se for a própria imagem de referência
        if file_path.name == ref_path.name:
            continue
            
        # Lê a imagem alvo
        tgt_img = cv2.imread(str(file_path))
        
        if tgt_img is not None:
            # Processa
            result_img = transfer_color_stats(ref_img, tgt_img)
            
            # Sobrescreve o arquivo
            cv2.imwrite(str(file_path), result_img)
            count += 1
            
    print(f"Equalização concluída: {count} imagens processadas.")

def apply_downsampling(stack_dir, factor):
    """
    Redimensiona todas as imagens da pasta.
    factor: 0.5 (50%), 0.25 (25%), etc.
    """
    folder = Path(stack_dir)
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    files = sorted([f for f in folder.iterdir() if f.suffix.lower() in valid_extensions])
    
    count = 0
    for file_path in files:
        img = cv2.imread(str(file_path))
        if img is not None:
            # Calcula novas dimensões
            new_width = int(img.shape[1] * factor)
            new_height = int(img.shape[0] * factor)
            dim = (new_width, new_height)
            
            # INTER_AREA é o melhor algoritmo para redução de tamanho (evita moiré)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            
            # Sobrescreve o arquivo
            cv2.imwrite(str(file_path), resized)
            count += 1
    return count

def apply_blur(stack_dir, kernel_size):

    """
    Aplica filtro de mediana para remover ruído (salt & pepper).
    kernel_size: Deve ser ímpar (3, 5, 7...)
    """
    if kernel_size % 2 == 0:
        kernel_size += 1 # Garante que seja ímpar
        
    folder = Path(stack_dir)
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    files = sorted([f for f in folder.iterdir() if f.suffix.lower() in valid_extensions])
    
    count = 0
    for file_path in files:
        img = cv2.imread(str(file_path))
        if img is not None:
            # Aplica o blur
            blurred = cv2.medianBlur(img, kernel_size)
            cv2.imwrite(str(file_path), blurred)
            count += 1
    return count

def crop_single_image(args):
        file_path, x, y, w, h = args
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        
        if img is not None:
            img_h, img_w = img.shape[:2]
            
            if x >= img_w or y >= img_h:
                return False
            
            x_end = min(x + w, img_w)
            y_end = min(y + h, img_h)
            
            cropped_img = img[y:y_end, x:x_end]
            
            if cropped_img.size > 0:
                cv2.imwrite(str(file_path), cropped_img)
                return True
        return False

def apply_crop(stack_dir, x, y, w, h, num_workers=4):
    """
    Versão paralela do apply_crop usando multiprocessing.
    Até 4x mais rápido em CPUs multi-core.
    """
    from multiprocessing import Pool
    
    folder = Path(stack_dir)
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    files = sorted([f for f in folder.iterdir() if f.suffix.lower() in valid_extensions])
    
    # Cria lista de argumentos
    args_list = [(f, x, y, w, h) for f in files]
    
    # Processa em paralelo
    with Pool(processes=num_workers) as pool:
        results = pool.map(crop_single_image, args_list)
    
    return sum(results)

def apply_inversion(stack_dir):
    """
    Inverte as cores de todas as imagens (Negativo).
    Útil para transformar fundo branco em fundo preto (transparente no 3D).
    """
    folder = Path(stack_dir)
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    files = sorted([f for f in folder.iterdir() if f.suffix.lower() in valid_extensions])
    
    count = 0
    for file_path in files:
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        
        if img is not None:
            # bitwise_not funciona para uint8 e uint16
            inverted = cv2.bitwise_not(img)
            cv2.imwrite(str(file_path), inverted)
            count += 1
            
    return count

def get_statistics(stack_dir):
    """
    Retorna estatísticas básicas (média, desvio padrão) das imagens na pasta.
    """
    folder = Path(stack_dir)
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    files = sorted([f for f in folder.iterdir() if f.suffix.lower() in valid_extensions])
    
    files= sorted(f for f in folder.iterdir() if f.suffix.lower() in valid_extensions)
    num_images = len(files)

    if num_images == 0:
        return None
    
    # Tamanho em Disco
    total_size_bt = sum(f.stat().st_size for f in files)
    total_size_kb = total_size_bt / (1024)
    total_size_mb = total_size_bt / (1024*1024)

    # Metadados de Imagem
    sample_img = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
    height, width = sample_img.shape[:2] # Altura e Largura    
    channels = 1 if len(sample_img.shape) == 2 else sample_img.shape[2] # Canais de cor

    # Bit depth (dtype)
    dtype = sample_img.dtype
    if dtype == "uint8":
        bytes = 1
        bit_depth = "8-bit"
    elif dtype == "uint16":
        bytes = 2
        bit_depth = "16-bit"
    else:
        bytes = 4
        bit_depth = str(dtype)

    # Memória estimada
    # W * H * Canais * Bytes  * Número de imagens 
    raw_ram_bytes = width * height * channels * bytes * num_images
    raw_ram_mb = raw_ram_bytes / (1024 * 1024)

    # Voxel
    total_voxels = width * height * num_images

    # Formatando em HTML
    info_text = (
        f"<b>GEOMETRIA</b><br>"
        f"  Dimensões: {width} x {height} px<br>"
        f"  Fatias (Z): {num_images} imagens<br>"
        f"  Total Estimado de Voxels (1px/img)s: {total_voxels:,} vx<br><br>" # A vírgula formata milhar (1,000,000)
        
        f"<b>ARQUIVO</b><br>"
        f"  Disco Total: {total_size_mb:.2f} MB<br>"
        f"  Espaço por Imagem: {(total_size_kb/num_images):.2f} KB<br>"
        f"  Canais de cor: {channels}<br>"
        f"  Profundidade: {bit_depth}<br><br>"
        
        f"<b>PERFORMANCE</b><br>"
        f" RAM Estimada: <span style='color:blue'>{raw_ram_mb:.2f} MB</span><br>"
        f"<i>(Tamanho descomprimido na memória)</i>"
    )
    return info_text
    
def apply_grayscale(stack_dir):
    """
    Converte todas as imagens para Escala de Cinza (1 canal).
    Reduz o tamanho em memória por 3.
    """
    folder = Path(stack_dir)
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    files = sorted([f for f in folder.iterdir() if f.suffix.lower() in valid_extensions])
    
    count = 0
    for file_path in files:
        img = cv2.imread(str(file_path))
        if img is not None:
            # Converte BGR para GRAY
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Sobrescreve o arquivo
            cv2.imwrite(str(file_path), gray)
            count += 1
    return count

def generate_gif(stack_dir, output_path, duration_ms=100):
    """
    Gera um GIF animado a partir das imagens na pasta.
    duration: tempo entre frames em milissegundos.
    """
    folder = Path(stack_dir)
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    files = sorted([f for f in folder.iterdir() if f.suffix.lower() in valid_extensions])
    
    if not files:
        print("Nenhuma imagem encontrada na pasta para gerar o GIF.")
        return False

    images = []
    for file_path in files:
        img = Image.open(file_path)
        images.append(img)

    if images:
        images[0].save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0
        )
        print(f"GIF salvo em: {output_path}")
    else:
        return False
    return True