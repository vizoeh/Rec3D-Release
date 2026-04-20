import csv
import numpy as np
from image_processing import get_dim

def load_csv_data(filepath):
    """ 
    Lê um CSV e retorna um dicionário {filename: value}.
    Ignora erros de formatação e foca em ler linhas com pelo menos 2 colunas.
    """
    data = {}
    try:
        with open(filepath, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                # Precisa ter pelo menos Nome e Valor
                if len(row) >= 2:
                    key = row[0].strip()
                    val_str = row[1].strip()
                    try:
                        # Tenta converter para float, se falhar ignora
                        data[key] = float(val_str)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Erro ao ler CSV: {e}")
    return data

def save_csv_data(filepath, data_list):
    """ 
    Salva uma lista de tuplas [(filename, value), ...] em um arquivo CSV.
    """
    try:
        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Opcional: Escrever cabeçalho
            writer.writerow(["Arquivo", "Valor"]) 
            writer.writerows(data_list)
        return True
    except Exception as e:
        print(f"Erro ao salvar CSV: {e}")
        return False

def calculate(values, strategy, fac=1):
    """
    Versão 100% vetorizada usando NumPy - até 10x mais rápida.
    """
    import numpy as np
    
    n = len(values)
    if n == 0: 
        return []

    pos = np.array(values, dtype=float) * float(fac)
    thickness = np.zeros(n)

    if strategy == 'troughNext':
        # Vetor de diferenças
        thickness[:-1] = np.abs(np.diff(pos))
        # Último elemento extrapola
        thickness[-1] = thickness[-2] if n > 1 else 1.0 * fac

    elif strategy == 'centerAvg':
        # Primeira camada
        if n > 1:
            thickness[0] = np.abs(pos[1] - pos[0]) / 2
        else:
            thickness[0] = 1.0 * fac
        
        # Camadas do meio (vetorizado)
        if n > 2:
            thickness[1:-1] = np.abs(pos[2:] - pos[:-2]) / 2
        
        # Última camada
        if n > 1:
            thickness[-1] = np.abs(pos[-1] - pos[-2]) / 2

    elif strategy == 'troughPrevious':
        # Primeiras diferenças
        thickness[1:] = np.abs(np.diff(pos))
        # Primeiro elemento extrapola
        thickness[0] = thickness[1] if n > 1 else 1.0 * fac

    else:
        print(f'Estratégia "{strategy}" desconhecida')
        return []

    # Converte para inteiros e cria tuplas
    thickness_px = np.maximum(1, np.round(thickness)).astype(int)
    starts = np.insert(np.cumsum(thickness_px[:-1]), 0, 0)
    
    return list(zip(starts, thickness_px))

def generate_report(data, reference_path =None):
    """
    Gera um relatório HTML com estatísticas do volume calculado.
    
    Args:
        data (list): Lista de tuplas [(start, size), ...]
    
    Returns:
        str: String formatada em HTML.
    """
    if not data:
        return "<i>Sem dados calculados.</i>"

    width,height = 0,0
    if reference_path:
        width, height = get_dim(reference_path)

    # 1. Extração de Estatísticas
    count = len(data)
    
    # Extrai apenas as espessuras (o segundo item da tupla)
    sizes = [item[1] for item in data]
    
    # Altura Total = Onde começa a última + O tamanho da última
    # (Ou simplesmente a soma de todas as espessuras)
    total_z = sum(sizes)
    
    min_z = min(sizes)
    max_z = max(sizes)
    avg_z = sum(sizes) / count

    voxel_str = "N/A"
    
    if width > 0 and height > 0:
        total_voxels = width * height * total_z
        
        # Formatação bonita (Bilhões/Milhões)
        if total_voxels > 1_000_000_000:
            voxel_str = f"{total_voxels / 1_000_000_000:.2f} Bilhões"
        elif total_voxels > 1_000_000:
            voxel_str = f"{total_voxels / 1_000_000:.2f} Milhões"
        else:
            voxel_str = f"{total_voxels:,}".replace(",", ".")

    # 2. Formatação HTML
    # Usamos CSS inline simples para deixar bonito
    html = f"""
    <h3 style="color: #2c3e50; margin-bottom: 5px;">Relatório de Geometria</h3>
    
    <table width="100%" cellspacing="2">
        <tr>
            <td><b>Camadas Processadas:</b></td>
            <td align="right">{count}</td>
        </tr>
        <tr>
            <td><b>Altura Total (Z):</b></td>
            <td align="right" style="color: #2980b9;"><b>{total_z} px</b></td>
        </tr>
        <tr>
            <td><b>Total Estimado de Voxels (Z):</b></td>
            <td align="right" style="color: #2980b9;"><b>{voxel_str} vx</b></td>
        </tr>
    </table>
    
    <hr style="border: 0; border-top: 1px solid #ccc; margin: 5px 0;">
    
    <div style="font-size: 11px; color: #555;">
        <b>Análise de Espessura:</b><br>
        • Mínima: <b>{min_z}</b> px<br>
        • Máxima: <b>{max_z}</b> px<br>
        • Média:  {avg_z:.2f} px
    </div>
    """
    
    # 3. Validação Simples (Alerta Visual)
    if min_z <= 0:
        html += '<br><b style="color: red;">⚠ ALERTA: Camadas com espessura 0 ou negativa detectadas!</b>'
    elif max_z > avg_z * 6:
        html += '<br><b style="color: orange;">⚠ AVISO: Variação brusca de espessura detectada.</b>'

    return html

if __name__ == "__main__":
    print(calculate([12.7, 11.6, 11.5, 11.4, 11.2, 11, 10.8, 10.5, 10, 9.5], 'centerAvg', 10))