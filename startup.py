from pathlib import Path
import shutil, sys

def clean_folder(folder):
    """ Remove todos os arquivos em uma pasta """
    for item in folder.iterdir():
        try:
            if item.is_file():
                item.unlink()  # Remove o arquivo
            elif item.is_dir():
                shutil.rmtree(item)  # Remove a pasta e todo o seu conteúdo
        except Exception as e:
            print(f"Erro ao limpar {item}: {e}")

def startup():
    ### Define o caminho base
    base_path = Path.home() / "Documents" / "Rec3D"

    # Caminho de origem (Onde o script/exe está)
    # Roda .py e o .exe
    if getattr(sys, 'frozen', False):
        # Se for .exe, a origem é a pasta interna
        source_path = Path(sys._MEIPASS)
    else:
        # Se for script normal, a origem é a pasta atual do arquivo ESSA PARTE PODE SER REMOVIDA COM O .EXE!
        source_path = Path(__file__).parent

    ### Estrutura de Diretórios
    folders = [
        base_path,                 # Raiz
            base_path / "backup",
            base_path / "example_stack",
            base_path / "image_stack",
            base_path / "temp",
            base_path / "assets"
    ]

    ### Criação de diretórios em C:/user/Documents/Rec3D
    for folder in folders:
        try:
            # parents=True: Cria pastas pai se faltarem (ex.: cria Documents se não existir)
            # exist_ok=True: NÃO dá erro se a pasta já existir
            folder.mkdir(parents=True, exist_ok=True)
            print(f"Verificao: {folder}")
            
        except PermissionError:
            print(f"ERRO: Sem permissão para criar {folder}")
        except Exception as e:
            print(f"Erro: {e}")

    ### Limpeza de diretórios
    toClean = [folders[1],folders[3],folders[4]] # backup, image_stack, temp
    for folder in toClean:
        clean_folder(folder)
    print('Pastas limpas.')

    ### Transferência de stack de calibração "example_stack"
    # Só copia se a origem existir
    transfer_task = [
        ('example_stack', folders[2], True),
        ('assets', folders[5], True)
    ]

    print("--- Verificando imagens de exemplo ---")

    for src_name, dest_folder, force_overwrite in transfer_task:
        src_folder = source_path / src_name
        
        # Só tentamos copiar se a pasta de origem existir no seu projeto
        if src_folder.exists():
            # Verifica se precisa copiar
            is_empty = not any(dest_folder.glob("*.*"))
            
            if is_empty or force_overwrite:
                action = "Atualizando assets" if force_overwrite else "Copiando exemplos"
                print(f"{action} em: {dest_folder.name}...")
                
                # Copia todos os PNGs (e JPGs se tiver)
                for file in src_folder.glob("*.*"):
                    # Filtro simples para não copiar lixo de sistema (opcional)
                    if file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.ico']:
                        try:
                            shutil.copy2(file, dest_folder / file.name)
                        except Exception as e:
                            print(f"Erro ao copiar {file.name}: {e}")
            else:
                print(f"Pulado: {dest_folder.name} já contém arquivos.")
        else:
            print(f"AVISO: Pasta de origem '{src_name}' não encontrada no programa.")

    return list(folders)


if __name__ == "__main__":
    startup()