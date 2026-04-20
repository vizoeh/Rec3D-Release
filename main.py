from startup import startup
from image_processing import run_stack_equalization, apply_downsampling, apply_blur, get_statistics, apply_grayscale, apply_inversion
from depth_processing import load_csv_data, save_csv_data, calculate, generate_report
from volume_processing import VolumeManager

import os, sys, shutil, ctypes
from pathlib import Path

import pyvista as pv
import numpy as np

from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
                             QListWidget, QListWidgetItem, QFileDialog, QLabel, QFrame,
                             QMainWindow, QStackedWidget, QAction, QToolBar, QApplication, 
                             QSpinBox, QTextEdit, QDialog, QMessageBox, QSizePolicy, 
                             QSizeGrip, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, 
                             QGraphicsRectItem, QTableWidget, QHeaderView, QCheckBox,
                             QTableWidgetItem, QRadioButton, QButtonGroup, QGroupBox, QComboBox,
                             QDoubleSpinBox)
from PyQt5.QtCore import pyqtSignal, QSize, Qt, QRectF, QPointF, QEvent
from PyQt5.QtGui import QIcon, QPixmap, QMovie, QPen, QColor, QBrush, QPainter
from pyvistaqt import QtInteractor
from superqt import QRangeSlider

####### IMPORTAÇÃO E PROCEESSAMENTO DE IMAGENS #########

class ImportWidget(QWidget):
    folder_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        
        ### Caminho para cópia de arquvios (sandbox)
        self.stack_dir = Path.home() / "Documents" / "Rec3D" / "image_stack"
        self.backup_dir = Path.home() / "Documents" / "Rec3D" / "backup"
        self.temp_dir = Path.home() / "Documents" / "Rec3D" / "temp"
        self.selected_dir = None
        
        # Tela em esquerda e Direita
        self.layout = QHBoxLayout(self)
        self.setup_ui()

        self.imageList.itemDoubleClicked.connect(self.open_viewer)

    def setup_ui(self):
        ######### Esquerda: Imagens e importação #########
        left_layout = QVBoxLayout()
        
        # Galeria de imagens
        self.imageList = QListWidget()
        self.imageList.setViewMode(QListWidget.IconMode) # Modo Galeria
        self.imageList.setIconSize(QSize(150, 150))      # Tamanho da miniatura
        self.imageList.setResizeMode(QListWidget.Adjust) # Reorganiza ao redimensionar
        self.imageList.setSpacing(17)                    # Espaço entre itens
        left_layout.addWidget(self.imageList)

        # Botão de carregar
        self.loadButton = QPushButton("Selecionar Pasta com Stack de Imagens")
        self.loadButton.setToolTip("Para o bom funcionamento do programa, as imagens devem ser nomeadas com o sufixo 00, 01, 02...")
        self.loadButton.clicked.connect(self.select_folder)
        self.loadButton.setMinimumHeight(50)
        left_layout.addWidget(self.loadButton)

        self.layout.addLayout(left_layout, stretch=3) # Ocupa 3 partes da tela

        ######### Direita: Painel de Opções #########
        self.options_panel = QFrame()
        self.options_panel.setFrameShape(QFrame.StyledPanel)
        self.options_layout = QVBoxLayout(self.options_panel)
        
        ### Recarregar e voltar

        self.refresh_reload_layout = QHBoxLayout()

        # Botão de recarregar imagens
        self.refreshButton = (QPushButton('Recarregar imagens', clicked=self.refresh_display))
        self.refresh_reload_layout.addWidget(self.refreshButton)

        # Botão de Voltar
        self.reloadButton = (QPushButton('Original'))
        self.reloadButton.clicked.connect(lambda: self.upload_images(self.selected_dir))
        self.reloadButton.setEnabled(False)
        self.refresh_reload_layout.addWidget(self.reloadButton)

        self.options_layout.addLayout(self.refresh_reload_layout)

        ### Texto
        self.infoBox = QTextEdit()
        self.infoBox.setReadOnly(True)
        self.infoBox.setMaximumHeight(200)
        self.infoBox.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; font-size: 11px;")
        self.infoBox.setHtml("<i>Nenhuma imagem carregada.</i>")
        self.options_layout.addWidget(self.infoBox)

         ### Título

        self.options_layout.addWidget(QLabel("<b>Processamento de Imagem</b>"))

        ### Botão de Equalização
        self.equalization_layout = QVBoxLayout()
        self.equalization_layout.addWidget(QLabel("Equalização:"))

        self.equalizationButton = QPushButton("Selecione a referência")
        self.equalizationButton.setEnabled(False)
        self.equalizationButton.setToolTip("Equaliza os canais de cor das imagens importadas com base em uma imagem selecionada")
        self.equalizationButton.clicked.connect(self.run_equalization) # Conecta o botão a função
        self.equalization_layout.addWidget(self.equalizationButton)
        self.options_layout.addLayout(self.equalization_layout)

        ### Botão de escala de cinza
        self.grayscaleLabel = QLabel("Escala de Cinza:")

        self.grayscaleButton = QPushButton("Converter para Escala de Cinza")
        self.grayscaleButton.setToolTip("Converte todas as imagens para escala de cinza (1 canal). Reduz o tamanho em memória por 3.")
        self.grayscaleButton.clicked.connect(self.run_grayscale)
        self.grayscaleButton.setEnabled(False)

        self.options_layout.addWidget(self.grayscaleLabel)
        self.options_layout.addWidget(self.grayscaleButton)

        ### Botão de Inversão
        self.invertLabel = QLabel("Inversão de Cores:")

        self.invertButton = QPushButton("Inverter Cores")
        self.invertButton.setToolTip("Inverte as cores das imagens (efeito negativo).")
        self.invertButton.setEnabled(False)
        self.invertButton.clicked.connect(self.run_invert)

        self.options_layout.addWidget(self.invertLabel)
        self.options_layout.addWidget(self.invertButton)

        ### Botão de downsampling
        self.downsample_layout = QVBoxLayout()
        self.downsample_layout.addWidget(QLabel("Downsampling:"))

        self.downsample_buttons_layout = QHBoxLayout()

        self.downsample_50Button = QPushButton("50%")
        self.downsample_50Button.setToolTip("Reduz as imagens para 50% do tamanho original")
        self.downsample_50Button.setEnabled(False)

        self.downsample_25Button = QPushButton("25%")
        self.downsample_25Button.setToolTip("Reduz as imagens para 25% do tamanho original")
        self.downsample_25Button.setEnabled(False)

        self.restoreButton = QPushButton("Restaurar")
        self.restoreButton.setToolTip("Restaura as imagens para o estado original antes do downsampling")
        self.restoreButton.setEnabled(False) # Desativado por enquanto

        self.downsample_buttons_layout.addWidget(self.downsample_50Button)
        self.downsample_buttons_layout.addWidget(self.downsample_25Button)
        self.downsample_buttons_layout.addWidget(self.restoreButton)
        self.downsample_layout.addLayout(self.downsample_buttons_layout)
        self.options_layout.addLayout(self.downsample_layout)

        ### Blur Spinbox

        self.blur_layout = QVBoxLayout()
        self.blur_layout.addWidget(QLabel("Desfoque:"))

        self.blur_layout_buttons_layout = QHBoxLayout()
        self.blurSpinbox = QSpinBox()
        self.blurSpinbox.setRange(3, 11)
        self.blurSpinbox.setSingleStep(2) 
        self.blurSpinbox.setValue(3)
        self.blurSpinbox.setSuffix(" px")
        self.blurSpinbox.setToolTip("Define o tamanho do kernel para o filtro utilizar. Útil para remover ruídos da imagem.")
        self.blur_layout_buttons_layout.addWidget(self.blurSpinbox)

        self.blurButton = QPushButton("Aplicar")
        self.blurButton.clicked.connect(self.run_blur)
        self.blurButton.setEnabled(False)
        self.blur_layout_buttons_layout.addWidget(self.blurButton)

        self.blur_layout.addLayout(self.blur_layout_buttons_layout)
        self.options_layout.addLayout(self.blur_layout)

        ###### GIF button
        self.inspect_layout = QHBoxLayout()
        self.gifButton = QPushButton("Gerar GIF Animado")
        self.gifButton.setIcon(QIcon.fromTheme("image-x-generic"))
        self.gifButton.setToolTip("Exporta as imagens como um GIF animado")
        self.gifButton.setEnabled(False)
        self.gifButton.clicked.connect(self.open_gif_dialog)
        self.inspect_layout.addWidget(self.gifButton)
      
        self.btn_visual_crop = QPushButton(" Definir Recorte")
        # Ícone de tesoura ou crop ajuda na UX
        self.btn_visual_crop.setIcon(QIcon.fromTheme("transform-crop")) 
        self.btn_visual_crop.setToolTip("Abre a primeira imagem para desenhar a área de corte.")
        self.btn_visual_crop.clicked.connect(self.open_crop_dialog)
        self.btn_visual_crop.setEnabled(False) # Começa travado
        
        self.inspect_layout.addWidget(self.btn_visual_crop)

        self.options_layout.addLayout(self.inspect_layout)

        # Conecta os botões às funções
        self.imageList.itemSelectionChanged.connect(self.toggle_equalization_button) # Quando a seleçã da lista mudar, chame toggle_equalization_button
        self.downsample_50Button.clicked.connect(lambda: self.downsample(0.5))
        self.downsample_25Button.clicked.connect(lambda: self.downsample(0.25))
        self.restoreButton.clicked.connect(self.restoreBackup)
 
        self.options_layout.addStretch() # Empurra tudo para cima

        self.layout.addWidget(self.options_panel, stretch=1) # Ocupa 1 parte

    def run_equalization(self):
        selection = self.imageList.selectedItems()
        if not selection: # Se não há nada selecionado, pare
            return
        else:
            source = selection[0].toolTip() 
            print(f"Equalizando com base em: {source}")
            run_stack_equalization(self.stack_dir, source)
            self.refresh_display()

    def open_gif_dialog(self):
        dialog = GifExportDialog(self.stack_dir, self.temp_dir, self)
        dialog.exec_()

    def open_crop_dialog(self):
        """ Abre a primeira imagem do stack para definir o corte. """
        # Pega o caminho da primeira imagem da lista
        if self.imageList.count() == 0: return
        first_image_path = self.imageList.item(0).toolTip()
        
        # Cria o dialog
        self.crop_dialog = CropSelectionDialog(first_image_path, self)
        
        # CONEXÃO CRÍTICA: Conecta o sinal do dialog à nossa função de execução
        # Quando o dialog emitir 'crop_confirmed', a função 'execute_crop_from_dialog' roda.
        self.crop_dialog.crop_confirmed.connect(self.execute_crop_from_dialog)
        
        # Abre a janela em modo modal (trava o fundo)
        self.crop_dialog.exec_()

    def execute_crop_from_dialog(self, x, y, w, h):
        """ Função chamada automaticamente (SLOT) quando o usuário confirma o corte visual. """
        print(f"Recebido comando de corte: X={x}, Y={y}, W={w}, H={h}")
        
        # Feedback visual na janela principal
        self.btn_visual_crop.setText("Processando Recorte...")
        self.btn_visual_crop.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # Força a interface a atualizar para mostrar o texto "Processando..." antes de travar
        QApplication.processEvents() 

        try:  
            # 2. Executa o corte (usando a mesma função de antes)
            from image_processing import apply_crop
            count = apply_crop(self.stack_dir, x, y, w, h)
            
            # 3. Atualiza a tela
            if count > 0:
                QMessageBox.information(self, "Sucesso", f"Recorte aplicado em {count} imagens.")
                self.refresh_display()
            else:
                QMessageBox.warning(self, "Atenção", "O recorte falhou ou estava fora dos limites da imagem.")

        except Exception as e:
            print(f"Erro crítico no recorte: {e}")
            QMessageBox.critical(self, "Erro", f"Falha no processamento:\n{e}")
            
        finally:
            # Restaura o botão
            self.btn_visual_crop.setText(" Definir Recorte Visualmente...")
            self.btn_visual_crop.setEnabled(True)
            QApplication.restoreOverrideCursor()

    def run_blur(self):
        ksize = self.blurSpinbox.value()

        # Visual
        self.blurButton.setText("Aplicando...")
        self.blurButton.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            apply_blur(self.stack_dir, ksize)
            self.refresh_display()
        except Exception as e:
            print(f"Erro ao aplicar desfoque: {e}")
        finally:
            QApplication.restoreOverrideCursor()
            self.blurButton.setText("Aplicar")
            self.blurButton.setEnabled(True)

    def open_viewer(self, item):
        """ Abre a imagem clicada em tela cheia, com zoom e pan"""
        image_path = item.toolTip()
        if image_path:
            viewer = ImageInspectionDialog(image_path, self)
            viewer.exec_()

    def toggle_equalization_button(self):
        # Verifica se existe algum item selecionado na lista
        if len(self.imageList.selectedItems()) > 0:
            self.equalizationButton.setEnabled(True)  # Destrava
            self.equalizationButton.setText("Aplicar")
        else:
            self.equalizationButton.setEnabled(False) # Trava de novo
            self.equalizationButton.setText("Selecione a referência")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecione a Pasta")
        if folder: # String vazia (Cancelar) = False
            self.selected_dir = folder
            self.upload_images(folder)
            self.folder_selected.emit(folder)

    def upload_images(self, source_folder):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            valid_images = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
            source_path = Path(source_folder)

            ### Excluir arquivos de Rec3D/image_stack e .\backup
            try:
                if self.stack_dir.exists():
                    for file in self.stack_dir.iterdir():
                        if file.is_file():
                            file.unlink()
            except Exception as e:
                print(f"Erro ao limpar pasta de stack: {e}")
            
            try:
                if self.backup_dir.exists():
                    for file in self.backup_dir.iterdir():
                        if file.is_file():
                            file.unlink()
            except Exception as e:
                print(f"Erro ao limpar pasta de backup: {e}")
            
            ### Copiar para Rec3D/image_stack
            try:
                # Ordenamos os arquvios para manter a sequência
                files = sorted([f for f in source_path.iterdir() if f.is_file()])

                for file in files:
                    if file.suffix.lower() in valid_images:
                        
                        # Define o destino (Rec3D/image_stack)
                        dest_file = self.stack_dir / file.name
                        
                        # Copia eles
                        shutil.copy2(file, dest_file)
                        
                # Se quiser avisar o resto do programa que terminou:
                # self.folder_selected.emit(str(self.stack_dir))

            except Exception as e:
                print(f"Erro ao processar imagens: {e}")

            self.refresh_display()
        finally:
            QApplication.restoreOverrideCursor()

    def refresh_display(self):     
        self.imageList.clear() # Apaga os displays
        valid_images = {".png", ".jpg", ".jpeg", ".bmp", ".tif"} 
        try:
            if not self.stack_dir.exists():
                return
            
            files = sorted([f for f in self.stack_dir.iterdir() if f.is_file()])

            for file in files:
                if file.suffix.lower() in valid_images:
                    item = QListWidgetItem(file.name)
                    
                    # QPixMap entende strings, não Paths
                    pixmap = QPixmap(str(self.stack_dir / file.name)) 
                    if not pixmap.isNull():
                        pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        
                        item.setIcon(QIcon(pixmap))
                        item.setToolTip(str(self.stack_dir / file.name)) # Tooltip mostra o caminho novo
                        
                        self.imageList.addItem(item)
            
            try:
                if self.stack_dir.exists() and any(self.stack_dir.iterdir()):
                    stats_html = get_statistics(self.stack_dir)
                    self.infoBox.setHtml(stats_html)
                else:
                    self.infoBox.setHtml("<i>Nenhuma imagem carregada.</i>")
            except Exception as e:
                print(f"Erro ao obter estatísticas: {e}")
                self.infoBox.setHtml("<i>Erro ao obter estatísticas das imagens.</i>")

            has_images = self.imageList.count() > 0
            self.downsample_50Button.setEnabled(has_images)
            self.downsample_25Button.setEnabled(has_images)
            self.reloadButton.setEnabled(has_images)
            self.blurButton.setEnabled(has_images)
            self.grayscaleButton.setEnabled(has_images)
            self.gifButton.setEnabled(has_images)
            self.btn_visual_crop.setEnabled(has_images)
            self.invertButton.setEnabled(has_images)
        except Exception as e:
            print(f"Erro ao atualizar display: {e}")

    def protect_backup(self):
        """
        Verifica se um backup já foi feito (imagens na pasta backup).
        Se não, copia os arquivos de image_stack para backup.
        """

        if any(self.backup_dir.iterdir()):
            # Já tem arquivos lá, não fazer nada
            return # Sai da função

        print('Criando backup de image_stack')
        try:
            for file in self.stack_dir.iterdir():
                if file.is_file():
                    shutil.copy2(file, self.backup_dir / file.name)
            print("Backup concluído com sucesso!")
            self.restoreButton.setEnabled(True) # Ativa o botão após o backup

        except Exception as e:
            print(f"Erro ao criar backup: {e}")

    def run_invert(self):
        self.invertButton.setText("Invertendo...")
        self.invertButton.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            # Processamento
            count = apply_inversion(self.stack_dir)
            
            # 3. Atualiza tela
            print(f"Inversão aplicada em {count} imagens.")
            self.refresh_display()
            
        except Exception as e:
            print(f"Erro na inversão: {e}")
            QMessageBox.critical(self, "Erro", f"Falha ao inverter cores:\n{e}")
            
        finally:
            self.invertButton.setText("Inverter Cores")
            self.invertButton.setEnabled(True)
            QApplication.restoreOverrideCursor()

    def restoreBackup(self):
        """
        Restaura os arquivos de backup para image_stack.
        Sobrescreve os arquivos existentes em image_stack.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            try:
                for file in self.backup_dir.iterdir():
                    if file.is_file():
                        shutil.copy2(file, self.stack_dir / file.name)
                        file.unlink() # Remove o arquivo de backup após restaurar
                print("Restauração concluída com sucesso!")
                self.refresh_display()
                self.restoreButton.setEnabled(False) # Desativa o botão após restaurar
            except Exception as e:
                print(f"Erro ao restaurar backup: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def downsample(self, factor):
        """
        Faz backup de image_stack se necessário.
        Aplica downsampling a todas as imagens na pasta image_stack.
        Sobrescreve as imagens existentes.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.protect_backup()
            apply_downsampling(self.stack_dir, factor)
            self.refresh_display()
        finally:
            QApplication.restoreOverrideCursor()

    def run_grayscale(self):
        """
        Converte todas as imagens na pasta image_stack para escala de cinza.
        Sobrescreve as imagens existentes.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.grayscaleButton.setText("Convertendo...")
        self.grayscaleButton.setEnabled(False)
        try:
            apply_grayscale(self.stack_dir)
            self.refresh_display()
        finally:
            QApplication.restoreOverrideCursor()
            self.grayscaleButton.setText("Converter para Escala de Cinza")
            self.grayscaleButton.setEnabled(True)

class GifExportDialog(QDialog):
    def __init__(self, stack_dir, temp_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Exportar GIF Animado")
        self.resize(500, 600)
        
        self.stack_dir = stack_dir
        self.temp_dir = temp_dir
        self.preview_path = self.temp_dir / "preview.gif"
        
        # Layout Principal
        layout = QVBoxLayout(self)
        
        # 1. Área de Visualização (Onde o GIF vai tocar)
        self.lbl_preview = QLabel("Gerando Preview...")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_preview.setStyleSheet("background-color: #222; color: white;")
        layout.addWidget(self.lbl_preview)
        
        # 2. Controles
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Velocidade (FPS):"))
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 60)
        self.spin_fps.setValue(10) # Padrão 10 FPS
        self.spin_fps.setSuffix(" fps")
        self.spin_fps.valueChanged.connect(self.generate_preview) # Regenera se mudar velocidade
        controls_layout.addWidget(self.spin_fps)
        
        controls_layout.addStretch()
        
        self.btn_save = QPushButton("Salvar GIF no Computador")
        self.btn_save.clicked.connect(self.save_file)
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        controls_layout.addWidget(self.btn_save)
        
        layout.addLayout(controls_layout)
        
        # Gera o primeiro preview ao abrir
        self.generate_preview()

    def generate_preview(self):
        fps = self.spin_fps.value()
        duration_ms = int(1000 / fps) # Converte FPS para milissegundos (ex: 10fps = 100ms)
        
        self.lbl_preview.setText("Gerando...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            from image_processing import generate_gif
            success = generate_gif(self.stack_dir, self.preview_path, duration_ms)
            
            if success:
                self.show_animation()
            else:
                self.lbl_preview.setText("Erro ao gerar GIF.")
                
        except Exception as e:
            self.lbl_preview.setText(f"Erro: {e}")
            print(e)
            
        QApplication.restoreOverrideCursor()

    def show_animation(self):
        # Usa QMovie para tocar o GIF na interface
        self.movie = QMovie(str(self.preview_path))

        self.movie.jumpToFrame(0) # Começa do primeiro frame
        self.gif_original_size = self.movie.currentImage().size()

        self.lbl_preview.setMovie(self.movie)
        self.movie.start()

    def save_file(self):
        # Abre diálogo para salvar
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Salvar GIF", "stack.gif", "GIF Files (*.gif)", options=options) # Retorna o file_path e o filtro selecionado
        
        if file_path:
            try:
                shutil.copy2(self.preview_path, file_path)

                QMessageBox.information(self, "Sucesso", f"GIF salvo em:\n{file_path}")
                self.accept() # Fecha a janela
            except Exception as e:
                print(f"Erro ao salvar: {e}")

    def resizeEvent(self, event):
        """Chamado automaticamente sempre que o diálogo é redimensionado."""
        # Chama a implementação padrão primeiro
        super().resizeEvent(event)
        
        # Verifica se temos o filme e se já capturamos o tamanho original
        if hasattr(self, 'movie') and self.movie and hasattr(self, 'gif_original_size'):
            
            # 1. Qual o tamanho disponível no label agora?
            available_size = self.lbl_preview.size()
            
            # 2. Calcula o novo tamanho mantendo a proporção.
            # O método .scaled() do QSize faz a matemática difícil para nós.
            # Qt.KeepAspectRatio é a chave aqui.
            new_size = self.gif_original_size.scaled(available_size, Qt.KeepAspectRatio)
            
            # 3. Aplica o novo tamanho calculado ao filme
            self.movie.setScaledSize(new_size)

class ZoomGraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        # Configurações padrão de UX
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)

    def wheelEvent(self, event):
        """
        Sobrescreve o evento da roda do mouse.
        """
        # Zoom In/Out baseado na direção da roda
        if event.angleDelta().y() > 0:
            factor = 1.15
        else:
            factor = 1 / 1.15
        
        self.scale(factor, factor)

        event.accept()

class ImageInspectionDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Inspeção: {Path(image_path).name}")
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        self.scene = QGraphicsScene()
        pixmap = QPixmap(str(image_path))
        self.scene.addItem(QGraphicsPixmapItem(pixmap))
        
        # Usa a sua classe ZoomGraphicsView que já está no código
        self.view = ZoomGraphicsView(self.scene)
        layout.addWidget(self.view)

class CropSelectionDialog(QDialog):
    """ Dialog para visualizar imagem e definir região de corte interativamente. """
    # Sinal que envia as coordenadas (x, y, w, h) de volta para a janela principal
    crop_confirmed = pyqtSignal(int, int, int, int)

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Definir Recorte: {Path(image_path).name}")
        self.resize(900, 700)
        
        # --- ESTADOS INTERNOS ---
        self.start_point = None # Onde o clique começou
        self.current_rect_item = None # O retângulo visual vermelho
        self.is_drawing = False
        
        layout = QVBoxLayout(self)

        # --- CENA E VIEW ---
        self.scene = QGraphicsScene()
        
        # Carrega imagem
        self.pixmap = QPixmap(str(image_path))
        self.image_item = QGraphicsPixmapItem(self.pixmap)
        self.scene.addItem(self.image_item)
        # Define o tamanho da cena igual ao da imagem para facilitar coordenadas
        self.scene.setSceneRect(QRectF(self.pixmap.rect())) 

        # Configura a View
        # Usamos QGraphicsView padrão, pois vamos controlar os eventos manualmente
        self.view = ZoomGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)

        self.view.setDragMode(QGraphicsView.NoDrag) # Desativa o Pan padrão para podermos desenhar
        self.view.setCursor(Qt.CrossCursor) # Cursor de mira para indicar modo de desenho
        
        # Instala o filtro de eventos para interceptar o mouse na view
        self.view.viewport().installEventFilter(self)

        layout.addWidget(self.view)
        
        # Barra inferior com instruções
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("<i>Instruções: Clique e arraste na imagem para desenhar a área de recorte.</i>"))
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        info_layout.addStretch()
        info_layout.addWidget(btn_cancel)
        layout.addLayout(info_layout)

    def eventFilter(self, source, event):
        """ Intercepta eventos do mouse na View para desenhar o retângulo. """
        if source == self.view.viewport():
            
            # --- CORREÇÃO AQUI ---
            # Não calculamos scene_pos aqui fora, pois eventos como 'Paint' não têm .pos()
            
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                # Agora sabemos que é um clique, então é seguro pedir a posição
                scene_pos = self.view.mapToScene(event.pos())
                self.start_drawing(scene_pos)
                return True # Consumimos o evento

            elif event.type() == QEvent.MouseMove and self.is_drawing:
                # Agora sabemos que é movimento do mouse
                scene_pos = self.view.mapToScene(event.pos())
                self.update_drawing(scene_pos)
                return True

            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self.is_drawing:
                # Agora sabemos que soltou o botão
                scene_pos = self.view.mapToScene(event.pos())
                self.finish_drawing(scene_pos)
                return True
                
        return super().eventFilter(source, event)

    def start_drawing(self, scene_pos):
        self.is_drawing = True
        self.start_point = scene_pos
        
        # Remove retângulo anterior se existir
        if self.current_rect_item:
            self.scene.removeItem(self.current_rect_item)
            self.current_rect_item = None

        # Cria o novo retângulo visual (Vermelho, semi-transparente)
        self.current_rect_item = QGraphicsRectItem()
        pen = QPen(QColor(255, 0, 0), 2) # Borda vermelha grossa
        brush = QBrush(QColor(255, 0, 0, 50)) # Preenchimento vermelho transparente
        self.current_rect_item.setPen(pen)
        self.current_rect_item.setBrush(brush)
        # Define o retângulo inicial com tamanho 0
        self.current_rect_item.setRect(QRectF(self.start_point, self.start_point))
        self.scene.addItem(self.current_rect_item)

    def update_drawing(self, scene_pos):
        # Calcula o retângulo entre o ponto inicial e o atual
        # .normalized() garante que funciona mesmo arrastando para trás/cima
        rect = QRectF(self.start_point, scene_pos).normalized()
        self.current_rect_item.setRect(rect)

    def finish_drawing(self, scene_pos):
        self.is_drawing = False
        
        # Obtém o retângulo final
        final_rect = self.current_rect_item.rect().normalized()
        
        # Converte para inteiros (pixels)
        x = int(final_rect.x())
        y = int(final_rect.y())
        w = int(final_rect.width())
        h = int(final_rect.height())

        # Validação mínima (evita cliques acidentais minúsculos)
        if w < 5 or h < 5:
            self.scene.removeItem(self.current_rect_item)
            return

        # Pergunta ao usuário se confirma
        msg = f"Confirmar área de recorte?\nX={x}, Y={y}\nLargura={w}, Altura={h}"
        resp = QMessageBox.question(self, "Confirmar Recorte", msg, QMessageBox.Yes | QMessageBox.No)

        if resp == QMessageBox.Yes:
            # Dispara o sinal com os números e fecha a janela
            self.crop_confirmed.emit(x, y, w, h)
            self.accept()
        else:
            # Se cancelar, só remove o retângulo visual e deixa tentar de novo
            self.scene.removeItem(self.current_rect_item)

####### PROFUNDIDADES E CONFIGURAÇÕES SIMILARES #######

class DepthWidget(QWidget):

    depth_data = pyqtSignal(list)

    def __init__(self):
        super().__init__()

        self.stack_dir = Path.home() / "Documents" / "Rec3D" / "image_stack"
        self.build_dir = Path.home() / "Documents" / "Rec3D" / "assets"
        self.calculated_data = []

        self.setup_ui()
        self.update_diagram()
        self.connect_change()

    def setup_ui(self):
        main_layout = QHBoxLayout(self)

        ###### ESQUERDA: TABELA #######
        table_layout = QVBoxLayout()

        self.temp = QLabel("Tela de Profundidade- Em Construção")


        ### Tabela
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Imagem", "Nome" ,"Valor", "Start (px)", "Size (px)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.setColumnWidth(0,100)
        self.table.verticalHeader().setDefaultSectionSize(100)
        self.table.setIconSize(QSize(90, 90))

        table_layout.addWidget(self.table)
        ### Botão Importar/Exportar CSV
        self.csvImpButton = QPushButton("Importar Valores (CSV)")
        self.csvImpButton.clicked.connect(self.import_csv)
        table_layout.addWidget(self.csvImpButton)

        self.csvExpButton = QPushButton("Exportar Valores (CSV)")
        self.csvExpButton.clicked.connect(self.export_csv)
        table_layout.addWidget(self.csvExpButton)


        main_layout.addLayout(table_layout, stretch=2)

        ###### DIREITA: PAINEL DE OPÇÕES ######
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_layout = QVBoxLayout(control_panel)

        control_layout.addWidget(QLabel("<b>Configurações de Profundidade</b>"))

        # Espaçaento Uniforme

        self.uniformGroup = QGroupBox('Modo Automático (uniforme):')
        uniform_layout = QHBoxLayout()

        self.uniformCheck = QCheckBox("Ativar modo Uniforme")
        self.uniformCheck.setChecked(False)

        self.uniformSpinbox = QSpinBox()
        self.uniformSpinbox.setRange(1, 100)
        self.uniformSpinbox.setValue(1)
        self.uniformSpinbox.setSuffix(" px")
        self.uniformSpinbox.setEnabled(False)
        self.uniformCheck.toggled.connect(self.toggle_uniform)

        uniform_layout.addWidget(self.uniformCheck)
        uniform_layout.addWidget(self.uniformSpinbox)

        # Calcular
        self.calculateButton = QPushButton("Calcular")
        self.calculateButton.clicked.connect(self.calculate_uniform)
        uniform_layout.addWidget(self.calculateButton)

        self.uniformGroup.setLayout(uniform_layout)
        control_layout.addWidget(self.uniformGroup)

        main_layout.addWidget(control_panel, stretch=2)

        ####################### OPÇÕES NÃO UNIFORMES.

        ### Direção

        self.directionGroup = QGroupBox("Direção de Empilhamento")
        dir_layout = QHBoxLayout()

        self.groupDirButton = QButtonGroup(self)

        self.incRadio = QRadioButton('Crescente (Z+)')
        self.incRadio.setToolTip('Ex:0, 10, 20...')
        self.incRadio.setChecked(True)

        self.decRadio = QRadioButton("Decrescente (Z-)")
        self.decRadio.setToolTip("Ex: 100, 90, 80...")

        self.groupDirButton.addButton(self.incRadio)
        self.groupDirButton.addButton(self.decRadio)

        dir_layout.addWidget(self.incRadio)
        dir_layout.addWidget(self.decRadio)

        self.directionGroup.setLayout(dir_layout)
        control_layout.addWidget(self.directionGroup)

        ### Estratégia de Interpretação

        self.strategyGroup = QGroupBox("Estratégia de Interpretação:")
        strategy_layout = QVBoxLayout()
        strategy_button_layout = QHBoxLayout()

        self.combo_strategy = QComboBox()
        # Adiciona item com Texto Visual e Dado Interno (UserRole)
        self.combo_strategy.addItem("Até Seguinte", "troughNext")
        self.combo_strategy.addItem("Média Central", "centerAvg")
        self.combo_strategy.addItem("Até Anterior", "troughPrevious")
        self.combo_strategy.setToolTip("Define como o valor da posição se relaciona com a fatia física.")
        self.combo_strategy.currentIndexChanged.connect(self.update_diagram)
        strategy_button_layout.addWidget(self.combo_strategy, stretch=3)

        #Fac
        self.facLabel = QLabel("Fator Z:")
        self.facLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.facLabel.setToolTip('Multiplicador para converter o valor do CSV em pixels.\nEx: Se CSV está em mm e 1mm = 10px, use 10.0')
        self.facSpin = QDoubleSpinBox()
        self.facSpin.setRange(0.01, 10000.0) # Permite de micro a macro
        self.facSpin.setDecimals(2)            # Precisão de 4 casas
        self.facSpin.setSingleStep(0.1)        # Passo da setinha
        self.facSpin.setValue(1.0)             # Valor padrão
        self.facSpin.setPrefix("x ")           # Sufixo visual

        self.facSpin.valueChanged.connect(self.on_modification)
        strategy_button_layout.addWidget(self.facLabel, stretch= 1) # Stretch menor
        strategy_button_layout.addWidget(self.facSpin, stretch = 1)

        strategy_layout.addLayout(strategy_button_layout)


        # Diagrama
        self.diagramLabel = QLabel()
        self.diagramLabel.setAlignment(Qt.AlignCenter)
        self.diagramLabel.setMinimumHeight(120)
        #self.diagramLabel.setStyleSheet("border: 1px solid #ccc; background-color: white; border-radius: 4px;")
        strategy_layout.addWidget(self.diagramLabel)

        self.strategyGroup.setLayout(strategy_layout)
        control_layout.addWidget(self.strategyGroup)

        # Salvar para geração de imagens.

        self.saveButton = QPushButton('Salvar')
        self.saveButton.clicked.connect(self.save_depth)
        control_layout.addWidget(self.saveButton)

        self.infoBox = QTextEdit()
        self.infoBox.setReadOnly(True)
        self.infoBox.setMaximumHeight(200)
        self.infoBox.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; font-size: 11px;")
        self.infoBox.setHtml("<i>Salve as configurações antes.</i>")
        control_layout.addWidget(self.infoBox)

        control_layout.addStretch()

 # Empurra tudo para cima
    def get_data(self):
        """
        Varre a table linha por linha, extraindo os dados.
        Retorna duas listas:
            - filenames: nomes dos arquivos (str)
            - values: valores (float)
        """

        filenames = []
        values = []

        rows = self.table.rowCount()

        if not rows: return []

        for i in range(rows):
            item_name = self.table.item(i, 1) # Nome
            item_value = self.table.item(i, 2) # Valor

            try:
                val = float(item_value.text().replace(',','.'))
            except ValueError:
                val = 0.0
        
            filenames.append(item_name.text())
            values.append(val)

        return filenames, values

    def table_feedback(self, data):
        self.table.blockSignals(True)

        for i, (start, thick) in enumerate(data):
            if i < self.table.rowCount():
                # Coluna 3: Onde começa
                item_start = QTableWidgetItem(str(start))
                item_start.setFlags(item_start.flags() ^ Qt.ItemIsEditable) # Read-only
                item_start.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, 3, item_start)

                # Coluna 4: Espessura
                item_thick = QTableWidgetItem(str(thick))
                item_thick.setFlags(item_thick.flags() ^ Qt.ItemIsEditable)
                item_thick.setTextAlignment(Qt.AlignCenter)
                
                self.table.setItem(i, 4, item_thick)
        
        self.table.setHorizontalHeaderLabels(["Imagem", "Nome", "Valor", "Start (px)", "Size (px)"])
        self.table.blockSignals(False)

    def toggle_uniform(self, checked):
        self.uniformSpinbox.setEnabled(checked)
        self.table.setEnabled(not checked)

        if checked:
            self.incRadio.setChecked(True)
            self.directionGroup.setEnabled(False)
        else: self.directionGroup.setEnabled(True)

    def calculate_uniform(self):
        uniform = self.uniformCheck.isChecked()
        step = self.uniformSpinbox.value()

        if not self.stack_dir.exists(): return

        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
        files = sorted([f.name for f in self.stack_dir.iterdir() if f.suffix.lower() in valid_extensions])

        if uniform:
            for i in range(len(files)):
                val = i * step
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i,2, item)
        
        self.on_modification()

    def connect_change(self):
        self.table.itemChanged.connect(self.on_modification)

        self.groupDirButton.buttonClicked.connect(self.on_modification)
        self.combo_strategy.currentIndexChanged.connect(self.on_modification)

        #self.saveButton.setStyleSheet("background-color: #E74C3C; color: white; font-weight: bold;")

    def on_modification(self):
        self.saveButton.setEnabled(True)
        self.saveButton.setText("Salvar *")

    def save_depth(self):
        try:
            filenames, values = self.get_data()
        except Exception as e:
            print(f"ERRO EM get_data: {e}")
            return
        
        if not values: return

        direction = "inc" if self.incRadio.isChecked() else 'dec'
        strategy = self.combo_strategy.currentData()

        #Spinbox fator
        try:
            self.calculated_data = calculate(values, strategy, float(self.facSpin.value()))
            self.table_feedback(self.calculated_data)
            self.depth_data.emit(self.calculated_data)
            print('Dados de camada salvos.')

            ref_image = None
            if self.stack_dir.exists():
                valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
                # Pega o primeiro arquivo válido encontrado
                ref_image = next(
                    (f for f in self.stack_dir.iterdir() if f.suffix.lower() in valid_exts), 
                    None
                )

            report_html = generate_report(self.calculated_data, ref_image )
            self.infoBox.setHtml(report_html)

            self.saveButton.setText("Salvo")
            self.saveButton.setEnabled(False)
            self.on_modification()
            #self.saveButton.setStyleSheet("background-color: #2ECC71; color: white; font-weight: bold;") # Verde
        except Exception as e:
            QMessageBox.critical(self, "Erro de Cálculo", f"Falha ao processar geometria:\n{e}")

    def load_table(self):
        """ 
        Lê a pasta image_stack e atualiza a tabela.
        Versão Debug: Imprime no console o que encontrou e força a cor do texto.
        """
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
        
        if not self.stack_dir.exists(): 
            print("AVISO: Pasta image_stack não existe.")
            return

        # 1. BACKUP: Salva valores digitados
        saved_values = {}
        rows = self.table.rowCount()
        for i in range(rows):
            item_name = self.table.item(i, 1)
            item_val = self.table.item(i, 2)
            if item_name and item_val:
                saved_values[item_name.text()] = item_val.text()

        # 2. LISTAGEM DO DISCO
        try:
            disk_files = sorted([f.name for f in self.stack_dir.iterdir() if f.suffix.lower() in valid_extensions])
            print(f"Debug: Encontrados {len(disk_files)} arquivos na pasta.")
        except Exception as e:
            print(f"Erro ao listar arquivos: {e}")
            return

        # 3. CONFIGURA TABELA
        self.table.blockSignals(True)
        self.table.setRowCount(len(disk_files))

        # 4. RECONSTRUÇÃO (Com proteção de cor e validação)
        for i, filename in enumerate(disk_files):
            fPath = str(self.stack_dir / filename)
            
            # --- COLUNA 0: IMAGEM ---
            try:
                pixmap = QPixmap(fPath)
                if not pixmap.isNull():
                    lbl_img = QLabel()
                    scaled_pixmap = pixmap.scaled(90, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    lbl_img.setPixmap(scaled_pixmap)
                    lbl_img.setAlignment(Qt.AlignCenter)
                    lbl_img.setStyleSheet("background-color: transparent;")
                    self.table.setCellWidget(i, 0, lbl_img)
            except Exception as e:
                print(f"Erro ao carregar imagem {filename}: {e}")

            # --- COLUNA 1: NOME (Blindado) ---
            item_name = QTableWidgetItem(str(filename))
            
            # Força a cor preta para garantir que não é texto branco em fundo branco
            item_name.setForeground(QBrush(QColor("black"))) 
            
            # Define flags explicitamente (Ativado + Selecionável) em vez de XOR
            item_name.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item_name.setTextAlignment(Qt.AlignCenter)
            
            self.table.setItem(i, 1, item_name)
            
            # --- COLUNA 2: VALOR ---
            val_to_restore = saved_values.get(filename, "0.0")
            item_val = QTableWidgetItem(val_to_restore)
            item_val.setTextAlignment(Qt.AlignCenter)
            
            # Força cor preta aqui também
            item_val.setForeground(QBrush(QColor("black")))
            
            self.table.setItem(i, 2, item_val)

        self.table.blockSignals(False)
        self.saveButton.setEnabled(True)
        self.saveButton.setText("Salvar")

    def import_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Importar CSV", "", "CSV Files (*.csv);;All Files (*)")

        if not file_path: return

        if self.uniformCheck.isChecked():
            self.uniformCheck.setChecked(False)

        data_dict = load_csv_data(file_path)

        if not data_dict:
            QMessageBox.warning(self, "Aviso", "Não foi possível ler dados válidos deste CSV.")
            return
        
        rows = self.table.rowCount()

        for i in range(rows):
            #
            item_name =  self.table.item(i, 1)

            if item_name:
                fname = item_name.text()
                if fname in data_dict:
                    new_val = data_dict[fname]

                    item_val = QTableWidgetItem(str(new_val))
                    item_val.setTextAlignment(Qt.AlignCenter)

                    self.table.setItem(i, 2, item_val)
        
        self.on_modification()

    def export_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Exportar CSV", "depth_data.csv", "CSV Files (*.csv)")

        if not file_path: return

        filenames, values = self.get_data()

        data_to_save = list(zip(filenames, values))

        success = save_csv_data(file_path, data_to_save)

        if success:
            QMessageBox.information(self, "Sucesso", f"Arquivo salvo em:\n{file_path}")
        else:
            QMessageBox.critical(self, "Erro", "Falha ao gravar o arquivo. Verifique permissões.")

    def update_diagram(self):
        strategy = self.combo_strategy.currentData()

        filename = f'strategy_{strategy}.png'
        filepath = self.build_dir / filename

        if filepath.exists():
            pixmap = QPixmap(str(filepath))

            if pixmap.width() > 280: pixmap = pixmap.scaledToWidth(280, Qt.SmoothTransformation)

            self.diagramLabel.setPixmap(pixmap)
            self.diagramLabel.setText("")
        else:
            self.diagramLabel.setText(f"Imagem não encontrada:\n{filename}")
            self.diagramLabel.setStyleSheet("border: 2px dashed red; color: red;")

####### VOLUME. #######

class ScaleManager:
    """
    Gerencia escalas independentes para X, Y, Z.
    Usado para representação científica correta.
    """
    def __init__(self):
        # Escalas em px/unidade (quantos pixels = 1 unidade real)
        self.scale_x = 1.0  
        self.scale_y = 1.0
        self.scale_z = 1.0
        
        # Unidades
        self.unit_x = "px"
        self.unit_y = "px"
        self.unit_z = "px"
    
    def set_scale(self, axis, pixels_per_unit, unit_name):
        """
        Define escala para um eixo.
        
        Args:
            axis: 'x', 'y' ou 'z'
            pixels_per_unit: Quantos pixels = 1 unidade (ex: 10px = 1mm)
            unit_name: Nome da unidade ('mm', 'μm', 'cm', etc)
        """
        axis = axis.lower()
        if axis == 'x':
            self.scale_x = float(pixels_per_unit)
            self.unit_x = unit_name
        elif axis == 'y':
            self.scale_y = float(pixels_per_unit)
            self.unit_y = unit_name
        elif axis == 'z':
            self.scale_z = float(pixels_per_unit)
            self.unit_z = unit_name
    
    def pixels_to_real(self, pixels, axis):
        """Converte pixels em unidades reais (unidade/px)"""
        axis = axis.lower()
        if axis == 'x':
            return pixels * self.scale_x
        elif axis == 'y':
            return pixels * self.scale_y
        elif axis == 'z':
            return pixels * self.scale_z
        return pixels
    
    def get_scale_factors(self):
        """
        Retorna fatores de escala para aplicar no PyVista.
        
        Returns:
            tuple: (scale_x, scale_y, scale_z) normalizados
        """
        # Normaliza pela menor escala
        min_scale = min(self.scale_x, self.scale_y, self.scale_z)
        
        return (
            self.scale_x / min_scale,
            self.scale_y / min_scale,
            self.scale_z / min_scale
        )
    
    def format_distance(self, dist_x, dist_y, dist_z):
        """
        Formata distância 3D com unidades corretas.
        
        Args:
            dist_x, dist_y, dist_z: Distâncias em pixels
        
        Returns:
            dict com distâncias formatadas
        """
        real_x = self.pixels_to_real(dist_x, 'x')
        real_y = self.pixels_to_real(dist_y, 'y')
        real_z = self.pixels_to_real(dist_z, 'z')
        
        # Distância euclidiana só faz sentido se unidades forem iguais
        if self.unit_x == self.unit_y == self.unit_z:
            euclidean = np.sqrt(real_x**2 + real_y**2 + real_z**2)
            euclidean_str = f"{euclidean:.3f} {self.unit_x}"
        else:
            euclidean_str = "N/A (unidades diferentes)"
        
        return {
            'x': f"{real_x:.3f} {self.unit_x}",
            'y': f"{real_y:.3f} {self.unit_y}",
            'z': f"{real_z:.3f} {self.unit_z}",
            'euclidean': euclidean_str
        }

class MeasurementTool:
    """
    Permite medir distância entre dois pontos clicados no PyVista.
    """
    def __init__(self, plotter, scale_manager):
        self.plotter = plotter
        self.scale_manager = scale_manager
        self.points = []
        self.line_actor = None
        self.text_actor = None
        self.sphere_actors = []
        self.measuring = False
    
    def start_measurement(self):
        """Inicia modo de medição"""
        self.points = []
        self.measuring = True
        self.clear_measurement()
        
        self.plotter.enable_surface_point_picking(
            callback=self.on_point_picked,
            show_message=True,
            color='red',
            point_size=10
        )
        
        print("Modo medição ativado. Clique 2 pontos no modelo.")
    
    def on_point_picked(self, point):
        """Callback quando usuário clica um ponto"""
        if not self.measuring or len(self.points) >= 2:
            return
        
        self.points.append(point)
        
        # Adiciona esfera no ponto
        sphere = pv.Sphere(radius=2.0, center=point)
        actor = self.plotter.add_mesh(
            sphere, 
            color='red',
            name=f'measure_point_{len(self.points)}'
        )
        self.sphere_actors.append(actor)
        
        print(f"Ponto {len(self.points)}: {point}")
        
        # Se temos 2 pontos, desenha linha e calcula distância
        if len(self.points) == 2:
            self.finish_measurement()
    
    def finish_measurement(self):
        """Finaliza medição e mostra resultado"""
        p1, p2 = self.points
        
        # Cria linha
        line = pv.Line(p1, p2)
        self.line_actor = self.plotter.add_mesh(
            line,
            color='yellow',
            line_width=3,
            name='measure_line'
        )
        
        # Calcula distâncias
        dist_x = abs(p2[0] - p1[0])
        dist_y = abs(p2[1] - p1[1])
        dist_z = abs(p2[2] - p1[2])
        
        # Formata com unidades
        result = self.scale_manager.format_distance(dist_x, dist_y, dist_z)
        
        # Cria texto no meio da linha
        mid_point = [(p1[i] + p2[i])/2 for i in range(3)]
        
        text = (
            f"ΔX: {result['x']}\n"
            f"ΔY: {result['y']}\n"
            f"ΔZ: {result['z']}\n"
            f"Distância: {result['euclidean']}"
        )
        
        # Adiciona label 3D
        self.text_actor = self.plotter.add_point_labels(
            [mid_point],
            [text],
            point_size=0,
            font_size=12,
            text_color='yellow',
            name='measure_text'
        )
        
        print("\n" + "="*50)
        print("MEDIÇÃO COMPLETA")
        print("="*50)
        print(text)
        print("="*50)
        
        self.measuring = False
        self.plotter.disable_picking()
    
    def clear_measurement(self):
        """Remove medição anterior"""
        # Remove linha
        if self.line_actor:
            self.plotter.remove_actor(self.line_actor)
            self.line_actor = None
        
        # Remove texto
        if self.text_actor:
            self.plotter.remove_actor(self.text_actor)
            self.text_actor = None
        
        # Remove esferas
        for actor in self.sphere_actors:
            self.plotter.remove_actor(actor)
        self.sphere_actors = []
        
        self.points = []

class VolumeWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.path = Path.home() / 'Documents' / 'Rec3D' / 'image_stack'
        self.memory = None
        self.measurement_tool = None

        self.plane_meshes = {}
        self._last_box_dims = None
        
        # --- 1. Backend Validado ---
        self.manager = VolumeManager()
        
        # Dicionário para gerenciar os atores (telas 3D)
        self.actors = {} 
        
        # Estado dos cortes (0 a 100%)
        self.cut_ranges = {
            'x': (0, 100), 
            'y': (0, 100), 
            'z': (0, 100)
        }

        # ======================================================== Estado do modo de visualização
        self.visualization_mode = 'planes'  # 'planes' ou 'voxels'
        
        # Parâmetros do modo voxel
        self.voxel_threshold = 128
        self.voxel_resolution = 1
        self.voxel_limit = 500000
        
        # Cache de voxels
        self.voxel_actor = None
        self.voxel_mesh = None
        # =======================================================

        # --- 2. Interface Gráfica ---
        self.main_layout = QHBoxLayout(self)
        
        # A. Visualizador 3D
        self.frame_3d = QFrame()
        self.layout_3d = QVBoxLayout(self.frame_3d)
        self.plotter = QtInteractor(self.frame_3d)
        self.layout_3d.addWidget(self.plotter.interactor)
        
        # Configuração Visual
        self.plotter.set_background("white")
        self.plotter.show_axes()
        
        # B. Painel Lateral
        self.controls_panel = QFrame()
        self.controls_panel.setFixedWidth(250)
        self.controls_layout = QVBoxLayout(self.controls_panel)

        # --- NOVO BOTÃO DE ATUALIZAR ---
        # Ele chama a função que relê a pasta image_stack
        self.btn_reload = QPushButton("↻ Atualizar Visualização")
        self.btn_reload.setToolTip("Recarrega as imagens da pasta caso tenham sido alteradas.")
        self.btn_reload.clicked.connect(self.reload_volume)
        self.controls_layout.addWidget(self.btn_reload)
        self.controls_layout.addSpacing(15)
        # -------------------------------
        
        # Sliders Duplos (SuperQt)
        self.sliderGroup = QGroupBox('Cortes')
        self.slider_layout = QVBoxLayout()

        self.create_slider("Corte X", 'x')
        self.create_slider("Corte Y", 'y')
        self.create_slider("Corte Z", 'z')

        self.sliderGroup.setLayout(self.slider_layout)
        self.controls_layout.addWidget(self.sliderGroup)

        # Interpolação
        self.interpolGroup = QGroupBox('Interpolação')
        interpol_layout = QHBoxLayout()

        self.interpolCheck = QCheckBox('OFF')
        self.interpolCheck.setChecked(False)
        interpol_layout.addWidget(self.interpolCheck)
        self.interpolCheck.stateChanged.connect(self.sync_interpol)

        self.interpolCombo = QComboBox()
        self.interpolCombo.addItem("Zoom (Recomendado)", "zoom")
        self.interpolCombo.addItem("Adaptativo", "adaptive")
        self.interpolCombo.addItem("Cúbico", "cubic")
        self.interpolCombo.setEnabled(False)
        self.interpolCombo.currentTextChanged.connect(self.sync_interpol)
        interpol_layout.addWidget(self.interpolCombo)


        self.interpolGroup.setLayout(interpol_layout)
        self.controls_layout.addWidget(self.interpolGroup)


        # ========================================================
        # --- GRUPO VOXELIZADO ---
        self.voxelGroup = QGroupBox('Modo Voxelizado')
        voxel_layout = QVBoxLayout()

        # Botão ativar/desativar
        self.btn_voxel_mode = QPushButton("Ativar Modo Voxel")
        self.btn_voxel_mode.setCheckable(True)
        self.btn_voxel_mode.setToolTip("Apenas disponível para imagens em escala de cinza")
        self.btn_voxel_mode.clicked.connect(self.toggle_voxel_mode)
        self.btn_voxel_mode.setEnabled(False)
        voxel_layout.addWidget(self.btn_voxel_mode)

        # Threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(128)
        self.threshold_spin.valueChanged.connect(self.on_voxel_param_changed)
        threshold_layout.addWidget(self.threshold_spin)
        voxel_layout.addLayout(threshold_layout)

        # Resolução
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("Resolução:"))
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(1, 8)
        self.resolution_spin.setValue(2)
        self.resolution_spin.setSuffix("px")
        self.resolution_spin.setToolTip("Tamanho do voxel (1=máxima, 4=baixa)")
        self.resolution_spin.valueChanged.connect(self.on_voxel_param_changed)
        resolution_layout.addWidget(self.resolution_spin)
        voxel_layout.addLayout(resolution_layout)

        # Limite
        limit_layout = QHBoxLayout()
        limit_layout.addWidget(QLabel("Limite:"))
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(10000, 2000000)
        self.limit_spin.setSingleStep(50000)
        self.limit_spin.setValue(500000)
        limit_layout.addWidget(self.limit_spin)
        voxel_layout.addLayout(limit_layout)

        # Info label
        self.voxel_info_label = QLabel("<i>Ative o modo para ver stats</i>")
        self.voxel_info_label.setWordWrap(True)
        self.voxel_info_label.setStyleSheet("font-size: 10px;")
        voxel_layout.addWidget(self.voxel_info_label)

        self.voxelGroup.setLayout(voxel_layout)
        self.controls_layout.addWidget(self.voxelGroup)

        #########################################################
        # --- CONTROLE DE ESCALA ---
        self.scaleGroup = QGroupBox('Escala Científica')
        scale_layout = QVBoxLayout()
        
        # X
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.scale_x_spin = QDoubleSpinBox()
        self.scale_x_spin.setRange(0.001, 10000.0)
        self.scale_x_spin.setValue(1.0)
        self.scale_x_spin.setDecimals(3)
        self.scale_x_spin.setSuffix(" unit/px")
        self.scale_x_spin.valueChanged.connect(self.on_scale_changed)
        x_layout.addWidget(self.scale_x_spin)
        
        self.unit_x_combo = QComboBox()
        self.unit_x_combo.addItems(["px", "mm", "μm", "cm", "nm"])
        self.unit_x_combo.currentTextChanged.connect(self.on_scale_changed)
        x_layout.addWidget(self.unit_x_combo)
        scale_layout.addLayout(x_layout)
        
        # Y
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.scale_y_spin = QDoubleSpinBox()
        self.scale_y_spin.setRange(0.001, 10000.0)
        self.scale_y_spin.setValue(1.0)
        self.scale_y_spin.setDecimals(3)
        self.scale_y_spin.setSuffix(" unit/px")
        self.scale_y_spin.valueChanged.connect(self.on_scale_changed)
        y_layout.addWidget(self.scale_y_spin)
        
        self.unit_y_combo = QComboBox()
        self.unit_y_combo.addItems(["px", "mm", "μm", "cm", "nm"])
        self.unit_y_combo.currentTextChanged.connect(self.on_scale_changed)
        y_layout.addWidget(self.unit_y_combo)
        scale_layout.addLayout(y_layout)
        
        # Z
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z:"))
        self.scale_z_spin = QDoubleSpinBox()
        self.scale_z_spin.setRange(0.001, 10000.0)
        self.scale_z_spin.setValue(1.0)
        self.scale_z_spin.setDecimals(3)
        self.scale_z_spin.setSuffix(" unit/px")
        self.scale_z_spin.valueChanged.connect(self.on_scale_changed)
        z_layout.addWidget(self.scale_z_spin)
        
        self.unit_z_combo = QComboBox()
        self.unit_z_combo.addItems(["px", "mm", "μm", "cm", "nm"])
        self.unit_z_combo.currentTextChanged.connect(self.on_scale_changed)
        z_layout.addWidget(self.unit_z_combo)
        scale_layout.addLayout(z_layout)
        
        # Botão aplicar
        #self.apply_scale_btn = QPushButton("Aplicar Escala ao Modelo")
        #self.apply_scale_btn.clicked.connect(self.apply_scale_to_model)
        #scale_layout.addWidget(self.apply_scale_btn)
        
        self.scaleGroup.setLayout(scale_layout)
        self.controls_layout.addWidget(self.scaleGroup)

        #######################################################################

        self.measureGroup = QGroupBox('Medição de Distância')
        measure_layout = QVBoxLayout()
        
        self.btn_measure = QPushButton("📏 Medir Distância (2 cliques)")
        self.btn_measure.clicked.connect(self.start_measurement)
        measure_layout.addWidget(self.btn_measure)
        
        self.btn_clear_measure = QPushButton("Limpar Medição")
        self.btn_clear_measure.clicked.connect(self.clear_measurement)
        measure_layout.addWidget(self.btn_clear_measure)
        
        self.measure_result = QLabel("<i>Clique no botão acima para iniciar</i>")
        self.measure_result.setWordWrap(True)
        self.measure_result.setStyleSheet("font-size: 10px; padding: 5px;")
        measure_layout.addWidget(self.measure_result)
        
        self.measureGroup.setLayout(measure_layout)
        self.controls_layout.addWidget(self.measureGroup)

        ########################################################################
        # Desabilita controles inicialmente
        self.threshold_spin.setEnabled(False)
        self.resolution_spin.setEnabled(False)
        self.limit_spin.setEnabled(False)
        # =====================================
        
        self.controls_layout.addStretch()

        self.main_layout.addWidget(self.frame_3d, stretch=4)
        self.main_layout.addWidget(self.controls_panel, stretch=1)

        self.scale_manager = ScaleManager()

    def sync_interpol(self):
        is_active = self.interpolCheck.isChecked()
        self.interpolCombo.setEnabled(is_active)

        if not is_active:
            self.manager.set_interpolation('none')
            self.interpolCheck.setText('OFF')
        else:
            selected_mode = self.interpolCombo.currentData()
            self.manager.set_interpolation(selected_mode)
            self.interpolCheck.setText('ON')

    def create_slider(self, label_text, axis_key):
        lbl = QLabel(f"<b>{label_text}</b>")
        self.slider_layout.addWidget(lbl)
        
        slider = QRangeSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue((0, 100))
        
        # Lambda para identificar qual eixo mudou
        slider.valueChanged.connect(lambda val: self.update_range(val, axis_key))
        
        self.slider_layout.addWidget(slider)
        self.slider_layout.addSpacing(15)

    def reload_volume(self):
        """Lê o disco novamente e atualiza a textura sem mexer na câmera"""
        if not self.path:
            print("Nenhuma pasta definida para atualizar.")
            return

        print(f"Recarregando imagens de: {self.path}")
        
        # Força o Backend a reler os arquivos do disco
        success = self.manager.load_stack(self.path)
        
        if success:
            # Atualiza a geometria (caso o número de imagens tenha mudado) e as texturas
            if self.memory is not None:
                self.manager.set_layer_data(self.memory)
                        # Verifica grayscale e habilita botão voxel
        if self.manager.is_grayscale:
            self.btn_voxel_mode.setEnabled(True)
        else:
            self.btn_voxel_mode.setEnabled(False)
            if self.btn_voxel_mode.isChecked():
                self.btn_voxel_mode.setChecked(False)
                self.toggle_voxel_mode()
        
        # Atualiza visualização correta
        if self.visualization_mode == 'planes':
            self.update_volume_view()
        else:
            self.update_voxel_view()
            self.update_volume_view()
            print("Visualização 3D sincronizada com o disco.")

    def update_range(self, value, axis):
        """Atualiza estado interno e redesenha"""
        self.cut_ranges[axis] = value 
        self.update_volume_view()

    def receive_geometry_data(self, data):
        """Slot: Recebe dados da Aba 2"""
        print("VolumeWidget: Recebendo geometria...")
        self.memory = data
        self.manager.set_layer_data(data)
        self.update_volume_view()

    def update_volume_view(self):
        """
        Versão SEM PISCAR: Atualiza apenas as texturas, mantendo os atores vivos.
        
        COMO FUNCIONA:
        1. Na primeira execução: Cria os 6 planos e salva referências
        2. Nas próximas: Apenas atualiza as texturas dos planos existentes
        3. Se dimensões mudarem: Recria tudo (caso raro)
        """
        
        if self.manager.volume is None: 
            return

        # Desliga renderização temporária para evitar frames parciais
        self.plotter.pause_render = True
        
        try:
            # 1. Dimensões do Volume
            total_z_pixels = self.manager.total_z_height
            d, h, w = self.manager.dims[:3]
            
            # 2. Lê Sliders (0 a 100)
            xm, xM = self.cut_ranges['x']
            ym, yM = self.cut_ranges['y']
            zm, zM = self.cut_ranges['z']

            # 3. Converte para índices (com clamping)
            ix_min = max(0, int(xm/100 * w))
            ix_max = min(w, int(xM/100 * w))
            
            iy_min = max(0, int(ym/100 * h))
            iy_max = min(h, int(yM/100 * h))
            
            iz_min = max(0, int(zm/100 * total_z_pixels))
            iz_max = min(total_z_pixels, int(zM/100 * total_z_pixels))

            # Garante pelo menos 1 pixel
            if ix_max <= ix_min: ix_max = ix_min + 1
            if iy_max <= iy_min: iy_max = iy_min + 1
            if iz_max <= iz_min: iz_max = iz_min + 1

            # 4. Dimensões da caixa
            box_w = ix_max - ix_min
            box_h = iy_max - iy_min
            box_d = iz_max - iz_min

            # 5. Centro da caixa
            cx = (ix_min + ix_max) / 2
            cy = (iy_min + iy_max) / 2
            cz = (iz_min + iz_max) / 2

            # =================================================================
            # DECISÃO: CRIAR OU ATUALIZAR?
            # =================================================================
            
            # Verifica se já existem planos criados E se as dimensões não mudaram
            needs_rebuild = (
                not hasattr(self, 'plane_meshes') or 
                not self.plane_meshes or
                getattr(self, '_last_box_dims', None) != (box_w, box_h, box_d)
            )
            
            if needs_rebuild:
                # PRIMEIRA VEZ ou MUDANÇA DE DIMENSÃO → Recria tudo
                self._rebuild_planes(cx, cy, cz, box_w, box_h, box_d)
            
            # SEMPRE atualiza texturas (rápido!)
            self._update_textures(ix_min, ix_max, iy_min, iy_max, iz_min, iz_max, xm, xM, ym, yM, zm, zM)
            
            # Salva dimensões para próxima comparação
            self._last_box_dims = (box_w, box_h, box_d)
            
        finally:
            # Religa renderização
            self.plotter.pause_render = False
            self.plotter.render()

# ===================================================================
# MÉTODO AUXILIAR 1: Criar planos (chamado raramente)
# ===================================================================

    def _rebuild_planes(self, cx, cy, cz, box_w, box_h, box_d):
        """
        Recria todos os 6 planos do zero.
        Chamado apenas quando:
        - É a primeira vez
        - As dimensões da caixa mudaram (usuário moveu os sliders para tamanhos diferentes)
        """
        print("Recriando geometria dos planos...")
        
        # Remove atores antigos se existirem
        if hasattr(self, 'actors'):
            for actor in self.actors.values():
                self.plotter.remove_actor(actor)
        
        self.actors = {}
        self.plane_meshes = {}
        
        # Configuração dos 6 planos
        planes_config = {
            'Top': {
                'center': (cx, cy, cz - box_d/2),
                'direction': (0, 0, 1),
                'i_size': box_w,
                'j_size': box_h
            },
            'Bottom': {
                'center': (cx, cy, cz + box_d/2),
                'direction': (0, 0, -1),
                'i_size': box_w,
                'j_size': box_h
            },
            'Left': {
                'center': (cx + box_w/2, cy, cz),
                'direction': (-1, 0, 0),
                'i_size': box_d,
                'j_size': box_h
            },
            'Right': {
                'center': (cx - box_w/2, cy, cz),
                'direction': (1, 0, 0),
                'i_size': box_d,
                'j_size': box_h
            },
            'Front': {
                'center': (cx, cy + box_h/2, cz),
                'direction': (0, 1, 0),
                'i_size': box_d,
                'j_size': box_w
            },
            'Back': {
                'center': (cx, cy - box_h/2, cz),
                'direction': (0, -1, 0),
                'i_size': box_d,
                'j_size': box_w
            }
        }
        
        # Cria os planos (sem textura ainda)
        for name, config in planes_config.items():
            plane = pv.Plane(
                center=config['center'],
                direction=config['direction'],
                i_size=config['i_size'],
                j_size=config['j_size']
            )
            
            # Cria textura vazia inicial (1x1 pixel preto)
            dummy_texture = pv.numpy_to_texture(np.zeros((1, 1, 3), dtype=np.uint8))
            
            # Adiciona ao plotter
            actor = self.plotter.add_mesh(
                plane,
                texture=dummy_texture,
                show_scalar_bar=False,
                lighting=False,
                name=name,
                reset_camera=False
            )
            
            # Salva referências
            self.actors[name] = actor
            self.plane_meshes[name] = plane

# ===================================================================
# MÉTODO AUXILIAR 2: Atualizar texturas (chamado SEMPRE, muito rápido)
# ===================================================================

    def _update_textures(self, ix_min, ix_max, iy_min, iy_max, iz_min, iz_max, xm, xM, ym, yM, zm, zM):
        """
        Atualiza apenas as texturas dos planos existentes.
        Esta função é RÁPIDA porque não mexe na geometria.
        """
        
        # Configuração de texturas (qual fatia pegar e como processar)
        texture_config = {
            'Top': {
                'slice': ('z', zM),
                'crop': lambda tex: tex[iy_min:iy_max, ix_min:ix_max],
                'transform': lambda tex: np.rot90(tex, k=2)
            },
            'Bottom': {
                'slice': ('z', zm),
                'crop': lambda tex: tex[iy_min:iy_max, ix_min:ix_max],
                'transform': lambda tex: np.flipud(tex)
            },
            'Left': {
                'slice': ('x', xm),
                'crop': lambda tex: tex[iz_min:iz_max, iy_min:iy_max],
                'transform': lambda tex: np.rot90(np.flipud(tex), k=1)
            },
            'Right': {
                'slice': ('x', xM),
                'crop': lambda tex: tex[iz_min:iz_max, iy_min:iy_max],
                'transform': lambda tex: np.rot90(tex, k=1)
            },
            'Front': {
                'slice': ('y', yM),
                'crop': lambda tex: tex[iz_min:iz_max, ix_min:ix_max],
                'transform': lambda tex: np.rot90(tex, k=1)
            },
            'Back': {
                'slice': ('y', ym),
                'crop': lambda tex: tex[iz_min:iz_max, ix_min:ix_max],
                'transform': lambda tex: np.flipud(np.rot90(tex, k=1))
            }
        }
        
        # Atualiza cada textura
        for name, config in texture_config.items():
            if name not in self.actors:
                continue
            
            try:
                # 1. Pega a fatia
                axis, pct = config['slice']
                texture_array = self.manager.get_slice_texture(axis, pct)
                
                if texture_array is None or texture_array.size == 0:
                    continue
                
                # 2. Recorta
                texture_cropped = config['crop'](texture_array)
                
                if texture_cropped.size == 0:
                    continue
                
                # 3. Aplica transformação (rotação/flip)
                texture_final = config['transform'](texture_cropped)
                
                # 4. ATUALIZA a textura do ator existente
                new_texture = pv.numpy_to_texture(texture_final)
                
                # CRITICAL: Atualiza a propriedade de textura do ator
                actor = self.actors[name]
                actor.SetTexture(new_texture)
                
            except Exception as e:
                print(f"Erro ao atualizar textura {name}: {e}")
                continue

    def update_volume_view(self):
        """Versão corrigida com orientações e posições corretas"""
        if self.manager.volume is None: 
            return

        self.plotter.pause_render = True

        # 1. Limpeza
        for actor in self.actors.values():
            self.plotter.remove_actor(actor)
        self.actors.clear()

        # 2. Dimensões do Volume
        total_z_pixels = self.manager.total_z_height
        d, h, w = self.manager.dims[:3]  # (Depth, Height, Width)
        
        # 3. Ler Sliders (0 a 100)
        xm, xM = self.cut_ranges['x']
        ym, yM = self.cut_ranges['y']
        zm, zM = self.cut_ranges['z']

        # 4. Converter para índices (com clamping para evitar out of bounds)
        # X (Largura)
        ix_min = max(0, int(xm/100 * w))
        ix_max = min(w, int(xM/100 * w))
        
        # Y (Altura da imagem)
        iy_min = max(0, int(ym/100 * h))
        iy_max = min(h, int(yM/100 * h))
        
        # Z (Espessura física)
        iz_min = max(0, int(zm/100 * total_z_pixels))
        iz_max = min(total_z_pixels, int(zM/100 * total_z_pixels))

        # Evita colapso (garante pelo menos 1 pixel)
        if ix_max <= ix_min: ix_max = ix_min + 1
        if iy_max <= iy_min: iy_max = iy_min + 1
        if iz_max <= iz_min: iz_max = iz_min + 1

        # 5. Dimensões da caixa recortada
        box_w = ix_max - ix_min
        box_h = iy_max - iy_min
        box_d = iz_max - iz_min

        # 6. Centro da caixa no espaço 3D
        cx = (ix_min + ix_max) / 2
        cy = (iy_min + iy_max) / 2
        cz = (iz_min + iz_max) / 2

        # =========================================================================
        # RECORTE DAS TEXTURAS COM ORIENTAÇÕES CORRETAS
        # =========================================================================

        planes_config = []

        # --- FACES Z (Topo e Fundo) ---
        # CORREÇÃO: Topo (z=100) deve estar EM CIMA, Fundo (z=0) deve estar EMBAIXO
        
        # Topo (Z máximo) - AGORA EM CIMA
        tex_top = self.manager.get_slice_texture('z', zM)
        if tex_top is not None and tex_top.size > 0:
            tex_top_cropped = tex_top[iy_min:iy_max, ix_min:ix_max]
            
            if tex_top_cropped.size > 0:
                #tex_top_cropped = np.flipud(tex_top_cropped)
                tex_top_cropped = np.rot90(tex_top_cropped, k=2)
                
                plane = pv.Plane(
                    center=(cx, cy, cz - box_d/2),  # POSITIVO = CIMA
                    direction=(0, 0, 1), 
                    i_size=box_w, 
                    j_size=box_h
                )
                planes_config.append(("Top", tex_top_cropped, plane))

        # Fundo (Z mínimo) - AGORA EMBAIXO
        tex_bot = self.manager.get_slice_texture('z', zm)
        if tex_bot is not None and tex_bot.size > 0:
            tex_bot_cropped = tex_bot[iy_min:iy_max, ix_min:ix_max]
            
            if tex_bot_cropped.size > 0:
                tex_bot_cropped = np.flipud(tex_bot_cropped)
                
                plane = pv.Plane(
                    center=(cx, cy, cz + box_d/2),  # NEGATIVO = BAIXO
                    direction=(0, 0, -1), 
                    i_size=box_w, 
                    j_size=box_h
                )
                planes_config.append(("Bottom", tex_bot_cropped, plane))


        # --- FACES X (Esquerda e Direita) ---
        # CORREÇÃO: Direita e Esquerda estavam trocados
        
        # Esquerda (X mínimo) - CORRIGIDO
        tex_left = self.manager.get_slice_texture('x', xm)
        if tex_left is not None and tex_left.size > 0:
            tex_left_cropped = tex_left[iz_min:iz_max, iy_min:iy_max]
            
            if tex_left_cropped.size > 0:
                tex_left_cropped = np.flipud(tex_left_cropped)
                tex_left_cropped = np.rot90(tex_left_cropped, k=1)
                
                plane = pv.Plane(
                    center=(cx + box_w/2, cy, cz),  # NEGATIVO = ESQUERDA
                    direction=(-1, 0, 0), 
                    i_size=box_d, 
                    j_size=box_h
                )
                planes_config.append(("Left", tex_left_cropped, plane))

        # Direita (X máximo) - CORRIGIDO
        tex_right = self.manager.get_slice_texture('x', xM)
        if tex_right is not None and tex_right.size > 0:
            tex_right_cropped = tex_right[iz_min:iz_max, iy_min:iy_max]
            
            if tex_right_cropped.size > 0:
                tex_right_cropped = np.rot90(tex_right_cropped, k=1)
                
                plane = pv.Plane(
                    center=(cx - box_w/2, cy, cz),  # POSITIVO = DIREITA
                    direction=(1, 0, 0), 
                    i_size=box_d, 
                    j_size=box_h
                )
                planes_config.append(("Right", tex_right_cropped, plane))


        # --- FACES Y (Frente e Trás) ---
        # CORREÇÃO: Frente estava rotacionada 90° e invertida, Trás estava rotacionada 90°
        
        # Frente (Y máximo) - CORRIGIDO com rotação
        tex_front = self.manager.get_slice_texture('y', yM)
        if tex_front is not None and tex_front.size > 0:
            tex_front_cropped = tex_front[iz_min:iz_max, ix_min:ix_max]
            
            if tex_front_cropped.size > 0:
                # CORREÇÃO: Rotação de 90° no sentido anti-horário + flipud
                # rot90(k=1) = 90° anti-horário
                tex_front_cropped = np.rot90(tex_front_cropped, k=1)
                
                plane = pv.Plane(
                    center=(cx, cy + box_h/2, cz), 
                    direction=(0, 1, 0), 
                    i_size=box_d, 
                    j_size=box_w
                )
                planes_config.append(("Front", tex_front_cropped, plane))

        # Trás (Y mínimo) - CORRIGIDO com rotação
        tex_back = self.manager.get_slice_texture('y', ym)
        if tex_back is not None and tex_back.size > 0:
            tex_back_cropped = tex_back[iz_min:iz_max, ix_min:ix_max]
            
            if tex_back_cropped.size > 0:
                # CORREÇÃO: Rotação de 90° no sentido horário
                # rot90(k=-1) = 90° horário = rot90(k=3) anti-horário
                tex_back_cropped = np.rot90(tex_back_cropped, k=1)
                tex_back_cropped = np.flipud(tex_back_cropped)
                
                plane = pv.Plane(
                    center=(cx, cy - box_h/2, cz), 
                    direction=(0, -1, 0), 
                    i_size=box_d, 
                    j_size=box_w
                )
                planes_config.append(("Back", tex_back_cropped, plane))


        # 7. Renderização
        for name, texture_array, plane in planes_config:
            try:
                tex = pv.numpy_to_texture(texture_array)
                #tex.InterpolateOn() # <--- Isso ativa a interpolação linear (Bilinear filtering)
                actor = self.plotter.add_mesh(
                    plane, 
                    texture=tex, 
                    show_scalar_bar=False, 
                    lighting=False,
                    name=name,
                    reset_camera = False
                )
                self.actors[name] = actor
                
            except Exception as e:
                print(f"Erro ao renderizar plano {name}: {e}")
                continue
        
        self.plotter.pause_render = False
        self.plotter.render()

# ====================================================================

    def toggle_voxel_mode(self):
        '''Alterna entre modo planos e voxels'''
        if self.btn_voxel_mode.isChecked():
            # ATIVANDO
            self.visualization_mode = 'voxels'
            self.btn_voxel_mode.setText("⬜ Desativar Voxels")
            
            # Habilita controles
            self.threshold_spin.setEnabled(True)
            self.resolution_spin.setEnabled(True)
            self.limit_spin.setEnabled(True)
            
            # Atualiza
            self.update_voxel_view()
            
        else:
            # DESATIVANDO
            self.visualization_mode = 'planes'
            self.btn_voxel_mode.setText("Ativar Modo Voxel")
            
            # Desabilita controles
            self.threshold_spin.setEnabled(False)
            self.resolution_spin.setEnabled(False)
            self.limit_spin.setEnabled(False)
            
            # Remove voxels
            if self.voxel_actor:
                self.plotter.remove_actor(self.voxel_actor)
                self.voxel_actor = None
            
            self.voxel_info_label.setText("<i>Modo desativado</i>")
            self.update_volume_view()

    def on_voxel_param_changed(self):
        '''Callback para mudanças nos parâmetros'''
        if self.visualization_mode == 'voxels':
            self.voxel_threshold = self.threshold_spin.value()
            self.voxel_resolution = self.resolution_spin.value()
            self.voxel_limit = self.limit_spin.value()
            self.update_voxel_view()

    def update_voxel_view(self):
        '''Renderiza voxels'''
        if self.manager.volume is None or not self.manager.is_grayscale:
            return
        
        self.plotter.pause_render = True
        
        try:
            # Remove planos
            for actor in self.actors.values():
                self.plotter.remove_actor(actor)
            self.actors.clear()
            
            # Remove voxel anterior
            if self.voxel_actor:
                self.plotter.remove_actor(self.voxel_actor)
                self.voxel_actor = None
            
            # Lê cortes
            xm, xM = self.cut_ranges['x']
            ym, yM = self.cut_ranges['y']
            zm, zM = self.cut_ranges['z']
            
            # Gera mesh
            QApplication.setOverrideCursor(Qt.WaitCursor)
            voxel_mesh, stats = self._generate_voxel_mesh(
                xm, xM, ym, yM, zm, zM,
                self.voxel_threshold,
                self.voxel_resolution,
                self.voxel_limit
            )
            QApplication.restoreOverrideCursor()
            
            if voxel_mesh is None:
                self.voxel_info_label.setText("<b style='color:red;'>Nenhum voxel!</b><br>Reduza threshold.")
                return
            
            # Renderiza
            self.voxel_actor = self.plotter.add_mesh(
                voxel_mesh,
                color='white',
                show_edges=False,
                lighting=True,
                reset_camera=False
            )
            
            # Info
            self.voxel_info_label.setText(
                f"<b>Voxels:</b> {stats['count']:,}<br>"
                f"<b>%:</b> {stats['percentage']:.1f}%<br>"
                f"<small>{stats['message']}</small>"
            )
            
        finally:
            self.plotter.pause_render = False
            self.plotter.render()

    def _generate_voxel_mesh(self, xm, xM, ym, yM, zm, zM, threshold, resolution, max_voxels):
        """
        Versão OTIMIZADA com broadcasting NumPy.
        """
        d, h, w = self.manager.dims[:3]
        total_z = self.manager.total_z_height
        
        # Índices
        ix_min = max(0, int(xm/100 * w))
        ix_max = min(w, int(xM/100 * w))
        iy_min = max(0, int(ym/100 * h))
        iy_max = min(h, int(yM/100 * h))
        iz_min = max(0, int(zm/100 * total_z))
        iz_max = min(total_z, int(zM/100 * total_z))
        
        # Slices
        slice_min = np.searchsorted(self.manager.z_cumulative, iz_min, side='right')
        slice_max = np.searchsorted(self.manager.z_cumulative, iz_max, side='right')
        slice_min = min(slice_min, d-1)
        slice_max = min(slice_max, d-1)
        
        if slice_max <= slice_min:
            slice_max = slice_min + 1
        
        # Lista para acumular coordenadas
        all_coords = []
        
        for slice_idx in range(slice_min, slice_max):
            # Recorta imagem
            img_slice = self.manager.volume[slice_idx, iy_min:iy_max, ix_min:ix_max]
            
            # Threshold + downsample
            if resolution > 1:
                img_slice = img_slice[::resolution, ::resolution]
            
            mask_2d = img_slice >= threshold
            
            # Pixels que passaram
            pixels_yx = np.argwhere(mask_2d)
            
            if len(pixels_yx) == 0:
                continue
            
            # Geometria da camada
            z_start = self.manager.z_starts[slice_idx]
            z_size = self.manager.z_thicknesses[slice_idx]
            
            # Voxels em Z
            num_z = max(1, int(z_size / resolution))
            z_positions = np.arange(num_z) * resolution + z_start
            
            # Broadcasting mágico do NumPy!
            # Cria grade (num_pixels, num_z_voxels)
            num_pixels = len(pixels_yx)
            
            # Expande pixels para cada Z
            y_coords = np.repeat(pixels_yx[:, 0], num_z) * resolution + iy_min
            x_coords = np.repeat(pixels_yx[:, 1], num_z) * resolution + ix_min
            z_coords = np.tile(z_positions, num_pixels)
            
            # Empilha [z, y, x]
            coords = np.column_stack([z_coords, y_coords, x_coords])
            all_coords.append(coords)
        
        if not all_coords:
            return None, None
        
        # Concatena tudo
        all_coords = np.vstack(all_coords)
        total_visible = len(all_coords)
        
        # Limita
        if total_visible > max_voxels:
            indices = np.random.choice(total_visible, max_voxels, replace=False)
            all_coords = all_coords[indices]
            used = max_voxels
            msg = f"Limitado: {max_voxels:,}/{total_visible:,}"
        else:
            used = total_visible
            msg = "Todos renderizados"
        
        # Cria mesh
        voxel_mesh = self._create_voxel_poly(all_coords, resolution)
        
        stats = {
            'count': used,
            'total': total_visible,
            'percentage': 100.0,
            'message': msg
        }
        
        return voxel_mesh, stats

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

    def _create_voxel_poly(self, coords, size):
        '''Cria PolyData de voxels'''
        cube = pv.Cube(bounds=(
            -size/2, size/2,
            -size/2, size/2,
            -size/2, size/2
        ))
        
        cubes = []
        for z, y, x in coords:
            c = cube.copy()
            c.translate([x, y, z], inplace=True)
            cubes.append(c)
        
        combined = cubes[0]
        for c in cubes[1:]:
            combined = combined.merge(c)
        
        return combined
    
    def on_scale_changed(self):
        """Atualiza o scale_manager quando valores mudam"""
        self.scale_manager.set_scale(
            'x', 
            self.scale_x_spin.value(), 
            self.unit_x_combo.currentText()
        )
        self.scale_manager.set_scale(
            'y', 
            self.scale_y_spin.value(), 
            self.unit_y_combo.currentText()
        )
        self.scale_manager.set_scale(
            'z', 
            self.scale_z_spin.value(), 
            self.unit_z_combo.currentText()
        )
    
    def apply_scale_to_model(self):
        """
        Aplica fatores de escala ao modelo 3D no PyVista.
        Isso distorce o modelo para refletir escalas reais.
        """
        scale_x, scale_y, scale_z = self.scale_manager.get_scale_factors()
        
        # Aplica transformação de escala usando a propriedade nativa do PyVista
        for actor in self.actors.values():
            actor.scale = (scale_x, scale_y, scale_z)
        
        # Se tem voxel, aplica também
        if self.voxel_actor:
            self.voxel_actor.scale = (scale_x, scale_y, scale_z)
        
        self.plotter.render()
        print(f"Escala aplicada: X={scale_x:.3f}, Y={scale_y:.3f}, Z={scale_z:.3f}")

    def start_measurement(self):
        if self.measurement_tool is None:
            self.measurement_tool = MeasurementTool(self.plotter, self.scale_manager)
        
        self.measurement_tool.start_measurement()
        self.measure_result.setText("<b>Clique 2 pontos no modelo 3D</b>")
        
        print("Modo medição ativado. Clique 2 pontos no modelo.")
    def clear_measurement(self):
        if self.measurement_tool:
            self.measurement_tool.clear_measurement()
        self.measure_result.setText("<i>Medição limpa</i>")
        

    

####### JANELA PRINCIPAL #######

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rec3D")
        self.resize(1000, 700)

        self.setWindowIcon(QIcon("icon.ico"))

        # 1. O StackedWidget (O "Baralho" de telas)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # 2. Instancia as suas telas
        self.page_import = ImportWidget()
        self.page_depth = DepthWidget()
        self.page_volume = VolumeWidget()

        # 3. Adiciona as telas ao baralho
        self.stack.addWidget(self.page_import) # Índice 0
        self.stack.addWidget(self.page_depth)  # Índice 1
        self.stack.addWidget(self.page_volume) # Índice 2


        # 4. Criação da Barra de Ferramentas (Menu Superior) para Navegar
        toolbar = QToolBar("Navegação")
        self.addToolBar(toolbar)

        # Ação: Ir para Importação
        btn_go_import = QAction("Importar Imagens", self)
        btn_go_import.triggered.connect(lambda: self.stack.setCurrentIndex(0))
        toolbar.addAction(btn_go_import)

        # Ação: Ir para Dados
        btn_go_volume = QAction("Dados de Profundidade", self)
        btn_go_volume.triggered.connect(self.go_to_depth_page)
        toolbar.addAction(btn_go_volume)

        # Ação: Ir para Volue
        btn_go_volume = QAction("Volume e Reconstrução", self)
        btn_go_volume.triggered.connect(lambda: self.stack.setCurrentIndex(2))
        toolbar.addAction(btn_go_volume)


        # Começa na tela de importação
        self.stack.setCurrentIndex(0)

        #self.page_import.folder_selected.connect(self.page_volume.set_path)

        self.page_depth.depth_data.connect(self.page_volume.receive_geometry_data)

#
    
    def go_to_depth_page(self):
        self.page_depth.load_table()
        self.stack.setCurrentIndex(1)

    #def update_volume_path(self, path):
        # Essa função age como uma ponte
        #print(f"Caminho selecionado: {path}")
        
        # Atualiza a tela de volume
        #self.page_volume.set_path(path)


if __name__ == "__main__":

    basedir = os.path.dirname(__file__)
    icon_path = os.path.join(basedir, 'icon.ico')

    # --- NOVO: Truque para o ícone aparecer na Barra de Tarefas do Windows ---
    try:
        # Define um ID único para o app. Pode ser qualquer string no formato 'empresa.produto.subproduto.versao'
        myappid = 'zoehler.rec3d.visualizer.2.1' 
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except ImportError:
        pass # Se não for Windows, ignora
    # -----------------------------------------------------------------------

    startup()

    app = QApplication(sys.argv)

    if not os.path.exists(icon_path):
        print(f"ERRO: Ícone não encontrado no caminho: {icon_path}")
    else:
        # Define o ícone no App E na Janela explicitamente
        app_icon = QIcon(icon_path)
        app.setWindowIcon(app_icon)

    window = MainWindow()

    window.setWindowIcon(QIcon(icon_path))

    window.show()
    sys.exit(app.exec_())