############################################# PARTE DA IA ##############################################
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, datasets, layers #Modelos, Banco de Dados, Camadas
from tensorflow.keras.models import load_model #Bilioteca capaz carregar a IA salvada

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
veiculos = [8, 13, 41, 48, 58, 69, 81, 85, 89, 90] #['bicycle' , 'bus',  'lawn_mower', 'motorcycle', 'pickup_truck', 'rocket', 'streetcar', 'tank', 'tractor', 'train']
index_train = [i for i in range(len(y_train)) if y_train[i] not in veiculos]
index_test = [i for i in range(len(y_test)) if y_test[i] not in veiculos]
x_test = np.delete(x_test, index_test, axis=0)
y_test = np.delete(y_test, index_test, axis=0)
x_test = x_test / 255
for i in range(1, len(veiculos)+1):
  #y_train[y_train == veiculos[i-1]] = i
  y_test[y_test == veiculos[i-1]] = i
Vei = models.Sequential([
    layers.Flatten(input_shape = (32,32,3)), #Usamos Flatten para tranformar os dados em linha, pois uma camada densa espera vetor linha
    layers.Dense(100, activation='relu'),  #(Numero de neuronios, função de ativação)
    layers.Dense(50, activation='relu'),  #(Numero de neuronios, função de ativação)
    layers.Dense(11, activation='softmax'),
])
Vei.compile(optimizer='adam',  #Otimizador do Modelo, usa algoritmo Adam
            loss='sparse_categorical_crossentropy',  #Calcula a perda de entropia cruzada entre os rótulos e as previsões.
            metrics=['accuracy'])  #Calcula com que frequência as previsões equivalem aos rótulos.
model=load_model(os.path.join(os.path.dirname(__file__), "data/cifar100_models.keras"))
class_names = ['bicycle' , 'bus',  'lawn_mower', 'motorcycle', 'pickup_truck', 'rocket', 'streetcar', 'tank', 'tractor', 'train']

############################################# FIM D APARTE DA IA #######################################


import random
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QFrame, QVBoxLayout, QHBoxLayout, QWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItemGroup, QPushButton
from PyQt6.QtGui import QPixmap, QPainter, QIcon
from PyQt6.QtCore import Qt
import sys


class Vehicle:
    def __init__(self, pixmap, x, y, vetor, classification, exit_time=None):
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.pixmap_item.setPos(x, y)
        self.exit_time = exit_time
        self.vetor = vetor
        self.classification = classification

class DraggableGraphicsPixmapItem(QGraphicsPixmapItem):
    def __init__(self, pixmap):
        super().__init__(pixmap)
        self.setFlag(QGraphicsPixmapItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsPixmapItem.GraphicsItemFlag.ItemSendsScenePositionChanges)

class SmartParkingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SmartParking")
        self.vehicle_objects = []
        self.indice = 0
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "data/icon_darkmode.png")))

        # Definir vetores de imagens
        self.normal = ["pick_up 2.png", "streetcar 1.png", "streetcar 2.png", "streetcar 3.png", "streetcar 4.png", "streetcar 5.png", "streetcar 6.png"]
        self.mini = ["bicicleta 1.png", "bicicleta 2.png", "moto 1.png", "moto 2.png", "moto 3.png"]
        self.large = ["pick_up 1.png", "bus 1.png", "bus 2.png", "truck 1.png", "truck 2.png", "truck 3.png", "truck 4.png"]

        # Definir fileiras e suas configurações
        self.fileiras = [(200, 195), (285, 195), (370, 195), (455, 195), (540, 195), (625, 195), (710, 195), (795, 195), (880, 195), (965, 195), (1050, 195), 
                         (1135, 195), (1220, 195), (1305, 195), (1390, 195), (1475, 195), (1560, 195), (1645, 195), (1730, 195), (1815, 195), (1900, 195), 
                         (1985, 195), (2070, 195), (370, 650), (455, 650), (540, 650), (625, 650), (710, 650), (795, 650), (880, 650), (965, 650), (1050, 650), 
                         (1135, 650), (1220, 650), (1305, 650), (1390, 650), (1475, 650), (1560, 650), (1645, 650), (1730, 650), (1815, 650), (1900, 650), 
                         (1985, 650), (2070, 650), (2155, 650), (2240, 650), (370, 820), (455, 820), (540, 820), (625, 820), (710, 820), (795, 820), (880, 820), 
                         (965, 820), (1050, 820), (1135, 820), (1220, 820), (1305, 820), (1390, 820), (1475, 820), (1560, 820), (1645, 820), (1730, 820), 
                         (1815, 820), (1900, 820), (1985, 820), (2070, 820), (2155, 820), (2240, 820), (2325, 820), (370, 1270), (455, 1270), (540, 1270), 
                         (625, 1270), (710, 1270), (795, 1270), (880, 1270), (965, 1270), (1050, 1270), (1135, 1270), (1220, 1270), (1305, 1270), (1390, 1270), 
                         (1475, 1270), (1560, 1270), (1645, 1270), (1730, 1270), (1815, 1270), (1900, 1270), (1985, 1270), (2070, 1270), (2155, 1270), 
                         (2240, 1270), (2325, 1270), (370, 1440), (455, 1440), (540, 1440), (625, 1440), (710, 1440), (795, 1440), (880, 1440), (965, 1440), 
                         (1050, 1440), (1135, 1440), (1220, 1440), (1305, 1440), (1390, 1440), (1475, 1440), (1560, 1440), (1645, 1440), (1730, 1440), 
                         (1815, 1440), (1900, 1440), (1985, 1440), (2070, 1440), (2155, 1440), (2240, 1440), (2325, 1440), (370, 1880), (455, 1880), (540, 1880), 
                         (625, 1880), (710, 1880), (795, 1880), (880, 1880), (965, 1880), (1050, 1880), (1135, 1880), (1220, 1880), (1305, 1880), (1390, 1880), 
                         (1475, 1880), (1560, 1880), (1645, 1880), (1730, 1880), (1815, 1880), (1900, 1880), (1985, 1880), (2070, 1880), (2155, 1880), 
                         (2240, 1880), (2325, 1880)]
        
        self.fileiraL = [(420, 2060), (535, 2060), (650, 2060), (765, 2060), (880, 2060), (995, 2060), (1110, 2060), (1225, 2060), (1340, 2060), (1455, 2060), 
                         (1570, 2060), (1685, 2060), (1800, 2060), (1915, 2060), (2030, 2060), (2145, 2060), (2260, 2060)]
        
        self.fileiraM = [(2712, 202), (2712, 259), (2712, 316), (2712, 373), (2712, 430), (2712, 487), (2712, 544), (2712, 601), (2712, 658), (2712, 715), 
                         (2712, 772), (2712, 829), (2712, 886), (2712, 943), (2712, 1000), (2712, 1057), (2712, 1114), (2712, 1171), (2712, 1228), (2712, 1285), 
                         (2712, 1342), (2712, 1399), (2712, 1456), (2712, 1513), (2712, 1570), (2712, 1627), (2712, 1684), (2712, 1741), (2712, 1798), 
                         (2712, 1855), (2712, 1912), (2712, 1969)]

        # Configuração do widget central
        central_widget = QWidget(self)
        central_widget.setStyleSheet("background-color: #484848;")
        self.setCentralWidget(central_widget)

        # Layout principal
        layout = QVBoxLayout(central_widget)

        # Espaço superior - 10%
        self.top_frame = QFrame()
        self.top_frame.setStyleSheet("background-color: #484848;")
        layout.addWidget(self.top_frame)

        # Criar um QGraphicsView para mostrar a imagem
        self.graphics_view = QGraphicsView()
        self.graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Remover barras de rolagem
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Criar uma cena para o QGraphicsView
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)

        # Criar um grupo para as imagens do estacionamento e dos veículos
        self.image_group = QGraphicsItemGroup()

        # Adicionar a imagem do estacionamento ao grupo
        self.add_parking_image()

        # Adicionar o grupo à cena
        self.scene.addItem(self.image_group)

        # Conectar evento de roda do mouse para zoom
        self.graphics_view.wheelEvent = self.wheel_event

        # Habilitar a movimentação do grupo
        self.image_group.setFlag(QGraphicsItemGroup.GraphicsItemFlag.ItemIsMovable)


        # Adicionar o QGraphicsView em um layout horizontal com margens
        h_layout = QHBoxLayout()
        h_layout.setContentsMargins(50, 0, 50, 0)
        h_layout.addWidget(self.graphics_view)
        layout.addLayout(h_layout)

        # Espaço inferior - 20%
        self.bottom_frame = QFrame()
        self.bottom_frame.setStyleSheet("background-color: #484848;")
        layout.addWidget(self.bottom_frame)

        self.containers = []
        self.containersT = []
        for i in range(5):
            self.container = QLabel(self.bottom_frame)
            self.container.setFixedSize(128, 128)  # Defina o tamanho de cada contêiner
            self.container.setStyleSheet("background-color: #606060; margin: 5px;")  # Estilo dos contêineres
            self.containerT = QLabel(f"imagem {i}")
            self.containers.append(self.container)
            self.containersT.append(self.containerT)


        # Botão para próximo turno
        self.next_turn_button = QPushButton("Próximo Turno", self.bottom_frame)
        self.next_turn_button.setFixedSize(200, 150)  # Definindo o tamanho do botão
        self.next_turn_button.setStyleSheet("background-color: #493F73;")
        self.next_turn_button.clicked.connect(self.proximo_turno)
        button_layout = QHBoxLayout(self.bottom_frame)
        for i in range(5):
            button_layout.addWidget(self.containers[i])
            button_layout.addWidget(self.containersT[i])
        button_layout.addWidget(self.next_turn_button)

        # Definir alturas iniciais
        self.adjust_heights()

        

    def predicao(self):
        from PIL import Image
        random_idx = np.random.randint(0, x_test.shape[0])
        new_input = x_test[random_idx].reshape(32, 32, 3)  # Mantém no formato (32, 32, 3)
        
        # Converte o array numpy para uma imagem PIL
        image = Image.fromarray((new_input * 255).astype(np.uint8))  # Multiplica por 255 para garantir valores de cor [0, 255]
        
        # Salva a imagem
        
        image.save(os.path.join(os.path.dirname(__file__), f"data/imagem{self.indice}.png"))
        new_input= x_test[random_idx].reshape(1, 32, 32, 3) #Redimensiona para formado esperado pelo modelo
        prediction= Vei.predict(new_input)
        predict_class=np.argmax(prediction)

        #Obtem a classe original
        original_class=y_test[random_idx]
        return(class_names[predict_class-1]), (class_names[original_class[0]-1])



    def adjust_heights(self):
        if hasattr(self, 'top_frame') and hasattr(self, 'graphics_view') and hasattr(self, 'bottom_frame'):
            self.graphics_view.setFixedHeight(int(self.height() * 0.7))
            self.bottom_frame.setFixedHeight(int(self.height() * 0.2))

    def resizeEvent(self, event):
        self.adjust_heights()
        super().resizeEvent(event)

    def wheel_event(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.image_group.setScale(self.image_group.scale() * factor)

    def reset_image_group(self):
        # Resetar a escala do grupo de imagens para 1.0
        self.image_group.setScale(1.0)
        
        # Resetar a posição do grupo de imagens para a posição original (0, 0)
        self.image_group.setPos(0, 0)

    def add_parking_image(self):
        image_path = os.path.join(os.path.dirname(__file__), "data/estacionamento.png")
        parking_pixmap = QPixmap(image_path)
        if parking_pixmap.isNull():
            print("Erro ao carregar a imagem do estacionamento. Verifique o caminho.")
        else:
            self.parking_item = DraggableGraphicsPixmapItem(parking_pixmap)
            self.image_group.addToGroup(self.parking_item)
            self.parking_item.setScale(1.0)
            self.parking_item.setPos(0, 0)

    def add_vehicle(self, vehicle_type):
        current_pos = self.image_group.pos()
        current_scale = self.image_group.scale()
        self.reset_image_group()
        if vehicle_type == self.normal:
            if self.fileiras:  # Verifica se self.fileiraL não está vazio
                posicao = random.choice(self.fileiras)
                classification = "normal"
            else:
                self.image_group.setPos(current_pos)
                self.image_group.setScale(current_scale)
                return
            self.fileiras.remove(posicao)

        elif vehicle_type == self.large:
            if self.fileiraL:  # Verifica se self.fileiraL não está vazio
                posicao = random.choice(self.fileiraL)
                classification = "large"
            else:
                self.image_group.setPos(current_pos)
                self.image_group.setScale(current_scale)
                return
            self.fileiraL.remove(posicao)

        elif vehicle_type == self.mini:
            if self.fileiraM:  # Verifica se self.fileiraL não está vazio
                posicao = random.choice(self.fileiraM)
                classification = "mini"
            else:
                self.image_group.setPos(current_pos)
                self.image_group.setScale(current_scale)
                return
            self.fileiraM.remove(posicao)    
        x = posicao[0]
        y = posicao[1]

        # Escolher imagem de veículo aleatoriamente a partir do vetor fornecido
        vehicle_image = random.choice(vehicle_type)
        vehicle_image_path = os.path.join(os.path.dirname(__file__), "data", vehicle_image)

        # Carregar e adicionar o veículo
        vehicle_pixmap = QPixmap(vehicle_image_path)
        if vehicle_pixmap.isNull():
            print(f"Erro ao carregar a imagem do veículo {vehicle_image}. Verifique o caminho.")
        else:
            # Gerar um número aleatório para exit_time
            exit_time = random.randint(10, 25)

            # Criar e adicionar o veículo
            vehicle = Vehicle(vehicle_pixmap, x, y, posicao, classification, exit_time)
            self.image_group.addToGroup(vehicle.pixmap_item)
            self.vehicle_objects.append(vehicle)
            self.vehicle_objects = sorted(self.vehicle_objects, key=lambda vehicle: vehicle.exit_time)

        # Restaurar a posição e zoom originais
        self.image_group.setPos(current_pos)
        self.image_group.setScale(current_scale)

    def proximo_turno(self):
        self.indice = 0
        
        # Adicionar 5 novos veículos
        for _ in range(5):
            adivinha, original = self.predicao()
            if adivinha in ['pickup_truck', 'streetcar']:
                tipo = self.normal
                self.add_vehicle(tipo)
            elif adivinha in ['bicycle',  'lawn_mower', 'motorcycle']:
                tipo = self.mini
                self.add_vehicle(tipo)
            elif adivinha in ['bus','tank', 'tractor']:
                tipo = self.large 
                self.add_vehicle(tipo)

            
            image_path = os.path.join(os.path.dirname(__file__), f"data/imagem{self.indice}.png")
            imagem = QPixmap(image_path)
            if imagem.isNull():
                print("Erro ao carregar a imagem do estacionamento. Verifique o caminho.")
            else:
                imagem = imagem.scaled(128, 128, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.containers[self.indice].setPixmap(imagem)
                self.containersT[self.indice].setText(f"Original:{original}\nPredição:{adivinha}")
            self.indice += 1
            

        # Diminuir o exit_time de cada veículo
        for vehicle in self.vehicle_objects:
            vehicle.exit_time -= 1

        # Verificar se algum veículo deve ser removido
        for vehicle in self.vehicle_objects[:]:  # Criar uma cópia da lista para evitar problemas de modificação
            if vehicle.exit_time <= 0:
                self.remove_vehicle(vehicle)  # Chama a função para remover veículo específico

    def remove_vehicle(self, vehicle=None):
        if vehicle.classification == "normal":
            self.fileiras.append(vehicle.vetor)
            self.vehicle_objects.remove(vehicle)  # Remove o veículo específico
            self.image_group.removeFromGroup(vehicle.pixmap_item)
            self.scene.removeItem(vehicle.pixmap_item)

        if vehicle.classification == "large":
            self.fileiraL.append(vehicle.vetor)
            self.vehicle_objects.remove(vehicle)  # Remove o veículo específico
            self.image_group.removeFromGroup(vehicle.pixmap_item)
            self.scene.removeItem(vehicle.pixmap_item)

        if vehicle.classification == "mini":
            self.fileiraM.append(vehicle.vetor)
            self.vehicle_objects.remove(vehicle)  # Remove o veículo específico
            self.image_group.removeFromGroup(vehicle.pixmap_item)
            self.scene.removeItem(vehicle.pixmap_item)

# Inicializar a aplicação
app = QApplication(sys.argv)
window = SmartParkingApp()
window.showMaximized()
sys.exit(app.exec())
