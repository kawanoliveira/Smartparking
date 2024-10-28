# -*- coding: utf-8 -*-
"""Work.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fcwK1ixXgzjfgByo3t8NYkvggOtKOq4R
"""

##Processamento de grandes matrizes, juntamente com uma grande coleção de funções matemáticas de alto nível para operar sobre estas matrizes
#Uso para possibilidar a alteração a base de dados original que está usando matrizes numpy
import numpy as np

##Criação de gráficos e visualizações de dados em geral
#Usei anteriormente para verificar imagens
import matplotlib.pyplot as plt

##Biblioteca de aprendizado de máquina de código aberto para a linguagem de programação
#O módulo implementa diversas funções de perda, pontuação e utilidade para medir o desempenho da classificação
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

##Biblioteca de código aberto para aprendizado de máquina aplicável a uma ampla variedade de tarefas
import tensorflow as tf
#keras => API de alto nível do TensorFlow para criar e treinar modelos de aprendizado profundo
from tensorflow.keras import models, datasets, layers #Modelos, Banco de Dados, Camadas
from tensorflow.keras.models import load_model #Bilioteca capaz carregar a IA salvada

#Usar keras.datasets para carregar a base de bados cifar100, já separado em x(imagens) e y(classificação) de treino e de teste
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

#Uso somente para verificar sua forma antes de alterá-lo
print(x_train.shape) #(Quantidade, formato...)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

###Retirar tudo não relacionado a veículos da base de dados

#Crio uma lista com o numero correspondente de cada classificação de veículo
veiculos = [8, 13, 41, 48, 58, 69, 81, 85, 89, 90] #['bicycle' , 'bus',  'lawn_mower', 'motorcycle', 'pickup_truck', 'rocket', 'streetcar', 'tank', 'tractor', 'train']

##Crio uma Máscara para filtrar tudo que não está dentro da lista de veiculos no treino e no teste
index_train = [i for i in range(len(y_train)) if y_train[i] not in veiculos]
index_test = [i for i in range(len(y_test)) if y_test[i] not in veiculos]

##Deleta tudo que a mascara confirmar não ser da lista veiculo
#Usa numpy pois datasets usa lista numpy, usa sua função delete
x_train = np.delete(x_train, index_train, axis=0)
y_train = np.delete(y_train, index_train, axis=0)

x_test = np.delete(x_test, index_test, axis=0)
y_test = np.delete(y_test, index_test, axis=0)

#Verificar se sua forma alterou e ver nova quantidade de imagens.
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#Normalizacão de dados imagens
x_train = x_train / 255
x_test = x_test / 255

#Troca valores numericos de Y para facilidar sua analise, numeros altos geram matrices de saída enormes (90 do 'train' gerava 90 colunas de saída)
for i in range(1, len(veiculos)+1):
  y_train[y_train == veiculos[i-1]] = i
  y_test[y_test == veiculos[i-1]] = i

###Para a classificação de imagens iremos usar MLP

##Usa Sequential, funçaõ que cria modelo especificando cada camada o tipo, quantidade neuronios e funcao de ativação
Vei = models.Sequential([
    #DENSE(MLP)=> camada densa(totalmente conectada), um tipo de camada em que cada neurônio está conectado a todos os neurônios da camada anterior
    layers.Flatten(input_shape = (32,32,3)), #Usamos Flatten para tranformar os dados em linha, pois uma camada densa espera vetor linha
    layers.Dense(100, activation='relu'),  #(Numero de neuronios, função de ativação)
    layers.Dense(50, activation='relu'),  #(Numero de neuronios, função de ativação)
    layers.Dense(11, activation='softmax'),
])


#Configura modelo para treinamento.
Vei.compile(optimizer='adam',  #Otimizador do Modelo, usa algoritmo Adam
            loss='sparse_categorical_crossentropy',  #Calcula a perda de entropia cruzada entre os rótulos e as previsões.
            metrics=['accuracy'])  #Calcula com que frequência as previsões equivalem aos rótulos.

#Começa treino com modelo criado, inserindo as imagens(x), classificaçoes(y) e numero de repetiçoes do treino
Vei.fit(x_train, y_train, epochs=15)

#Avalia o medelo geral, mostrando uma porcentagem de acuracia nas suas previsões
Vei.evaluate(x_test,y_test)

##Salvar modelo completo(arquitetura + peso) na pasta atual
Vei.save('cifar100_models.keras')

##Carrega o modelo
model=load_model('cifar100_models.keras')

#Informa sobre nosso modelo
Vei.summary()

###Matriz de confusão do conjunto de treino(Traz uma ideia do numero de acertos da IA)

#Faz previsões no conjunto de teste
y_pred = np.argmax(Vei.predict(x_test), axis=1)

#Calcula matrix de confusao
conf_matrix = confusion_matrix(y_test, y_pred)

#Exibe matrix de confusão
ConfusionMatrixDisplay(conf_matrix).plot(cmap='Blues')

#Nome das classes  [8, 13, 48, 58, 69, 81, 85, 89, 90] => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
class_names = ['bicycle' , 'bus',  'lawn_mower', 'motorcycle', 'pickup_truck', 'rocket', 'streetcar', 'tank', 'tractor', 'train']

##Fazendo previsões com modelo carregado

#Simula nova entrada: seleciona uma imagem aleatoria do conjuto de teste
random_idx = np.random.randint(0, x_test.shape[0])
new_input= x_test[random_idx].reshape(1, 32, 32, 3) #Redimensiona para formado esperado pelo modelo

#Faz previsão
prediction= Vei.predict(new_input)
predict_class=np.argmax(prediction)

#Obtem a classe original
original_class=y_test[random_idx]
print(original_class)

#Exibe imagem entrada e suas classes
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(new_input[0], cmap='gray')
plt.title("Classe original: "+class_names[original_class[0]-1])
plt.axis('off')

plt.subplot(1,2,2)
plt.bar([0,1],[prediction[0, original_class[0]],1-prediction[0, original_class[0]]], color=['blue', 'red'])
plt.xticks([0,1], ['Classe predita', 'Probabilidade'])
plt.title(f'Classe predita: {class_names[predict_class-1]}')
plt.show()

print("Classe prevista: "+class_names[predict_class-1])
print("Classe original: "+class_names[original_class[0]-1])