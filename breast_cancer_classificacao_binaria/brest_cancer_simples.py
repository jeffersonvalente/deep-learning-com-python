import pandas as pd

#importando dados
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

#dividindo dados em teste e resultado correto
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

import tensorflow as tf 
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
import keras
from keras.models import Sequential
from keras.layers import Dense

#criando a rede neural
classificador = Sequential()

#primeira camada oculta
classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform', input_dim = 30))

#segunda camada oculta
classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform'))

#camada de saida
classificador.add(Dense(units = 1, activation = 'sigmoid'))

#fazendo os testes

otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.0001, clipvalue = 0.5) 
classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 10, epochs = 100)

#adicionando os pesos
#posição entrada para rede neuronio 1
pesos0 = classificador.layers[0].get_weights()
#posição rede neuronio 1 para rede neuronio 2
pesos1 = classificador.layers[1].get_weights()
#posição rede neuronio 2 para saida
pesos2 = classificador.layers[2].get_weights()

#validação cruzada


#vendo resultados de forma manual
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

resultado = classificador.evaluate(previsores_teste, classe_teste)

