import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


#importando dados
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

#função apra validação cruzada
def criarRede():
    #criando a rede neural
    classificador = Sequential()

    #primeira camada oculta
    classificador.add(Dense(units = 8, activation = 'relu', 
                            kernel_initializer = 'normal', input_dim = 30))
    classificador.add(Dropout(0.2))

    #segunda camada oculta
    classificador.add(Dense(units = 8, activation = 'relu', 
                            kernel_initializer = 'normal'))
    classificador.add(Dropout(0.2))

    #camada de saida
    classificador.add(Dense(units = 1, activation = 'sigmoid'))

    #fazendo os testes

    otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.0001, clipvalue = 0.5) 
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy',
                          metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede,
                                epochs = 100,
                                batch_size = 10)
resultados= cross_val_score(estimator = classificador,
                            X = previsores, y = classe,
                            cv = 10, scoring= 'accuracy')
media = resultados.mean()
desvio = resultados.std()
