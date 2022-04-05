import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential 
import tensorflow as tf 
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import backend as k 

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

classificador = Sequential([
tf.keras.layers.Dense(units= 8, activation = 'relu', kernel_initializer = 'normal', input_dim=30),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(units= 8, activation = 'relu', kernel_initializer = 'normal'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(units=1, activation = 'sigmoid')])
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                      metrics = ['binary_accuracy'])
classificador.fit(previsores, classe, batch_size = 10, epochs = 100)

#salvando a rede
classificador_json = classificador.to_json()
with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)

#salvando os pesos
classificador.save_weights('classificador_breast.h5')