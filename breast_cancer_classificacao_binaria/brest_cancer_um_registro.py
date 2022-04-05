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

novo = np.array([[1.80,1.34, 0.18, 500, 0.005, 0.16, 0.01, 0.1, 0.1,
                  0.20, 0.05, 200, 0.87, 100, 125.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 218, 0.14, 0.185,
                  0.84, 158, 0.363]])
previsao = classificador.predict(novo)
previsao = (previsao > 0.9)
