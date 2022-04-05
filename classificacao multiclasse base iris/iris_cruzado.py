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
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix

#importa a base
base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
labelenconder = LabelEncoder()
classe  = labelenconder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criar_rede():
    #estrutura da rede neural
    classificador = Sequential([
                   tf.keras.layers.Dense(units=4, activation = 'relu', input_dim=4),
                   tf.keras.layers.Dropout(0.2),
                   tf.keras.layers.Dense(units=4, activation = 'relu'),
                   tf.keras.layers.Dropout(0.2),
                   tf.keras.layers.Dense(units=3, activation = 'softmax')])
    classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                          metrics = ['categorical_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criar_rede,
                                epochs = 1000,
                                batch_size = 10)
resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe,
                             cv = 10,
                             scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()

