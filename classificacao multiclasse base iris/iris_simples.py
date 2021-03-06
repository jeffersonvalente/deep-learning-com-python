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
#cria a divisão entre treinamento e teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size = 0.25)

#estrutura da rede neural
classificador = Sequential([
               tf.keras.layers.Dense(units=4, activation = 'relu', input_dim=4),
               tf.keras.layers.Dense(units=4, activation = 'relu'),
               tf.keras.layers.Dense(units=3, activation = 'softmax')])
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                      metrics = ['categorical_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10,
                  epochs = 1000)

resultado = classificador.evaluate(previsores_teste, classe_teste)
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)
classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]

matriz = confusion_matrix(previsoes2, classe_teste2)