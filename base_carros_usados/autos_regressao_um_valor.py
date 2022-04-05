import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential 
import tensorflow as tf 
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import backend as k 


base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)

#base['name'].value_counts()
base = base.drop('name', axis = 1)
#base['seller'].value_counts()
base = base.drop('seller', axis = 1)
#base['offerType'].value_counts()
base = base.drop('offerType', axis = 1)

i1 = base.loc[base.price <= 10]
base = base[base.price > 10]
i2 = base.loc[base.price > 350000]
base = base.loc[base.price < 350000]

base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() #limousine

base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() #manuell

base.loc[pd.isnull(base['model'])]
base['model'].value_counts() #golf

base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts() #benzin

base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() #nein

valores = {'vehicleType':'limousine', 'gearbox' : 'manuell', 'model': 'golf',
          'fuelType': 'benzin', 'notRepairedDamage': 'nein'}
base = base.fillna(value= valores)

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:, 10])

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray() 

regressor = Sequential([
tf.keras.layers.Dense(units= 158, activation = 'relu', input_dim=316),
tf.keras.layers.Dense(units= 158, activation = 'relu'),
tf.keras.layers.Dense(units=1, activation = 'linear')])
regressor.compile(optimizer = 'adam', loss = 'mean_absolute_error', 
                      metrics = ['mean_absolute_error'])
regressor.fit(previsores, preco_real, batch_size = 300, epochs = 100)

previsoes = regressor.predict(previsores)
preco_real.mean()
previsoes.mean()





















