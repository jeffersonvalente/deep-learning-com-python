import pandas as pd
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Dense, Dropout, Activation, Input # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.models import Model # atualizado: tensorflow==2.0.0-beta1

base = pd.read_csv('games.csv')
base = base.drop('Other_Sales', axis = 1)
#base = base.drop('Global_Sales', axis = 1)
base = base.drop('Developer', axis = 1)
base = base.drop('NA_Sales', axis = 1)
base = base.drop('EU_Sales', axis = 1)
base = base.drop('JP_Sales', axis = 1)


base = base.dropna(axis = 0)
base = base.loc[base['Global_Sales'] > 1]
#base = base.loc[base['EU_Sales'] > 1]

base['Name'].value_counts()
nome_jogos = base.Name
base = base.drop('Name', axis = 1)

previsores = base.iloc[:, [0,1,2,3,5,6,7,8,9]].values
valor_vendas = base.iloc[:,4].values
#venda_eu = base.iloc[:,5].values
#venda_jp = base.iloc[:,6].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

#camada_entrada = Input(shape=(99,))
#camada_oculta1 = Dense(units = 50, activation= 'sigmoid')(camada_entrada)
#camada_oculta2 = Dense(units = 50, activation= 'sigmoid')(camada_oculta1)
#camada_saida1 = Dense(units = 1, activation= 'linear')(camada_oculta2)
#camada_saida2 = Dense(units = 1, activation= 'linear')(camada_oculta2)
#camada_saida3 = Dense(units = 1, activation= 'linear')(camada_oculta2)
camada_entrada = Input(shape=(99,))
ativacao = Activation(activation = 'sigmoid')
camada_oculta1 = Dense(units = 50, activation=ativacao)(camada_entrada)
camada_oculta2 = Dense(units = 50, activation=ativacao)(camada_oculta1)
camada_saida = Dense(units = 1, activation='linear')(camada_oculta2)
regressor = Model(inputs = camada_entrada, outputs=[camada_saida])
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(previsores, valor_vendas, epochs = 5000, batch_size=100)
previsoes = regressor.predict(previsores)

#regressor = Model(inputs = camada_entrada,
#                  outputs = [camada_saida1,camada_saida2,camada_saida3,])
#regressor.compile(optimizer = 'adam',
#                  loss = 'mse')
#regressor.fit(previsores, [venda_na, venda_eu, venda_jp],
#              epochs = 5000, batch_size = 100)
#previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores)