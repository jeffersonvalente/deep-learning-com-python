import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.utils import np_utils 
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization

#carrega as imagens
(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

#altera a imagem para cinza
plt.imshow(X_treinamento[0], cmap = 'gray')
plt.title('Classe ' +str(y_treinamento[0]))

#converte os dados para o tensorflow ler
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0],
                                               28, 28,1)
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

#converte para valores entre 0 e 1
previsores_treinamento /= 255
previsores_teste /= 255

#cria os dummys e encondig /operador de convolução
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

#cria os kernels das imagens
classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape=(28 , 28, 1),
                         activation = 'relu'))
classificador.add(BatchNormalization())

#polling
classificador.add(MaxPooling2D(pool_size = (2,2)))

#segunda camada
classificador.add(Conv2D(32, (3,3),activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
classificador.add(Flatten())

#gera rede neura densa
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
#segunda camada oculta
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 10, 
                        activation = 'softmax'))
classificador.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam', metrics = ['accuracy'])
#faz o treinametno
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 128, epochs = 5,
                  validation_data = (previsores_teste, classe_teste))
resultado = classificador.evaluate(previsores_teste, classe_teste)