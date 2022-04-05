from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization

classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape=(64 , 64, 3),
                         activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(32, (3,3), input_shape=(64 , 64, 3),
                         activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, 
                        activation = 'sigmoid'))

classificador.compile(loss = 'binary_crossentropy',
                      optimizer = 'adam', metrics = ['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                         rotation_range =7,
                                         horizontal_flip= True,
                                         shear_range = 0.2,
                                         height_shift_range= 0.07,
                                         zoom_range = 0.2)
gerador_teste = ImageDataGenerator(rescale = 1./255)

base_treinamento = gerador_treinamento.flow_from_directory('F:/curso redes neurais/redes_convolucionais/gatos_chachorros/dataset/training_set',
                                                           target_size = (64,64),
                                                           batch_size = 32,
                                                           class_mode = 'binary')