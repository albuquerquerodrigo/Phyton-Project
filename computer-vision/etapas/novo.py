import tensorflow as tf
import numpy as np
import pickle
from keras.applications import DenseNet121
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_data(data_dir):
    x_train = []
    y_train = []

    for i in range(1, 6):
        batch = unpickle(f"{data_dir}/data_batch_{i}")
        x_train.append(batch[b'data'])
        y_train.append(batch[b'labels'])

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    
    test_batch = unpickle(f"{data_dir}/test_batch")
    x_test = test_batch[b'data']
    y_test = test_batch[b'labels']

    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)

data_dir = 'assets/cifar-10-batches-py'
(x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)

# Normalizar os dados
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Converter rótulos para one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Carregar o modelo base
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Adicionar camadas no topo do modelo base
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)  # Diminuir o número de neurônios para 512
predictions = Dense(10, activation='softmax')(x)

# Criar o modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar o modelo
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo por 10 épocas
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Salvar o modelo treinado
model.save('cifar10_model_with_densenet_less_neurons.h5')
