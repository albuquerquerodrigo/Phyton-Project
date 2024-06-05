import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications import MobileNet
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras._tf_keras.keras.models import load_model
from tkinter import filedialog
from PIL import Image, ImageTk

##################
# primeira parte #
##################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Visualizando algumas imagens
# for i in range(9):
#     plt.subplot(330 + 1 + i)
#     plt.imshow(x_train[i])
# plt.show()

##################
# segunda parte #
##################

datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

##################
# terceira parte #
##################

model = MobileNet(weights='imagenet')

##################
# quarta parte #
##################

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

##################
# quinta parte #
##################

def preprocess_image(image):
    image = cv2.resize(image, (32, 32))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def classify_image(model, image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return np.argmax(prediction)
 
##################
# quinta parte #
##################

model.save('my_model.h5')
model = load_model('my_model.h5')


def load_image():
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)
    image = image.resize((200, 200))
    image = ImageTk.PhotoImage(image)
    panel = tk.Label(root, image=image)
    panel.image = image
    panel.pack()
    
    # Classificar a imagem
    image = cv2.imread(file_path)
    label = classify_image(model, image)
    result_label.config(text="Resultado: " + str(label))

root = tk.Tk()
root.title("Classificação de Imagens")
root.geometry("300x400")

button = tk.Button(root, text="Carregar Imagem", command=load_image)
button.pack()

result_label = tk.Label(root, text="Resultado: ")
result_label.pack()

root.mainloop()

