import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from keras._tf_keras.keras.models import load_model

# Carregar o modelo treinado
model = load_model('cifar10_model.h5')

# Definindo o mapeamento de classes
class_names = ['Avião', 'Automóvel', 'Pássaro', 'Gato', 'Veado', 'Cachorro', 'Sapo', 'Cavalo', 'Navio', 'Caminhão']

# Função para pré-processar a imagem
def preprocess_image(image):
    image = cv2.resize(image, (32, 32))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Função para classificar a imagem
def classify_image(model, image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    return class_names[predicted_class]

# Função para carregar imagem
def load_image():
    global panel, result_label
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = image.resize((200, 200))
        image = ImageTk.PhotoImage(image)
        if panel is not None:
            panel.config(image=image)
            panel.image = image
        else:
            panel = tk.Label(root, image=image)
            panel.image = image
            panel.pack()
        
        # Classificar a imagem
        img_cv2 = cv2.imread(file_path)
        label = classify_image(model, img_cv2)
        result_label.config(text="Resultado: " + label)

# Função para remover imagem
def remove_image():
    global panel, result_label
    if panel is not None:
        panel.config(image='')
        panel.image = None
        result_label.config(text="Resultado: ")

root = tk.Tk()
root.title("Classificação de Imagens")
root.geometry("300x400")

panel = None

button_load = tk.Button(root, text="Carregar Imagem", command=load_image)
button_load.pack(pady=10)

button_remove = tk.Button(root, text="Remover Imagem", command=remove_image)
button_remove.pack(pady=10)

result_label = tk.Label(root, text="Resultado: ")
result_label.pack(pady=10)

root.mainloop()
