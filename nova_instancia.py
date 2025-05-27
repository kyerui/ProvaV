import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pickle import load

SCALER_PATH = 'Classificador/dados/image_scaler.pkl'
MODEL_PATH = 'Classificador/dados/image_tree_model.pkl'
IMG_DIR = 'img_classificar'

scaler = load(open(SCALER_PATH, 'rb'))
model = load(open(MODEL_PATH, 'rb'))

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128)) 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    img_eq = cv2.merge([gray_eq, gray_eq, gray_eq])
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)
    flat = img_blur.flatten().astype(np.float32)
    return flat, img_blur

for file in os.listdir(IMG_DIR):
    if file.lower().endswith('.jpg'):
        img_path = os.path.join(IMG_DIR, file)
        features, preprocessed_img = preprocess_image(img_path)
        features_scaled = scaler.transform([features])

        pred_index = model.predict(features_scaled)[0]
        pred_class = model.classes_[pred_index]

        # Exibir imagem com a classe prevista
        plt.imshow(cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Classe: {pred_class}")
        plt.axis('off')
        plt.show()
