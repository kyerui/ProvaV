import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pickle import load

# Caminhos dos modelos salvos
SCALER_PATH = 'Classificador/dados/image_scaler.pkl'
MODEL_PATH = 'Classificador/dados/image_tree_model.pkl'
IMG_DIR = 'img_classificar'

# Carregar scaler e modelo
scaler = load(open(SCALER_PATH, 'rb'))
model = load(open(MODEL_PATH, 'rb'))

# Função de pré-processamento
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))

    # Converter para tons de cinza para equalização
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    # Substituir o canal V do HSV pela equalização (opcional) ou apenas usar o equalizado para todas as bandas
    img_eq = cv2.merge([gray_eq, gray_eq, gray_eq])  # Forçar para 3 canais

    # Aplicar filtro gaussiano
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)

    # Flatten para o modelo
    flat = img_blur.flatten().astype(np.float32)
    return flat, img_blur

# Classificar e exibir as imagens
for file in os.listdir(IMG_DIR):
    if file.lower().endswith('.jpg'):
        img_path = os.path.join(IMG_DIR, file)
        features, preprocessed_img = preprocess_image(img_path)
        features_scaled = scaler.transform([features])

        pred = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        confidence = np.max(proba)

        # Exibir imagem com label
        plt.imshow(cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Classe: {pred} (Confiança: {confidence:.2f})")
        plt.axis('off')
        plt.show()
