import os
import glob
import numpy as np
import cv2
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
from pickle import dump

BASE_DIR = 'dados'            
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR  = os.path.join(BASE_DIR, 'test')
IMG_SIZE = (128, 128)           
#
def load_image_folder_cv2(folder_path, label):
    X, y = [], []
    pattern = os.path.join(folder_path, '*.jpg')
    for img_path in glob.glob(pattern):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue  
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
        arr = img_resized.astype(np.float32).flatten()
        X.append(arr)
        y.append(label)
    return X, y

X_train, y_train = [], []
X_test,  y_test  = [], []

for cls, lbl in [('cats', 0), ('dogs', 1)]:
    Xt, yt = load_image_folder_cv2(os.path.join(TRAIN_DIR, cls), lbl)
    X_train += Xt;  y_train += yt

for cls, lbl in [('cats', 0), ('dogs', 1)]:
    Xt, yt = load_image_folder_cv2(os.path.join(TEST_DIR, cls), lbl)
    X_test  += Xt;  y_test  += yt

X_train = np.vstack(X_train)
y_train = np.array(y_train)
X_test  = np.vstack(X_test)
y_test  = np.array(y_test)

print("Distribuição original (treino):", Counter(y_train))
print("Distribuição original (teste):",  Counter(y_test))

sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
print("Após SMOTE (treino):", Counter(y_train_bal))

pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('clf', DecisionTreeClassifier(random_state=42))
])

param_grid = {
    'clf__criterion':        ['gini', 'entropy', 'log_loss'],
    'clf__splitter':         ['best', 'random'],
    'clf__max_depth':        [None, 5, 10, 20],
    'clf__min_samples_split':[2, 5, 0.5, 0.7],
    'clf__min_samples_leaf': [1, 2, 0.5, 0.7],
    'clf__max_features':     ['auto', 'sqrt', 'log2', None, 0.5, 0.7]
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='accuracy',
    return_train_score=True,
    verbose=2,
    n_jobs=-1
)
grid.fit(X_train_bal, y_train_bal)

print("Melhores parâmetros:", grid.best_params_)
print("Melhor acurácia no CV (treino):", grid.best_score_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

acc   = accuracy_score(y_test, y_pred)
prec  = precision_score(y_test, y_pred, average='macro')
rec   = recall_score(y_test, y_pred, average='macro')
f1    = f1_score(y_test, y_pred, average='macro')

print(f"Acurácia: {acc:.4f}")
print(f"Precisão (macro): {prec:.4f}")
print(f"Recall    (macro): {rec:.4f}")
print(f"F1-score  (macro): {f1:.4f}\n")

print("Relatório completo:\n", classification_report(y_test, y_pred, target_names=['cats','dogs']))

disp = ConfusionMatrixDisplay.from_estimator(
    best_model, X_test, y_test,
    display_labels=['cats','dogs'],
    normalize=None
)
plt.title("Matriz de Confusão")
plt.show()

os.makedirs('Classificador/dados', exist_ok=True)
dump(grid.best_estimator_.named_steps['scaler'],
     open('Classificador/dados/image_scaler.pkl', 'wb'))
dump(best_model.named_steps['clf'],
     open('Classificador/dados/image_tree_model.pkl', 'wb'))
