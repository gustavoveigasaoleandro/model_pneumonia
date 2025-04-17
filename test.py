import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dotenv import load_dotenv
import os

# Carrega variáveis de ambiente do .env
load_dotenv()

# Caminhos sensíveis via variáveis de ambiente
base_dir = os.getenv('BASE_DIR')
model_path = os.getenv('BEST_MODEL_CHECKPOINT')

# 1. Carrega o modelo treinado
model = tf.keras.models.load_model(model_path)

# 2. Carrega e pré-processa o dataset de teste
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=os.path.join(base_dir, "test"),
    image_size=(224, 224),
    color_mode="rgb",
    label_mode="int",
    batch_size=32,
    shuffle=False  # ← manter ordem para análise de métricas
)

# Aplica preprocessamento do DenseNet
test_ds = test_ds.map(lambda x, y: (
    tf.keras.applications.densenet.preprocess_input(x), y))

# 3. Extrai os dados reais e as previsões
y_true = []
y_probs = []

for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    y_probs.extend(probs.squeeze())
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_probs = np.array(y_probs)

# 4. Avalia para diferentes thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

print("\n====== Avaliação por Threshold ======\n")
for t in thresholds:
    y_pred = (y_probs >= t).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Threshold: {t}")
    print(f"Acurácia    : {acc:.4f}")
    print(f"Precisão    : {prec:.4f}")
    print(f"Sensibilidade (Recall): {rec:.4f}")
    print(f"F1-Score    : {f1:.4f}")
    print("Matriz de Confusão:")
    print(cm)
    print("-" * 40)
