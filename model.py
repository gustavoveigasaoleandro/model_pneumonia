
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras._tf_keras.keras import layers, models
import os
from dotenv import load_dotenv


base_dir = os.getenv('BASE_DIR')
model_save_path = os.getenv('MODEL_SAVE_PATH')
checkpoint_path = os.getenv('BEST_MODEL_CHECKPOINT')
# 1. Data Augmentation com maior variedade
# Melhor diversidade para ajudar o modelo a generalizar


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])

# 2. Carrega imagens normais com augmentação + replicação
normal_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=os.path.join(base_dir, "train"),
    labels='inferred',
    label_mode='int',
    image_size=(224, 224),  # padrão para EfficientNetB0
    color_mode='rgb',       # RGB para modelos pré-treinados
    batch_size=32,
    shuffle=True
).filter(lambda x, y: tf.equal(y[0], 0))

normal_ds = normal_ds.unbatch().batch(32, drop_remainder=True)

augmented_normal_ds = normal_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y))

'''
# Replicação para balanceamento
augmented_normal_ds = augmented_normal_ds.concatenate(
    augmented_normal_ds).concatenate(augmented_normal_ds)
'''


# 3. Carrega imagens com pneumonia (sem aumento)
pneumonia_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=os.path.join(base_dir, "train"),
    labels='inferred',
    label_mode='int',
    image_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    shuffle=True
).filter(lambda x, y: tf.equal(y[0], 1))

pneumonia_ds = pneumonia_ds.unbatch().batch(32, drop_remainder=True)

# 4. Divide em treino/validação
normal_batches = sum(1 for _ in augmented_normal_ds)
pneumonia_batches = sum(1 for _ in pneumonia_ds)

normal_train_size = int(0.8 * normal_batches)
pneumonia_train_size = int(0.8 * pneumonia_batches)

normal_train = augmented_normal_ds.take(normal_train_size)
normal_val = augmented_normal_ds.skip(normal_train_size)

pneumonia_train = pneumonia_ds.take(pneumonia_train_size)
pneumonia_val = pneumonia_ds.skip(pneumonia_train_size)

train_ds_final = normal_train.concatenate(pneumonia_train)
val_ds = normal_val.concatenate(pneumonia_val).shuffle(500)

# 5. Pré-processamento usando função da densenet
train_ds = train_ds_final.map(lambda x, y: (
    tf.keras.applications.densenet.preprocess_input(x), tf.cast(y, tf.float32)))
val_ds = val_ds.map(lambda x, y: (
    tf.keras.applications.densenet.preprocess_input(x), tf.cast(y, tf.float32)))
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=os.path.join(base_dir, "train"),
    image_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    label_mode="int")
test_ds = test_ds.map(lambda x, y: (
    tf.keras.applications.densenet.preprocess_input(x), y))

train_steps = sum(1 for _ in train_ds)
train_ds = train_ds.shuffle(
    buffer_size=1000).repeat().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


class_weights = {
    0: 1.944,
    1: 0.673
}
print("class_weights:", class_weights)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_auc',         # ou 'val_auc', se quiser priorizar
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)


# 1. Carrega EfficientNetB0 sem o top (camada final de classificação do ImageNet)
base_model = tf.keras.applications.DenseNet121(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
base_model.trainable = False  # ← ETAPA 1: Congelado

# 2. Cria o modelo completo com "cabeça personalizada"
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# 3. Compila o modelo com taxa de aprendizado padrão
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# 4. Treina apenas a cabeça do modelo
print("\n==== ETAPA 1: Treinando só a cabeça ====\n")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    steps_per_epoch=train_steps,
    callbacks=[early_stopping],
    class_weight=class_weights,
    verbose=1
)

# 5. Descongela parcialmente a EfficientNetB0 (últimos blocos)
# Escolhe quantas camadas finais quer treinar
fine_tune_at = 400  # Ex: descongelar da camada 200 em diante

base_model.trainable = True
# ← Descongela só as últimas 20 camadas
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# 6. Compila novamente com taxa de aprendizado menor
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # ← taxa menor!
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc', curve='ROC')]
)

# 7. Fine-tuning: treina o modelo completo (ajuste fino)
print("\n==== ETAPA 2: Fine-tuning ====\n")
historico = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    steps_per_epoch=train_steps,
    callbacks=[checkpoint],
    class_weight=class_weights,
    verbose=1
)

# Avaliação
print("\n== AVALIAÇÃO NO TESTE ==")
test_loss, test_accuracy, auc = model.evaluate(test_ds)
print(f"Loss no teste: {test_loss:.4f}")
print(f"Acurácia no teste: {test_accuracy:.4f}")

# Salva modelo
model.save(model_save_path)

# Gráficos
plt.figure()
plt.plot(historico.history['accuracy'])
plt.plot(historico.history['val_accuracy'])
plt.title('Acurácia por época')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend(['Treino', 'Validação'])
plt.grid(True)
plt.show()

plt.figure()
plt.plot(historico.history['loss'])
plt.plot(historico.history['val_loss'])
plt.title('Perda (Loss) por época')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend(['Treino', 'Validação'])
plt.grid(True)
plt.show()
