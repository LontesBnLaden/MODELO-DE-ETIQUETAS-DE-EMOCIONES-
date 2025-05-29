import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 1. Cargar y preparar datos CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalizar imágenes a rango [0,1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# y_train y y_test están en formato (n,1), los dejamos así porque Keras lo acepta

# 2. Definir modelo CNN simple
def create_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = create_cnn_model()

# 3. Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Entrenar el modelo
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test),
                    batch_size=64)

# 5. Evaluar en test
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Precisión en test: {test_acc:.4f}")

# 6. Guardar modelo entrenado si quieres
model.save("modelo_cnn_cifar10.h5")
