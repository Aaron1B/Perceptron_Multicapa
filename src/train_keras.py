"""
Entrenamiento y evaluación de un modelo MLP en MNIST usando el compilador.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Oculta warnings de TensorFlow

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from .compiler import compile_model


def load_mnist(flatten=False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    if flatten:
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def train_and_evaluate(arch, epochs=5, batch_size=128, flatten=True):
    (x_train, y_train), (x_test, y_test) = load_mnist(flatten=flatten)
    input_shape = (x_train.shape[1],) if flatten else (28, 28)

    model = compile_model(arch, input_shape=input_shape)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(x_train, y_train, validation_split=0.1,
                        epochs=epochs, batch_size=batch_size)

    print("\nEvaluación final en test:")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Loss={test_loss:.4f}, Test Accuracy={test_acc:.4f}")

    # Graficar evolución
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.show()

    return model, history
