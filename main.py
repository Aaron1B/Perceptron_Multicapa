"""
Punto de entrada del proyecto.
Ejecuta un MLP con MNIST y la sintaxis definida.
"""

from src.train_keras import train_and_evaluate

if __name__ == "__main__":
    architecture = "Dense(256, relu) -> Dense(128, relu) -> Dense(10, softmax)"
    train_and_evaluate(architecture, epochs=5, flatten=True)
