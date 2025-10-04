"""
Implementación OOP de un MLP simple (solo forward) usando NumPy.
"""

import numpy as np


class Activation:
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        ex = np.exp(x - np.max(x, axis=1, keepdims=True))
        return ex / np.sum(ex, axis=1, keepdims=True)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def get(name):
        name = (name or "linear").lower()
        mapping = {
            "sigmoid": Activation.sigmoid,
            "relu": Activation.relu,
            "softmax": Activation.softmax,
            "tanh": Activation.tanh,
            "linear": Activation.linear,
        }
        return mapping.get(name, Activation.linear)


class DenseLayer:
    def __init__(self, input_dim, units, activation=None, seed=None):
        rng = np.random.RandomState(seed)
        limit = np.sqrt(6.0 / (input_dim + units))
        self.W = rng.uniform(-limit, limit, size=(input_dim, units))
        self.b = np.zeros((1, units))
        self.activation_name = activation
        self.activation = Activation.get(activation)

    def forward(self, X):
        z = X.dot(self.W) + self.b
        return self.activation(z)

    def __repr__(self):
        return f"DenseLayer(in={self.W.shape[0]}, out={self.W.shape[1]}, act={self.activation_name})"


class MLP:
    def __init__(self):
        self.layers = []

    def add(self, layer: DenseLayer):
        self.layers.append(layer)

    def predict(self, X):
        out = X
        for l in self.layers:
            out = l.forward(out)
        return out

    def summary(self):
        total = 0
        for l in self.layers:
            params = l.W.size + l.b.size
            total += params
            print(f"{l} - params={params}")
        print(f"Total params: {total}")
