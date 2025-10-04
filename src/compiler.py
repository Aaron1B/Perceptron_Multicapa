"""
Compilador de descripciones de redes -> tf.keras.Sequential.
Ejemplo de entrada:
"Dense(256, relu) -> Dense(128, relu) -> Dense(10, softmax)"
"""

import re
import tensorflow as tf

LAYER_PATTERN = re.compile(r'(\w+)\s*\(\s*([0-9]+)\s*(?:,\s*([a-zA-Z0-9_]+)\s*)?\)')


def parse_layer(token: str):
    m = LAYER_PATTERN.match(token.strip())
    if not m:
        raise ValueError(f"Token inválido: {token}")
    layer_type, units, act = m.group(1), int(m.group(2)), (m.group(3) or "linear").lower()
    return {"type": layer_type, "units": units, "activation": act}


def compile_model(arch_string: str, input_shape=None):
    tokens = [t.strip() for t in arch_string.split("->")]
    model = tf.keras.Sequential()

    if input_shape is not None:
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        if len(input_shape) > 1:
            model.add(tf.keras.layers.Flatten())

    for t in tokens:
        spec = parse_layer(t)
        if spec["type"].lower() == "dense":
            model.add(tf.keras.layers.Dense(spec["units"], activation=spec["activation"]))
        else:
            raise NotImplementedError(f"Capa no soportada: {spec['type']}")

    return model
