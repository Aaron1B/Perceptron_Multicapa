import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Oculta warnings de TensorFlow

def compile_model(arch, input_shape):
    # Implementación básica de ejemplo
    import tensorflow as tf
    from tensorflow.keras import layers, models
    model = models.Sequential()
    for layer_def in arch.split('->'):
        layer_def = layer_def.strip()
        if layer_def.startswith('Dense'):
            params = layer_def[len('Dense('):-1].split(',')
            units = int(params[0].strip())
            activation = params[1].strip() if len(params) > 1 else None
            if not model.layers:
                model.add(layers.Dense(units, activation=activation, input_shape=input_shape))
            else:
                model.add(layers.Dense(units, activation=activation))
    return model
