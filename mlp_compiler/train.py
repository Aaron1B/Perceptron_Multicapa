def train_and_save_model():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Oculta warnings de TensorFlow
    from .interpreter import compile_model
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    # Cargar MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], -1)) / 255.0
    x_test = x_test.reshape((x_test.shape[0], -1)) / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    # Arquitectura por defecto
    arch = "Dense(256, relu) -> Dense(128, relu) -> Dense(10, softmax)"
    model = compile_model(arch, input_shape=(x_train.shape[1],))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_split=0.1, epochs=5, batch_size=128)
    model.save("mlp_model.h5")
    return model
