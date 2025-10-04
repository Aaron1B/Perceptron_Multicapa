from mlp_compiler.interpreter import compile_model
from mlp_compiler.train import train_and_save_model
from webapp import app
import tensorflow as tf
import os

if __name__ == "__main__":
    if os.path.exists("mlp_model.h5"):
        model = tf.keras.models.load_model("mlp_model.h5")
    else:
        model = train_and_save_model()

    app.set_model(model)

    app.app.run(debug=True)
