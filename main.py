import os
from src.train_keras import train_and_evaluate
from webapp.app import app
import tensorflow as tf

MODEL_PATH = os.path.join("model", "mnist_mlp.h5")
os.makedirs("model", exist_ok=True)

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(">>> No se encontró el modelo, entrenando uno nuevo...")
        arch = "Dense(256, relu) -> Dense(128, relu) -> Dense(10, softmax)"
        model, _ = train_and_evaluate(arch, epochs=5, flatten=True)
        model.save(MODEL_PATH)
        print(f">>> Modelo entrenado y guardado en {MODEL_PATH}")
    else:
        print(f">>> Modelo encontrado en {MODEL_PATH}, cargando...")

    
    print(">>> Iniciando servidor Flask en http://127.0.0.1:5000")
    app.run(debug=True)
