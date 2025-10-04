import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = "webapp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = None

def set_model(loaded_model):
    """Permite inyectar el modelo entrenado desde main.py"""
    global model
    model = loaded_model

def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array  
    img_array = img_array.astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No se envió archivo"})
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Nombre de archivo vacío"})

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Predecir
        img_array = preprocess_image(filepath)
        preds = model.predict(img_array)
        pred_digit = int(np.argmax(preds))

        return jsonify({"prediction": pred_digit})

    return render_template("index.html")
