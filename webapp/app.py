import os
import base64
import io
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "webapp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = None

def set_model(loaded_model):
    global model
    model = loaded_model
    print(" Modelo cargado en Flask")

def preprocess_image(image):
    img = image.convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = img_array.astype("float32") / 255.0
    img_array = img_array.reshape(1, 784)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if "file" not in request.files:
                return jsonify({"error": "No se envió archivo"})
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "Nombre de archivo vacío"})

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            try:
                img = Image.open(filepath)
            except Exception as e:
                print(f" Error al abrir la imagen: {e}")
                return jsonify({"error": "No se pudo abrir la imagen. El formato puede no ser válido."}), 400
            img_array = preprocess_image(img)

            preds = model.predict(img_array)
            pred_digit = int(np.argmax(preds))

            return jsonify({"prediction": pred_digit})
        except Exception as e:
            print(" Error en index:", e)
            return jsonify({"error": str(e)}), 500

    return render_template("index.html")

@app.route("/predict_base64", methods=["POST"])
def predict_base64():
    try:
        data = request.get_json()
        if "image" not in data:
            return jsonify({"error": "No se recibió la imagen"}), 400

        img_data = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_data)

        img = Image.open(io.BytesIO(img_bytes))
        img_array = preprocess_image(img)

        preds = model.predict(img_array)
        pred_digit = int(np.argmax(preds))

        return jsonify({"prediction": pred_digit})
    except Exception as e:
        print(" Error en predict_base64:", e)
        return jsonify({"error": str(e)}), 500
