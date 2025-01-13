from flask import Flask, request, render_template, jsonify, redirect
import numpy as np
from PIL import Image
import io
import base64
import os
import json
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

headers = {"content-type": "application/json"}
label_to_text = {0: 'Ira', 1: 'Odio', 2: 'Miedo', 3: 'Felicidad', 4: 'Tristeza', 5: 'Sorpresa', 6: 'Neutral'}

def preprocess_image(img_path, target_size):
    img = Image.open(img_path).convert('L')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array

def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return redirect(request.url)

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        img_facialexpression = preprocess_image(file_path, target_size=(48, 48))
        try:
            img_facialexpr_list = img_facialexpression.tolist()
            data = json.dumps({"signature_name": "serving_default", "instances": img_facialexpr_list})
            json_response2 = requests.post('https://tfexpressions-v1.onrender.com/v1/models/saved_model/versions/2:predict', data=data, headers=headers)
            json_response2.raise_for_status()
            facial_express = np.argmax(json.loads(json_response2.text)['predictions'], axis=1)
            emotion_text = label_to_text[int(facial_express)]
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"No se pudo conectar con el servidor de TensorFlow Serving. Detalles: {e}"})

        return jsonify({"emotion": emotion_text})

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"error": "No file uploaded"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    img_facialexpression = preprocess_image(file_path, target_size=(48, 48))
    try:
        img_facialexpr_list = img_facialexpression.tolist()
        data = json.dumps({"signature_name": "serving_default", "instances": img_facialexpr_list})
        json_response2 = requests.post('https://tfexpressions-v1.onrender.com/v1/models/saved_model/versions/2:predict', data=data, headers=headers)
        json_response2.raise_for_status()
        facial_express = np.argmax(json.loads(json_response2.text)['predictions'], axis=1)
        emotion_text = label_to_text[int(facial_express)]
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"No se pudo conectar con el servidor de TensorFlow Serving. Detalles: {e}"})

    return jsonify({"emotion": emotion_text})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=os.getenv('PORT', default=5000))
