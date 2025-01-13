from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
import io
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Página para cargar el archivo JSON

@app.route('/predict', methods=['POST'])
def predict():
    # Leer el archivo JSON cargado desde el formulario
    json_file = request.files['file']

    # Leer el contenido del archivo JSON
    try:
        img_data = json_file.read()
        img = Image.open(io.BytesIO(img_data))
        img_array = np.array(img)

        # Usar DeepFace para analizar la emoción
        result = DeepFace.analyze(img_array, actions=['emotion'])

        # Extraer la emoción con mayor probabilidad
        dominant_emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][dominant_emotion]

        # Devolver el resultado en formato JSON
        return jsonify({'emotion': dominant_emotion, 'confidence': confidence})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8500)
