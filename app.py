from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

model = load_model('model.h5')
class_labels = ['Benign', 'Malignant', 'Normal']

@app.route('/')
def home():
    return "Lung Prediction Model is live!"

@app.route('/predict', methods=['POST'])
def predict():
    print(f"Request method: {request.method}")
    print(f"Files: {request.files}")
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    print("Image received successfully")
    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')
    
    # Resize based on model input shape (for example 224x224)
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 224, 224, 3)

    prediction = model.predict(image_array)
    predicted_index = int(np.argmax(prediction))
    predicted_label = class_labels[predicted_index]

    return jsonify({'result': predicted_label})
@app.route('/test', methods=['POST'])
def test():
    print("POST request arrived at /test")
    return jsonify({"message": "POST received"})
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)