from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model
model = load_model('model.h5')

# Class labels in the same order the model was trained with
class_labels = ['Benign', 'Malignant', 'Normal']


@app.route('/')
def home():
    return "Lung Prediction Model is live!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Make sure input is a list of numbers
    features = np.array(data['features']).reshape(1, -1)

    prediction = model.predict(features)  # Output: probabilities
    predicted_index = int(np.argmax(prediction))
    predicted_label = class_labels[predicted_index]

    return jsonify({'result': predicted_label})


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
