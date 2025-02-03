from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from io import BytesIO
from werkzeug.utils import secure_filename
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained models (ensure models exist in the 'models/' directory)
MODEL_DIR = os.path.join(os.getcwd(), 'models')

if not os.path.exists(MODEL_DIR):
    raise Exception("Model directory not found. Ensure models are present in 'models/'.")

tumor_detection_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'Detection_Model.keras'))
tumor_classification_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'Classification_Model.keras'))

# Define the tumor types (the class labels used during training)
class_labels = ['glioma', 'meningioma', 'pituitary', 'notumor']

# Define allowed extensions for file upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_tumor_and_type(img):
    """Predicts whether a tumor is present and classifies its type if found."""
    # Preprocess the image for the model
    img = img.resize((224, 224))  
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    # Tumor detection prediction (binary classification)
    detection_prediction = tumor_detection_model.predict(img_array)

    if detection_prediction <= 0.5:  
        # If tumor is detected, classify the tumor type
        classification_prediction = tumor_classification_model.predict(img_array)
        predicted_class = np.argmax(classification_prediction)  
        tumor_type = class_labels[predicted_class]
        return f"Tumor detected. Type: {tumor_type}"

    return "No tumor detected."

@app.route('/')
def index():
    return render_template('index.html')  # Frontend form for file upload

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        img = image.load_img(BytesIO(file.read()))  
        
        # Run prediction
        prediction_result = predict_tumor_and_type(img)

        return prediction_result

    return "Invalid file type", 400

if __name__ == '__main__':
    # Run Flask app on 0.0.0.0 to allow external connections
    app.run(debug=True, host="0.0.0.0", port=8080)
