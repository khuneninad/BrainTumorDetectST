import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO

# Load the trained models
tumor_detection_model = tf.keras.models.load_model('models/Detection_Model.keras')
tumor_classification_model = tf.keras.models.load_model('models/Classification_Model.keras')

# Define the tumor types (the class labels used during training)
class_labels = ['glioma', 'meningioma', 'pituitary', 'notumor']

# Define allowed extensions for the file upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_tumor_and_type(img):
    # Load and preprocess the image for tumor detection
    img = img.resize((224, 224))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Convert to array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make the tumor detection prediction (binary classification)
    detection_prediction = tumor_detection_model.predict(img_array)

    if detection_prediction <= 0.5:  # Threshold for binary classification (tumor detected)
        # If tumor is detected, classify the tumor type
        classification_prediction = tumor_classification_model.predict(img_array)
        predicted_class = np.argmax(classification_prediction)  # Get the class with highest probability
        tumor_type = class_labels[predicted_class]
        return f"Tumor detected. Type: {tumor_type}"

    return "No tumor detected."

# Streamlit App
st.title("Tumor Detection and Classification")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Use BytesIO to handle the image in memory without saving it
    img = image.load_img(BytesIO(uploaded_file.read()))  # Load image from the uploaded file
    
    # Run prediction
    result = predict_tumor_and_type(img)

    # Display result
    st.image(img, caption='Uploaded Image', use_container_width=True)
    st.write(result)
