import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
import os

# Ensure model directory exists
MODEL_DIR = os.path.join(os.getcwd(), 'models')

if not os.path.exists(MODEL_DIR):
    raise Exception("Model directory not found. Ensure models are present in 'models/'.")

# Load the trained models
tumor_detection_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'Detection_Model.keras'))
tumor_classification_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'Classification_Model.keras'))

# Define the tumor types (the class labels used during training)
class_labels = ['glioma', 'meningioma', 'pituitary', 'notumor']

def predict_tumor_and_type(img):
    """Predicts whether a tumor is present and classifies its type if found."""
    img = img.resize((224, 224))  
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    # Tumor detection prediction (binary classification)
    detection_prediction = tumor_detection_model.predict(img_array)

    if detection_prediction <= 0.5:  
        classification_prediction = tumor_classification_model.predict(img_array)
        predicted_class = np.argmax(classification_prediction)  
        tumor_type = class_labels[predicted_class]
        return f"Tumor detected. Type: {tumor_type}"

    return "No tumor detected."

# Streamlit App
st.title("Tumor Detection and Classification")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    img = image.load_img(BytesIO(uploaded_file.read()))  
    
    # Run prediction
    result = predict_tumor_and_type(img)

    # Display result
    st.image(img, caption='Uploaded Image', use_container_width=True)
    st.write(result)

