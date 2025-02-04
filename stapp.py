# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from io import BytesIO
# import os

# # Ensure model directory exists
# MODEL_DIR = os.path.join(os.getcwd(), 'models')

# if not os.path.exists(MODEL_DIR):
#     raise Exception("Model directory not found. Ensure models are present in 'models/'.")

# # Load the trained models
# tumor_detection_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'Detection_Model.keras'))
# tumor_classification_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'Classification_Model.keras'))

# # Define the tumor types (the class labels used during training)
# class_labels = ['glioma', 'meningioma', 'pituitary', 'notumor']

# def predict_tumor_and_type(img):
#     """Predicts whether a tumor is present and classifies its type if found."""
#     img = img.resize((224, 224))  
#     img_array = np.array(img) / 255.0  
#     img_array = np.expand_dims(img_array, axis=0)  

#     # Tumor detection prediction (binary classification)
#     detection_prediction = tumor_detection_model.predict(img_array)

#     if detection_prediction <= 0.5:  
#         classification_prediction = tumor_classification_model.predict(img_array)
#         predicted_class = np.argmax(classification_prediction)  
#         tumor_type = class_labels[predicted_class]
#         return f"Tumor detected. Type: {tumor_type}"

#     return "No tumor detected."

# # Streamlit App
# st.set_page_config(page_title="Tumor Detection", page_icon="🧠", layout="centered")
# st.title("Tumor Detection and Classification")

# # Upload Image
# uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

# if uploaded_file is not None:
#     img = image.load_img(BytesIO(uploaded_file.read()))  
    
#     # Run prediction
#     result = predict_tumor_and_type(img)

#     # Display result
#     st.image(img, caption='Uploaded Image', use_container_width=True)
#     st.write(result)
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
import os

MODEL_DIR = os.path.join(os.getcwd(), 'models')

# Check if model directory exists
if not os.path.exists(MODEL_DIR):
    st.error("Model directory not found. Ensure models are present in 'models/'.")
    st.stop()

# Print contents of model directory for debugging
st.write("Model directory contents:", os.listdir(MODEL_DIR))

# Load the detection model
try:
    detection_model_path = os.path.join(MODEL_DIR, 'Detection_Model.keras')
    if not os.path.exists(detection_model_path):
        st.error(f"Detection model file not found at {detection_model_path}")
        st.stop()

    tumor_detection_model = tf.keras.models.load_model(detection_model_path)
    st.success("Detection model loaded successfully.")

except Exception as e:
    st.error(f"Error loading Detection Model: {str(e)}")
    st.stop()

# Load the classification model
try:
    classification_model_path = os.path.join(MODEL_DIR, 'Classification_Model.keras')
    if not os.path.exists(classification_model_path):
        st.error(f"Classification model file not found at {classification_model_path}")
        st.stop()

    tumor_classification_model = tf.keras.models.load_model(classification_model_path)
    st.success("Classification model loaded successfully.")

except Exception as e:
    st.error(f"Error loading Classification Model: {str(e)}")
    st.stop()

# Define the tumor types (the class labels used during training)
class_labels = ['glioma', 'meningioma', 'pituitary', 'notumor']

def predict_tumor_and_type(img):
    """Predicts whether a tumor is present and classifies its type if found."""
    try:
        img = img.resize((224, 224))  
        img_array = np.array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  

        # Tumor detection prediction (binary classification)
        detection_prediction = tumor_detection_model.predict(img_array)

        if detection_prediction <= 0.5:  
            classification_prediction = tumor_classification_model.predict(img_array)
            predicted_class = np.argmax(classification_prediction)  
            tumor_type = class_labels[predicted_class]
            return f"🧠 Tumor detected. Type: {tumor_type}"

        return "✅ No tumor detected."

    except Exception as e:
        return f"Error processing image: {str(e)}"

# Streamlit App
st.set_page_config(page_title="Tumor Detection", page_icon="🧠", layout="centered")
st.title("Tumor Detection and Classification")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    try:
        img = image.load_img(BytesIO(uploaded_file.getvalue()))  # Corrected image loading method
        
        # Run prediction
        result = predict_tumor_and_type(img)

        # Display result
        st.image(img, caption='Uploaded Image', use_container_width=True)
        st.write(result)

    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

# Streamlit automatically runs, no need for `st.run()`

# # Ensure Streamlit runs on port 8080
# if __name__ == "__main__":
#     st.write("Running on AWS with port 8080")
#     st.run()
