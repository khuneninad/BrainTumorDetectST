import streamlit as st
import os
import django
import pandas as pd
from PIL import Image
from datetime import datetime

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'brain_tumor_ml.settings')
django.setup()

from Dashboard.models import Data  # Import Django model

# Set Streamlit page configuration
st.set_page_config(page_title="Brain Tumor Detection Dashboard", layout="wide")

# Title
st.title("🧠 Brain Tumor Detection Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["🏠 Home", "📊 View Predictions", "📤 Upload Image"])

# 🏠 Home Page (User Input Form)
if options == "🏠 Home":
    st.write("## 📝 Enter Patient Details for Prediction")

    # Create an input form
    with st.form("user_input_form"):
        name = st.text_input("👤 Name", "")
        age = st.number_input("🔢 Age", min_value=1, max_value=120, step=1)
        sex = st.radio("⚧ Sex", ["Male", "Female"])
        prediction = st.text_input("🔮 Prediction Result", "")
        
        # Convert sex to numeric format for database storage
        sex_numeric = 1 if sex == "Male" else 0

        # Submit button
        submit = st.form_submit_button("✅ Save Record")

        # If form is submitted, save to database
        if submit:
            if name and prediction:
                new_record = Data.objects.create(
                    name=name,
                    age=age,
                    sex=sex_numeric,
                    predictions=prediction,
                    date=datetime.now()
                )
                new_record.save()
                st.success("🎉 Record saved successfully!")
            else:
                st.error("⚠ Please fill in all required fields.")

# 📊 View Predictions Page
elif options == "📊 View Predictions":
    st.write("## 📊 Prediction Results")

    predictions = Data.objects.all()
    
    if predictions.exists():
        # Convert data to Pandas DataFrame for table display
        data_list = []
        for record in predictions:
            data_list.append({
                "Name": record.name,
                "Age": record.age,
                "Sex": "Male" if record.sex == 1 else "Female",
                "Prediction": record.predictions,
                "Date": record.date,
                "Image URL": record.tumor_Img.url if record.tumor_Img else None  # Handle optional images
            })

        df = pd.DataFrame(data_list)

        # Display Table
        st.dataframe(df[["Name", "Age", "Sex", "Prediction", "Date"]])

        # Display Images
        st.write("### 📷 Tumor Images")
        for record in predictions:
            if record.tumor_Img:
                st.image(record.tumor_Img.url, caption=f"Tumor Image of {record.name}", width=120)
    else:
        st.warning("⚠ No predictions available.")

# 📤 Upload Image Page
elif options == "📤 Upload Image":
    st.write("## 📤 Upload an Image for Prediction")
    
    uploaded_file = st.file_uploader("Choose a tumor MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success("✅ Image successfully uploaded. Please use the Django app for predictions.")

# Footer
st.sidebar.info("💡 Django Integrated Streamlit App")
