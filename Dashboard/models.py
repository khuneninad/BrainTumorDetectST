from django.db import models
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from django.core.files.storage import default_storage

# Define constants for gender and tumor types
GENDER = (
    (0, 'Female'),
    (1, 'Male'),
)

TUMOR_TYPES = {
    0: 'Glioma',
    1: 'Meningioma',
    2: 'Pituitary'
}

class Data(models.Model):
    name = models.CharField(max_length=100, null=True)
    age = models.PositiveIntegerField(null=True)
    sex = models.PositiveIntegerField(choices=GENDER, null=True)
    tumor_Img = models.ImageField(upload_to='tumor_images/')
    predictions = models.CharField(max_length=100, blank=True)
    date = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        # Load both detection and classification models
        detection_model = load_model('ml_model/Detection_Model.keras')  # Ensure correct path
        classification_model = load_model('ml_model/Classification_Model.keras')  # Ensure correct path

        # Save image temporarily to process it
        img_path = default_storage.save(self.tumor_Img.name, self.tumor_Img)
        img_full_path = os.path.join(default_storage.location, img_path)

        try:
            # Load and preprocess the image
            img = Image.open(img_full_path).convert('RGB').resize((224, 224))  # Ensure RGB format
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Debug: Show the image array shape and the first few values
            print(f"Image array shape: {img_array.shape}")
            print(f"Image array first 5 values: {img_array[0][0][:5]}")  # Show first 5 pixel values

            # Detect if there is a tumor (detection model)
            tumor_probability = detection_model.predict(img_array)[0][0]  # Probability of having a tumor
            print(f"Tumor detection probability: {tumor_probability}")  # Debugging output

            if tumor_probability > 0.3:  # If the model detects a tumor with a probability above 0.5
                # Classify the tumor (classification model)
                classification_result = classification_model.predict(img_array)
                print(f"Classification result: {classification_result}")  # Debugging output

                # Get the predicted class (0-2 for tumor types)
                class_label = np.argmax(classification_result, axis=1)[0]
                print(f"Predicted class label: {class_label}")  # Debugging output

                # Map the label to tumor type and set the predictions field
                self.predictions = TUMOR_TYPES.get(class_label, 'Unknown Type')
            else:
                # No tumor detected, no need to set anything
                self.predictions = "No tumor detected"

            print(f"Predictions: {self.predictions}")  # Final prediction result

        except Exception as e:
            print(f"Error during prediction: {e}")

        finally:
            # Clean up the saved image after processing
            if os.path.exists(img_full_path):
                os.remove(img_full_path)

        # Save the instance
        super().save(*args, **kwargs)

    class Meta:
        ordering = ['-date']

    def __str__(self):
        return self.name
