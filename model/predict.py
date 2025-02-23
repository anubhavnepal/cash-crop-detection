import numpy as np
import pandas as pd
from django.http import HttpResponseServerError
from pathlib import Path
import os
import pickle
from PIL import Image
import cupy as cp

model_path = Path("model/model_files/88_cnn_model.pkl")
diseases_file = Path("model/diseases.xlsx")

def load_model_custom(model_path):
    ext = os.path.splitext(model_path)[1].lower()
    if ext == '.pkl':
        try:
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == '__main__':
                        try:
                            import model.cnn as cnn
                            
                            return getattr(cnn, name)
                        except AttributeError:
                            pass

                    return super().find_class(module, name)
            
            with open(model_path, 'rb') as file:
                model = CustomUnpickler(file).load()
            print("Model loaded using custom pickle unpickler.")
        except Exception as e:
            return HttpResponseServerError(f"Error loading pkl model: {e}")
    else:
        return HttpResponseServerError("Only .pkl files are supported for model loading.")
    return model


# Load the trained model using the custom loader
model = load_model_custom(model_path)

# Define class names (you can later load these from your Excel file if needed)
class_names = [
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'coffee_healthy', 'coffee_miner', 'coffee_phoma', 'sugarcane_Bacterial Blight',
    'sugarcane_Healthy', 'sugarcane_Red Rot', 'tea_bird eye spot',
    'tea_healthy', 'tea_red leaf spot'
]

num_classes = len(class_names)

# Preprocess the input image
def preprocess_image(image_path):
    """Preprocess the image for prediction."""
    img = Image.open(image_path)
    img = img.resize((224, 224))  
    img_array = np.array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    return img_array

# Predict the class of an image
def predict_disease(image_path):
    """Predict the class of an input image."""
    img_array = preprocess_image(image_path)
    
    # predictions is a tuple: (predicted_classes, outputs)
    predicted_classes, outputs = model.predict(img_array)
    
    # Convert predicted_classes to a numpy array (if it's a cupy array)
    predicted_classes = cp.asnumpy(predicted_classes)
    
    print("Predicted classes:", predicted_classes)
    
    # Use the predicted class directly
    if predicted_classes[0] >= len(class_names):
        raise ValueError(f"Invalid predicted class index: {predicted_classes[0]}")
    
    predicted_class = class_names[predicted_classes[0]]
    
    # If you want to include the raw outputs in your response, you can convert them as well.
    outputs_np = cp.asnumpy(outputs)
    
    return {"class": predicted_class, "probabilities": outputs_np.tolist()}

