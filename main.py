import os
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Ensure the model file exists
model_path = 'cnn_model.keras'
if not os.path.exists(model_path):
    raise ValueError(f"Model file not found: {model_path}")

# Load the trained model
model = load_model(model_path)

# Define class names for CIFAR-10 dataset
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def preprocess_image(image_bytes):
    # Open the image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Resize the image to 32x32
    img = img.resize((32, 32))
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Normalize the image
    img_array = img_array.astype('float32') / 255.0
    # Expand dimensions to match the model input shape
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/Predict")

async def predict(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    # Preprocess the image
    img_array = preprocess_image(contents)
    # Make a prediction
    predictions = model.predict(img_array)
    # Get the class with the highest probability
    predicted_class = class_names[np.argmax(predictions[0])]
    return {"predicted_class": predicted_class}



 