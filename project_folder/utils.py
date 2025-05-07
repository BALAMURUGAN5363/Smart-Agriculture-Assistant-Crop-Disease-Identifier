import cv2
import numpy as np
import tensorflow as tf
import os

# Load model once during import
MODEL_PATH = "model/crop_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Labels should exactly match your model’s output
labels = ["Blight","Common_Rust","Gray_Leaf_Spot","Healthy"]  # Adjust as per your model

def predict_crop_and_disease(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    try:
        # Read and preprocess image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224)) / 255.0
        image = np.expand_dims(image, axis=0)

        # Model prediction
        predictions = model.predict(image)
        idx = np.argmax(predictions)
        label = labels[idx]
        confidence = float(predictions[0][idx]) * 100

        # Split label
        crop, disease = label.split("_")
        return crop.capitalize(), disease.capitalize(), round(confidence, 2)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Unknown", "Unknown", 0.0
