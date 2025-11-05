from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# =====================================================
# Flask App Configuration
# =====================================================
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# =====================================================
# Model Loading Section
# =====================================================
MODEL_DIR = os.path.join(os.getcwd(), "models")  # Use backend/models/
print("📁 Loading models from:", MODEL_DIR)

# Load all your models
models = {
    "VGG16": load_model(os.path.join(MODEL_DIR, "VGG16_model.h5")),
    "ResNet50": load_model(os.path.join(MODEL_DIR, "ResNet50_model.h5")),
    "MobileNetV2": load_model(os.path.join(MODEL_DIR, "MobileNetV2_model.h5")),
}

# =====================================================
# Class Labels
# =====================================================
class_labels = ['Benign', 'Malignant', 'Normal']

# =====================================================
# Image Preprocessing Function
# =====================================================
def preprocess_image(img_path):
    """Load and preprocess an image for model prediction."""
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0,1]
    return img_array

# =====================================================
# API ROUTE: Predict Endpoint
# =====================================================
@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload and returns predictions."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    model_name = request.form.get('model')  # e.g., 'VGG16'

    if model_name not in models:
        return jsonify({"error": f"Invalid model name. Choose from {list(models.keys())}"}), 400

    # Save image temporarily
    img_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(img_path)

    # Preprocess image
    img_array = preprocess_image(img_path)

    # Prediction
    model = models[model_name]
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    # Clean up
    os.remove(img_path)

    return jsonify({
        "model": model_name,
        "predicted_class": predicted_class,
        "confidence": float(np.max(predictions))
    })

# =====================================================
# Run the Flask App
# =====================================================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
