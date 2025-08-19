import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = "saved_models/best_model.keras"

# Hyperparameters
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Adenocarcinoma", "Large Cell Carcinoma", "Squamous Cell Carcinoma"]

def preprocess_image(img_path, img_size=IMG_SIZE):
    """Load and preprocess an image for prediction."""
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    img_array = img_array / 255.0  # normalize
    return img_array, img

def predict_image(img_path):
    print(f"🔹 Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"🔹 Processing image: {img_path}")
    img_array, img = preprocess_image(img_path)

    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    # Show results
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prediction: {CLASS_NAMES[pred_class]} ({confidence*100:.2f}%)")
    plt.show()

if __name__ == "__main__":
    # Example usage: change the path to a test image from your dataset
    test_image_path = "dataset/adenocarcinoma/adenocarcinoma_001.jpg"
    predict_image(test_image_path)