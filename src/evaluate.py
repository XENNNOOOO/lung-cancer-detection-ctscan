import tensorflow as tf
import numpy as np
from utils.data_loader import get_datasets
from utils.metrics import plot_confusion_matrix, print_classification_report

# Master class list
MASTER_CLASSES = [
    "Normal",
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Squamous Cell Carcinoma",
    "Benign",
    "Malignant",
    "Other"
]

def evaluate_model():
    # Load test dataset from combined datasets
    _, _, test_ds, class_names = get_datasets(data_dir="../datasets")

    # Load the trained combined model
    model = tf.keras.models.load_model("../models/final_model.keras")
    print("Loaded trained model")

    # Prepare ground-truth labels and predictions
    y_true = []
    y_pred = []

    for images, labels in test_ds.unbatch():  # unbatch to get individual samples
        preds = model.predict(tf.expand_dims(images, axis=0), verbose=0)
        y_true.append(labels.numpy())
        y_pred.append(np.argmax(preds, axis=1)[0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Show evaluation metrics
    print_classification_report(y_true, y_pred, MASTER_CLASSES)
    plot_confusion_matrix(y_true, y_pred, MASTER_CLASSES, save_path="../results/confusion_matrix.png")
    print("Confusion matrix saved to ../results/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()