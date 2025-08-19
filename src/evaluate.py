import tensorflow as tf
import numpy as np
from utils.data_loader import get_datasets
from utils.metrics import plot_confusion_matrix, print_classification_report

def evaluate_model():
    # Load test dataset
    _, _, test_ds, class_names = get_datasets(data_dir="../dataset")

    # Load the best model
    model = tf.keras.models.load_model("../models/best_model.keras")
    print("Loaded trained model")

    # Prepare ground-truth labels and predictions
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Show evaluation metrics
    print_classification_report(y_true, y_pred, class_names)
    plot_confusion_matrix(y_true, y_pred, class_names, save_path="../results/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()
