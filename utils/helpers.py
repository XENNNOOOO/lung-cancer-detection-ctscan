import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf

def create_dir(path):
    """Create directory if it doesn’t exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def save_history(history, filename="training_history.json"):
    """Save training history as JSON."""
    hist_dict = history.history
    with open(filename, "w") as f:
        json.dump(hist_dict, f)

def plot_history(history, save_path=None):
    """Plot training accuracy and loss curves."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Model Accuracy")

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Model Loss")

    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def preprocess(ds, num_classes):
    return ds.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)))

def augment(ds, data_augmentation):
    return ds.map(lambda x, y: (data_augmentation(x, training=True), y))
