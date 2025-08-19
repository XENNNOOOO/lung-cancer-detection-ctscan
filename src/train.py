import os
print("Current working directory:", os.getcwd())
print("Expected dataset path:", os.path.abspath("./dataset"))

import tensorflow as tf
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.data_loader import get_datasets
from models.model import build_model

def train_model(fine_tune=False):
    # Load datasets
    train_ds, val_ds, test_ds, class_names = get_datasets(data_dir="./dataset")

    # Build model
    # Use smaller learning rate if fine-tuning
    learning_rate = 1e-4 if not fine_tune else 1e-5
    model = build_model(num_classes=len(class_names), learning_rate=learning_rate, fine_tune=fine_tune)

    # Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="./models/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    )

    lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1
    )

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,  # increase epochs for fine-tuning
        callbacks=[checkpoint_cb, earlystop_cb, lr_scheduler_cb]
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save final model
    model.save("./models/final_model.keras")
    print("Model saved to ./models/final_model.keras")

    return history, model, class_names

if __name__ == "__main__":
    # Set fine_tune=True to enable fine-tuning
    train_model(fine_tune=True)
    
# python3 -m src.train