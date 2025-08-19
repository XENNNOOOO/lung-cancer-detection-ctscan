import os
import tensorflow as tf
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.data_loader import get_datasets
from models.model import build_model

def train_model():
    # Load datasets
    train_ds, val_ds, test_ds, class_names = get_datasets(data_dir="./dataset")

    # Optimize dataset performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build model with fine-tuning enabled
    model = build_model(num_classes=len(class_names), fine_tune=True)

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
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save final model
    model.save("./models/final_model.keras")
    print("Model saved to ./models/final_model.keras")

    return history, model, class_names

if __name__ == "__main__":
    train_model()
    
# python3 -m src.train