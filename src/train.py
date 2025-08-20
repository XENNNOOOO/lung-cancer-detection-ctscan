import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.data_loader import get_datasets
from models.model import build_model

BATCH_SIZE = 16
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 20


def compute_class_weights(dataset, class_names):
    """Compute class weights for imbalanced dataset."""
    y_list = []

    for batch in dataset:
        images, labels = batch
        # Flatten batch labels to 1D numpy array
        y_list.append(labels.numpy().flatten())

    y_array = np.concatenate(y_list)

    # Only consider valid labels (0..len(class_names)-1)
    y_array = y_array[(y_array >= 0) & (y_array < len(class_names))]

    unique_classes = np.unique(y_array)
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y_array
    )

    class_weights = {i: 1.0 for i in range(len(class_names))}
    for cls, w in zip(unique_classes, weights):
        class_weights[int(cls)] = w

    return class_weights


def format_dataset(image, label):
    """
    Ensures the label is a scalar tf.int32.
    """
    label = tf.cast(label, tf.int32)
    label.set_shape([]) 
    return image, label

def train_model():
    # Load combined datasets (4 datasets with potentially different classes)
    train_ds, val_ds, test_ds, class_names = get_datasets(main_dir="./datasets")

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(format_dataset).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(format_dataset).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(format_dataset).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    for images, labels in train_ds.take(1):
        print("Sample labels:", labels.numpy())
        print("Label shape:", labels.shape)

    class_weights = compute_class_weights(train_ds, class_names)

    print("\n[INFO] Computed Class Weights:")
    for idx, cls_name in enumerate(class_names):
        print(f"  Class {idx} ({cls_name}): {class_weights[idx]:.4f}")

    # Stage 1: Train top layers only
    model = build_model(num_classes=len(class_names), fine_tune=False, learning_rate=3e-4)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="./models/best_model_stage1.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    print("\n=== Stage 1: Training top layers only ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE1,
        class_weight=class_weights,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
    )

    # Stage 2: Fine-tune last layers
    model = build_model(num_classes=len(class_names), fine_tune=True, learning_rate=5e-5)

    checkpoint_cb2 = tf.keras.callbacks.ModelCheckpoint(
        filepath="./models/best_model_stage2.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    print("\n=== Stage 2: Fine-tuning last layers ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE2,
        class_weight=class_weights,
        callbacks=[checkpoint_cb2, earlystop_cb, reduce_lr_cb]
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\n[RESULT] Test Accuracy: {test_acc:.4f}")

    # Save final model
    model.save("./models/final_model.keras")
    print("[INFO] Model saved to ./models/final_model.keras")

    return model, class_names


if __name__ == "__main__":
    train_model()

# To run this script, use the command:
# python3 -m src.train