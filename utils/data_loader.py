import os
import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

def get_datasets(data_dir="dataset"):
    """
    Loads train, validation, and test datasets from the specified folder.
    Applies data augmentation to the training dataset.

    Args:
        data_dir (str): Path to the dataset folder containing 'train', 'valid', 'test' subfolders.

    Returns:
        train_ds, val_ds, test_ds, class_names
    """

    # Resolve data_dir relative to this file
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_path, data_dir)

    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "valid"),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "test"),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names

    # Apply data augmentation only to training dataset
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    # Optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
