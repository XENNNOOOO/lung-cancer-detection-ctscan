import os
import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def get_datasets(data_dir="dataset"):
    """
    Loads train, validation, and test datasets from the specified folder.

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

    # Optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
