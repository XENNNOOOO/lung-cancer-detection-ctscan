import os
import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def get_datasets(data_dir="../dataset"):
    """
    Loads train, validation, and test datasets from the given folder structure.
    Folder must have:
        dataset/train/<class_name>
        dataset/valid/<class_name>
        dataset/test/<class_name>
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "valid")
    test_dir  = os.path.join(data_dir, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names
    print("🔹 Classes:", class_names)

    # Normalize datasets
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds  = test_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds, test_ds, class_names