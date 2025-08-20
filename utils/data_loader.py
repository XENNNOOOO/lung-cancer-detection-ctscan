import os
import numpy as np
import tensorflow as tf

# Config
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])

# Normalization
normalization_layer = tf.keras.layers.Rescaling(1. / 255)

def preprocess(image, label):
    """Normalize images and return (image, label)."""
    return normalization_layer(image), label


# Class Definitions
MASTER_CLASSES = [
    "Normal",
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Squamous Cell Carcinoma",
    "Benign",
    "Malignant"
]

DATASET_LABEL_MAP = {
    "dataset_1": {
        "adenocarcinoma": "Adenocarcinoma",
        "large.cell.carcinoma": "Large Cell Carcinoma",
        "normal": "Normal",
        "squamous.cell.carcinoma": "Squamous Cell Carcinoma",
        "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib": "Adenocarcinoma",
        "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa": "Squamous Cell Carcinoma",
        "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa": "Large Cell Carcinoma"
    },
    "dataset_2": {
        "adenocarcinoma": "Adenocarcinoma",
        "BenginCases": "Benign",
        "Bengin cases": "Benign",
        "large.cell.carcinoma": "Large Cell Carcinoma",
        "MalignantCases": "Malignant",
        "Malignant cases": "Malignant",
        "normal": "Normal",
        "squamous.cell.carcinoma": "Squamous Cell Carcinoma",
        "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib": "Adenocarcinoma",
        "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa": "Squamous Cell Carcinoma",
        "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa": "Large Cell Carcinoma"
    },
    "dataset_3": {
        "adenocarcinoma": "Adenocarcinoma",
        "large.cell.carcinoma": "Large Cell Carcinoma",
        "normal": "Normal",
        "squamous.cell.carcinoma": "Squamous Cell Carcinoma",
        "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib": "Adenocarcinoma",
        "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa": "Squamous Cell Carcinoma",
        "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa": "Large Cell Carcinoma"
    },
    "dataset_4": {
        "adenocarcinoma": "Adenocarcinoma",
        "large.cell.carcinoma": "Large Cell Carcinoma",
        "normal": "Normal",
        "squamous.cell.carcinoma": "Squamous Cell Carcinoma",
        "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib": "Adenocarcinoma",
        "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa": "Squamous Cell Carcinoma",
        "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa": "Large Cell Carcinoma"
    }
}

# Dataset Loader
def get_datasets(main_dir="datasets"):
    """Load, preprocess, and merge datasets into train/val/test sets."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_dir = os.path.join(base_path, main_dir)

    train_ds_list, val_ds_list, test_ds_list = [], [], []

    for dataset_name, label_map in DATASET_LABEL_MAP.items():
        dataset_path = os.path.join(main_dir, dataset_name)

        ds_train = tf.keras.utils.image_dataset_from_directory(
            os.path.join(dataset_path, "train"),
            image_size=IMAGE_SIZE,
            batch_size=None,
            labels="inferred",
            label_mode="int"
        )
        ds_val = tf.keras.utils.image_dataset_from_directory(
            os.path.join(dataset_path, "valid"),
            image_size=IMAGE_SIZE,
            batch_size=None,
            labels="inferred",
            label_mode="int"
        )
        ds_test = tf.keras.utils.image_dataset_from_directory(
            os.path.join(dataset_path, "test"),
            image_size=IMAGE_SIZE,
            batch_size=None,
            labels="inferred",
            label_mode="int"
        )

        # Map dataset labels to MASTER_CLASSES index (integer)
        label_names = ds_train.class_names
        dataset_label_names = {i: label_map.get(name, None) for i, name in enumerate(label_names)}
        print(f"[INFO] {dataset_name} label map: {dataset_label_names}")

        def map_labels(y):
            def py_map(y_np):
                label_name = dataset_label_names.get(int(y_np))
                if label_name in MASTER_CLASSES:
                    return np.int32(MASTER_CLASSES.index(label_name))
                return np.int32(-1)
            mapped = tf.py_function(py_map, [y], tf.int32)
            mapped.set_shape([])  
            return mapped

        def apply_mapping(ds):
            ds = ds.map(lambda x, y: (x, map_labels(y)))
            ds = ds.filter(lambda x, y: tf.not_equal(y, -1))
            return ds

        ds_train, ds_val, ds_test = map(apply_mapping, [ds_train, ds_val, ds_test])

        train_ds_list.append(ds_train)
        val_ds_list.append(ds_val)
        test_ds_list.append(ds_test)

    def concat_datasets(ds_list):
        if not ds_list:
            return None
        combined = ds_list[0]
        for d in ds_list[1:]:
            combined = combined.concatenate(d)
        return combined

    train_ds = concat_datasets(train_ds_list)
    val_ds = concat_datasets(val_ds_list)
    test_ds = concat_datasets(test_ds_list)

    # Augmentation + preprocessing
    if train_ds:
        train_ds = train_ds.map(preprocess)
        train_ds = train_ds.cache()
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    if val_ds:
        val_ds = val_ds.map(preprocess)

    if test_ds:
        test_ds = test_ds.map(preprocess)

    return train_ds, val_ds, test_ds, MASTER_CLASSES