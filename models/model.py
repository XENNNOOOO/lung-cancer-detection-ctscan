import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def build_model(
    img_size=(224, 224, 3),
    num_classes=6,  
    learning_rate=1e-4,
    fine_tune=False
):
    """
    Builds an EfficientNetB0-based CNN model for multi-class classification.
    Supports optional fine-tuning of the last 50 layers.

    Args:
        img_size (tuple): Input image size (default: 224x224x3).
        num_classes (int): Number of output classes (default: 7, matching MASTER_CLASSES).
        learning_rate (float): Learning rate for the optimizer.
        fine_tune (bool): If True, unfreeze last 50 layers of base model for fine-tuning.

    Returns:
        tf.keras.Model: Compiled CNN model ready for training.
    """

    # Load EfficientNetB0 without top layers
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=img_size
    )

    # Freeze base if not fine-tuning
    base_model.trainable = fine_tune

    # Classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)  # Increased dropout for regularization
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # If fine-tuning, freeze all layers except the last 50
    if fine_tune:
        base_model.trainable = True
        for layer in base_model.layers[:-80]:
            layer.trainable = False

    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
