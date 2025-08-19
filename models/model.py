import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def build_model(img_size=(224, 224, 3), num_classes=4, learning_rate=1e-4, fine_tune=False):
    """
    Builds an EfficientNetB0-based CNN model for multi-class classification.
    Can optionally fine-tune the base model.

    Args:
        img_size (tuple): Input image size (default: 224x224x3).
        num_classes (int): Number of output classes (default: 4).
        learning_rate (float): Learning rate for the optimizer.
        fine_tune (bool): If True, unfreeze some base model layers for fine-tuning.

    Returns:
        model (tf.keras.Model): Compiled CNN model.
    """

    # Load base EfficientNetB0 (without top layers)
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=img_size)
    base_model.trainable = False  # Freeze base model by default

    if fine_tune:
        # Fine-tune last 50 layers only
        base_model.trainable = True
        for layer in base_model.layers[:-50]:
            layer.trainable = False

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)  # Dropout for regularization
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",  # matches integer labels
        metrics=["accuracy"]
    )

    return model