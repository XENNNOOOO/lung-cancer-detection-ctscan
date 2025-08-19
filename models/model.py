import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def build_model(img_size=(224, 224, 3), num_classes=4, learning_rate=0.0001):
    """
    Builds an EfficientNetB0-based CNN model for multi-class classification.

    Args:
        img_size (tuple): Input image size (default: 224x224x3).
        num_classes (int): Number of output classes (default: 4).
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        model (tf.keras.Model): Compiled CNN model.
    """

    # Load base EfficientNetB0 (without top layers)
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=img_size)
    base_model.trainable = False  # Freeze base model for transfer learning

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)  # Dropout for regularization
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
