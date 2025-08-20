import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

def grad_cam(model, img_array, layer_name):
    """
    Generate Grad-CAM heatmap for an image.
    
    Args:
        model: tf.keras.Model, trained model
        img_array: preprocessed image array with shape (1, H, W, C)
        layer_name: string, name of the convolutional layer to inspect

    Returns:
        heatmap: 2D numpy array normalized to [0,1]
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

    for i in range(pooled_grads.shape[-1]):
        heatmap += pooled_grads[i] * conv_outputs[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    return heatmap

def display_grad_cam(img_path, heatmap, alpha=0.4, cmap="jet"):
    """
    Overlay Grad-CAM heatmap on the original image using Pillow and Matplotlib.
    
    Args:
        img_path: path to original image
        heatmap: 2D numpy array from grad_cam()
        alpha: transparency for overlay
        cmap: color map for heatmap
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize((heatmap.shape[1], heatmap.shape[0]))
    img_np = np.array(img)

    # Convert heatmap to RGB using colormap
    cmap_func = plt.get_cmap(cmap)
    heatmap_rgb = cmap_func(heatmap)[..., :3]  # drop alpha
    heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)

    # Blend images
    superimposed = (img_np * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(superimposed)
    plt.axis("off")
    plt.title("Grad-CAM Overlay")
    plt.show()