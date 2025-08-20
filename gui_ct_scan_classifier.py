import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from PIL.Image import Resampling  # CHANGED: Use modern Resampling
import numpy as np
import tensorflow as tf

# --- Configuration ---

MODEL_PATH = "./models/final_model.keras"
class_names = [
    "Normal",
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Squamous Cell Carcinoma",
    "Benign",
    "Malignant"
]
IMAGE_SIZE = (224, 224)

# --- Model Loading ---

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    messagebox.showerror("Model Loading Error", f"Failed to load the model:\n{e}")
    exit()

# --- Core Functions ---

def preprocess_image(img_path):
    """Load and preprocess image for prediction."""
    try:
        img = Image.open(img_path).convert("RGB")
        # CHANGED: Use modern, high-quality resampling method
        img = img.resize(IMAGE_SIZE, Resampling.LANCZOS)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        messagebox.showerror("Image Preprocessing Error", f"Failed to preprocess image:\n{e}")
        return None

def predict_image(img_path):
    """Predict CT scan image and return the main result and detailed probabilities."""
    img_array = preprocess_image(img_path)
    if img_array is None:
        return "Error", "Could not process the image."

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_index = np.argmax(predictions)
    
    predicted_label = class_names[predicted_index]
    confidence = predictions[predicted_index]
    
    main_result = f"{predicted_label} ({confidence*100:.2f}%)"
    
    prob_details = "\n".join(
        [f"{name}: {prob*100:.2f}%" for name, prob in zip(class_names, predictions)]
    )
    return main_result, prob_details

def open_file():
    """Open file dialog to select an image and display predictions."""
    file_path = filedialog.askopenfilename(
        title="Select CT Scan Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return

    # Display the selected image
    try:
        img = Image.open(file_path)
        img.thumbnail((350, 350)) # Slightly larger image preview
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
    except Exception as e:
        messagebox.showerror("Image Display Error", f"Failed to load image:\n{e}")
        return

    # Predict and display the results
    main_pred, detailed_probs = predict_image(file_path)
    main_result_label.config(text=main_pred)
    detailed_result_label.config(text=f"Class Probabilities:\n{detailed_probs}")

# --- GUI Setup ---

root = tk.Tk()
root.title("Lung CT Scan Classifier")
root.geometry("700x800") 
root.configure(bg="#f0f0f0") # Light gray background

# Main frame for better organization
main_frame = tk.Frame(root, bg="#f0f0f0", padx=20, pady=20)
main_frame.pack(expand=True, fill="both")

# --- Widgets ---

# Button to select an image
btn_select = tk.Button(
    main_frame, text="Select CT Scan Image", command=open_file, 
    font=("Arial", 16, "bold"), bg="#6b80eb", fg="blue", relief="flat", padx=10, pady=10
)
btn_select.pack(pady=(0, 20))

# Label to display the image
image_label = tk.Label(main_frame, bg="#f0f0f0", text="Please select an image to begin", font=("Arial", 12))
image_label.pack(pady=10)

# Frame for results to group them
results_frame = tk.Frame(main_frame, bg="#ffffff", relief="sunken", borderwidth=1)
results_frame.pack(pady=20, fill="x")

# Label for the main prediction (large and bold)
main_result_label = tk.Label(
    results_frame, text="", font=("Arial", 24, "bold"), justify="center", bg="white", fg="#007aff"
)
main_result_label.pack(pady=(10, 5))

# Label for detailed class probabilities
detailed_result_label = tk.Label(
    results_frame, text="", font=("Arial", 11), justify="left", wraplength=600, bg="white"
)
detailed_result_label.pack(pady=(5, 10))

root.mainloop()