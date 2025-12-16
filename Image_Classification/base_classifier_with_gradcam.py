"""
Enhanced Image Classifier with Grad-CAM Visualization
This program classifies images using MobileNetV2 and visualizes attention regions using Grad-CAM.
"""

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights="imagenet")

# Create a model that returns both predictions and the last convolutional layer output for Grad-CAM
last_conv_layer_name = "Conv_1"
grad_model = Model(
    inputs=model.input,
    outputs=[model.get_layer(last_conv_layer_name).output, model.output]
)


def make_gradcam_heatmap(img_array, pred_index=None):
    """
    Generate Grad-CAM heatmap for the image.
    
    Args:
        img_array: Preprocessed image array
        pred_index: Index of the class to visualize (None = top prediction)
    
    Returns:
        heatmap: Grad-CAM heatmap (0-255 range)
    """
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array, training=False)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = tf.cast(heatmap * 255, tf.uint8)
    
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, output_path="gradcam_heatmap.png"):
    """
    Overlay Grad-CAM heatmap on original image and save.
    
    Args:
        img_path: Path to original image
        heatmap: Grad-CAM heatmap
        output_path: Path to save the visualization
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    cv2.imwrite(output_path, superimposed_img)
    print(f"  Grad-CAM visualization saved as '{output_path}'")


def classify_image_with_gradcam(image_path):
    """
    Classify image and generate Grad-CAM visualization.
    
    Args:
        image_path: Path to the image file
    """
    try:
        # Load and preprocess image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get predictions
        predictions = model.predict(img_array, verbose=0)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
        print("\n" + "="*60)
        print(f"Top-3 Predictions for: {image_path}")
        print("="*60)
        for i, (_, label, score) in enumerate(decoded_predictions):
            print(f"  {i + 1}. {label.upper():<20} Confidence: {score:.4f}")
        
        # Generate Grad-CAM for top prediction
        print("\nGenerating Grad-CAM visualization...")
        heatmap = make_gradcam_heatmap(img_array, pred_index=0)
        
        # Save visualization
        base_name = Path(image_path).stem
        output_file = f"{base_name}_gradcam.png"
        save_and_display_gradcam(image_path, heatmap, output_file)
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"Error processing '{image_path}': {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("IMAGE CLASSIFIER WITH GRAD-CAM VISUALIZATION")
    print("="*60)
    print("Type 'exit' to quit\n")
    
    while True:
        image_path = input("Enter image filename: ").strip()
        if image_path.lower() == "exit":
            print("Goodbye!")
            break
        
        if not Path(image_path).exists():
            print(f"File not found: {image_path}\n")
            continue
        
        classify_image_with_gradcam(image_path)
