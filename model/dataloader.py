import os
import time
import cupy as cp 
from PIL import Image
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2

# -------------------
# Enhanced Data Loading and Augmentation
# -------------------
def load_dataset(directory, target_size=(224, 224), augment=False):
    images = []
    labels = []
    class_names = sorted(os.listdir(directory))
    
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(directory, class_name)
        if not os.path.isdir(class_folder):
            continue
            
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            
            # Load and preprocess
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size)
            img_array = cp.array(img) / 255.0  # Normalization
            
            # Basic augmentation for all images
            if augment:
                img_array = augment_image(img_array)
                
            images.append(img_array)
            one_hot = cp.zeros(len(class_names))
            one_hot[label] = 1
            labels.append(one_hot)
    
    return cp.array(images), cp.array(labels), class_names

import cupy as cp
import numpy as np
from PIL import Image, ImageEnhance
import cv2

def augment_image(image):
    """Improved image augmentation with advanced transformations"""
    
    # Convert Cupy array to NumPy (if needed for OpenCV/PIL)
    image_np = image.get()
    
    # Add vertical flip
    if cp.random.rand() > 0.5:
        image_np = np.flipud(image_np)
    
    # Random horizontal flip
    if cp.random.rand() > 0.5:
        image_np = np.fliplr(image_np)

    # Add shear transformation
    if cp.random.rand() > 0.5:
        shear_factor = cp.random.uniform(-0.2, 0.2)  # Increased range
        rows, cols, ch = image_np.shape
        M = np.array([[1, shear_factor, 0], [shear_factor, 1, 0]])
        image_np = cv2.warpAffine(image_np, M, (cols, rows))

    # Random rotation (-20 to +20 degrees)
    if cp.random.rand() > 0.5:
        angle = cp.random.uniform(-20, 20)  # Increased range
        image_np = np.array(Image.fromarray(image_np).rotate(angle))

    # Random brightness adjustment
    if cp.random.rand() > 0.5:
        factor = cp.random.uniform(0.6, 1.4)  # More variability
        image_np = np.array(ImageEnhance.Brightness(Image.fromarray(image_np)).enhance(factor))
    
    # Random contrast adjustment
    if cp.random.rand() > 0.5:
        factor = cp.random.uniform(0.6, 1.4)  # Increase contrast range
        image_np = np.array(ImageEnhance.Contrast(Image.fromarray(image_np)).enhance(factor))
    
    # Apply Gaussian Noise
    if cp.random.rand() > 0.5:
        noise = np.random.normal(0, 0.05, image_np.shape) * 255
        image_np = np.clip(image_np + noise, 0, 255).astype(np.uint8)

    # Random zoom crop with variable zoom factor
    if cp.random.rand() > 0.5:
        zoom = cp.random.uniform(0.7, 1.0)  # Increased zoom range
        h, w = image_np.shape[:2]
        new_h, new_w = int(h * zoom), int(w * zoom)
        start_h = cp.random.randint(0, h - new_h + 1)
        start_w = cp.random.randint(0, w - new_w + 1)
        image_np = image_np[start_h:start_h+new_h, start_w:start_w+new_w]
        image_np = np.array(Image.fromarray(image_np).resize((h, w)))

    # Elastic distortion
    if cp.random.rand() > 0.5:
        sigma = cp.random.uniform(8, 15)  # Randomize distortion strength
        alpha = cp.random.uniform(30, 60)
        dx = cv2.GaussianBlur((np.random.rand(*image_np.shape[:2]) * 2 - 1), (17, 17), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(*image_np.shape[:2]) * 2 - 1), (17, 17), sigma) * alpha
        x, y = np.meshgrid(np.arange(image_np.shape[1]), np.arange(image_np.shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        image_np = cv2.remap(image_np, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Cutout (Random Erasing)
    if cp.random.rand() > 0.5:
        num_patches = cp.random.randint(1, 4)  # Random number of cutout patches
        for _ in range(num_patches):
            patch_h = cp.random.randint(10, 50)  # Patch size
            patch_w = cp.random.randint(10, 50)
            y1 = cp.random.randint(0, image_np.shape[0] - patch_h)
            x1 = cp.random.randint(0, image_np.shape[1] - patch_w)
            image_np[y1:y1+patch_h, x1:x1+patch_w] = 0  # Black patch
    
    # Convert back to Cupy array
    image = cp.array(image_np)
    
    return image


def batch_generator(images, labels, batch_size):
    """
    Generate batches of data.
    
    Args:
        images (np.array): Array of images.
        labels (np.array): Array of labels.
        batch_size (int): Number of samples per batch.
    
    Yields:
        batch_images (np.array): Batch of images.
        batch_labels (np.array): Batch of labels.
    """
    num_samples = len(images)
    indices = cp.arange(num_samples)
    cp.random.shuffle(indices)  # Shuffle the data
    
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield images[batch_indices], labels[batch_indices]
