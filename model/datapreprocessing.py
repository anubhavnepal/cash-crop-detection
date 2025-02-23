import os
import time
import cupy as cp
from PIL import Image, ImageEnhance
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2

import os
import hashlib

def find_duplicates(directory):
    class_duplicates = {}  # Dictionary to store duplicates for each class

    if not os.path.exists(directory):
        print(f"❌ Warning: {directory} does not exist!")
        return class_duplicates

    for class_name in sorted(os.listdir(directory)):  # Sort for consistency
        class_folder = os.path.join(directory, class_name)

        if os.path.isdir(class_folder):  # Ensure it's a folder
            hashes = {}  # Dictionary to store file hashes
            duplicates = []  # List to store duplicate file paths

            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                try:
                    with open(img_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()

                    if file_hash in hashes:
                        duplicates.append(img_path)  # Duplicate found
                    else:
                        hashes[file_hash] = img_path

                except Exception as e:
                    print(f"⚠️ Error processing {img_path}: {e}")

            if duplicates:
                class_duplicates[class_name] = duplicates  # Store duplicates for this class

    return class_duplicates

# Find duplicates in each dataset
train_duplicates = find_duplicates(train_dir)
val_duplicates = find_duplicates(val_dir)
test_duplicates = find_duplicates(test_dir)

# Print results
print("\n✅ Duplicate images in Training Set:")
for class_name, duplicates in train_duplicates.items():
    print(f"  {class_name}: {duplicates}")

print("\n✅ Duplicate images in Validation Set:")
for class_name, duplicates in val_duplicates.items():
    print(f"  {class_name}: {duplicates}")

print("\n✅ Duplicate images in Test Set:")
for class_name, duplicates in test_duplicates.items():
    print(f"  {class_name}: {duplicates}")
    
    
    
def check_image_sizes(directory):
    class_sizes = {}  # Dictionary to store sizes for each class

    if not os.path.exists(directory):
        print(f"❌ Warning: {directory} does not exist!")
        return class_sizes

    for class_name in sorted(os.listdir(directory)):  # Sort for consistency
        class_folder = os.path.join(directory, class_name)

        if os.path.isdir(class_folder):  # Ensure it's a folder
            sizes = set()

            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                try:
                    with Image.open(img_path) as img:
                        sizes.add(img.size)  # Add unique image size
                except Exception as e:
                    print(f"⚠️ Error processing {img_path}: {e}")

            class_sizes[class_name] = sizes  # Store sizes for this class

    return class_sizes

# Get image sizes for each dataset
train_sizes = check_image_sizes(train_dir)
val_sizes = check_image_sizes(val_dir)
test_sizes = check_image_sizes(test_dir)

# Print results
print("\n✅ Unique image sizes in Training Set:")
for class_name, sizes in train_sizes.items():
    print(f"  {class_name}: {sizes}")

print("\n✅ Unique image sizes in Validation Set:")
for class_name, sizes in val_sizes.items():
    print(f"  {class_name}: {sizes}")

print("\n✅ Unique image sizes in Test Set:")
for class_name, sizes in test_sizes.items():
    print(f"  {class_name}: {sizes}")
    

# -------------------
# Calculating Standard Deviation and Mean
# -------------------
import os
import hashlib
from PIL import Image
import cupy as cp

def find_duplicates(directory):
    hashes = {}
    duplicates = []

    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            with open(path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash in hashes:
                duplicates.append(path)
            else:
                hashes[file_hash] = path
    return duplicates

def calculate_mean_std(directory, target_size=(224, 224)):
    pixel_values = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):  # Only process image files
                try:
                    img = Image.open(os.path.join(root, file)).convert('RGB')
                    img = img.resize(target_size)  # Resize to the target size
                    img_array = cp.array(img) / 255.0  # Normalize pixel values to [0, 1]
                    pixel_values.append(img_array)  # Add the CuPy array of the image
                except Exception as e:
                    print(f"Error loading {file}: {e}")

    # Check if pixel_values is empty
    if not pixel_values:
        raise ValueError("No valid image files found in the directory.")

    # Stack all pixel arrays into a single CuPy array for efficient computation
    pixels = cp.stack(pixel_values)

    # Flatten the pixels to compute mean and std over all pixels
    flattened_pixels = pixels.reshape(-1, 3)

    # Compute mean and std per channel (axis=0)
    mean = cp.mean(flattened_pixels, axis=0)
    std = cp.std(flattened_pixels, axis=0)

    return mean, std


directory = r"C:\Users\abann\OneDrive\Desktop\Real_Data\Real_Data\train"
try:
    mean, std = calculate_mean_std(directory)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")
except ValueError as e:
    print(e)

def get_class_distribution(directory):
    class_counts = {}
    class_names = sorted(os.listdir(directory))
    for class_name in class_names:
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(os.listdir(class_path))
    return class_counts

# Usage in main:
train_counts = get_class_distribution(train_dir)
val_counts = get_class_distribution(val_dir)
test_counts = get_class_distribution(test_dir)

print(f"{'Class':<30} {'Train':<6} {'Val':<6} {'Test':<6}")
for cls in class_names:
    print(f"{cls:<30} {train_counts[cls]:<6} {val_counts[cls]:<6} {test_counts[cls]:<6}")
    
    
import os
from PIL import Image
import matplotlib.pyplot as plt

def display_samples(dataset_dir):
    class_names = sorted(os.listdir(dataset_dir))
    n_classes = len(class_names)
    n_images_per_class = 3

    # Set grid dimensions dynamically
    n_rows = n_classes
    n_cols = n_images_per_class

    plt.figure(figsize=(5 * n_cols, 5 * n_rows))

    for i, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_dir, class_name)
        images = os.listdir(class_path)[:n_images_per_class]  # Show 3 per class

        for j, img_name in enumerate(images):
            # Calculate the subplot index (matplotlib subplot indices start at 1)
            index = i * n_cols + j + 1
            ax = plt.subplot(n_rows, n_cols, index)

            # Open and display the image
            img = Image.open(os.path.join(class_path, img_name))
            plt.imshow(img)
            plt.title(f"{class_name}\n{img.size}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example calls for each dataset
display_samples(train_dir)
display_samples(val_dir)
display_samples(test_dir)
