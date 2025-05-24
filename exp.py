#%%
import os
import cv2
import numpy as np

# Directory containing input images
input_dir = "inputs"
output_dir = "noisy_outputs"
os.makedirs(output_dir, exist_ok=True)

# Define noise levels (standard deviations for Gaussian noise)
noise_levels = [5, 15, 30, 60]
# Define blur kernel sizes (must be odd numbers)
blur_levels = [3, 7, 11, 15]

def add_gaussian_noise(image, std):
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy_img = image.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

def add_blur(image, ksize):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue
        # Add noise
        for std in noise_levels:
            noisy_image = add_gaussian_noise(image, std)
            name, ext = os.path.splitext(filename)
            out_filename = f"{name}_noise{std}{ext}"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, noisy_image)
        # Add blur
        for ksize in blur_levels:
            blurred_image = add_blur(image, ksize)
            name, ext = os.path.splitext(filename)
            out_filename = f"{name}_blur{ksize}{ext}"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, blurred_image)