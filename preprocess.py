import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

input_dir = 'images'
output_dir = 'preprocessed'
os.makedirs(output_dir, exist_ok=True)

def rgb_to_hsi(image):
    image = image.astype(np.float32) / 255.0
    R, G, B = image[..., 2], image[..., 1], image[..., 0]

    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B)*(G - B)) + 1e-8
    theta = np.arccos(np.clip(num / den, -1, 1))
    
    H = np.where(B <= G, theta, 2*np.pi - theta)
    H = H / (2 * np.pi)  # Normalize to [0, 1]

    min_rgb = np.minimum(np.minimum(R, G), B)
    sum_rgb = R + G + B + 1e-8
    S = 1 - (3 * min_rgb / sum_rgb)
    I = sum_rgb / 3

    return H, S, I

def compute_HI(H, I):
    HI = H / (I + 1e-8)
    HI_transformed = np.where(HI < 1, HI, 1 + np.log(HI))
    return HI_transformed

def normalize_minmax(channel):
    return (channel - np.min(channel)) / (np.max(channel) - np.min(channel) + 1e-8)

def normalize_hi_95(HI_transformed):
    p95 = np.percentile(HI_transformed, 95)
    HI_clipped = np.clip(HI_transformed, 0, p95)
    return normalize_minmax(HI_clipped)

for filename in tqdm(os.listdir(input_dir)):
    if not (filename.endswith('.png') or filename.endswith('.jpg')):
        continue

    path = os.path.join(input_dir, filename)
    img = cv2.imread(path)
    if img is None:
        print(f"Error reading {filename}")
        continue

    # STEP 1: RGB to HSI
    H, S, I = rgb_to_hsi(img)

    # STEP 2: Gaussian Filtering
    H = gaussian_filter(H, sigma=1)
    S = gaussian_filter(S, sigma=1)
    I = gaussian_filter(I, sigma=1)

    # STEP 3: Compute HI
    HI = compute_HI(H, I)

    # STEP 4: Normalize
    S_norm = normalize_minmax(S)
    I_norm = normalize_minmax(I)
    HI_norm = normalize_hi_95(HI)

    # Combine to shape (H, W, 3) → [HI, I, S]
    processed = np.stack([HI_norm, I_norm, S_norm], axis=-1)
    np.save(os.path.join(output_dir, filename.replace('.png', '.npy').replace('.jpg', '.npy')), processed)