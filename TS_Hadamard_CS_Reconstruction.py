# =============================================================================
#  Block-based Compressive Sensing using 2D Total Sequency (TS) Ordered Hadamard Matrix
# =============================================================================
#
# Authors       : Dr. Mohammad Amiri,
#                 Alireza Ghafari
#
# Description   : Official implementation of the 2D Total Sequency (TS) ordering 
#                 method for fast, deterministic, block-based 
#                 compressive sensing reconstruction.
# =============================================================================

import os
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from PIL import Image
from tqdm import tqdm  
from scipy.linalg import hadamard
import time


# ----------------------------------------------------------------------
# Image I/O
# ----------------------------------------------------------------------
def load_image(image_path):
    """Load image and normalize to float in range [0, 1]."""
    img = Image.open(image_path)
    img = np.array(img)
    return img / 255.0  


# ----------------------------------------------------------------------
# Measurement matrix construction (once per run)
# ----------------------------------------------------------------------
def compress_matrix(H, MR):
    """Keep only the top MR proportion of rows (measurement matrix Φ)."""
    n_rows = H.shape[0]
    num_to_keep = round(n_rows * MR)
    return H[:num_to_keep, :]

def find_nearest_factors(n):
    """Find two factors of n with minimal difference (for 2D reshaping)."""
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            factor1 = i
            factor2 = n // i
    return factor1, factor2

def calculate_zero_crossings(row):
    """Count sign changes in a 1D sequence."""
    return np.sum(np.diff(row) != 0)

def calculate_total_sequency(H, row_index, nearest_factors):
    """Total Sequency (TS) = Row Sequency + Column Sequency after 2D reshape."""
    row = H[row_index, :]
    rows, cols = nearest_factors
    reshaped = row.reshape(rows, cols) 
    RS = sum(calculate_zero_crossings(reshaped[i, :]) for i in range(rows))
    CS = sum(calculate_zero_crossings(reshaped[:, j]) for j in range(cols))
    TS = RS + CS
    return TS

def sort_hadamard_by_total_sequency(H):
    """Reorder Hadamard rows in ascending order of Total Sequency (core contribution)."""
    n = H.shape[0]
    nearest_factors = find_nearest_factors(n)
    total_sequencies = []
    for i in tqdm(range(n), desc="Calculating TS"):
        TS = calculate_total_sequency(H, i, nearest_factors)
        total_sequencies.append(TS)
    sorted_indices = sorted(range(n), key=lambda i: (total_sequencies[i], i))
    sorted_H = H[sorted_indices, :]
    return sorted_H


# ----------------------------------------------------------------------
# Forward and inverse operators per block
# ----------------------------------------------------------------------
def calculate_S(image, compressed_H):
    """y = Φx  (compressive measurements for one block)."""
    S = np.dot(compressed_H, image.flatten())
    return S

def reconstruct_image(S, compressed_H_pinv, shape):
    """x̂ = Φ⁺ y  (direct reconstruction using pre-computed pseudo-inverse)."""
    return np.reshape(np.dot(compressed_H_pinv, S), shape)


# ----------------------------------------------------------------------
# Block handling utilities
# ----------------------------------------------------------------------
def split_image_into_blocks(image, block_size):
    """Divide image into non-overlapping blocks; pad border blocks with zeros."""
    blocks = []
    rows, cols = image.shape[:2]
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.shape[0] < block_size or block.shape[1] < block_size:
                if len(image.shape) == 2:
                    padded_block = np.zeros((block_size, block_size))
                else:  
                    padded_block = np.zeros((block_size, block_size, image.shape[2]))
                padded_block[:block.shape[0], :block.shape[1]] = block
                blocks.append(padded_block)
            else:
                blocks.append(block)
    return blocks

def combine_blocks_into_image(blocks, image_shape, block_size):
    """Reassemble reconstructed blocks into full image (crop padded borders)."""
    rows, cols = image_shape[:2]
    if len(image_shape) == 2:  
        image = np.zeros((rows, cols))
    else: 
        image = np.zeros(image_shape)
    block_idx = 0
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = blocks[block_idx]
            block_rows, block_cols = block.shape[:2]
            actual_rows = min(block_rows, rows - i)
            actual_cols = min(block_cols, cols - j)
            if len(image_shape) == 2:
                image[i:i+actual_rows, j:j+actual_cols] = block[:actual_rows, :actual_cols]
            else:
                image[i:i+actual_rows, j:j+actual_cols] = block[:actual_rows, :actual_cols]
            block_idx += 1
    return image


# ----------------------------------------------------------------------
# Full image processing pipeline (sampling + reconstruction)
# ----------------------------------------------------------------------
def process_image(image, new_compressed_H, compressed_H_pinv, block_size):
    """Process one image: block-wise sampling → reconstruction (timed)."""
    all_compressed_S = []
    total_block_reconstruction_time = 0   
    
    if len(image.shape) == 2: 
        # Grayscale path
        channel_blocks = split_image_into_blocks(image, block_size)
        for block in channel_blocks:
            compressed_S = calculate_S(block, new_compressed_H)
            all_compressed_S.append(compressed_S)
        reconstructed_blocks = []
        start_time = time.time()
        for compressed_S in all_compressed_S:
            reconstructed_block = reconstruct_image(compressed_S, compressed_H_pinv, (block_size, block_size))
            reconstructed_blocks.append(reconstructed_block)
        total_block_reconstruction_time = time.time() - start_time
        reconstructed_image = combine_blocks_into_image(reconstructed_blocks, image.shape, block_size)
        
    else:  
        # RGB path (channel-independent processing)
        channels = []
        for channel in range(3):  
            channel_blocks = split_image_into_blocks(image[:, :, channel], block_size)
            channel_compressed_S = [calculate_S(block, new_compressed_H) for block in channel_blocks]
            reconstructed_blocks = []
            start_time = time.time()
            for compressed_S in channel_compressed_S:
                reconstructed_block = reconstruct_image(compressed_S, compressed_H_pinv, (block_size, block_size))
                reconstructed_blocks.append(reconstructed_block)
            total_block_reconstruction_time += time.time() - start_time
            reconstructed_channel = combine_blocks_into_image(reconstructed_blocks, image[:, :, channel].shape, block_size)
            channels.append(reconstructed_channel)
        reconstructed_image = np.stack(channels, axis=-1)
        
    return reconstructed_image, all_compressed_S, total_block_reconstruction_time


# ----------------------------------------------------------------------
# Main experiment (fixed parameters as used in the paper)
# ----------------------------------------------------------------------
MR = 0.1
block_size = 32

# 1. Build full Hadamard matrix
start_time_hadamard = time.time()
hadamard_matrix = hadamard(block_size**2)
time_hadamard = time.time() - start_time_hadamard

# 2. Total Sequency ordering (one-time cost)
start_time_reordering = time.time()
new_reordered_matrix = sort_hadamard_by_total_sequency(hadamard_matrix)
time_reordering = time.time() - start_time_reordering

# 3. Compressed measurement matrix Φ
new_compressed_H = compress_matrix(new_reordered_matrix, MR)

# 4. Pre-compute pseudo-inverse Φ⁺ (used for all images)
start_time_pinv = time.time()
compressed_H_pinv = np.linalg.pinv(new_compressed_H)
time_pinv = time.time() - start_time_pinv

# ----------------------------------------------------------------------
# Dataset paths (change to your local paths)
# ----------------------------------------------------------------------
input_folder  = 'data/Set11'
output_folder = 'results/'
csv_file      = 'results/result.csv'

os.makedirs(output_folder, exist_ok=True)

results = []
total_reconstruction_time = 0
image_files = [f for f in os.listdir(input_folder)
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]

for image_file in tqdm(image_files, desc="Processing Images", unit="image"):
    image_path = os.path.join(input_folder, image_file)
    image = load_image(image_path)
    
    reconstructed_image, _, image_recon_time = process_image(
        image, new_compressed_H, compressed_H_pinv, block_size)
    total_reconstruction_time += image_recon_time

    # Quality metrics
    if len(image.shape) == 2:
        ssim_val = ssim(image, reconstructed_image)
        psnr_val = psnr(image * 255, reconstructed_image * 255, data_range=255)
    else:
        ssim_val = np.mean([ssim(image[..., c] * 255, reconstructed_image[..., c] * 255, data_range=255)
                           for c in range(3)])
        psnr_val = np.mean([psnr(image[..., c] * 255, reconstructed_image[..., c] * 255, data_range=255)
                           for c in range(3)])
        
    results.append([image_file, ssim_val, psnr_val])

    # Save reconstructed image
    out_path = os.path.join(output_folder, f'reconstructed_{os.path.splitext(image_file)[0]}.png')
    reconstructed_image = np.clip(reconstructed_image, 0, 1)
    Image.fromarray((reconstructed_image * 255).astype(np.uint8)).save(out_path)


# ----------------------------------------------------------------------
# Save CSV results
# ----------------------------------------------------------------------
df = pd.DataFrame(results, columns=['Image', 'SSIM', 'PSNR (dB)'])
avg_ssim = df['SSIM'].mean()
avg_psnr = df['PSNR (dB)'].mean()
df.loc[len(df)] = ['Average', avg_ssim, avg_psnr]
df.to_csv(csv_file, index=False)

# ----------------------------------------------------------------------
# Summary of experiment parameters and results
# ----------------------------------------------------------------------
print("=" * 60)
print("  Total Sequency (TS) Ordered Hadamard Compressive Sensing")
print("=" * 60)
print(f"Block size          : {block_size} × {block_size}  (N = {block_size**2})")
print(f"Sampling ratio (MR) : {MR:.2f}")
print(f"Number of images    : {len(image_files)}")
print("-" * 60)
print(f"Average SSIM        : {avg_ssim:.4f}")
print(f"Average PSNR        : {avg_psnr:.2f} dB")
print("-" * 60)
print(f"Hadamard generation : {time_hadamard:.4f} s")
print(f"TS reordering       : {time_reordering:.4f} s")
print(f"Pseudo-inverse comp : {time_pinv:.4f} s")
print(f"Avg recon. time/img : {total_reconstruction_time / len(image_files):.6f} s")
print("=" * 60, "\n" )