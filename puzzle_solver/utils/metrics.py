# puzzle_solver/utils/metrics.py

import torch
import numpy as np
import math
from pytorch_msssim import ssim

def calculate_ssim(img1, img2):
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).float() / 255.
        img2 = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).float() / 255.
    with torch.no_grad():
        score = ssim(img1, img2, data_range=1.0, size_average=True)
    return score.item()

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def patch_error_map(img1, img2, patch_size=30):
    H, W, _ = img1.shape
    error_map = np.zeros((H // patch_size, W // patch_size))
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch1 = img1[i:i+patch_size, j:j+patch_size]
            patch2 = img2[i:i+patch_size, j:j+patch_size]
            mse = np.mean((patch1 - patch2) ** 2)
            error_map[i//patch_size, j//patch_size] = mse
    return error_map
