# utils/image_loader.py
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.patch_utils import PatchManager
from config.config import IMAGE_SIZE, PATCH_GRID, DIFFICULTY_MAP

class PuzzleDataset(Dataset):
    def __init__(self, image_dir, transform=None, level='easy'):
        self.image_dir = image_dir
        self.transform = transform
        self.level = level
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        self.k = DIFFICULTY_MAP[level]

        parent_dir = os.path.dirname(image_dir)
        self.output_dir = os.path.join(parent_dir, f'shuffled_{level}')
        os.makedirs(self.output_dir, exist_ok=True)

        self.patch_manager = PatchManager()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        img = Image.open(image_path).convert('RGB').resize(IMAGE_SIZE)
        img_np = np.array(img)

        patches = self.patch_manager.divide_into_patches(img_np)
        shuffled_patches = self.patch_manager.shuffle_patches(patches, k=self.k)
        reconstructed_img_np = self.patch_manager.reconstruct_image(shuffled_patches)

        shuffled_image_path = os.path.join(self.output_dir, f"shuffled_{image_name}")
        Image.fromarray(reconstructed_img_np).save(shuffled_image_path)

        shuffled_tensor = torch.from_numpy(reconstructed_img_np).permute(2, 0, 1).float() / 255.0
        original_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        return shuffled_tensor, original_tensor


