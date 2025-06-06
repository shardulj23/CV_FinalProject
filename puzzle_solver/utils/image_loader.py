# utils/image_loader.py

import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from patch_utils import PatchManager

class PuzzleDataset(Dataset):
    def __init__(self, image_dir, transform=None, level='easy'):
        self.image_dir = image_dir
        self.transform = transform
        self.level = level
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Set shuffle difficulty
        self.k = {'easy': 2, 'medium': 4, 'difficult': 6}[level]

        # Create output folder as a **sibling**, not child**
        parent_dir = os.path.dirname(image_dir)
        self.output_dir = os.path.join(parent_dir, f'shuffled_{level}')
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize patch manager
        self.patch_manager = PatchManager(patch_size=(4, 4), image_size=(256, 256))


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Load and resize image
        img = Image.open(image_path).convert('RGB').resize((256, 256))
        img_np = np.array(img)

        # Divide and shuffle patches
        patches = self.patch_manager.divide_into_patches(img_np)
        shuffled_patches = self.patch_manager.shuffle_patches(patches, k=self.k)
        reconstructed_img_np = self.patch_manager.reconstruct_image(shuffled_patches)

        # Save shuffled image
        shuffled_image_name = f"shuffled_{image_name}"
        shuffled_image_path = os.path.join(self.output_dir, shuffled_image_name)
        Image.fromarray(reconstructed_img_np).save(shuffled_image_path)

        # Apply transform if provided
        if self.transform:
            img_tensor = self.transform(Image.fromarray(reconstructed_img_np))
        else:
            img_tensor = torch.from_numpy(reconstructed_img_np).permute(2, 0, 1).float() / 255.0

        return img_tensor, image_name
