# utils/patch_utils.py

import numpy as np
import random
import copy

class PatchManager:
    def __init__(self, patch_size=(4, 4), image_size=(256, 256)):
        self.patch_rows, self.patch_cols = patch_size
        self.img_height, self.img_width = image_size
        self.patch_height = self.img_height // self.patch_rows
        self.patch_width = self.img_width // self.patch_cols

    def divide_into_patches(self, image):
        """
        Splits the image into non-overlapping 64x64 patches (assuming 256x256 input and 4x4 grid).
        Returns:
            patches (list of np.array): list of 64x64 patches
        """
        patches = []
        for i in range(self.patch_rows):
            for j in range(self.patch_cols):
                top = i * self.patch_height
                left = j * self.patch_width
                patch = image[top:top + self.patch_height, left:left + self.patch_width]
                patches.append(patch)
        return patches


    def shuffle_patches(self, patches, k):
        """
        Randomly selects k unique patches and shuffles them such that none remain in the original position.
        Args:
            patches (list of np.array): list of 16 patches
            k (int): number of patches to shuffle
        Returns:
            list of np.array: new list with k patches shuffled
        """
        assert k <= len(patches), "k must be less than or equal to number of patches"
    
        indices = list(range(len(patches)))
        selected_indices = random.sample(indices, k)
    
        # Keep shuffling until no patch stays in its original position
        while True:
            shuffled_indices = selected_indices.copy()
            random.shuffle(shuffled_indices)
            if all(orig != new for orig, new in zip(selected_indices, shuffled_indices)):
                break
    
        # Apply the shuffle
        shuffled_patches = copy.deepcopy(patches)
        for orig_idx, new_idx in zip(selected_indices, shuffled_indices):
            shuffled_patches[orig_idx] = patches[new_idx]
    
        return shuffled_patches


    def reconstruct_image(self, patches):
        """
        Reconstructs the original image from 64x64 patches.
        Args:
            patches (list of np.array): list of 16 patches
        Returns:
            np.array: reconstructed 256x256 image
        """
        rows = []
        for i in range(self.patch_rows):
            row_patches = patches[i * self.patch_cols:(i + 1) * self.patch_cols]
            row = np.concatenate(row_patches, axis=1)
            rows.append(row)
        reconstructed_image = np.concatenate(rows, axis=0)
        return reconstructed_image
