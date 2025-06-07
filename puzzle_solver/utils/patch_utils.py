# utils/patch_utils.py
import numpy as np
import random
import copy
from config.config import IMAGE_SIZE, PATCH_GRID

class PatchManager:
    def __init__(self):
        self.patch_rows, self.patch_cols = PATCH_GRID
        self.img_height, self.img_width = IMAGE_SIZE
        self.patch_height = self.img_height // self.patch_rows
        self.patch_width = self.img_width // self.patch_cols

    def divide_into_patches(self, image):
        patches = []
        for i in range(self.patch_rows):
            for j in range(self.patch_cols):
                top = i * self.patch_height
                left = j * self.patch_width
                patch = image[top:top + self.patch_height, left:left + self.patch_width]
                patches.append(patch)
        return patches

    def shuffle_patches(self, patches, k):
        assert k <= len(patches), "k must be <= number of patches"
        indices = list(range(len(patches)))
        selected_indices = random.sample(indices, k)

        while True:
            shuffled_indices = selected_indices.copy()
            random.shuffle(shuffled_indices)
            if all(orig != new for orig, new in zip(selected_indices, shuffled_indices)):
                break

        shuffled_patches = copy.deepcopy(patches)
        for orig_idx, new_idx in zip(selected_indices, shuffled_indices):
            shuffled_patches[orig_idx] = patches[new_idx]

        return shuffled_patches

    def reconstruct_image(self, patches):
        rows = []
        for i in range(self.patch_rows):
            row_patches = patches[i * self.patch_cols:(i + 1) * self.patch_cols]
            row = np.concatenate(row_patches, axis=1)
            rows.append(row)
        return np.concatenate(rows, axis=0)
