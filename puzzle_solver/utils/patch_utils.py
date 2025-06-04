# utils/patch_utils.py

import torch
import random
import math

class PatchManager:
    def __init__(self, patch_size=(4, 4), image_size=(256, 256)):
        self.patch_size = patch_size
        self.image_size = image_size

        self.num_patches_y = image_size[0] // patch_size[0]
        self.num_patches_x = image_size[1] // patch_size[1]
        self.total_patches = self.num_patches_y * self.num_patches_x

    def divide_into_patches(self, image):
        """
        image: torch.Tensor of shape (C, H, W)
        returns: list of patches of shape (C, patch_H, patch_W)
        """
        C, H, W = image.shape
        ph, pw = self.patch_size
        patches = []
        for i in range(0, H, ph):
            for j in range(0, W, pw):
                patch = image[:, i:i+ph, j:j+pw]
                patches.append(patch)
        return patches

    def shuffle_patches(self, patches, k):
        """
        patches: list of torch.Tensor patches
        k: number of patches to shuffle
        returns: list with k patches shuffled
        """
        patches = patches.copy()
        indices = random.sample(range(len(patches)), k)
        shuffled = indices[:]
        random.shuffle(shuffled)
        for i, j in zip(indices, shuffled):
            patches[i], patches[j] = patches[j], patches[i]
        return patches

    def reconstruct_image(self, patches):
        """
        patches: list of torch.Tensor patches of shape (C, patch_H, patch_W)
        returns: torch.Tensor image of shape (C, H, W)
        """
        C = patches[0].shape[0]
        ph, pw = self.patch_size
        ny, nx = self.num_patches_y, self.num_patches_x

        rows = []
        for y in range(ny):
            row = [patches[y * nx + x] for x in range(nx)]
            rows.append(torch.cat(row, dim=2))  # concat along width
        return torch.cat(rows, dim=1)  # concat along height

