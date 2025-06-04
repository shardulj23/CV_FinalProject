from patch_utils import PatchManager
from PIL import Image
import numpy as np

# Step 1: Load the image and resize to 256x256
image_path = '/Users/kavanamanvi/Desktop/ComputerVision/FinalProject/img2.jpg'
img = Image.open(image_path).convert('RGB').resize((256, 256))
img_np = np.array(img)

# Step 2: Initialize PatchManager
patch_manager = PatchManager(patch_size=(4, 4), image_size=(256, 256))

# Step 3: Divide into patches
patches = patch_manager.divide_into_patches(img_np)
assert len(patches) == 16, f"Expected 16 patches, got {len(patches)}"

# Step 4: Shuffle k=4 patches
shuffled_patches = patch_manager.shuffle_patches(patches, k=5)
assert len(shuffled_patches) == 16, "Shuffled patches list should still contain 16 patches."

# Step 5: Reconstruct original and shuffled images
original_reconstructed = patch_manager.reconstruct_image(patches)
shuffled_reconstructed = patch_manager.reconstruct_image(shuffled_patches)

# Step 6: Save images for visual inspection
Image.fromarray(original_reconstructed).save('/Users/kavanamanvi/Desktop/ComputerVision/FinalProject/original_reconstructed.png')
Image.fromarray(shuffled_reconstructed).save('/Users/kavanamanvi/Desktop/ComputerVision/FinalProject/shuffled_reconstructed.png')

print("Test completed. Saved 'original_reconstructed.png' and 'shuffled_reconstructed.png'.")
