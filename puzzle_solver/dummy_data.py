import os
import random
from torchvision.datasets import Places365
from torchvision.transforms import ToPILImage

dataset = Places365(root="places365", split='val', small=True, download=True)

sample_indices = random.sample(range(len(dataset)), 200)

os.makedirs("data/train", exist_ok=True)
os.makedirs("data/val", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

to_pil = ToPILImage()

for i, idx in enumerate(sample_indices):
    img, _ = dataset[idx]

    if i < 120:
        img.save(f"data/train/img_{i}.jpg")
    elif i < 150:
        img.save(f"data/val/img_{i}.jpg")
    else:
        img.save(f"data/test/img_{i}.jpg")

print("Sample dataset created in /data/train, /val, /test.")
