import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from math import log10
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

from models.autoencoder import Autoencoder
from utils.image_loader import PuzzleDataset
from loss.composite_loss import CompositeLoss
from config.config import *


def mse(img1, img2):
    return ((img1 - img2) ** 2).mean()

def psnr(img1, img2):
    mse_val = mse(img1, img2)
    return 20 * log10(1.0) - 10 * log10(mse_val + 1e-8)

def compute_ssim(img1, img2):
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    return ssim_fn(img1_np, img2_np, channel_axis=2, data_range=1.0)


def train():
    train_dataset = PuzzleDataset(image_dir=TRAIN_DIR, level=DIFFICULTY)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = PuzzleDataset(image_dir=VAL_DIR, level=DIFFICULTY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Autoencoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = CompositeLoss(use_spatial_loss=USE_SPATIAL_LOSS,
                            spatial_weight=SPATIAL_WEIGHT,
                            patch_size=PATCH_SIZE)

    os.makedirs(SAVE_DIR, exist_ok=True)
    to_pil = ToPILImage()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            shuffled, original = batch
            shuffled = shuffled.to(DEVICE)
            original = original.to(DEVICE)

            optimizer.zero_grad()
            output = model(shuffled)
            loss = loss_fn(output, original)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"\n[Train] Epoch {epoch+1}: Loss = {avg_train_loss:.4f}")

        model.eval()
        val_mse = val_psnr = val_ssim = 0.0

        with torch.no_grad():
            for val_batch in val_loader:
                shuffled, original = batch
                shuffled = shuffled.to(DEVICE)
                original = original.to(DEVICE)

                output = model(shuffled)

                for i in range(output.size(0)):
                    recon_img = to_pil(output[i].cpu().clamp(0, 1))
                    target_img = to_pil(original[i].cpu())

                    img1 = np.array(recon_img).astype(np.float32) / 255.
                    img2 = np.array(target_img).astype(np.float32) / 255.

                    val_mse += mse(img1, img2)
                    val_psnr += psnr(img1, img2)
                    val_ssim += compute_ssim(recon_img, target_img)

        N = len(val_loader.dataset)
        print(f"[Val] MSE: {val_mse/N:.4f} | PSNR: {val_psnr/N:.2f} | SSIM: {val_ssim/N:.4f}")

        ckpt_path = os.path.join(SAVE_DIR, f"autoencoder_{DIFFICULTY}_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)

if __name__ == "__main__":
    train()
