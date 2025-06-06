import torch
import torch.nn as nn
import torch.nn.functional as F

class CompositeLoss(nn.Module):
    def __init__(self, use_spatial_loss=False, spatial_weight=0.0, patch_size=64):
        super(CompositeLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.use_spatial_loss = use_spatial_loss
        self.spatial_weight = spatial_weight
        self.patch_size = patch_size

    def forward(self, output, target):
        content_loss = self.mse(output, target)
        if self.use_spatial_loss:
            spatial_loss = self._spatial_loss(output, target)
            return content_loss + self.spatial_weight * spatial_loss
        return content_loss

    def _spatial_loss(self, output, target):
        B, C, H, W = output.shape
        ps = self.patch_size
        total_loss = 0.0
        patch_count = 0

        for i in range(0, H, ps):
            for j in range(0, W, ps):
                # Extract patches and flatten
                out_patch = output[:, :, i:i+ps, j:j+ps].reshape(B, -1)
                tgt_patch = target[:, :, i:i+ps, j:j+ps].reshape(B, -1)

                # Normalize the patches
                out_patch = F.normalize(out_patch, dim=1)
                tgt_patch = F.normalize(tgt_patch, dim=1)

                cosine_sim = torch.sum(out_patch * tgt_patch, dim=1)  # shape: [B]
                patch_loss = 1.0 - cosine_sim  # higher sim = lower loss
                total_loss += torch.sum(patch_loss)
                patch_count += B

        return total_loss / patch_count
