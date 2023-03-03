import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
  '''
  Create patch embeddings using hybrid architecture as described in section 3.1
  '''
  def __init__(self, in_channels=3, patch_size=16, embedding_dim=768):
    super(PatchEmbedding, self).__init__()
    self.patch_size = patch_size
    # create input embedding from flattened patch feature maps.
    self.embedding = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0), # [B, 3, 224, 244] -> [B, 768, 14, 14]
        nn.Flatten(start_dim=2, end_dim=3) # [B, 768, 14, 14] -> [B, 768, 196]
    )

  def forward(self, x):
    # input spatial dimensions should be divided without remainder into 16x16 patches
    height, width = x.shape[-2:] #  x = [C, H, W]
    assert(height % self.patch_size == 0 and width % self.patch_size == 0)
    x = self.embedding(x) # [B, (P^2 . C), (HW / P^2)]
    x = x.permute(0, 2, 1) # [B, (HW / P^2), (P^2 . C)]
    return x