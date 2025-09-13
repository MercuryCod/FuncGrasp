"""
PointNet++ Encoder using PyTorch Geometric's built-in PointNet2.
This implementation relies on torch_geometric and does not provide a fallback.
"""

import torch
import torch.nn as nn


class PointNet2Encoder(nn.Module):
    """Adapter around torch_geometric.nn.models.PointNet2.
    Produces per-point features [B,N,Cgeo] and a global mean [B,Cgeo].
    """
    def __init__(self, in_c=3, cgeo=256):
        super().__init__()
        try:
            from torch_geometric.nn.models import PointNet2
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "torch_geometric is required. Please install PyTorch Geometric to use PointNet++ (pip install torch-geometric)."
            ) from e

        # Configure PointNet2 to output per-point embeddings with 'out_channels=cgeo'.
        self.backbone = PointNet2(in_channels=in_c, out_channels=cgeo)

    def forward(self, pts):
        # pts: [B,N,3]
        B, N, _ = pts.shape
        pos = pts.reshape(B * N, 3)
        batch = torch.arange(B, device=pts.device).repeat_interleave(N)
        # Forward pass: some PyG versions accept positional args, others kwargs
        try:
            F_geo_flat = self.backbone(None, pos, batch)  # [B*N, Cgeo]
        except TypeError:
            F_geo_flat = self.backbone(x=None, pos=pos, batch=batch)
        F_geo = F_geo_flat.view(B, N, -1)
        g = F_geo.mean(dim=1)
        return F_geo, g
