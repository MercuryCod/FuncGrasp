"""
PointNet++ Encoder for point cloud feature extraction.
Minimal implementation with FPS sampling and shared MLPs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def fps(x, M):
    """
    Farthest Point Sampling
    
    Args:
        x: [B, N, 3] point coordinates
        M: number of points to sample
    
    Returns:
        indices: [B, M] indices of sampled points
    """
    B, N, _ = x.shape
    device = x.device
    
    centroids = torch.zeros(B, M, dtype=torch.long, device=device)
    dist = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), device=device)
    batch = torch.arange(B, device=device)
    
    for i in range(M):
        centroids[:, i] = farthest
        centroid = x[batch, farthest].unsqueeze(1)  # [B, 1, 3]
        d = ((x - centroid) ** 2).sum(-1)           # [B, N]
        dist = torch.minimum(dist, d)
        farthest = torch.max(dist, dim=1).indices
    
    return centroids


class SharedMLP(nn.Sequential):
    """Shared MLP with ReLU activations"""
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.Linear(in_c, out_c),
            nn.ReLU(),
            nn.Linear(out_c, out_c),
            nn.ReLU()
        )


class PointNet2Encoder(nn.Module):
    """
    Minimal PointNet++-style encoder.
    Outputs: per-point F_geo [B,N,Cgeo] and global g [B,Cgeo].
    """
    def __init__(self, in_c=3, cgeo=256, hidden=128, groups=32, samples=512):
        super().__init__()
        self.groups = groups
        self.samples = samples
        
        # First MLP for initial features
        self.mlp1 = SharedMLP(in_c, hidden)
        
        # Second MLP for final features
        self.mlp2 = SharedMLP(hidden, cgeo)

    def forward(self, pts):
        """
        Forward pass through PointNet++ encoder.
        
        Args:
            pts: [B, N, 3] point cloud (optionally append normals/RGB)
        
        Returns:
            F_geo: [B, N, Cgeo] per-point features
            g: [B, Cgeo] global features
        """
        # Initial feature extraction
        x = self.mlp1(pts)  # [B, N, H]
        B, N, H = x.shape
        
        # Very light grouping (FPS + local mean), keeps everything differentiable
        # Sample subset of points as anchors
        idx = fps(pts[..., :3], min(self.samples, N))  # [B, S]
        
        # Gather features at sampled points
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, H)  # [B, S, H]
        anchors = x.gather(1, idx_expanded)  # [B, S, H]
        
        # Project anchors back to all points by nearest-anchor assignment
        # Get coordinates of sampled points
        batch_indices = torch.arange(B, device=pts.device)[:, None]
        sampled_pts = pts[batch_indices, idx]  # [B, S, 3]
        
        # Compute distances from all points to sampled points
        dists = torch.cdist(pts[..., :3], sampled_pts[..., :3])  # [B, N, S]
        
        # Soft assignment weights
        w = torch.softmax(-dists, dim=-1)  # [B, N, S]
        
        # Weighted combination of anchor features
        x = w @ anchors  # [B, N, H]
        
        # Final feature extraction
        F_geo = self.mlp2(x)  # [B, N, Cgeo]
        
        # Global pooling
        g = F_geo.mean(dim=1)  # [B, Cgeo]
        
        return F_geo, g