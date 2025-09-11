"""
Contact Head for per-point contact prediction.
Predicts contact probability for each point in the point cloud.
"""

import torch
import torch.nn as nn


class ContactHead(nn.Module):
    """
    Per-point contact prediction head.
    
    Takes fused features and predicts contact probability for each point.
    """
    def __init__(self, c_fuse, k=1):
        """
        Args:
            c_fuse: Dimension of input fused features
            k: Number of contact classes (1 for binary)
        """
        super().__init__()
        
        # Two-layer MLP with ReLU activation
        self.net = nn.Sequential(
            nn.Linear(c_fuse, c_fuse // 2),
            nn.ReLU(),
            nn.Linear(c_fuse // 2, k)
        )
    
    def forward(self, f_fuse):
        """
        Forward pass to predict contact logits.
        
        Args:
            f_fuse: [B, N, Cfuse] fused features for each point
        
        Returns:
            logits_c: [B, N, k] contact logits (k=1 for binary)
        """
        return self.net(f_fuse)