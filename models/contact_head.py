"""
Contact Head for per-point contact prediction.
Predicts 7-class contact logits for each point in the point cloud
(thumb, index, middle, ring, little, palm, no_contact).
"""

import torch
import torch.nn as nn


class ContactHead(nn.Module):
    """
    Per-point contact prediction head.
    
    Takes fused features and predicts per-point contact logits (7 classes).
    """
    def __init__(self, c_fuse, k=7):
        """
        Args:
            c_fuse: Dimension of input fused features
            k: Number of contact classes (default 7)
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
        Forward pass to predict contact logits (7 classes).
        
        Args:
            f_fuse: [B, N, Cfuse] fused features for each point
        
        Returns:
            logits_c: [B, N, 7] contact logits
        """
        return self.net(f_fuse)