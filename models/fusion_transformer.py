"""
Fusion Transformer for combining semantic and geometric features.
Implements broadcast & concatenate followed by 1D transformer across points.
"""

import torch
import torch.nn as nn


class FusionTransformer1D(nn.Module):
    """
    1D Fusion Transformer that combines semantic and geometric features.
    
    Baseline (no bottleneck): Broadcast semantic features to all points,
    concatenate with geometric features, and run a Transformer across points
    with model dimension d_model = c_geo + c_sem.
    """
    def __init__(self, c_geo, c_sem, depth=4, heads=8):
        """
        Args:
            c_geo: Dimension of geometric features
            c_sem: Dimension of semantic features
            depth: Number of transformer layers
            heads: Number of attention heads
        """
        super().__init__()
        
        c_fuse = c_geo + c_sem
        # Transformer encoder layers (no input projection bottleneck)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=c_fuse,
            nhead=heads,
            dim_feedforward=4 * c_fuse,
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
    
    def forward(self, f_geo, s):
        """
        Forward pass through fusion transformer.
        
        Args:
            f_geo: [B, N, Cgeo] per-point geometric features
            s: [B, Csem] global semantic features
        
        Returns:
            f_fuse: [B, N, Cfuse] fused features
        """
        B, N, _ = f_geo.shape
        
        # Broadcast semantic features to all points
        s_tile = s.unsqueeze(1).expand(B, N, -1)  # [B, N, Csem]
        
        # Concatenate geometric and semantic features (no projection)
        x = torch.cat([f_geo, s_tile], dim=-1)  # [B, N, Cgeo + Csem]
        
        # Apply transformer directly in concatenated dimension
        f_fuse = self.enc(x)  # [B, N, Cgeo + Csem]
        
        return f_fuse
