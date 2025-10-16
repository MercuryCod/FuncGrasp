"""
Conditional Flow Matching for grasp pose generation.
Implements rectified flow to transport from noise to grasp poses.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding1D(nn.Module):
    """
    Sinusoidal positional encoding for time embedding.
    """
    def __init__(self, d=64, maxw=10000.0):
        """
        Args:
            d: Dimension of encoding
            maxw: Maximum wavelength for sinusoidal encoding
        """
        super().__init__()
        self.d = d
        self.maxw = maxw
    
    def forward(self, t):
        """
        Generate positional encoding for time values.
        
        Args:
            t: [B, 1] time values in [0, 1]
        
        Returns:
            encoding: [B, d] positional encoding
        """
        device = t.device
        d = self.d // 2
        
        # Generate frequency bands
        freqs = torch.exp(
            torch.arange(d, device=device) * 
            (-math.log(self.maxw) / max(d - 1, 1))
        )
        
        # Apply sinusoidal encoding
        ang = 2 * math.pi * t * freqs  # [B, d]
        
        # Concatenate sin and cos
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [B, d*2]


class PoseFlow(nn.Module):
    """
    Conditional flow model for pose generation.
    Predicts velocity field v_θ(x_t, t, c) for rectified flow.
    """
    def __init__(self, d_pose, c_cond, hidden=1024):
        """
        Args:
            d_pose: Dimension of pose space
            c_cond: Dimension of conditioning vector
            hidden: Hidden layer dimension
        """
        super().__init__()
        
        # Time embedding network
        self.t_embed = nn.Sequential(
            PositionalEncoding1D(64),
            nn.Linear(64, 128),
            nn.SiLU()
        )
        
        # Main velocity prediction network
        self.net = nn.Sequential(
            nn.Linear(d_pose + 128 + c_cond, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_pose)
        )
    
    def forward(self, x_t, t, c):
        """
        Predict velocity field for flow matching.
        
        Args:
            x_t: [B, Dpose] current pose state
            t: [B] time values in [0, 1]
            c: [B, Ccond] conditioning vector
        
        Returns:
            v: [B, Dpose] predicted velocity
        """
        # Embed time
        te = self.t_embed(t.unsqueeze(-1))  # [B, 128]
        
        # Concatenate all inputs
        inputs = torch.cat([x_t, te, c], dim=-1)  # [B, Dpose + 128 + Ccond]
        
        # Predict velocity
        v = self.net(inputs)  # [B, Dpose]
        
        return v