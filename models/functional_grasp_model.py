"""
Full Functional Grasp Model integrating all components.
Combines Qwen2.5-VL, PointNet++, Fusion Transformer, Contact Head, and Flow Matching.
"""

import torch
import torch.nn as nn

from .semantics_qwen import QwenSemanticsEncoder
from .pointnet2_encoder import PointNet2Encoder
from .fusion_transformer import FusionTransformer1D
from .contact_head import ContactHead
from .flow_matching import PoseFlow


class FunctionalGraspModel(nn.Module):
    """
    Complete functional grasp model for predicting grasp poses.
    
    Pipeline:
    1. Semantic encoding: (Image, Text) → Qwen2.5-VL → s
    2. Geometric encoding: Point cloud → PointNet++ → F_geo
    3. Fusion: [F_geo; tile(s,N)] → Transformer → F_fuse
    4. Contact prediction: F_fuse → Contact Head → p_contact
    5. Conditioning: Contact-weighted pooling + MLP → c
    6. Pose generation: Flow Matching with c → grasp pose
    """
    def __init__(
        self,
        qwen_name="Qwen/Qwen2.5-VL-3B-Instruct",
        freeze_qwen=True,
        CSEM=256,
        CGEO=256,
        DPOSE=28,
        K_CONTACT=1
    ):
        """
        Args:
            qwen_name: Pretrained Qwen model name
            freeze_qwen: Whether to freeze Qwen backbone
            CSEM: Semantic feature dimension
            CGEO: Geometric feature dimension
            DPOSE: Pose dimension (wrist + joints)
            K_CONTACT: Number of contact classes
        """
        super().__init__()
        
        # Component models
        self.sem = QwenSemanticsEncoder(
            model_name=qwen_name,
            csem_proj=CSEM,
            freeze_backbone=freeze_qwen
        )
        
        self.pc = PointNet2Encoder(
            in_c=3,
            cgeo=CGEO
        )
        
        # CFUSE = CSEM + CGEO as per pipeline design
        CFUSE = CSEM + CGEO
        
        self.fuse = FusionTransformer1D(
            c_geo=CGEO,
            c_sem=CSEM,
            c_fuse=CFUSE
        )
        
        self.cm = ContactHead(
            c_fuse=CFUSE,
            k=K_CONTACT
        )
        
        self.flow = PoseFlow(
            d_pose=DPOSE,
            c_cond=CFUSE  # Conditioning is pooled fused features
        )
        
        # Store dimensions
        self.DPOSE = DPOSE

    def forward_backbone(self, images, texts, pts):
        """
        Forward pass through backbone to get features and conditioning.
        
        Args:
            images: List of PIL images
            texts: List of text prompts
            pts: [B, N, 3] point clouds
        
        Returns:
            f_fuse: [B, N, CFUSE] fused features
            logits_c: [B, N, 1] contact logits
            c: [B, CFUSE] conditioning vector (pooled fused)
        """
        # Semantic encoding
        s = self.sem(images, texts)  # [B, CSEM]
        
        # Geometric encoding
        f_geo, _ = self.pc(pts)  # [B, N, CGEO], global features not used in baseline
        
        # Fusion
        f_fuse = self.fuse(f_geo, s)  # [B, N, CFUSE]
        
        # Contact prediction
        logits_c = self.cm(f_fuse)  # [B, N, 1]
        
        # Simple mean pooling as in baseline design
        z = f_fuse.mean(dim=1)  # [B, CFUSE]
        c = z  # Conditioning is pooled fused features
        
        """
        Optional Advanced Features (from pipeline.md sections M.1-M.3):
        
        M.1 Contact-weighted pooling (focus on graspable regions):
        p = torch.sigmoid(logits_c)  # [B, N, 1]
        w = p / (p.sum(dim=1, keepdim=True) + 1e-6)
        z = (w * f_fuse).sum(dim=1)  # [B, CFUSE]
        
        # With warm-up mixing:
        z_mean = f_fuse.mean(dim=1)
        alpha = max(0.0, 1.0 - step / warmup_steps)
        z = alpha * z_mean + (1 - alpha) * z_weighted
        
        M.2 Global geometry skip path (preserve coarse shape context):
        f_geo, g = self.pc(pts)  # Get global features g
        c = torch.cat([z, s, g], dim=-1)  # [B, CFUSE + CSEM + CGEO]
        # Requires: self.cond = nn.Linear(CFUSE + CSEM + CGEO, CCOND)
        # And: self.flow = PoseFlow(d_pose=DPOSE, c_cond=CCOND)
        
        M.3 Input projection bottleneck is already implemented in FusionTransformer1D
        """
        
        return f_fuse, logits_c, c

    def forward_train(self, images, texts, pts):
        """
        Forward pass for training.
        
        Args:
            images: List of PIL images
            texts: List of text prompts
            pts: [B, N, 3] point clouds
        
        Returns:
            dict with:
                f_fuse: [B, N, CFUSE] fused features
                logits_c: [B, N, 1] contact logits
                cond: [B, CCOND] conditioning vector
        """
        # Get features and conditioning
        f_fuse, logits_c, c = self.forward_backbone(images, texts, pts)
        
        # Return outputs for loss computation
        return {
            'f_fuse': f_fuse,
            'logits_c': logits_c,
            'cond': c
        }

    def flow_step(self, x_t, t, c):
        """
        Single flow matching step.
        
        Args:
            x_t: [B, DPOSE] current pose state
            t: [B] time values
            c: [B, CCOND] conditioning
        
        Returns:
            v: [B, DPOSE] predicted velocity
        """
        return self.flow(x_t, t, c)
    
    def sample(self, images, texts, pts, num_steps=20, device='cuda'):
        """
        Sample grasp poses using flow matching.
        
        Args:
            images: List of PIL images
            texts: List of text prompts
            pts: [B, N, 3] point clouds
            num_steps: Number of integration steps
            device: Device to run on
        
        Returns:
            poses: [B, DPOSE] sampled grasp poses
        """
        self.eval()
        with torch.no_grad():
            # Get conditioning
            _, _, c = self.forward_backbone(images, texts, pts)
            B = c.size(0)
            
            # Start from noise
            x = torch.randn(B, self.DPOSE, device=device)
            
            # Integrate ODE
            dt = 1.0 / num_steps
            for k in range(num_steps):
                t = torch.full((B,), (k + 0.5) / num_steps, device=device)
                v = self.flow_step(x, t, c)
                x = x + dt * v
            
            return x