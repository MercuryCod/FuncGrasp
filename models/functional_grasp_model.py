"""
Full Functional Grasp Model integrating all components.
Combines Qwen2.5-VL, PointNet++, Fusion Transformer,
Contact Head, and Flow Matching. The semantic hidden size is read
from the loaded encoder dynamically (2048 for Qwen2.5-VL-3B).
"""

import torch
import torch.nn as nn

from .semantics_qwen import QwenSemanticsEncoder
from .pointnet2_encoder import PN2GeometryEncoder
from .fusion_transformer import FusionTransformer1D
from .contact_head import ContactHead
from .flow_matching import PoseFlow


class FunctionalGraspModel(nn.Module):
    """
    Complete functional grasp model for predicting grasp poses.
    
    Pipeline (fully batched):
    1. Semantic encoding: B×(Images, Text) → Qwen2.5-VL → s[B,CSEM]
    2. Geometric encoding: Point clouds[B,N,3] → PointNet++ → F_geo[B,N,CGEO]
    3. Fusion: [F_geo; tile(s,N)] → Transformer → F_fuse[B,N,CFUSE]
    4. Contact prediction: F_fuse → Contact Head → p_contact[B,N,1]
    5. Conditioning: Contact-weighted pooling → c[B,CFUSE]
    6. Pose generation: Flow Matching with c → poses[B,DPOSE]
    
    Each sample in the batch has its own semantic context (text + images).
    """
    def __init__(
        self,
        CSEM: int = 256,
        CGEO: int = 256,
        DPOSE: int = 28,
        K_CONTACT: int = 1
    ):
        """
        Args:
            CSEM: Semantic feature dimension
            CGEO: Geometric feature dimension
            DPOSE: Pose dimension (wrist + joints)
            K_CONTACT: Number of contact classes
        """
        super().__init__()
        
        # Component models
        # Qwen2.5-VL encoder; returns last_hidden_state [B, L, Dq]
        self.sem = QwenSemanticsEncoder()
        # Pool + project semantic hidden states to CSEM for fusion using dynamic Dq
        dq = int(getattr(self.sem, 'hidden_size', 0) or 0)
        if dq <= 0:
            raise ValueError("QwenSemanticsEncoder.hidden_size must be > 0")
        self.sem_proj = nn.Sequential(
            nn.LayerNorm(dq),
            nn.Linear(dq, CSEM)
        )
        
        self.pc = PN2GeometryEncoder(in_c=3, cgeo=CGEO)
        
        # CFUSE = CSEM + CGEO as per pipeline design
        CFUSE = CSEM + CGEO
        
        self.fuse = FusionTransformer1D(
            c_geo=CGEO,
            c_sem=CSEM
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

    def forward_backbone(self, images_list, texts_list, pts):
        """
        Forward pass through backbone to get features and conditioning.
        
        Args:
            images_list: List[List[PIL.Image]] - B lists of images (one per sample)
            texts_list: List[str] - B text prompts (one per sample)
            pts: [B, N, 3] point clouds
        
        Returns:
            f_fuse: [B, N, CFUSE] fused features
            logits_c: [B, N, 1] contact logits
            c: [B, CFUSE] conditioning vector (pooled fused)
        """
        B = pts.shape[0]
        
        # Semantic encoding: hidden states [B, L_max, Dq] with attention mask → masked-pooled s [B, CSEM]
        H, attention_mask = self.sem(images_list, texts_list)  # H: [B, L_max, Dq], attention_mask: [B, L_max]
        if attention_mask is not None:
            mask = attention_mask.to(H.dtype).unsqueeze(-1)   # [B, L_max, 1]
            denom = mask.sum(dim=1).clamp_min(1e-6)            # [B, 1]
            pooled = (H * mask).sum(dim=1) / denom             # [B, Dq]
        else:
            pooled = H.mean(dim=1)  # [B, Dq]
        
        # Ensure sem_proj is on the same device as pooled and convert to float32
        device = pooled.device
        self.sem_proj = self.sem_proj.to(device)
        
        # Convert to float32 for compatibility with linear layers
        pooled = pooled.float()
        
        s = self.sem_proj(pooled)  # [B, CSEM]
        
        # Geometric encoding
        f_geo, _ = self.pc(pts)  # [B, N, CGEO], global features not used in baseline
        
        # Fusion
        f_fuse = self.fuse(f_geo, s)  # [B, N, CFUSE]
        
        # Contact prediction
        logits_c = self.cm(f_fuse)  # [B, N, 1]
        
        # Contact‑weighted pooling (baseline)
        p = torch.sigmoid(logits_c)  # [B, N, 1]
        w = p / (p.sum(dim=1, keepdim=True) + 1e-6)
        z = (w * f_fuse).sum(dim=1)  # [B, CFUSE]
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

    def forward_train(self, images_list, texts_list, pts):
        """
        Forward pass for training.
        
        Args:
            images_list: List[List[PIL.Image]] - B lists of images (one per sample)
            texts_list: List[str] - B text prompts (one per sample)
            pts: [B, N, 3] point clouds
        
        Returns:
            dict with:
                f_fuse: [B, N, CFUSE] fused features
                logits_c: [B, N, 1] contact logits
                cond: [B, CCOND] conditioning vector
        """
        # Get features and conditioning
        f_fuse, logits_c, c = self.forward_backbone(images_list, texts_list, pts)
        
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
    
    def sample(self, images_list, texts_list, pts, num_steps=20, device='cuda'):
        """
        Sample grasp poses using flow matching.
        
        Args:
            images_list: List[List[PIL.Image]] - B lists of images (one per sample)
            texts_list: List[str] - B text prompts (one per sample)
            pts: [B, N, 3] point clouds
            num_steps: Number of integration steps
            device: Device to run on
        
        Returns:
            poses: [B, DPOSE] sampled grasp poses
        """
        self.eval()
        with torch.no_grad():
            # Get conditioning
            _, _, c = self.forward_backbone(images_list, texts_list, pts)
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
