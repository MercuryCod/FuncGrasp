# functional_grasp_model.py (corrected)
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from .semantics_qwen import QwenSemanticsEncoder
from .pointnet2_encoder import PN2GeometryEncoder
from .fusion_transformer import FusionTransformer1D
from .contact_head import ContactHead
from .flow_matching import PoseFlow

# Make Config optional
try:
    from config import Config
except Exception:
    class Config:
        NO_CONTACT_INDEX = 0  # fallback (no_contact:0, palm:1, fingers:2-6)

class FunctionalGraspModel(nn.Module):
    def __init__(
        self,
        CSEM: int = 256,
        CGEO: int = 256,
        DPOSE: int = 61,
        K_CONTACT: int = 7,
        qwen_tuning: str = 'lora',
        lora_cfg: dict = None,
        geo_cfg: dict = None,
        flow_cfg: dict = None,
        training_mode: str = 'joint'  # 'joint' or 'staged'
    ):
        super().__init__()

        self.sem = QwenSemanticsEncoder(tuning=qwen_tuning, lora_cfg=lora_cfg)
        self.flow_cfg = flow_cfg or {}
        dq = int(getattr(self.sem, 'hidden_size', 0) or 0)
        if dq <= 0:
            raise ValueError("QwenSemanticsEncoder.hidden_size must be > 0")

        self.sem_proj = nn.Sequential(
            nn.LayerNorm(dq),
            nn.Linear(dq, CSEM)
        )

        # Geometric encoder with centralized hyperparameters
        geo_cfg = geo_cfg or {}
        self.pc = PN2GeometryEncoder(
            in_c=geo_cfg.get('in_channels', 3),
            cgeo=CGEO,
            n1=geo_cfg.get('n_samples_sa1', 512),
            n2=geo_cfg.get('n_samples_sa2', 128),
            r1=geo_cfg.get('radius_sa1', 0.05),
            r2=geo_cfg.get('radius_sa2', 0.10),
            k_fp=geo_cfg.get('k_interpolate', 3),
            max_n1=geo_cfg.get('max_neighbors_sa1', 32),
            max_n2=geo_cfg.get('max_neighbors_sa2', 64)
        )

        CFUSE = CSEM + CGEO
        self.fuse = FusionTransformer1D(c_geo=CGEO, c_sem=CSEM)

        self.cm = ContactHead(c_fuse=CFUSE, k=K_CONTACT)
        if getattr(self.cm.net[-1], 'out_features', None) != 7:
            raise ValueError("ContactHead must output 7 classes (no_contact, palm, thumb, index, middle, ring, pinky)")

        self.flow = PoseFlow(d_pose=DPOSE, c_cond=CFUSE)

        self.DPOSE = DPOSE
        self.K_CONTACT = K_CONTACT
        self.training_mode = training_mode  # 'joint' or 'staged'
        self.training_stage = 'stage1'  # For staged mode: 'stage1' (contact only) or 'stage2' (joint)

    def _model_device(self):
        return next(self.parameters()).device
    
    def set_training_stage(self, stage):
        """Set training stage for staged mode: 'stage1' (contact only) or 'stage2' (joint)."""
        assert stage in ['stage1', 'stage2'], f"Invalid stage: {stage}"
        self.training_stage = stage
        print(f"[FunctionalGraspModel] Set training stage to: {stage}")

    def forward_backbone(self, images_list, texts_list, pts):
        """
        Returns:
            f_fuse: [B, N, CFUSE]
            logits_c: [B, N, 7]
            c: [B, CFUSE]
        """
        # Semantic encoding
        H, attention_mask = self.sem(images_list, texts_list)  # H: [B, L, Dq]
        if attention_mask is not None:
            mask = attention_mask.to(H.dtype).unsqueeze(-1)   # [B, L, 1]
            denom = mask.sum(dim=1).clamp_min(1e-6)           # [B, 1]
            pooled = (H * mask).sum(dim=1) / denom            # [B, Dq]
        else:
            pooled = H.mean(dim=1)                            # [B, Dq]

        pooled = pooled.float().to(self._model_device())
        s = self.sem_proj(pooled)                             # [B, CSEM]

        # Geometric encoding (ensure pts on model device)
        pts = pts.to(self._model_device())
        f_geo, _ = self.pc(pts)                               # [B, N, CGEO]

        # Fusion
        f_fuse = self.fuse(f_geo, s)                          # [B, N, CFUSE]

        # Contact prediction (always computed)
        logits_c = self.cm(f_fuse)                            # [B, N, 7]

        # Contact-weighted pooling for flow conditioning
        # Always use BCE regression approach (contact_regression removed)
        # Use independent sigmoid over the 6 contact parts (exclude no_contact at index 0)
        # Indices 1-6: palm, thumb, index, middle, ring, pinky
        part_probs = torch.sigmoid(logits_c[..., 1:])         # [B, N, 6]
        w = part_probs.max(dim=-1)[0]                         # [B, N]
        
        # Gradient isolation strategy depends on training mode:
        # - 'joint' mode: Always detach (prevents flow gradients affecting contact)
        # - 'staged' mode:
        #     * stage1: No flow, so detach doesn't matter
        #     * stage2: Contact is already well-trained, so we CAN allow gradients to flow
        #               (or still detach if we want to freeze contact learning)
        if self.training_mode == 'joint':
            # Joint training: detach to prevent gradient contamination
            w = w.detach()  
        elif self.training_mode == 'staged' and self.training_stage == 'stage2':
            # Staged mode, stage 2: Contact already trained, allow gradients (cleaner)
            # No detach - gradients can flow from flow loss to contact if desired
            pass
        else:
            # Staged mode, stage 1: Doesn't matter (no flow), but detach anyway for consistency
            w = w.detach()

        # Normalize and pool
        w = w / (w.sum(dim=1, keepdim=True) + 1e-6)
        c = (w.unsqueeze(-1) * f_fuse).sum(dim=1)             # [B, CFUSE]

        return f_fuse, logits_c, c

    def forward_train(self, images_list, texts_list, pts, mano_params_gt=None):
        """
        Training forward pass. Behavior depends on training mode and stage.
        
        Args:
            images_list: List of image lists for each batch element
            texts_list: List of text instructions
            pts: [B, N, 3] object point clouds
            mano_params_gt: [B, DPOSE] ground truth MANO parameters (for flow training)
        
        Returns:
            dict with keys depending on mode:
                - 'logits_c': [B, N, 7] contact logits (always)
                - 'cond': [B, CFUSE] conditioning vector (always)
                - 'mano_pred': [B, DPOSE] predicted MANO (only if flow is active)
        """
        f_fuse, logits_c, c = self.forward_backbone(images_list, texts_list, pts)
        
        outputs = {
            'logits_c': logits_c,
            'cond': c,
        }
        
        # Determine if we should compute flow
        compute_flow = False
        if self.training_mode == 'joint':
            # Joint mode: always compute flow
            compute_flow = True
        elif self.training_mode == 'staged':
            # Staged mode: only compute flow in stage2
            if self.training_stage == 'stage2':
                compute_flow = True
        
        # Compute flow prediction if needed
        if compute_flow:
            if mano_params_gt is None:
                raise ValueError("mano_params_gt required for flow training")
            
            # Flow matching forward pass (training)
            # Sample time and noise
            B = c.size(0)
            device = c.device
            
            # Random time in [0, 1]
            t = torch.rand(B, device=device)
            
            # Sample noise with parameter-specific scaling
            noise = torch.randn(B, self.DPOSE, device=device)
            noise[:, :48] *= self.flow_cfg.get('noise_scale_pose', 1.0)
            noise[:, 48:58] *= self.flow_cfg.get('noise_scale_shape', 1.0)
            noise[:, 58:61] *= self.flow_cfg.get('noise_scale_trans', 1.0)
            
            # Interpolate: x_t = (1-t)*noise + t*data
            t_expanded = t.view(B, 1)
            x_t = (1.0 - t_expanded) * noise + t_expanded * mano_params_gt
            
            # Predict velocity
            v_pred = self.flow_step(x_t, t, c)
            
            # Ground truth velocity (for rectified flow)
            v_gt = mano_params_gt - noise
            
            outputs['v_pred'] = v_pred
            outputs['v_gt'] = v_gt
            outputs['mano_pred'] = v_pred  # For compatibility
        
        return outputs

    def flow_step(self, x_t, t, c):
        return self.flow(x_t, t, c)

    def sample(self, images_list, texts_list, pts, num_steps=20):
        """
        Sample grasp poses with rectified-flow Euler integration.
        
        The flow network is trained directly in MANO parameter space (unnormalized).
        This allows the model to generalize to any MANO parameters without
        dataset-specific bias from normalization statistics.
        
        Returns:
            x: [B, DPOSE] MANO parameters (ready for MANO forward kinematics)
        """
        self.eval()
        device = self._model_device()
        with torch.no_grad():
            _, _, c = self.forward_backbone(images_list, texts_list, pts)   # c on model device
            B = c.size(0)
            
            # Sample from noise with parameter-specific scaling
            x = torch.randn(B, self.DPOSE, device=device)
            x[:, :48] *= self.flow_cfg.get('noise_scale_pose', 1.0)    # Pose
            x[:, 48:58] *= self.flow_cfg.get('noise_scale_shape', 1.0)  # Shape
            x[:, 58:61] *= self.flow_cfg.get('noise_scale_trans', 1.0)  # Translation
            
            dt = 1.0 / num_steps
            
            # Euler integration
            for k in range(num_steps):
                t = torch.full((B,), (k + 0.5) / num_steps, device=device)
                v = self.flow_step(x, t, c)
                x = x + dt * v
            
            # Return directly (no denormalization needed - trained in original space)
            return x
