# functional_grasp_model.py (corrected)
import torch
import torch.nn as nn

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
        DPOSE: int = 63,
        K_CONTACT: int = 7,
        qwen_tuning: str = 'frozen',
        lora_cfg: dict = None,
        contact_regression: bool = True,
        inference_threshold: float = 0.4
    ):
        super().__init__()

        self.sem = QwenSemanticsEncoder(tuning=qwen_tuning, lora_cfg=lora_cfg, dtype=torch.bfloat16)
        dq = int(getattr(self.sem, 'hidden_size', 0) or 0)
        if dq <= 0:
            raise ValueError("QwenSemanticsEncoder.hidden_size must be > 0")

        self.sem_proj = nn.Sequential(
            nn.LayerNorm(dq),
            nn.Linear(dq, CSEM)
        )

        self.pc = PN2GeometryEncoder(in_c=3, cgeo=CGEO)

        CFUSE = CSEM + CGEO
        self.fuse = FusionTransformer1D(c_geo=CGEO, c_sem=CSEM)

        self.cm = ContactHead(c_fuse=CFUSE, k=K_CONTACT)
        if getattr(self.cm.net[-1], 'out_features', None) != 7:
            raise ValueError("ContactHead must output 7 classes (thumb,index,middle,ring,little,palm,no_contact)")

        self.flow = PoseFlow(d_pose=DPOSE, c_cond=CFUSE)

        self.DPOSE = DPOSE
        self.K_CONTACT = K_CONTACT
        self.contact_regression = contact_regression
        self.inference_threshold = inference_threshold

    def _model_device(self):
        return next(self.parameters()).device

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

        # Contact prediction
        logits_c = self.cm(f_fuse)                            # [B, N, 7]

        # Pooling weights
        if self.contact_regression:
            # Use independent sigmoid over the 6 contact parts (exclude no_contact at index 0)
            # Indices 1-6: palm, thumb, index, middle, ring, pinky
            part_probs = torch.sigmoid(logits_c[..., 1:])     # [B, N, 6]
            w = part_probs.max(dim=-1)[0]                     # [B, N]
        else:
            probs = logits_c.softmax(dim=-1)                  # [B, N, 7]
            no_contact_index = int(getattr(Config, "NO_CONTACT_INDEX", 0))  # Default to 0
            p_nc = probs[..., no_contact_index]               # [B, N]
            w = 1.0 - p_nc

        # Normalize and pool
        w = w / (w.sum(dim=1, keepdim=True) + 1e-6)
        c = (w.unsqueeze(-1) * f_fuse).sum(dim=1)             # [B, CFUSE]

        return f_fuse, logits_c, c

    def forward_train(self, images_list, texts_list, pts):
        f_fuse, logits_c, c = self.forward_backbone(images_list, texts_list, pts)
        return {'f_fuse': f_fuse, 'logits_c': logits_c, 'cond': c}

    def flow_step(self, x_t, t, c):
        return self.flow(x_t, t, c)

    def sample(self, images_list, texts_list, pts, num_steps=20):
        """
        Sample grasp poses with rectified-flow Euler integration.
        """
        self.eval()
        device = self._model_device()
        with torch.no_grad():
            _, _, c = self.forward_backbone(images_list, texts_list, pts)   # c on model device
            B = c.size(0)
            x = torch.randn(B, self.DPOSE, device=device)
            dt = 1.0 / num_steps
            for k in range(num_steps):
                t = torch.full((B,), (k + 0.5) / num_steps, device=device)
                v = self.flow_step(x, t, c)
                x = x + dt * v
            return x
