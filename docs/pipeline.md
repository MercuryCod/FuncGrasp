A. Final pipeline (end‑to‑end)

Goal: predict a functional dexterous hand grasp pose conditioned on semantics (image+language) and geometry (object point cloud).

(Image(s), Text) ──► Qwen2.5‑VL (frozen) ──► s ∈ ℝ^{B×Csem}
Point cloud P ∈ ℝ^{B×N×d} ─► PointNet++ ─► F_geo ∈ ℝ^{B×N×Cgeo}

Broadcast & Concat: [F_geo ; tile(s,N)] ─► 1D Fusion Transformer (across points)
  └─► F_fuse ∈ ℝ^{B×N×Cfuse}

Per‑point contact head:
  F_fuse ─► logits_c ∈ ℝ^{B×N×1} ─► p_contact ∈ [0,1]

Contact‑weighted pooling:
  z = Σ_i softmax(p_contact)_i · F_fuse_i ∈ ℝ^{B×Cfuse}
  c = MLP([z; s; g]) ∈ ℝ^{B×Ccond}     (g = global geometry pool)

Conditional Flow Matching (rectified flow):
  Train vector field vθ(x_t,t,c) to transport N(0,I) → grasp pose y
  Inference: integrate dx/dt = vθ(x,t,c) from t=0→1 → pose ŷ ∈ ℝ^{B×Dpose}


Canonical shapes.
B batch; N points (e.g., 1024);
Csem from Qwen (projected); Cgeo from PointNet++; Cfuse fusion width; Ccond conditioning width; Dpose wrist + joints (e.g., 26–30 DOF).

B. Why Qwen2.5‑VL works well here (and how we use it)

HF Transformers includes first‑class support for Qwen2.5‑VL via Qwen2_5_VLForConditionalGeneration / Qwen2_5_VLModel and AutoProcessor; use the chat template + processor to pack images+text. It also exposes min_pixels/max_pixels to control image tokenization cost. 
Hugging Face

The model card provides a reference quickstart and notes to install a recent Transformers (otherwise you may hit the KeyError: 'qwen2_5_vl'). It also shows the recommended helper qwen-vl-utils to assemble images/videos payloads. 
Hugging Face

Config exposes image_token_id and other internals; we’ll simply mean‑pool the final hidden states under the attention mask to obtain a robust global semantic embedding s without modifying Qwen. 
Hugging Face

C. Dependencies & environment
# Core
pip install -U "torch>=2.3" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -U "transformers>=4.51" accelerate
pip install qwen-vl-utils[decord]==0.0.8   # helper for images/videos packing (optional but convenient)
# Geometry
pip install open3d pytorch3d  # or torch-geometric; choose your preference
# (Optional) quantization / speedups
pip install torchao  # weight-only quantization if ever needed


Qwen2.5‑VL support landed in recent transformers; upgrading avoids the qwen2_5_vl KeyError mentioned in the model card. 
Hugging Face

D. Implementation skeleton (modular PyTorch)

Folders

grasp/
  models/
    semantics_qwen.py
    pointnet2_encoder.py
    fusion_transformer.py
    contact_head.py
    flow_matching.py
    functional_grasp_model.py
  train.py
  utils.py
  cfg.py

D1) Qwen2.5‑VL semantics encoder (frozen features)
# grasp/models/semantics_qwen.py
import torch, torch.nn as nn
from transformers import AutoProcessor, Qwen2_5_VLModel
from qwen_vl_utils import process_vision_info  # from qwen-vl-utils

class QwenSemanticsEncoder(nn.Module):
    """
    Wraps Qwen2.5-VL to produce a pooled multimodal embedding s ∈ ℝ^{B×Csem_proj}.
    Default: freeze Qwen and learn only a projection.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                 csem_proj=256, min_pixels=None, max_pixels=None, device_map="auto", dtype=torch.bfloat16):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels, max_pixels=max_pixels
        )
        # Use the *backbone* (no LM head) to access hidden states
        self.backbone = Qwen2_5_VLModel.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device_map
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, csem_proj)
        )

    @torch.inference_mode()
    def _pack(self, images, texts):
        # images: list of PIL.Image or paths; texts: list[str], length B
        messages = [{"role": "user",
                     "content": [{"type": "image", "image": im},
                                 {"type": "text",  "text": txt}]}]
        # one-message-per-sample batching
        messages = [{"role":"user","content":[{"type":"image","image":im},{"type":"text","text":tx}]} 
                    for im,tx in zip(images, texts)]
        text_inputs = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                       for m in messages]
        image_inputs, video_inputs = zip(*[process_vision_info(m) for m in messages])
        proc = self.processor(
            text=text_inputs, images=list(image_inputs), videos=list(video_inputs),
            padding=True, return_tensors="pt"
        )
        return proc

    def forward(self, images, texts):
        """
        images: List[image]; texts: List[str]; returns s [B, Csem_proj]
        """
        inputs = self._pack(images, texts).to(next(self.proj.parameters()).device)
        out = self.backbone(**inputs, output_hidden_states=False, return_dict=True)
        # masked mean pool over tokens (text+vision) → global multimodal vector
        mask = inputs.attention_mask.float().unsqueeze(-1)  # [B, L, 1]
        pooled = (out.last_hidden_state * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1e-6))
        s = self.proj(pooled)  # [B, Csem_proj]
        return s


Notes: AutoProcessor + chat template and the min_pixels/max_pixels knobs are the officially supported way to control visual token counts. 
Hugging Face
+1

D2) PointNet++ encoder (per‑point features)

You can plug your favorite implementation. Below is a light, dependency‑free PointNet++‑like fallback (FPS + kNN groups with shared MLPs). Replace with your high‑performance version when ready.

# grasp/models/pointnet2_encoder.py
import torch, torch.nn as nn, torch.nn.functional as F

def fps(x, M):
    # x: [B,N,3] → indices [B,M]
    B, N, _ = x.shape
    centroids = torch.zeros(B, M, dtype=torch.long, device=x.device)
    dist = torch.full((B, N), 1e10, device=x.device)
    farthest = torch.randint(0, N, (B,), device=x.device)
    batch = torch.arange(B, device=x.device)
    for i in range(M):
        centroids[:, i] = farthest
        centroid = x[batch, farthest].unsqueeze(1)  # [B,1,3]
        d = ((x - centroid)**2).sum(-1)             # [B,N]
        dist = torch.minimum(dist, d)
        farthest = torch.max(dist, dim=1).indices
    return centroids

class SharedMLP(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(nn.Linear(in_c, out_c), nn.ReLU(), nn.Linear(out_c, out_c), nn.ReLU())

class PointNet2Encoder(nn.Module):
    """
    Minimal PointNet++-style encoder.
    Outputs: per-point F_geo [B,N,Cgeo] and global g [B,Cgeo].
    """
    def __init__(self, in_c=3, cgeo=256, hidden=128, groups=32, samples=512):
        super().__init__()
        self.groups, self.samples = groups, samples
        self.mlp1 = SharedMLP(in_c, hidden)
        self.mlp2 = SharedMLP(hidden, cgeo)

    def forward(self, pts):
        # pts: [B,N,3] (optionally append normals/RGB)
        x = self.mlp1(pts)                    # [B,N,H]
        B, N, _ = x.shape
        # very light grouping (FPS + local mean), keeps everything differentiable
        idx = fps(pts[..., :3], min(self.samples, N))       # [B,S]
        anchors = x.gather(1, idx.unsqueeze(-1).expand(-1,-1,x.size(-1)))  # [B,S,H]
        # project anchors back to all points by nearest-anchor assignment
        dists = torch.cdist(pts[..., :3], pts[torch.arange(B)[:,None], idx][..., :3]) # [B,N,S]
        w = torch.softmax(-dists, dim=-1)
        x = (w @ anchors)                     # [B,N,H]
        F_geo = self.mlp2(x)                  # [B,N,Cgeo]
        g = F_geo.mean(dim=1)                 # [B,Cgeo]
        return F_geo, g

D3) Fusion transformer & contact head
# grasp/models/fusion_transformer.py
import torch, torch.nn as nn

class FusionTransformer1D(nn.Module):
    def __init__(self, c_geo, c_sem, c_fuse=512, depth=4, heads=8):
        super().__init__()
        self.in_proj = nn.Linear(c_geo + c_sem, c_fuse)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=c_fuse, nhead=heads, dim_feedforward=4*c_fuse,
            batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
    def forward(self, f_geo, s):
        # f_geo: [B,N,Cgeo], s: [B,Csem]
        B, N, _ = f_geo.shape
        s_tile = s.unsqueeze(1).expand(B, N, -1)          # [B,N,Csem]
        x = self.in_proj(torch.cat([f_geo, s_tile], -1))  # [B,N,Cfuse]
        return self.enc(x)                                # [B,N,Cfuse]

# grasp/models/contact_head.py
import torch.nn as nn, torch

class ContactHead(nn.Module):
    def __init__(self, c_fuse, k=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_fuse, c_fuse//2), nn.ReLU(),
            nn.Linear(c_fuse//2, k)
        )
    def forward(self, f_fuse):            # [B,N,Cfuse]
        return self.net(f_fuse)          # [B,N,k]

D4) Conditional Flow Matching (rectified flow)
# grasp/models/flow_matching.py
import torch, torch.nn as nn, math

class PositionalEncoding1D(nn.Module):
    def __init__(self, d=64, maxw=10000.0):
        super().__init__(); self.d = d; self.maxw = maxw
    def forward(self, t):
        # t: [B,1] in [0,1]
        device = t.device; d = self.d // 2
        freqs = torch.exp(torch.arange(d, device=device) * (-math.log(self.maxw)/max(d-1,1)))
        ang = 2*math.pi * t * freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [B,d*2]

class PoseFlow(nn.Module):
    def __init__(self, d_pose, c_cond, hidden=1024):
        super().__init__()
        self.t_embed = nn.Sequential(PositionalEncoding1D(64), nn.Linear(64, 128), nn.SiLU())
        self.net = nn.Sequential(
            nn.Linear(d_pose + 128 + c_cond, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d_pose)
        )
    def forward(self, x_t, t, c):   # x_t: [B,Dpose], t: [B], c: [B,Ccond]
        te = self.t_embed(t.unsqueeze(-1))
        return self.net(torch.cat([x_t, te, c], -1))

D5) Full model wrapper
# grasp/models/functional_grasp_model.py
import torch, torch.nn as nn, torch.nn.functional as F
from .semantics_qwen import QwenSemanticsEncoder
from .pointnet2_encoder import PointNet2Encoder
from .fusion_transformer import FusionTransformer1D
from .contact_head import ContactHead
from .flow_matching import PoseFlow

class FunctionalGraspModel(nn.Module):
    def __init__(self, qwen_name="Qwen/Qwen2.5-VL-3B-Instruct",
                 CSEM=256, CGEO=256, CFUSE=512, CCOND=512,
                 DPOSE=28, K_CONTACT=1):
        super().__init__()
        self.sem = QwenSemanticsEncoder(model_name=qwen_name, csem_proj=CSEM)
        self.pc  = PointNet2Encoder(in_c=3, cgeo=CGEO)
        self.fuse= FusionTransformer1D(c_geo=CGEO, c_sem=CSEM, c_fuse=CFUSE)
        self.cm  = ContactHead(CFUSE, k=K_CONTACT)
        self.cond= nn.Linear(CFUSE + CSEM + CGEO, CCOND)
        self.flow= PoseFlow(d_pose=DPOSE, c_cond=CCOND)

    def forward_backbone(self, images, texts, pts):
        s = self.sem(images, texts)             # [B,CSEM]
        f_geo, g = self.pc(pts)                 # [B,N,CGEO], [B,CGEO]
        f_fuse = self.fuse(f_geo, s)            # [B,N,CFUSE]
        logits_c = self.cm(f_fuse)              # [B,N,1]
        p = torch.sigmoid(logits_c)             # [B,N,1]
        w = p / (p.sum(1, keepdim=True) + 1e-6) # normalized weights
        z = (w * f_fuse).sum(1)                 # [B,CFUSE]
        c = self.cond(torch.cat([z, s, g], -1)) # [B,CCOND]
        return f_fuse, logits_c, c

    def forward_train(self, images, texts, pts, y_pose):
        # contact map + CFM losses
        f_fuse, logits_c, c = self.forward_backbone(images, texts, pts)

        # Contact loss (binary)
        # Expect batch provides contact_labels [B,N] in caller; use BCE as example
        # Return logits_c and let caller compute BCE with labels to keep flexibility
        return dict(f_fuse=f_fuse, logits_c=logits_c, cond=c)

    def flow_step(self, x_t, t, c):
        return self.flow(x_t, t, c)

D6) Training loop (minimal)
# grasp/train.py
import torch, torch.nn.functional as F
from torch.optim import AdamW
from models.functional_grasp_model import FunctionalGraspModel

def bce_logits(logits, labels):
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())

def project_to_limits(x, lo, hi):   # x: [B,D]; lo/hi: vectors
    return lo + (torch.tanh(x) + 1) * 0.5 * (hi - lo)

def train_one_epoch(model, loader, opt, cfg, device="cuda"):
    model.train()
    for batch in loader:
        images, texts, pts = batch["images"], batch["texts"], batch["points"].to(device)
        y = batch["pose"].to(device)                            # [B,Dpose]
        contact_labels = batch["contact_labels"].to(device)     # [B,N]

        out = model.forward_train(images, texts, pts, y)
        logits_c, c = out["logits_c"], out["cond"]

        # Contact map loss
        Lc = bce_logits(logits_c, contact_labels)

        # Rectified Flow Matching
        x0 = torch.randn_like(y)
        t  = torch.rand(y.size(0), device=y.device)
        x_t = (1 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * y
        v_star = y - x0                                        # constant target
        v = model.flow_step(x_t, t, c)
        Lfm = ((v - v_star)**2).mean()

        loss = cfg["lambda_fm"]*Lfm + cfg["lambda_c"]*Lc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

def sample(model, images, texts, pts, cfg, device="cuda", K=20, M=4):
    model.eval()
    with torch.no_grad():
        _, logits_c, c = model.forward_backbone(images, texts, pts.to(device))
        B, D = c.size(0), cfg["Dpose"]
        best = []
        for m in range(M):
            x = torch.randn(B, D, device=device)
            for k in range(K):
                t = torch.full((B,), (k+0.5)/K, device=device)
                x = x + (1.0/K) * model.flow_step(x, t, c)
            best.append(x)
        return torch.stack(best, dim=1)   # [B,M,D]

E. Data interface you’ll want
# Each batch should provide:
batch = {
  "images": [PIL.Image.Image, ...],  # length B
  "texts":  [str, ...],              # length B, affordance prompt / recipe
  "points": torch.Tensor[B,N,3],     # object point cloud in camera or object frame
  "contact_labels": torch.Tensor[B,N], # 0/1 labels (or probabilities) for contact
  "pose": torch.Tensor[B,Dpose],     # wrist pose + joint angles, in consistent frame
}


Contact labels can be painted from demonstration contacts (nearest‑surface assignment).
Pose parameterization: (tx, ty, tz, rx, ry, rz) (axis‑angle) + finger joints; keep joint limits to post‑project or squash by tanh mapping.

F. Default hyper‑parameters (good first pass)

N = 1024 points (random uniform from mesh/pcd; oversample then FPS downsample is fine)

CSEM = 256, CGEO = 256, CFUSE = 512, CCOND = 512

DPOSE = 28 (adapt to your hand)

Fusion Transformer: 4 layers, 8 heads

Optimizer: AdamW, lr=1e-4 (backbone frozen), weight decay 0.05

Loss weights: λ_fm=1.0, λ_c=1.0

Qwen processor resolution: min_pixels=256*28*28, max_pixels=1024*28*28 as a balanced start (scale for memory). 
Hugging Face

G. Inference post‑processing & ranking (recommended)

Generate M poses, score each with:

Penetration penalty using a coarse SDF / signed ray distance against the object mesh;

Contact coverage Σ_i p_contact(i) · 𝟙(point i within d of any finger link);

Reachability and joint‑limit penalty.

Pick top‑K by weighted sum.

H. Ablations you can run (to validate choices)

Remove the fusion transformer (replace with 1×1 MLP) → expect a drop in functional placement accuracy.

Replace contact‑weighted pooling with average pooling.

Supply text‑only vs image+text to Qwen to quantify visual gain.

End‑to‑end fine‑tune a small LoRA on Qwen’s final layers (optional, after the base works).

I. Checklist of invariants (prevents silent bugs)

Coordinate frames are consistent (all in camera or object frame).

p_contact is normalized before pooling (softmax or L1).

Joint angles always projected back to limits after sampling.

t ~ U[0,1] every step; use the same (x0, t) for vθ and the target v*.

If you quantize Qwen weights for memory, keep the projection head and downstream modules in full precision.

J. Quick smoke test (one batch)
from grasp.models.functional_grasp_model import FunctionalGraspModel

model = FunctionalGraspModel(DPOSE=28)
dummy_images = [Image.open("example.jpg")] * 2     # two identical images just to test
dummy_texts  = ["how to use this object to pour water?"] * 2
dummy_pts    = torch.randn(2, 1024, 3)
out = model.forward_backbone(dummy_images, dummy_texts, dummy_pts)
for k,v in zip(["F_fuse","logits_c","cond"], out):
    print(k, v.shape)
# Expect: [B,N,Cfuse], [B,N,1], [B,Ccond]