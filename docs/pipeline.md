# High-level design for Functional Grasp Training Pipeline

## A. Overview

**Goal**: Predict functional dexterous hand grasp poses conditioned on semantics (image+language) and object geometry (point cloud).

> **Visual Reference**: See hand-drawn architecture sketches in `assets/`:
> - `overall.png` - Overall system architecture
> - `fusion.png` - Fusion mechanism details

### Architecture Flow
```
(Image, Text) ──► Qwen2.5-VL ──► H ∈ ℝ^{B×L_max×2048} ──► masked‑pool+proj ──► s ∈ ℝ^{B×256}
     ↓
Point Cloud ────► PointNet++ ──► F_geo ∈ ℝ^{B×N×256}          [Geometric features]
     ↓
Concat s (tiled over N) ──► Fusion Transformer (across points) ──► F_fuse ∈ ℝ^{B×N×(CSEM+CGEO)}
     ↓
Contact Head ──► logits_c ∈ ℝ^{B×N×7}                           [7-class contact logits]
     ↓
Pooling (1-p(no_contact)) ──► z ∈ ℝ^{B×(CSEM+CGEO)}            [Global fused]
     ↓
Flow Matching ──► pose ∈ ℝ^{B×28}                               [Grasp pose]
```

#### Caveat (semantics pooling)
When pooling the Qwen hidden states `H ∈ ℝ^{B×L_max×2048}` to obtain `s ∈ ℝ^{B×256}`, prefer attention‑mask aware pooling instead of a plain mean to avoid including padding tokens:

```python
# H: [B, L_max, 2048], attention_mask: [B, L_max]
mask = attention_mask.float().unsqueeze(-1)          # [B, L, 1]
pooled = (H * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
s = sem_proj(pooled)                                  # LayerNorm+Linear → [B, 256]
```
The current baseline uses a simple mean over tokens; switch to masked mean for strict correctness with variable sequence lengths.

### Key Dimensions
- B: Batch size (2-4 for CPU, 8-32 for GPU)
- N: Points per object (1024)
- Feature dimensions: CSEM=256, CGEO=256, CFUSE=CSEM+CGEO=512, CCOND=CFUSE
- Pose dimension: DPOSE=63 (21 joints × 3 coordinates = 63D flattened representation)

## B. Training Implementation (Core Focus)

### B.1 Main Training Loop

```python
# train.py - Core training logic
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from dataset.oakink_loader import create_oakink_loaders
from models.functional_grasp_model import FunctionalGraspModel

def train_functional_grasp(cfg):
    # Initialize model
    model = FunctionalGraspModel(CSEM=256, CGEO=256, DPOSE=63)
    
    # Data loaders with OakInk
    train_loader, val_loader = create_oakink_loaders(
        root_dir=cfg['data_path'],
        batch_size=cfg['batch_size'],  # 2-4 for CPU, 8-16 for GPU
        n_points=1024,
        contact_threshold=0.01  # 1cm for contact approximation
    )
    
    # Optimizer (all trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=0.05)
    
    for epoch in range(cfg['epochs']):
        for batch in train_loader:
            # Forward pass
            out = model.forward_train(
                images_list=batch['images_list'],
                texts_list=batch['texts_list'],  # From semantic attributes
                pts=batch['points']
            )
            
            # Loss computation (7-way classification)
            B, N = out['logits_c'].shape[:2]
            loss_contact = F.cross_entropy(
                out['logits_c'].view(B * N, 7), 
                batch['contact_labels'].view(B * N)
            )
            
            loss_flow = compute_flow_matching_loss(
                model, out['cond'], batch['pose']
            )
            
            loss = cfg['lambda_contact'] * loss_contact + cfg['lambda_flow'] * loss_flow
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
```

### B.2 Flow Matching Loss

```python
def compute_flow_matching_loss(model, conditioning, target_pose):
    """Rectified flow matching for pose generation"""
    B = target_pose.size(0)
    
    # Sample noise and time
    x0 = torch.randn_like(target_pose)  # Start: Gaussian noise
    t = torch.rand(B, device=target_pose.device)  # Time: U[0,1]
    
    # Interpolate between noise and target
    x_t = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * target_pose
    
    # Predict velocity field
    v_pred = model.flow_step(x_t, t, conditioning)
    
    # Target velocity (constant for rectified flow)
    v_target = target_pose - x0
    
    return F.mse_loss(v_pred, v_target)
```

### B.3 CPU Development Strategy

```python
# Configuration for CPU development
CPU_CONFIG = {
    'batch_size': 2,  # Small batch for CPU
    'num_workers': 0,  # Avoid multiprocessing issues
    'n_points': 512,   # Reduce points if needed
    'use_fp16': False, # Stay in FP32 for CPU
    'gradient_accumulation': 4,  # Simulate larger batches
    'checkpoint_freq': 100  # Frequent saves
}

# Memory-efficient training step
def train_step_cpu(model, batch, optimizer, grad_accum_steps=4):
    # Accumulate gradients over mini-batches
    for i in range(grad_accum_steps):
        mini_batch = get_mini_batch(batch, i, grad_accum_steps)
        loss = compute_loss(model, mini_batch)
        loss = loss / grad_accum_steps
        loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
```

## C. Environment Setup

### Already Installed Dependencies
```bash
# Core (in conda env 'grasp')
- torch, torchvision (CPU or CUDA)
- transformers>=4.51 (Qwen2.5-VL support)
- qwen-vl-utils==0.0.8
- trimesh (for mesh loading)

# Geometry backbone (required)
pip install torch-geometric  # Install per your Torch/CUDA version
# Guide: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# Additional requirements
pip install wandb  # For experiment tracking
pip install einops # For tensor operations
```

### Verify Installation
```python
# test.py - Already confirmed working
from transformers import Qwen2_5_VLForConditionalGeneration
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct"
)
print(model)  # Should show full architecture
```

## D. Project Structure

```
FuncGrasp/
├── models/
│   ├── semantics_qwen.py          # Qwen2.5-VL wrapper
│   ├── pointnet2_encoder.py       # Point cloud encoder
│   ├── fusion_transformer.py      # Multi-modal fusion (Transformer across points)
│   ├── contact_head.py            # Contact prediction
│   ├── flow_matching.py           # Pose generation
│   └── functional_grasp_model.py  # Full model
├── dataset/
│   ├── oakink_loader.py           # OakInk dataset loader
│   └── prepare_data.py            # Object rendering utilities
├── docs/
│   ├── pipeline.md
│   └── dataset.md
├── config.py                       # Configuration
├── train.py                        # Training script
```

## E. Model Architecture Components

### E.1 Qwen2.5-VL Semantics Encoder

```python
# models/semantics_qwen.py
import torch, torch.nn as nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

class QwenSemanticsEncoder(nn.Module):
    """
    Wraps Qwen2.5-VL and returns token hidden states H ∈ ℝ^{B×L_max×2048} and attention_mask.
    Masked pooling and projection to s ∈ ℝ^{B×CSEM} are performed inside FunctionalGraspModel.
    """
    # See models/semantics_qwen.py in the repo for the exact implementation
```

### E.2 PointNet++ Encoder

```python
# models/pointnet2_encoder.py
import torch, torch.nn as nn

def fps(x, M):
    # Farthest point sampling: x[B,N,3] → indices[B,M]
    B, N, _ = x.shape
    centroids = torch.zeros(B, M, dtype=torch.long, device=x.device)
    dist = torch.full((B, N), 1e10, device=x.device)
    farthest = torch.randint(0, N, (B,), device=x.device)
    batch = torch.arange(B, device=x.device)
    for i in range(M):
        centroids[:, i] = farthest
        centroid = x[batch, farthest].unsqueeze(1)
        d = ((x - centroid)**2).sum(-1)
        dist = torch.minimum(dist, d)
        farthest = torch.max(dist, dim=1).indices
    return centroids

class PN2GeometryEncoder(nn.Module):
    """PointNet++ encoder using torch_geometric.nn.models.PointNet2 (required)."""
    def __init__(self, in_c=3, cgeo=256):
        super().__init__()
        from models.pointnet2_encoder import PN2GeometryEncoder as Impl
        self.impl = Impl(in_c=in_c, cgeo=cgeo)
    def forward(self, pts):
        return self.impl(pts)
```

### E.3 Fusion Transformer & Contact Head

```python
# models/fusion_transformer.py
import torch, torch.nn as nn

class FusionTransformer1D(nn.Module):
    """Concatenate per-point geometry with broadcasted semantics and
    apply a Transformer encoder horizontally (across points).
    """
    def __init__(self, c_geo, c_sem, depth=4, heads=8):
        super().__init__()
        c_fuse = c_geo + c_sem
        enc_layer = nn.TransformerEncoderLayer(
            d_model=c_fuse, nhead=heads, dim_feedforward=4*c_fuse,
            batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        
    def forward(self, f_geo, s):
        # f_geo: [B,N,Cgeo], s: [B,Csem]
        B, N, _ = f_geo.shape
        s_tile = s.unsqueeze(1).expand(B, N, -1)
        x = torch.cat([f_geo, s_tile], -1)  # [B,N,Cgeo+Csem]
        return self.enc(x)

# models/contact_head.py
class ContactHead(nn.Module):
    def __init__(self, c_fuse, k=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_fuse, c_fuse//2), nn.ReLU(),
            nn.Linear(c_fuse//2, k)
        )
    def forward(self, f_fuse):
        return self.net(f_fuse)  # [B, N, 7]
```

### E.4 Conditional Flow Matching

```python
# models/flow_matching.py
import torch, torch.nn as nn, math

class PositionalEncoding1D(nn.Module):
    def __init__(self, d=64, maxw=10000.0):
        super().__init__()
        self.d = d
        self.maxw = maxw
        
    def forward(self, t):
        # t: [B,1] in [0,1]
        device = t.device
        d = self.d // 2
        freqs = torch.exp(torch.arange(d, device=device) * 
                         (-math.log(self.maxw)/max(d-1,1)))
        ang = 2*math.pi * t * freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

class PoseFlow(nn.Module):
    def __init__(self, d_pose, c_cond, hidden=1024):
        super().__init__()
        self.t_embed = nn.Sequential(
            PositionalEncoding1D(64), 
            nn.Linear(64, 128), 
            nn.SiLU()
        )
        self.net = nn.Sequential(
            nn.Linear(d_pose + 128 + c_cond, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d_pose)
        )
        
    def forward(self, x_t, t, c):
        te = self.t_embed(t.unsqueeze(-1))
        return self.net(torch.cat([x_t, te, c], -1))
```

### E.5 Full Model Integration

```python
# models/functional_grasp_model.py
import torch, torch.nn as nn
from .semantics_qwen import QwenSemanticsEncoder
from .pointnet2_encoder import PN2GeometryEncoder
from .fusion_transformer import FusionTransformer1D
from .contact_head import ContactHead
from .flow_matching import PoseFlow

class FunctionalGraspModel(nn.Module):
    def __init__(self, qwen_name="Qwen/Qwen2.5-VL-3B-Instruct",
                 CSEM=256, CGEO=256, DPOSE=63, K_CONTACT=7):
        super().__init__()
        self.sem = QwenSemanticsEncoder(model_name=qwen_name, csem_proj=CSEM)
        # PointNet++ via PyTorch Geometric (required)
        self.pc = PN2GeometryEncoder(in_c=3, cgeo=CGEO)
        CFUSE = CSEM + CGEO
        # Fusion uses a Transformer across points (horizontal)
        self.fuse = FusionTransformer1D(c_geo=CGEO, c_sem=CSEM)
        self.cm = ContactHead(CFUSE, k=K_CONTACT)
        self.flow = PoseFlow(d_pose=DPOSE, c_cond=CFUSE)

    def forward_backbone(self, images, texts, pts):
        s = self.sem(images, texts)             # [B,CSEM]
        f_geo, _ = self.pc(pts)                 # [B,N,CGEO]
        f_fuse = self.fuse(f_geo, s)            # [B,N,CFUSE]
        logits_c = self.cm(f_fuse)              # [B,N,7]
        # Pooling: 1 - p(no_contact) using softmax over 7 classes
        probs = logits_c.softmax(dim=-1)        # [B,N,7]
        p_nc = probs[..., 6]                    # no_contact index
        w = (1.0 - p_nc)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-6)
        z = (w.unsqueeze(-1) * f_fuse).sum(dim=1)   # [B,CFUSE]
        c = z
        return f_fuse, logits_c, c

    def forward_train(self, images, texts, pts):
        f_fuse, logits_c, c = self.forward_backbone(images, texts, pts)
        return dict(f_fuse=f_fuse, logits_c=logits_c, cond=c)

    def flow_step(self, x_t, t, c):
        return self.flow(x_t, t, c)
```

## F. Training Configuration

### F.1 Hyperparameters

```python
# config.py
CONFIG = {
    # Model architecture
    'model': {
        'qwen_name': 'Qwen/Qwen2.5-VL-3B-Instruct',
        'freeze_qwen': True,  # Set False to fine-tune backbone
        'CSEM': 256,
        'CGEO': 256,
        'DPOSE': 63,
        'n_points': 1024
    },
    
    # Training
    'training': {
        'batch_size': 4 if torch.cuda.is_available() else 2,
        'learning_rate': 1e-4,  # For new layers
        'weight_decay': 0.05,
        'epochs': 100,
        'gradient_clip': 1.0,
        'lambda_contact': 1.0,
        'lambda_flow': 1.0
    },
    
    # Data
    'data': {
        'root_dir': '/path/to/OakInk',
        'split_mode': 'split0',  # Object-based split
        'contact_threshold': 0.01,  # 1cm
        'use_cache': True
    },
    
    # CPU/GPU settings
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4 if torch.cuda.is_available() else 0,
    'fp16': torch.cuda.is_available()  # Only on GPU
}
```

### F.2 Inference Pipeline

```python
def inference(model, images, text, point_cloud, device='cpu'):
    """Generate grasp pose from inputs"""
    model.eval()
    with torch.no_grad():
        # Get conditioning
        _, _, conditioning = model.forward_backbone(
            images=images,           # List[PIL.Image]
            texts=[text],
            pts=point_cloud.unsqueeze(0)
        )
        
        # Sample poses via flow matching
        x = torch.randn(1, 63, device=device)  # Start from noise
        
        # Integrate ODE
        steps = 20
        for k in range(steps):
            t = torch.tensor([(k + 0.5) / steps], device=device)
            v = model.flow_step(x, t, conditioning)
            x = x + (1.0 / steps) * v
        
        return x  # Final pose
```

## G. Data Interface

```python
# Each batch should provide:
batch = {
  "images_list": List[List[PIL.Image.Image]],  # B lists of images (rendered views)
  "texts_list":  List[str],                    # B prompts from semantic attributes
  "points": torch.Tensor[B,N,3],               # object point cloud
  "contact_labels": torch.LongTensor[B,N],     # 7-way per-point labels (0..6)
  "pose": torch.Tensor[B,63],                  # hand pose (21 joints × 3)
}
```

## H. Quick Start Guide

### H.1 Start Training

```bash
# 1. Activate environment
conda activate grasp

# 2. Render clean object views (sample or full set)
python dataset/prepare_data.py --oakink_root /path/to/OakInk --sample --num_samples 3
# or render all objects (may take time)
python dataset/prepare_data.py --oakink_root /path/to/OakInk --output_dir rendered_objects

# 3. Start training (CPU mode)
python train.py \
    --data_path /path/to/OakInk \
    --batch_size 2 \
    --device cpu

# 4. Monitor with tensorboard
tensorboard --logdir ./logs
```

### H.2 Development Tips

1. **CPU Development**:
   - Use batch_size=1-2 for testing
   - Test with subset of data first
   - Use gradient accumulation for effective larger batches

   2. **Debugging**:
   ```python
   # Add shape checks
   def debug_forward(model, batch):
       print(f"Images: {len(batch['images'])}")
       print(f"Points: {batch['points'].shape}")
       print(f"Contact labels: {batch['contact_labels'].shape}")
       print(f"Pose: {batch['pose'].shape}")
       
       out = model.forward_train(
           images=batch['images'],
           texts=batch['texts'],
           pts=batch['points']
       )
       for k, v in out.items():
           print(f"{k}: {v.shape if hasattr(v, 'shape') else type(v)}")
   ```

3. **Memory Management**:
   - Clear cache regularly: `torch.cuda.empty_cache()`
   - Use checkpoint gradients for large models
   - Consider model quantization for inference

## I. Evaluation & Metrics

### I.1 Evaluation Metrics

```python
def evaluate_grasp(model, images, text, pts, gt_pose, gt_contact_labels):
    """Compute evaluation metrics"""
    metrics = {}
    
    # Pose error (MSE)
    with torch.no_grad():
        _, _, cond = model.forward_backbone([images], [text], pts.unsqueeze(0))
        # Sample a pose via flow matching sampler (or do teacher-forced one-step for speed)
        pred_pose = model.sample([images], [text], pts.unsqueeze(0), num_steps=20, device=pts.device)
    metrics['pose_error'] = F.mse_loss(pred_pose, gt_pose.unsqueeze(0)).item()
    
    # Contact prediction IoU
    with torch.no_grad():
        _, logits_c, _ = model.forward_backbone([images], [text], pts.unsqueeze(0))
        pred_contacts = logits_c.argmax(-1).squeeze(0)  # [N]
    metrics['contact_iou'] = compute_iou(pred_contacts, gt_contact_labels)
    
    # Optional: functional success proxy
    # metrics['grasp_stability'] = check_force_closure(pred_pose, pts)
    
    return metrics
```

### I.2 Post-processing

- Generate M=4 poses per object
- Score by: contact coverage + penetration penalty + joint limits
- Select best pose for execution

## J. Ablation Studies

1. **Fusion Transformer**: Replace with MLP → test multi-modal fusion importance
2. **Contact weighting**: Use average pooling → test contact prediction value
3. **Visual input**: Text-only vs Image+Text → quantify visual contribution
4. **Flow steps**: Vary integration steps (10, 20, 50) → speed vs quality

## K. Implementation Checklist

- [ ] Coordinate frames consistent (world/camera/object)
- [ ] Contact probabilities normalized (softmax/sigmoid)
- [ ] Joint limits enforced after generation
- [ ] Time sampling t ~ U[0,1] for flow matching
- [ ] Gradient clipping enabled (prevent instability)
- [ ] Qwen backbone frozen (only projection trainable)
- [ ] Data augmentation applied consistently
- [ ] Checkpoint saving every N iterations

## L. Testing & Validation

```python
from models.functional_grasp_model import FunctionalGraspModel
from PIL import Image

# Quick smoke test
model = FunctionalGraspModel(DPOSE=63, K_CONTACT=7)
dummy_images = [[Image.open("example.jpg")]] * 2
dummy_texts = ["grasp the bottle to pour water"] * 2
dummy_pts = torch.randn(2, 1024, 3)

out = model.forward_backbone(dummy_images, dummy_texts, dummy_pts)
for k,v in zip(["F_fuse","logits_c","cond"], out):
    print(k, v.shape)
# Expected: [B,N,Cfuse], [B,N,7], [B,Ccond]
```

## L. Alignment with Original Architecture Sketches

The implementation follows the hand-drawn architecture sketches in `assets/`:

### Sketch-to-Code Mapping

| Sketch Notation | Implementation | Description |
|----------------|----------------|-------------|
| LLM (red path) | Qwen2.5-VL | Vision-language model for semantics |
| F_sem (1×C_L) | s ∈ ℝ^{B×256} | Global semantic features |
| F_geo (1024×C_i) | F_geo ∈ ℝ^{B×1024×256} | Per-point geometric features |
| pts | Point cloud input | 3D object points |
| Fusion | FusionTransformer1D (across points) | Concatenate semantic+geometry then transform |
| Contact map | ContactHead | 7-way finger/palm contact classification |
| Pooling | Contact‑weighted pooling | Aggregate to global features |
| FM (Flow Matching) | PoseFlow | Generate grasp poses |

### Design Decisions

1. **Baseline uses contact‑weighted pooling**: Pool after contact prediction to emphasize graspable regions (matches sketches).
2. **Semantic broadcast + transformer**: Tile semantic features to all points, concatenate with `F_geo`, then process with a Transformer across points (horizontal self‑attention).
3. **Dimension alignment**: C_i=256 (CGEO), C_L=256 (CSEM), fusion produces 512D features

## M. Optional Design Variants (Advanced)

> These are drop-in alternatives you can enable later without changing the baseline in this doc. They reflect designs we discussed as potentially better under some settings.

### M.0 Qwen2.5-VL Backbone Tuning Modes

The pipeline supports three tuning modes for the Qwen backbone:

#### 1. Frozen (Default)
- **What**: Freeze entire Qwen backbone, only train new layers.
- **Why**: Fast training, low memory, preserves pre-trained capabilities.
- **How**: Set `qwen_tuning: 'frozen'` in config or `--qwen_tuning frozen`.
- **Impact**: ~15.4M trainable params, works on consumer GPUs.

#### 2. Full Fine-tuning
- **What**: Allow gradients through entire Qwen backbone.
- **Why**: Adapt vision-language representations to grasp domain.
- **How**: Set `qwen_tuning: 'full'` in config or `--qwen_tuning full`.
- **Impact**:
  - Trainable params: 15.4M → 3.77B
  - Memory: ~0.2GB → ~42GB
  - Requires 24GB+ GPU (A100, A6000, etc.)
- **Tips**:
  - Use gradient checkpointing (automatic)
  - Mixed precision (fp16/bf16) essential
  - Monitor for catastrophic forgetting

#### 3. LoRA (Low-Rank Adaptation)
- **What**: Apply LoRA adapters to specific Qwen modules.
- **Why**: Efficient fine-tuning with <1% trainable params.
- **How**: 
```bash
python train.py --qwen_tuning lora --lora_r 16 --lora_alpha 32
```
- **Config**:
```python
MODEL = {
    'qwen_tuning': 'lora',
}
LORA = {
    'r': 16,                    # Rank
    'alpha': 32,                # Scaling
    'dropout': 0.05,
    'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                      'gate_proj', 'up_proj', 'down_proj'],
    'bias': 'none',
    'use_8bit': False,          # Set True to use 8-bit base (saves memory)
}
```
- **Impact**:
  - Trainable params: ~15.4M + LoRA adapters (~0.1-1% of base)
  - Memory: Similar to frozen + small overhead
  - Works on consumer GPUs (16GB+)
- **CLI Options**:
  - `--lora_r`: LoRA rank (default: 16)
  - `--lora_alpha`: LoRA alpha (default: 32)
  - `--lora_dropout`: Dropout (default: 0.05)
  - `--lora_targets`: Comma-separated target modules
  - `--lora_bias`: Bias mode (none/all/lora_only)
  - `--lora_use_8bit`: Use 8-bit base model
- **Checkpoint Handling**:
  - LoRA adapters saved separately in `lora_adapters_step_{N}/`
  - Easy to swap adapters or revert to base model

### M.1 Plain mean pooling (ablation)

- **What**: Replace contact‑weighted pooling with an unweighted mean over points.
- **Why**: Provides a simple ablation/baseline without using contact predictions.
- **How**:
```python
z = f_fuse.mean(dim=1)
c = z
```
- **Stability notes**: Expect slower convergence and weaker pose quality vs. weighted pooling.

### M.2 Global geometry skip path (coarse shape context)

- **What**: Feed a global summary of the geometry branch directly to the conditioner alongside the fused features; this “skips” the fusion block.
- **Why**: Preserves coarse object cues (size, symmetry) even if fusion or pooling under‑represents them; often stabilizes early training.
- **How**:
```python
# In FunctionalGraspModel.__init__
CFUSE = CSEM + CGEO
self.flow = PoseFlow(d_pose=DPOSE, c_cond=CFUSE + CSEM + CGEO)

# In forward_backbone
f_geo, g = self.pc(pts)                  # g: [B, CGEO]
f_fuse = self.fuse(f_geo, s)             # [B,N,CFUSE]
z = f_fuse.mean(dim=1)                   # or use baseline weighted pooling
c = torch.cat([z, s, g], dim=-1)         # [B, CFUSE + CSEM + CGEO]
```

### M.3 Input projection bottleneck for fusion (capacity/control)

- **What**: Project `concat([F_geo, s_tile])` down to a chosen `CFUSE` before the transformer.
- **Why**: Controls compute/memory, adds learnable bottleneck; useful if `CSEM+CGEO` is large or you want tighter capacity.
- **How**:
```python
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
        B, N, _ = f_geo.shape
        x = torch.cat([f_geo, s.unsqueeze(1).expand(B, N, -1)], -1)
        return self.enc(self.in_proj(x))
```

- If you enable this, set `CFUSE=c_fuse` and adjust conditioner/flow dims accordingly.
