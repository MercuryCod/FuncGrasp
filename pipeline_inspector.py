#!/usr/bin/env python3
"""
Pipeline Shape Inspector
Visualizes data shapes at each stage of the FuncGrasp pipeline
"""

import torch
from PIL import Image
from models.functional_grasp_model import FunctionalGraspModel
from models.semantics_qwen import QwenSemanticsEncoder
from models.pointnet2_encoder import PN2GeometryEncoder
from models.fusion_transformer import FusionTransformer1D
from models.contact_head import ContactHead
from models.flow_matching import PoseFlow


def print_shape(name, tensor, indent=0):
    """Pretty print tensor shape"""
    prefix = "  " * indent + "→ " if indent > 0 else ""
    if isinstance(tensor, torch.Tensor):
        print(f"{prefix}{name}: {list(tensor.shape)} (dtype: {tensor.dtype})")
    elif isinstance(tensor, (list, tuple)) and len(tensor) > 0 and isinstance(tensor[0], torch.Tensor):
        print(f"{prefix}{name}: List[Tensor], length={len(tensor)}")
        if len(tensor) > 0:
            print(f"{prefix}  First element: {list(tensor[0].shape)}")
    else:
        print(f"{prefix}{name}: {type(tensor).__name__}")


def inspect_pipeline(B=2, N=1024, num_images_per_sample=[2, 3]):
    """
    Inspect shapes at each stage of the pipeline
    
    Args:
        B: Batch size
        N: Number of points per cloud (standard: 1024)
        num_images_per_sample: List of image counts per batch sample
    
    Note: Pipeline assumes 1024 points per cloud as standard input
    """
    print("="*80)
    print("FUNCGRASP PIPELINE SHAPE INSPECTION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Batch size (B): {B}")
    print(f"  Points per cloud (N): {N}")
    print(f"  Images per sample: {num_images_per_sample}")
    print(f"  Model dimensions: CSEM=256, CGEO=256, DPOSE=63, K_CONTACT=7")
    
    # Create model
    model = FunctionalGraspModel(CSEM=256, CGEO=256, DPOSE=63)
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    # ==================== INPUT DATA ====================
    print("\n" + "="*80)
    print("STAGE 1: INPUT DATA")
    print("="*80)
    
    # Create sample data
    images_list = []
    for i, num_imgs in enumerate(num_images_per_sample[:B]):
        imgs = [Image.new('RGB', (224, 224), color=(i*100, 0, 0)) for _ in range(num_imgs)]
        images_list.append(imgs)
    
    texts_list = [f"grasp object {i}" for i in range(B)]
    pts = torch.randn(B, N, 3).to(device)
    
    print("\nInputs:")
    print(f"  images_list: List[List[PIL.Image]], length={len(images_list)}")
    for i, imgs in enumerate(images_list):
        print(f"    Sample {i}: {len(imgs)} images")
    print(f"  texts_list: List[str], length={len(texts_list)}")
    print_shape("pts", pts, indent=1)
    
    # ==================== SEMANTIC ENCODING ====================
    print("\n" + "="*80)
    print("STAGE 2: SEMANTIC ENCODING (Qwen2.5-VL)")
    print("="*80)
    
    with torch.no_grad():
        # Get hidden states from semantics encoder
        H, attention_mask = model.sem(images_list, texts_list)
        
        print("\nQwen outputs:")
        print_shape("H (hidden states)", H, indent=1)
        print_shape("attention_mask", attention_mask, indent=1)
        
        # Show masked pooling
        if attention_mask is not None:
            mask = attention_mask.to(H.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1e-6)
            pooled = (H * mask).sum(dim=1) / denom
        else:
            pooled = H.mean(dim=1)
        
        print("\nMasked pooling:")
        print_shape("mask (expanded)", mask, indent=1)
        print_shape("pooled", pooled, indent=1)
        
        # Convert and project
        pooled = pooled.float()
        s = model.sem_proj(pooled)
        
        print("\nProjection:")
        print_shape("pooled (float32)", pooled, indent=1)
        print_shape("s (semantic features)", s, indent=1)
    
    # ==================== GEOMETRIC ENCODING ====================
    print("\n" + "="*80)
    print("STAGE 3: GEOMETRIC ENCODING (PointNet++)")
    print("="*80)
    
    with torch.no_grad():
        f_geo, g = model.pc(pts)
        
        print("\nPointNet++ outputs:")
        print_shape("f_geo (per-point features)", f_geo, indent=1)
        print_shape("g (global descriptor)", g, indent=1)
    
    # ==================== FUSION ====================
    print("\n" + "="*80)
    print("STAGE 4: FUSION TRANSFORMER")
    print("="*80)
    
    with torch.no_grad():
        # Show how semantic features are tiled
        s_tiled = s.unsqueeze(1).expand(-1, N, -1)
        print("\nSemantic tiling:")
        print_shape("s", s, indent=1)
        print_shape("s_tiled", s_tiled, indent=1)
        
        # Fusion
        f_fuse = model.fuse(f_geo, s)
        print("\nFusion output:")
        print_shape("f_fuse", f_fuse, indent=1)
    
    # ==================== CONTACT PREDICTION ====================
    print("\n" + "="*80)
    print("STAGE 5: CONTACT PREDICTION (7-way classification)")
    print("="*80)
    
    with torch.no_grad():
        logits_c = model.cm(f_fuse)
        probs = logits_c.softmax(dim=-1)
        
        print("\nContact head outputs:")
        print_shape("logits_c", logits_c, indent=1)
        print_shape("probs (softmax)", probs, indent=1)
        print("\nContact classes: thumb, index, middle, ring, little, palm, no_contact")
    
    # ==================== CONDITIONING ====================
    print("\n" + "="*80)
    print("STAGE 6: POOLING (1 - p(no_contact))")
    print("="*80)
    
    with torch.no_grad():
        # 1 - p(no_contact) pooling
        p_nc = probs[..., 6]  # no_contact is at index 6
        w = 1.0 - p_nc
        w = w / (w.sum(dim=1, keepdim=True) + 1e-6)
        c = (w.unsqueeze(-1) * f_fuse).sum(dim=1)
        
        print("\nWeighted pooling:")
        print_shape("p_nc (no_contact prob)", p_nc, indent=1)
        print_shape("w (1 - p_nc, normalized)", w, indent=1)
        print_shape("c (conditioning)", c, indent=1)
    
    # ==================== FLOW MATCHING ====================
    print("\n" + "="*80)
    print("STAGE 7: FLOW MATCHING")
    print("="*80)
    
    with torch.no_grad():
        # Show flow inputs/outputs
        x_t = torch.randn(B, model.DPOSE).to(device)
        t = torch.rand(B).to(device)
        
        print("\nFlow inputs:")
        print_shape("x_t (current pose)", x_t, indent=1)
        print_shape("t (time)", t, indent=1)
        print_shape("c (conditioning)", c, indent=1)
        
        # Velocity prediction
        v = model.flow_step(x_t, t, c)
        print("\nFlow output:")
        print_shape("v (velocity)", v, indent=1)
        
        # Show sampling
        print("\nSampling process:")
        poses = model.sample(images_list, texts_list, pts, num_steps=5, device=device)
        print_shape("final_poses", poses, indent=1)
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("SUMMARY OF SHAPES")
    print("="*80)
    print(f"""
Input:
  - Images: B={B} samples, variable images per sample
  - Texts: B={B} strings  
  - Points: [{B}, {N}, 3]

Semantic Path:
  - Qwen hidden: [{B}, L_max, 2048] (L_max varies by batch)
  - Masked pool: [{B}, 2048]
  - Project to: [{B}, 256]

Geometric Path:
  - PointNet++: [{B}, {N}, 3] → [{B}, {N}, 256]

Fusion:
  - Concat: [{B}, {N}, 256] + [{B}, {N}, 256] → [{B}, {N}, 512]

Contact:
  - Logits: [{B}, {N}, 7] (7-way classification)
  - Weighted pool: [{B}, 512] (using 1 - p(no_contact))

Flow:
  - Condition: [{B}, 512]
  - Output: [{B}, 63] (21 joints × 3 coordinates)
""")


def inspect_custom(images_list, texts_list, pts):
    """
    Inspect pipeline with custom inputs
    
    Args:
        images_list: List[List[PIL.Image]]
        texts_list: List[str]
        pts: torch.Tensor of shape [B, N, 3] where N should be 1024
    """
    B = len(images_list)
    N = pts.shape[1]
    
    print(f"\nCustom inspection: B={B}, N={N}")
    print(f"Images per sample: {[len(imgs) for imgs in images_list]}")
    
    # Create model and run inspection
    model = FunctionalGraspModel(CSEM=256, CGEO=256, DPOSE=63)
    device = pts.device
    model = model.to(device)
    
    with torch.no_grad():
        # Run through pipeline
        f_fuse, logits_c, c = model.forward_backbone(images_list, texts_list, pts)
        
        print("\nPipeline outputs:")
        print_shape("f_fuse", f_fuse)
        print_shape("logits_c", logits_c)
        print_shape("c (conditioning)", c)
        
        # Sample poses
        poses = model.sample(images_list, texts_list, pts, num_steps=10, device=device)
        print_shape("sampled_poses", poses)


if __name__ == "__main__":
    # Run default inspection with standard 1024 points
    inspect_pipeline(B=3, N=1024, num_images_per_sample=[1, 2, 3])
    
    # Example custom inspection
    print("\n" + "="*80)
    print("CUSTOM EXAMPLE")
    print("="*80)
    
    custom_images = [
        [Image.new('RGB', (224, 224))],  # 1 image
        [Image.new('RGB', (224, 224)), Image.new('RGB', (224, 224))]  # 2 images
    ]
    custom_texts = ["grasp the cup", "pick up the tool"]
    custom_pts = torch.randn(2, 1024, 3)  # Need at least 512 points for PointNet++
    
    inspect_custom(custom_images, custom_texts, custom_pts)
