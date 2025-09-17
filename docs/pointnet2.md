# PointNet++ Geometry Encoder

## Overview
The PointNet++ component encodes object point clouds into per-point and global geometry features for fusion with semantics. We implement the Single-Scale Grouping (SSG) variant using PyTorch Geometric building blocks: farthest point sampling (FPS), radius-based grouping, PointNetConv for local aggregation, and KNN interpolation for feature propagation.

- Input: points `pts ∈ ℝ^{B×N×3}` (XYZ), optional extras `x_extra ∈ ℝ^{B×N×Fin}`
- Output: per-point features `F_geo ∈ ℝ^{B×N×Cgeo}`, global descriptor `g ∈ ℝ^{B×Cgeo}`
- Class: `PN2GeometryEncoder` in `models/pointnet2_encoder.py`

## Architecture (SSG)
1) Set Abstraction 1 (SA-1)
- Sampling: FPS to ~`n1` points per batch
- Grouping: radius ball query with `r1`, capped to `max_n1` neighbors
- Local aggregation: `PointNetConv(local_nn=[in_c+3→64→64→128], global_nn=[128→256])`
- Output: `x1 @ N1_total × 256`

2) Set Abstraction 2 (SA-2)
- Sampling: FPS on SA-1 set to ~`n2` points per batch
- Grouping: radius ball query with `r2`, capped to `max_n2`
- Local aggregation: `PointNetConv(local_nn=[256+3→128→128→256], global_nn=[256→256])`
- Output: `x2 @ N2_total × 256`

3) Global Descriptor
- `g = MLP([256→512→Cgeo])(global_max_pool(x2, batch2))`
- Shape: `g ∈ ℝ^{B×Cgeo}`

4) Feature Propagation (FP)
- SA-2 → SA-1: `x1_up = knn_interpolate(x2, pos2, pos1, k=k_fp)` → concat skip `x1` → `fp1([256+256→256→256])`
- SA-1 → Input: `x0_up = knn_interpolate(x1_fp, pos1, pos, k=k_fp)` → concat raw `x0` → `fp0([256+in_c→256→Cgeo])`
- Final per-point features: `F_geo = reshape([B×N, Cgeo]) → ℝ^{B×N×Cgeo}`

## Parameters and Effects
Constructor: `PN2GeometryEncoder(in_c=3, cgeo=256, n1=512, n2=128, r1=0.2, r2=0.4, k_fp=3, max_n1=32, max_n2=64)`

- in_c (int): Input feature channels per point. Use 3 for XYZ-only, or 3+Fin if passing x_extra.
- cgeo (int): Output feature dimension for both F_geo and g. Higher increases capacity and cost.
- n1, n2 (ints): Target sampled points for SA-1/SA-2 via FPS ratios. Higher captures finer structure; increases compute.
- r1, r2 (floats): Ball query radii at SA-1/SA-2. Smaller focuses on fine details; larger captures broader context.
- k_fp (int): K for knn_interpolate during FP. Larger → smoother interpolation, higher cost.
- max_n1, max_n2 (ints): Max neighbors per center in radius grouping; bounds memory/compute.

Forward args:
- pts: ℝ^{B×N×3} (required)
- x_extra: ℝ^{B×N×Fin} (optional). If provided, set in_c = 3 + Fin.

## I/O Shapes and Dimension Flow
- Input: pts [B,N,3] (+ optional x_extra [B,N,Fin])
- SA-1: x1 [N1_total,256]
- SA-2: x2 [N2_total,256]
- Global: g [B,Cgeo]
- FP (N2→N1→N): F [B×N,Cgeo] → F_geo [B,N,Cgeo]

Return:
- F_geo: [B,N,Cgeo]
- g: [B,Cgeo]

## Integration in Pipeline
Used by FunctionalGraspModel:
- F_geo is concatenated with broadcasted semantics and processed by a Transformer across points
- Contact head predicts per-point contact; pooled fused features condition the flow model

Minimal usage:
```python
from models.pointnet2_encoder import PN2GeometryEncoder
encoder = PN2GeometryEncoder(in_c=3, cgeo=256)
pts = torch.randn(2, 1024, 3)
F_geo, g = encoder(pts)
# F_geo: [2,1024,256], g: [2,256]
```

## Tuning Guide
- More detail: ↑n1 (e.g., 768), ↓r1 if neighborhoods too large, ↑max_n1 if many neighbors clipped
- More context: ↑r2 and/or ↑n2
- Smoother FP: ↑k_fp (3→6 or 8)
- Capacity: ↑cgeo (256→512) if underfitting; monitor memory
- Scale: Adapt r1/r2 to the metric scale of your point clouds

## Best Practices
- Permutation invariance via symmetric pooling (global_max_pool)
- Robust sampling (FPS) and radius grouping with neighbor caps
- Skip connections in FP to preserve fine details
- Add training-time augmentations for rotation robustness (SO(3) rotations, jitter, scale)

## Notes vs Built-in Models
We intentionally do not use torch_geometric.nn.models.PointNet2 (classification-oriented). Our encoder uses PyG primitives (fps, radius, PointNetConv, knn_interpolate) to produce both per-point and global features with the clean interface required by the fusion stack.

## References
- PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space (Qi et al.)
- PyTorch Geometric docs: PointNetConv, fps, radius, knn_interpolate

## Changelog
- 2025-09-17: Initial documentation added for PN2GeometryEncoder (SSG)
