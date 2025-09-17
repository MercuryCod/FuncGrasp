import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv, fps, radius, knn_interpolate, global_max_pool


def mlp(ch):
    layers = []
    for i in range(len(ch) - 1):
        layers += [nn.Linear(ch[i], ch[i+1]), nn.BatchNorm1d(ch[i+1]), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

class PN2GeometryEncoder(nn.Module):
    """
    PointNet++ (SSG) geometry encoder:
      - SA1: sample 512, r=0.2, nsample<=32
      - SA2: sample 128, r=0.4, nsample<=64
      - Global SA: pool to 1, then MLP to cgeo
      - FP: N2->N1, N1->N with skip connections
    Inputs:
      pts: [B, N, 3]  (xyz; if you have extra point feats X, concat to xyz and set in_c=3+X)
    Outputs:
      F_geo: [B, N, cgeo]  (per-point geometry features)
      g:     [B, cgeo]     (global geometry descriptor)
    """
    def __init__(self, in_c=3, cgeo=256,
                 n1=512, n2=128, r1=0.2, r2=0.4, k_fp=3,
                 max_n1=32, max_n2=64):
        super().__init__()
        self.in_c = in_c
        self.cgeo = cgeo
        self.n1, self.n2 = n1, n2
        self.r1, self.r2 = r1, r2
        self.k_fp = k_fp
        self.max_n1, self.max_n2 = max_n1, max_n2

        # SA-1: outputs 256-d features at N1 points (to match common PointNet++ dims)
        self.sa1 = PointNetConv(
            local_nn=mlp([in_c + 3, 64, 64, 128]),
            global_nn=mlp([128, 256])
        )
        # SA-2: outputs 256-d features at N2 points
        self.sa2 = PointNetConv(
            local_nn=mlp([256 + 3, 128, 128, 256]),
            global_nn=mlp([256, 256])
        )
        # “Global SA”: take pooled 256 -> cgeo
        self.glob = mlp([256, 512, cgeo])

        # Feature Propagation heads (with skip connections)
        # N2 -> N1: concat interpolated top-level features (256) with SA-1 features (256) → 512
        self.fp1 = mlp([256 + 256, 256, 256])
        self.fp0 = mlp([256 + in_c, 256, cgeo])  # N1 -> N,  concat raw input feats

    def forward(self, pts, x_extra=None):
        """
        pts: [B, N, 3]
        x_extra: optional per-point features [B, N, Fin]; if provided, set in_c = 3 + Fin.
        """
        B, N, _ = pts.shape
        
        # Input validation
        if N < self.n1:
            raise ValueError(f"Input has {N} points, but n1={self.n1}. Need at least n1 points.")
        if pts.shape[2] != 3:
            raise ValueError(f"Expected pts shape [B, N, 3], got {pts.shape}")
        
        pos = pts.reshape(B * N, 3)  # [B*N, 3]
        batch = torch.arange(B, device=pts.device).repeat_interleave(N)  # [B*N]

        if x_extra is None:
            x0 = pos if self.in_c == 3 else None
            if x0 is None:
                raise ValueError("in_c != 3 but x_extra not provided")
        else:
            x0 = torch.cat([pos, x_extra.reshape(B * N, -1)], dim=1)
            assert x0.size(1) == self.in_c, "in_c must equal 3 + x_extra channels"

        # ---- SA-1: sample exactly n1 points per batch via ratio ----
        ratio1 = float(self.n1) / float(N)
        idx1 = fps(pos, batch, ratio=ratio1)                 # [N1_total across batch]
        row, col = radius(pos, pos[idx1], self.r1,
                          batch, batch[idx1], max_num_neighbors=self.max_n1)
        edge1 = torch.stack([col, row], dim=0)
        x1 = self.sa1(x0, (pos, pos[idx1]), edge1)           # [N1_total, 256]
        pos1, batch1 = pos[idx1], batch[idx1]                # Sampled positions and batch indices

        # ---- SA-2: sample n2 from the N1 set ----
        # estimate per-batch N1 to compute ratio robustly
        n1_per_batch = idx1.numel() // B
        ratio2 = float(self.n2) / float(max(n1_per_batch, 1))
        idx2 = fps(pos1, batch1, ratio=ratio2)               # [N2_total]
        row, col = radius(pos1, pos1[idx2], self.r2,
                          batch1, batch1[idx2], max_num_neighbors=self.max_n2)
        edge2 = torch.stack([col, row], dim=0)
        x2 = self.sa2(x1, (pos1, pos1[idx2]), edge2)         # [N2_total, 256]
        pos2, batch2 = pos1[idx2], batch1[idx2]

        # ---- Global descriptor from top SA ----
        g = global_max_pool(x2, batch2)                      # [B, 256]
        g = self.glob(g)                                     # [B, cgeo]

        # ---- Feature Propagation: N2 -> N1 ----
        x1_up = knn_interpolate(x2, pos2, pos1,
                                batch_x=batch2, batch_y=batch1, k=self.k_fp)   # [N1_total, 256]
        x1_fp = torch.cat([x1_up, x1], dim=1)                                    # [N1_total, 512]
        x1_fp = self.fp1(x1_fp)                                                  # [N1_total, 256]

        # ---- Feature Propagation: N1 -> N ----
        x0_up = knn_interpolate(x1_fp, pos1, pos,
                                batch_x=batch1, batch_y=batch, k=self.k_fp)     # [BN, 256]
        # concat original per-point input features (xyz or xyz+extras)
        x0_cat = torch.cat([x0_up, x0], dim=1)                                   # [BN, 256+in_c]
        F = self.fp0(x0_cat)                                                     # [BN, cgeo]

        F_geo = F.view(B, N, self.cgeo)                                          # [B, N, cgeo]
        return F_geo, g





class MultiScaleSA(nn.Module):
    """
    One PointNet++ MSG block:
      - Takes features on 'pos' (source/all points)
      - Samples 'pos[idx]' as centroids
      - For each scale: builds radius graph, applies PointNetConv with its own MLP
      - Concats per-scale outputs at the sampled points
    """
    def __init__(self, in_c, radii, nsamples, mlps):
        """
        in_c    : input feature dim at source points
        radii   : list[float], e.g. [0.1, 0.2, 0.4]
        nsamples: list[int],   e.g. [16, 32, 128]
        mlps    : list[list[int]], e.g. [[32,32,64],[64,64,128],[64,96,128]]
                  (we build local_nn as [in_c+3] + mlp_k for each scale)
        """
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.radii = radii
        self.nsamples = nsamples

        convs = []
        for m in mlps:
            convs.append(
                PointNetConv(
                    local_nn=mlp([in_c + 3] + m),   # +3 for relative xyz
                    global_nn=None                  # keep last dim = m[-1]
                )
            )
        self.convs = nn.ModuleList(convs)
        self.out_c = sum(m[-1] for m in mlps)

    def forward(self, x, pos, batch, idx):
        # x: [N, in_c]  (may be None; then we’ll use only relative xyz inside local_nn)
        # pos: [N, 3], batch: [N], idx: indices of sampled centroids (subset of pos)
        pos_s, batch_s = pos[idx], batch[idx]
        outs = []
        for r, k, conv in zip(self.radii, self.nsamples, self.convs):
            row, col = radius(pos, pos_s, r, batch, batch_s, max_num_neighbors=k)
            edge_index = torch.stack([col, row], dim=0)  # src=row (all), dst=col (sampled)
            outs.append(conv(x, (pos, pos_s), edge_index))  # [Ns, m_k[-1]]
        x_out = torch.cat(outs, dim=1)  # [Ns, sum_k m_k[-1]]
        return x_out, pos_s, batch_s

class PN2GeometryEncoderMSG(nn.Module):
    """
    PointNet++ MSG geometry encoder (2× MSG SA + global + FP).
    Returns:
      F_geo: [B, N, cgeo]  per-point geometry features
      g:     [B, cgeo]     global geometry descriptor
    """
    def __init__(
        self,
        in_c=3, cgeo=256,
        n1=512, n2=128,
        radii1=(0.1, 0.2, 0.4), ns1=(16, 32, 128), mlps1=((32,32,64),(64,64,128),(64,96,128)),
        radii2=(0.2, 0.4, 0.8), ns2=(32, 64, 128), mlps2=((64,64,128),(128,128,256),(128,128,256)),
        k_fp=3
    ):
        super().__init__()
        self.in_c = in_c
        self.cgeo = cgeo
        self.n1, self.n2 = n1, n2
        self.k_fp = k_fp

        # SA-1 (MSG): in_c -> 320 (=64+128+128)
        self.sa1 = MultiScaleSA(
            in_c=in_c, radii=list(radii1), nsamples=list(ns1), mlps=[list(m) for m in mlps1]
        )
        c1 = self.sa1.out_c  # 320

        # SA-2 (MSG): 320 -> 640 (=128+256+256)
        self.sa2 = MultiScaleSA(
            in_c=c1, radii=list(radii2), nsamples=list(ns2), mlps=[list(m) for m in mlps2]
        )
        c2 = self.sa2.out_c  # 640

        # "Global SA": pool top level to B×cgeo
        self.glob = mlp([c2, 512, cgeo])

        # Feature Propagation heads (N2->N1->N)
        self.fp1 = mlp([c2 + c1, 256, 256])       # concat (x2_up, x1)
        self.fp0 = mlp([256 + in_c, 256, cgeo])   # concat (x0_up, raw input feats)

    def forward(self, pts, x_extra=None):
        """
        pts: [B, N, 3]
        x_extra: optional per-point features [B, N, Fin]; if used, set in_c=3+Fin.
        """
        B, N, _ = pts.shape
        if N < self.n1:
            raise ValueError(f"Need at least n1={self.n1} points, got N={N}.")
        pos = pts.reshape(B * N, 3)
        batch = torch.arange(B, device=pts.device).repeat_interleave(N)

        if x_extra is None:
            x0 = pos if self.in_c == 3 else None
            if x0 is None:
                raise ValueError("in_c != 3 but x_extra not provided")
        else:
            x0 = torch.cat([pos, x_extra.reshape(B * N, -1)], dim=1)
            assert x0.size(1) == self.in_c, "in_c must equal 3 + x_extra channels"

        # -------- SA-1 (MSG) --------
        idx1 = fps(pos, batch, ratio=float(self.n1) / float(N))
        x1, pos1, batch1 = self.sa1(x0, pos, batch, idx1)   # [N1_tot, 320]

        # -------- SA-2 (MSG) --------
        n1_per_b = max(x1.size(0) // B, 1)
        idx2 = fps(pos1, batch1, ratio=float(self.n2) / float(n1_per_b))
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1, idx2) # [N2_tot, 640]

        # -------- Global descriptor --------
        g = self.glob(global_max_pool(x2, batch2))           # [B, cgeo]

        # -------- Feature Propagation --------
        # N2 -> N1
        x1_up = knn_interpolate(x2, pos2, pos1, batch_x=batch2, batch_y=batch1, k=self.k_fp)  # [N1_tot, 640]
        x1_fp = self.fp1(torch.cat([x1_up, x1], dim=1))                                       # [N1_tot, 256]
        # N1 -> N
        x0_up = knn_interpolate(x1_fp, pos1, pos, batch_x=batch1, batch_y=batch, k=self.k_fp) # [BN, 256]
        F = self.fp0(torch.cat([x0_up, x0], dim=1))                                           # [BN, cgeo]

        F_geo = F.view(B, N, self.cgeo)                                                       # [B, N, cgeo]
        return F_geo, g
