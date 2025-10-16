# pointnet2_encoder.py
import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv, fps, radius, knn_interpolate, global_max_pool


def mlp(ch, last_relu=True):
    """
    Simple MLP builder with optional final ReLU removal.
    Expects inputs of shape [*, C], compatible with PointNetConv.
    """
    layers = []
    for i in range(len(ch) - 1):
        in_c, out_c = ch[i], ch[i + 1]
        is_last = (i == len(ch) - 2)
        use_bias = is_last and not last_relu
        layers.append(nn.Linear(in_c, out_c, bias=use_bias))
        if not (is_last and not last_relu):
            layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# --------------------------- SSG: Single-Scale Grouping ---------------------------

class PN2GeometryEncoder(nn.Module):
    """
    PointNet++ SSG-style encoder that returns per-point geometric features [B, N, Cgeo]
    and a global descriptor [B, Cgeo].
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

        # SA blocks (use (in + 3) for local_nn to include relative coordinates)
        self.sa1 = PointNetConv(
            local_nn=mlp([in_c + 3, 64, 64, 128]),
            global_nn=mlp([128, 256])
        )
        self.sa2 = PointNetConv(
            local_nn=mlp([256 + 3, 128, 128, 256]),
            global_nn=mlp([256, 256])
        )
        self.glob = mlp([256, 512, cgeo], last_relu=False)

        # Feature propagation heads
        self.fp1 = mlp([256 + 256, 256, 256])
        self.fp0 = mlp([256 + in_c, 256, cgeo], last_relu=False)

    def forward(self, pts, x_extra=None):
        """
        Args:
            pts: [B, N, 3]
            x_extra: optional extra per-point inputs to concatenate with xyz,
                     so that in_c == 3 + x_extra_dim.

        Returns:
            F_geo: [B, N, Cgeo] per-point geometric features
            g:     [B, Cgeo] global descriptor
        """
        B, N, C = pts.shape
        if C != 3:
            raise ValueError(f"Expected pts shape [B, N, 3], got {pts.shape}")
        if N < self.n1:
            raise ValueError(f"Input has {N} points, but n1={self.n1}. Need at least n1 points.")

        pos = pts.reshape(B * N, 3)
        batch = torch.arange(B, device=pts.device).repeat_interleave(N)

        # Build initial per-point features x0
        if x_extra is None:
            if self.in_c == 3:
                x0 = pos  # use xyz as features
            else:
                raise ValueError("in_c != 3 but x_extra not provided")
        else:
            x0 = torch.cat([pos, x_extra.reshape(B * N, -1)], dim=1)
            if x0.size(1) != self.in_c:
                raise AssertionError(
                    f"in_c must equal 3 + x_extra channels (got in_c={self.in_c}, built={x0.size(1)})"
                )

        # ---- SA-1 (downsample to n1) ----
        ratio1 = float(self.n1) / float(N)
        idx1 = fps(pos, batch, ratio=ratio1)  # indices into pos/x0
        # radius returns (row, col): row indexes y=pos[idx1] (dst), col indexes x=pos (src)
        row1, col1 = radius(pos, pos[idx1], self.r1, batch, batch[idx1], max_num_neighbors=self.max_n1)
        # edge_index must be [src, dst] = [col, row]
        edge1 = torch.stack([col1, row1], dim=0)
        x1 = self.sa1((x0, x0[idx1]), (pos, pos[idx1]), edge1)
        pos1, batch1 = pos[idx1], batch[idx1]

        # ---- SA-2 (downsample to n2) ----
        n1_counts = torch.bincount(batch1, minlength=B)
        avg_n1 = float(n1_counts.float().mean().clamp_min(1.0))
        ratio2 = float(self.n2) / avg_n1
        idx2 = fps(pos1, batch1, ratio=ratio2)
        row2, col2 = radius(pos1, pos1[idx2], self.r2, batch1, batch1[idx2], max_num_neighbors=self.max_n2)
        edge2 = torch.stack([col2, row2], dim=0)  # [src, dst]
        x2 = self.sa2((x1, x1[idx2]), (pos1, pos1[idx2]), edge2)
        pos2, batch2 = pos1[idx2], batch1[idx2]

        # ---- Global descriptor ----
        g = self.glob(global_max_pool(x2, batch2))

        # ---- Feature Propagation (FP) ----
        x1_up = knn_interpolate(x2, pos2, pos1, batch_x=batch2, batch_y=batch1, k=self.k_fp)
        x1_fp = self.fp1(torch.cat([x1_up, x1], dim=1))

        x0_up = knn_interpolate(x1_fp, pos1, pos, batch_x=batch1, batch_y=batch, k=self.k_fp)
        x0_cat = torch.cat([x0_up, x0], dim=1)
        F = self.fp0(x0_cat)  # [B*N, Cgeo]

        F_geo = F.reshape(B, N, self.cgeo)
        return F_geo, g


# --------------------------- MSG: Multi-Scale Grouping ---------------------------

class MultiScaleSA(nn.Module):
    """
    Multi-scale set abstraction: for each sampled center, aggregate neighbors
    at multiple radii and concatenate the outputs.
    """
    def __init__(self, in_c, radii, nsamples, mlps):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.radii = list(radii)
        self.nsamples = list(nsamples)

        convs = []
        for m in mlps:
            convs.append(
                PointNetConv(
                    local_nn=mlp([in_c + 3] + list(m)),
                    global_nn=None
                )
            )
        self.convs = nn.ModuleList(convs)
        self.out_c = sum(list(m)[-1] for m in mlps)

    def forward(self, x, pos, batch, idx):
        """
        Args:
            x:    [M, Cin] features at all points (src), or None
            pos:  [M, 3]  positions of all points (src)
            batch:[M]     batch ids
            idx:  indices (into pos/x) of sampled centers (dst)

        Returns:
            x_out:  [len(idx_total), Cout]
            pos_s:  sampled positions (dst)
            batch_s: batch ids for sampled positions
        """
        pos_s, batch_s = pos[idx], batch[idx]
        x_dst = None if x is None else x[idx]

        outs = []
        for r, k, conv in zip(self.radii, self.nsamples, self.convs):
            # radius returns (row, col): row indexes dst(=pos_s), col indexes src(=pos)
            row, col = radius(pos, pos_s, r, batch, batch_s, max_num_neighbors=k)
            edge_index = torch.stack([col, row], dim=0)  # [src, dst]
            outs.append(conv((x, x_dst), (pos, pos_s), edge_index))
        x_out = torch.cat(outs, dim=1)
        return x_out, pos_s, batch_s


class PN2GeometryEncoderMSG(nn.Module):
    """
    PointNet++ MSG-style encoder that returns per-point geometric features [B, N, Cgeo]
    and a global descriptor [B, Cgeo].
    """
    def __init__(
        self,
        in_c=3, cgeo=256,
        n1=512, n2=128,
        radii1=(0.1, 0.2, 0.4), ns1=(16, 32, 128), mlps1=((32, 32, 64), (64, 64, 128), (64, 96, 128)),
        radii2=(0.2, 0.4, 0.8), ns2=(32, 64, 128), mlps2=((64, 64, 128), (128, 128, 256), (128, 128, 256)),
        k_fp=3
    ):
        super().__init__()
        self.in_c = in_c
        self.cgeo = cgeo
        self.n1, self.n2 = n1, n2
        self.k_fp = k_fp

        self.sa1 = MultiScaleSA(
            in_c=in_c, radii=radii1, nsamples=ns1, mlps=[list(m) for m in mlps1]
        )
        c1 = self.sa1.out_c

        self.sa2 = MultiScaleSA(
            in_c=c1, radii=radii2, nsamples=ns2, mlps=[list(m) for m in mlps2]
        )
        c2 = self.sa2.out_c

        self.glob = mlp([c2, 512, cgeo], last_relu=False)
        self.fp1 = mlp([c2 + c1, 256, 256])
        self.fp0 = mlp([256 + in_c, 256, cgeo], last_relu=False)

    def forward(self, pts, x_extra=None):
        """
        Args:
            pts: [B, N, 3]
            x_extra: optional per-point extra features to concatenate with xyz
        Returns:
            F_geo: [B, N, Cgeo]
            g:     [B, Cgeo]
        """
        B, N, C = pts.shape
        if C != 3:
            raise ValueError(f"Expected pts shape [B, N, 3], got {pts.shape}")
        if N < self.n1:
            raise ValueError(f"Need at least n1={self.n1} points, got N={N}.")

        pos = pts.reshape(B * N, 3)
        batch = torch.arange(B, device=pts.device).repeat_interleave(N)

        # Build initial per-point features x0
        if x_extra is None:
            if self.in_c == 3:
                x0 = pos
            else:
                raise ValueError("in_c != 3 but x_extra not provided")
        else:
            x0 = torch.cat([pos, x_extra.reshape(B * N, -1)], dim=1)
            if x0.size(1) != self.in_c:
                raise AssertionError(
                    f"in_c must equal 3 + x_extra channels (got in_c={self.in_c}, built={x0.size(1)})"
                )

        # SA-1
        idx1 = fps(pos, batch, ratio=float(self.n1) / float(N))
        x1, pos1, batch1 = self.sa1(x0, pos, batch, idx1)

        # SA-2
        n1_counts = torch.bincount(batch1, minlength=B)
        avg_n1 = float(n1_counts.float().mean().clamp_min(1.0))
        idx2 = fps(pos1, batch1, ratio=float(self.n2) / avg_n1)
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1, idx2)

        # Global descriptor
        g = self.glob(global_max_pool(x2, batch2))

        # Feature propagation
        x1_up = knn_interpolate(x2, pos2, pos1, batch_x=batch2, batch_y=batch1, k=self.k_fp)
        x1_fp = self.fp1(torch.cat([x1_up, x1], dim=1))

        x0_up = knn_interpolate(x1_fp, pos1, pos, batch_x=batch1, batch_y=batch, k=self.k_fp)
        F = self.fp0(torch.cat([x0_up, x0], dim=1))

        F_geo = F.reshape(B, N, self.cgeo)
        return F_geo, g
