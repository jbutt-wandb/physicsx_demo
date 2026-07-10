from __future__ import annotations

import math

import torch
import torch.nn as nn


class PhysicsAttention(nn.Module):
    """One Physics-Attention block: slice -> attend -> deslice, plus an FFN."""

    def __init__(self, hidden: int, n_slices: int, n_heads: int, ffn_expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        if hidden % n_heads != 0:
            raise ValueError("hidden must be divisible by n_heads")
        self.n_slices = n_slices
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads

        self.slice_proj = nn.Linear(hidden, n_slices)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.attn_out = nn.Linear(hidden, hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn_expansion * hidden),
            nn.GELU(),
            nn.Linear(ffn_expansion * hidden, hidden),
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def _slice_attend_deslice(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, c = x.shape
        m, h, d = self.n_slices, self.n_heads, self.head_dim

        w = torch.softmax(self.slice_proj(x), dim=-1)
        numer = torch.einsum("bnm,bnc->bmc", w, x)
        denom = w.sum(dim=1).unsqueeze(-1) + 1e-8
        z = numer / denom

        qkv = self.qkv(z).view(b, m, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        attn = torch.softmax(scores, dim=-1)
        z_out = (attn @ v).transpose(1, 2).contiguous().view(b, m, c)
        z_out = self.attn_out(z_out)

        return torch.einsum("bnm,bmc->bnc", w, z_out), w

    def forward(self, x: torch.Tensor, return_slice_weights: bool = False):
        attended, w = self._slice_attend_deslice(self.norm1(x))
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        if return_slice_weights:
            return x, w
        return x


class TinyTransolver(nn.Module):
    """Small Transolver: Physics-Attention blocks plus a global context path."""

    def __init__(
        self,
        in_dim: int = 5,
        out_dim: int = 3,
        hidden: int = 128,
        n_layers: int = 6,
        n_slices: int = 64,
        n_heads: int = 8,
        ffn_expansion: int = 2,
        dropout: float = 0.05,
        use_global_context: bool = True,
    ):
        super().__init__()
        self.use_global_context = use_global_context
        self.embed = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList(
            [PhysicsAttention(hidden, n_slices, n_heads, ffn_expansion, dropout=dropout) for _ in range(n_layers)]
        )
        if use_global_context:
            self.global_mlp = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
            self.global_gate = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.Sigmoid(),
            )
        self.out_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, features: torch.Tensor, return_slice_weights: bool = False):
        x = self.embed(features)
        all_w = []
        for block in self.blocks:
            if return_slice_weights:
                x, w = block(x, return_slice_weights=True)
                all_w.append(w)
            else:
                x = block(x)
        if self.use_global_context:
            context = self.global_mlp(x.mean(dim=1, keepdim=True))
            x = x + self.global_gate(x) * context
        out = self.head(self.out_norm(x))
        if return_slice_weights:
            return out, all_w
        return out


class PointwiseMLP(nn.Module):
    """Per-point baseline with no communication between mesh points."""

    def __init__(self, in_dim: int = 5, out_dim: int = 3, hidden: int = 128, depth: int = 4, dropout: float = 0.05):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout)])
            dim = hidden
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class TinyTransolverKpi(nn.Module):
    """TinyTransolver wrapper for one-vector CL/CD/CM prediction from design tokens."""

    def __init__(
        self,
        in_dim: int = 7,
        out_dim: int = 3,
        hidden: int = 64,
        n_layers: int = 3,
        n_slices: int = 8,
        n_heads: int = 4,
        ffn_expansion: int = 2,
        dropout: float = 0.05,
        use_global_context: bool = True,
    ):
        super().__init__()
        self.backbone = TinyTransolver(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=hidden,
            n_layers=n_layers,
            n_slices=n_slices,
            n_heads=n_heads,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
            use_global_context=use_global_context,
        )

    def forward(self, features: torch.Tensor, return_slice_weights: bool = False):
        if return_slice_weights:
            token_pred, weights = self.backbone(features, return_slice_weights=True)
            return token_pred.mean(dim=1), weights
        return self.backbone(features).mean(dim=1)


class KpiTokenMLP(nn.Module):
    """Pointwise-token baseline for KPI prediction with the same mean pooling."""

    def __init__(self, in_dim: int = 7, out_dim: int = 3, hidden: int = 64, depth: int = 3, dropout: float = 0.05):
        super().__init__()
        self.token_mlp = PointwiseMLP(in_dim=in_dim, out_dim=out_dim, hidden=hidden, depth=depth, dropout=dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.token_mlp(features).mean(dim=1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
