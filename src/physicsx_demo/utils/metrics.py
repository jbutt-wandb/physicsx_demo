from __future__ import annotations

import math

import numpy as np
import torch

from physicsx_demo.utils.data import AirfoilCase


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean per-case relative L2 over all points and output channels."""
    numerator = (pred - target).flatten(1).norm(dim=1)
    denominator = target.flatten(1).norm(dim=1).clamp(min=eps)
    return (numerator / denominator).mean()


def slice_regularization(
    all_w: list[torch.Tensor],
    usage_weight: float = 1.0,
    sharpness_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Healthy slices are sharp per point but balanced across the batch."""
    weighted_terms = []
    diagnostics = []
    for w in all_w:
        m = w.shape[-1]
        max_entropy = math.log(m)
        w_clamp = w.clamp(min=1e-8)
        point_entropy = -(w_clamp * w_clamp.log()).sum(dim=-1).mean()
        sharpness_loss = point_entropy / max_entropy

        usage = w.mean(dim=(0, 1)).clamp(min=1e-8)
        usage_entropy = -(usage * usage.log()).sum()
        usage_kl_uniform = max_entropy - usage_entropy
        usage_loss = usage_kl_uniform / max_entropy

        weighted_terms.append(sharpness_weight * sharpness_loss + usage_weight * usage_loss)
        diagnostics.append((point_entropy, usage_entropy, torch.exp(usage_entropy), sharpness_loss, usage_loss))

    aux = torch.stack(weighted_terms).mean()
    point_entropy, usage_entropy, effective_slices, sharpness_loss, usage_loss = diagnostics[-1]
    return aux, {
        "slice/point_entropy": float(point_entropy.detach().cpu()),
        "slice/usage_entropy": float(usage_entropy.detach().cpu()),
        "slice/effective_slices": float(effective_slices.detach().cpu()),
        "slice/sharpness_loss": float(sharpness_loss.detach().cpu()),
        "slice/usage_loss": float(usage_loss.detach().cpu()),
    }


def selection_score(pressure_rmse: float, field_relative_l2: float, lift_mae: float, drag_mae: float) -> float:
    """Single workshop score for checkpoint selection."""
    return pressure_rmse + 0.35 * field_relative_l2 + 0.20 * lift_mae + 0.10 * drag_mae


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ra = rankdata(np.asarray(a, dtype=np.float64))
    rb = rankdata(np.asarray(b, dtype=np.float64))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt(np.sum(ra**2) * np.sum(rb**2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(ra * rb) / denom)


def pressure_force_proxy(case: AirfoilCase, field: np.ndarray) -> dict[str, float]:
    """Approximate pressure-force coefficients from near-surface points."""
    pressure = field[:, 0]
    near = case.sdf <= np.quantile(case.sdf, 0.25)
    pts = case.points[near]
    p = pressure[near]
    radial = pts - np.array([0.5, 0.0], dtype=np.float32)
    normals = radial / (np.linalg.norm(radial, axis=1, keepdims=True) + 1e-6)
    force = (-p[:, None] * normals).mean(axis=0)
    flow_dir = np.array([math.cos(case.aoa), math.sin(case.aoa)], dtype=np.float32)
    lift_dir = np.array([-flow_dir[1], flow_dir[0]], dtype=np.float32)
    return {
        "drag_proxy": float(np.dot(force, flow_dir)),
        "lift_proxy": float(np.dot(force, lift_dir)),
    }


def proxy_metrics(cases: list[AirfoilCase], pred_fields: np.ndarray) -> dict[str, float]:
    truth_lift, pred_lift, truth_drag, pred_drag = [], [], [], []
    for case, pred in zip(cases, pred_fields):
        truth = np.concatenate([case.pressure[:, None], case.velocity], axis=1)
        truth_proxy = pressure_force_proxy(case, truth)
        pred_proxy = pressure_force_proxy(case, pred)
        truth_lift.append(truth_proxy["lift_proxy"])
        pred_lift.append(pred_proxy["lift_proxy"])
        truth_drag.append(truth_proxy["drag_proxy"])
        pred_drag.append(pred_proxy["drag_proxy"])

    truth_lift = np.asarray(truth_lift)
    pred_lift = np.asarray(pred_lift)
    truth_drag = np.asarray(truth_drag)
    pred_drag = np.asarray(pred_drag)
    return {
        "val/lift_proxy_mae": float(np.mean(np.abs(pred_lift - truth_lift))),
        "val/drag_proxy_mae": float(np.mean(np.abs(pred_drag - truth_drag))),
        "val/lift_proxy_spearman": spearman_corr(truth_lift, pred_lift),
        "val/drag_proxy_spearman": spearman_corr(truth_drag, pred_drag),
    }

