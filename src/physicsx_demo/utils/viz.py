from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from physicsx_demo.utils.data import AirfoilCase, HighLiftCase, high_lift_surfaces


def case_shape_params(case: HighLiftCase) -> tuple[float, float, float, float, float] | None:
    params = [
        getattr(case, key, None)
        for key in ("main_camber", "main_thickness", "slat_deflection", "flap_deflection", "flap_gap")
    ]
    if all(value is not None for value in params):
        return tuple(float(value) for value in params)
    return None


def fallback_high_lift_surface(case: HighLiftCase, n: int = 260) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Smooth a visual envelope from near-surface samples for legacy artifacts."""
    near_count = max(90, int(0.24 * case.points.shape[0]))
    near = case.points[np.argsort(case.sdf)[:near_count]]
    x_grid = np.linspace(float(np.min(near[:, 0])), float(np.max(near[:, 0])), n)
    upper = np.zeros_like(x_grid)
    lower = np.zeros_like(x_grid)
    for i, x0 in enumerate(x_grid):
        band = near[np.abs(near[:, 0] - x0) <= 0.075]
        if len(band) < 5:
            order = np.argsort(np.abs(near[:, 0] - x0))[: max(8, min(18, len(near)))]
            band = near[order]
        upper[i] = np.percentile(band[:, 1], 84)
        lower[i] = np.percentile(band[:, 1], 16)

    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64)
    kernel /= kernel.sum()
    upper = np.convolve(upper, kernel, mode="same")
    lower = np.convolve(lower, kernel, mode="same")
    camber = 0.5 * (upper + lower)
    half_thickness = np.maximum(0.5 * (upper - lower), 0.012)
    taper = np.sin(np.pi * x_grid) ** 0.35
    half_thickness = np.maximum(half_thickness * taper, 0.004 * taper)
    upper = camber + half_thickness
    lower = camber - half_thickness
    upper[[0, -1]] = camber[[0, -1]]
    lower[[0, -1]] = camber[[0, -1]]
    return {"reconstructed": (np.stack([x_grid, upper], axis=1), np.stack([x_grid, lower], axis=1))}


def fallback_airfoil_surface(case: AirfoilCase, n: int = 260) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves = fallback_high_lift_surface(case, n=n)
    upper, lower = curves["reconstructed"]
    return upper[:, 0], upper[:, 1], lower[:, 1]


def high_lift_surface(case: HighLiftCase, n: int = 260) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], bool]:
    params = case_shape_params(case)
    if params is None:
        return fallback_high_lift_surface(case, n=n), False
    main_camber, main_thickness, slat_deflection, flap_deflection, flap_gap = params
    return (
        high_lift_surfaces(
            main_camber=main_camber,
            main_thickness=main_thickness,
            slat_deflection=slat_deflection,
            flap_deflection=flap_deflection,
            flap_gap=flap_gap,
        ),
        True,
    )


def airfoil_surface(case: AirfoilCase, n: int = 260) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    curves, exact = high_lift_surface(case, n=n)
    upper = np.concatenate([pair[0] for pair in curves.values()], axis=0)
    lower = np.concatenate([pair[1] for pair in curves.values()], axis=0)
    order_u = np.argsort(upper[:, 0])
    order_l = np.argsort(lower[:, 0])
    x = np.linspace(float(min(upper[:, 0].min(), lower[:, 0].min())), float(max(upper[:, 0].max(), lower[:, 0].max())), n)
    upper_y = np.interp(x, upper[order_u, 0], upper[order_u, 1])
    lower_y = np.interp(x, lower[order_l, 0], lower[order_l, 1])
    return x, upper_y, lower_y, exact


def _style_high_lift_axis(ax, curves: dict[str, tuple[np.ndarray, np.ndarray]], pad_x: float = 0.04) -> None:
    all_pts = np.concatenate([np.concatenate([upper, lower], axis=0) for upper, lower in curves.values()], axis=0)
    x_min = float(np.min(all_pts[:, 0]))
    x_max = float(np.max(all_pts[:, 0]))
    y_min = float(np.min(all_pts[:, 1]))
    y_max = float(np.max(all_pts[:, 1]))
    span = max(y_max - y_min, 0.18)
    y_mid = 0.5 * (y_min + y_max)
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_mid - 0.62 * span, y_mid + 0.62 * span)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def _style_airfoil_axis(ax, x: np.ndarray, upper: np.ndarray, lower: np.ndarray, pad_x: float = 0.04) -> None:
    y_min = float(np.min(lower))
    y_max = float(np.max(upper))
    span = max(y_max - y_min, 0.18)
    y_mid = 0.5 * (y_min + y_max)
    ax.set_xlim(float(np.min(x)) - pad_x, float(np.max(x)) + pad_x)
    ax.set_ylim(y_mid - 0.62 * span, y_mid + 0.62 * span)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def draw_high_lift_body(ax, case: HighLiftCase, *, show_flow: bool = False, show_camber: bool = True):
    curves, exact = high_lift_surface(case)
    if show_flow:
        flow = case.points[case.sdf <= np.quantile(case.sdf, 0.55)]
        ax.scatter(flow[:, 0], flow[:, 1], s=4, c="#94a3b8", alpha=0.16, linewidths=0, zorder=1)
    colors = {"slat": "#0891b2", "main": "#0f766e", "flap": "#7c3aed", "reconstructed": "#0f766e"}
    for name, (upper, lower) in curves.items():
        polygon = np.concatenate([upper, lower[::-1]], axis=0)
        ax.fill(polygon[:, 0], polygon[:, 1], color=colors.get(name, "#0f766e"), alpha=0.92, linewidth=0, zorder=3)
        ax.plot(upper[:, 0], upper[:, 1], color="#082f49", lw=1.45, zorder=4)
        ax.plot(lower[:, 0], lower[:, 1], color="#082f49", lw=1.45, zorder=4)
        if show_camber and len(upper) == len(lower):
            camber = 0.5 * (upper + lower)
            ax.plot(camber[:, 0], camber[:, 1], color="#f59e0b", lw=0.9, alpha=0.9, zorder=5)
    _style_high_lift_axis(ax, curves)
    return curves, exact


def draw_airfoil_body(ax, case: AirfoilCase, *, show_flow: bool = False, show_camber: bool = True):
    curves, exact = draw_high_lift_body(ax, case, show_flow=show_flow, show_camber=show_camber)
    x, upper, lower, _ = airfoil_surface(case)
    return x, upper, lower, exact


def high_lift_profile_figure(case: HighLiftCase, title: str | None = None):
    """Render a clean high-lift section silhouette."""
    fig, ax = plt.subplots(figsize=(6.4, 2.7), dpi=190)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")
    _, exact = draw_high_lift_body(ax, case, show_flow=True, show_camber=True)
    label = "synthetic CRM-HL section" if exact else "reconstructed section"
    flap = getattr(case, "flap_deflection", None)
    flap_label = f" | flap {math.degrees(flap):.0f} deg" if flap is not None else ""
    fig.suptitle(title or "High-lift section", fontsize=13, fontweight="bold", color="#0f172a", y=0.98)
    fig.text(
        0.5,
        0.875,
        f"AoA {math.degrees(case.aoa):.1f} deg | Re {case.re:.2e}{flap_label} | {label}",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#334155",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.80], pad=0.45)
    return fig


def airfoil_profile_figure(case: AirfoilCase, title: str | None = None):
    return high_lift_profile_figure(case, title=title)


def high_lift_sections_figure(case: HighLiftCase, title: str | None = None):
    """Render a high-lift silhouette plus local section cuts."""
    fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=190)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")
    curves, _ = draw_high_lift_body(ax, case, show_flow=False, show_camber=True)
    upper = curves.get("main", next(iter(curves.values())))[0]
    lower = curves.get("main", next(iter(curves.values())))[1]
    order_u = np.argsort(upper[:, 0])
    order_l = np.argsort(lower[:, 0])
    x = np.linspace(float(upper[:, 0].min()), float(upper[:, 0].max()), 220)
    upper_y = np.interp(x, upper[order_u, 0], upper[order_u, 1])
    lower_y = np.interp(x, lower[order_l, 0], lower[order_l, 1])
    stations = np.array([0.16, 0.32, 0.50, 0.68, 0.82], dtype=np.float32)
    colors = ["#2563eb", "#7c3aed", "#db2777", "#ea580c", "#16a34a"]
    for x0, color in zip(stations, colors):
        y_hi = float(np.interp(x0, x, upper_y))
        y_lo = float(np.interp(x0, x, lower_y))
        thickness = y_hi - y_lo
        ax.plot([x0, x0], [y_lo, y_hi], color=color, lw=2.4, solid_capstyle="round", zorder=7)
        ax.text(
            x0,
            y_hi + 0.012,
            f"{thickness:.2f}c",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=color,
            fontweight="bold",
        )
    fig.suptitle(title or "High-lift section cuts", fontsize=13, fontweight="bold", color="#0f172a", y=0.98)
    fig.text(0.5, 0.875, "Main-element thickness sections plus slat/flap context", ha="center", va="center", fontsize=8.8, color="#334155")
    fig.tight_layout(rect=[0, 0, 1, 0.80], pad=0.45)
    return fig


def airfoil_sections_figure(case: AirfoilCase, title: str | None = None):
    return high_lift_sections_figure(case, title=title)


def high_lift_gallery_figure(cases: list[HighLiftCase], title: str, max_cases: int = 12):
    """Compact gallery of high-lift sections for workshop browsing."""
    shown = list(cases[:max_cases])
    cols = 4
    rows = int(math.ceil(len(shown) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 1.55), dpi=180)
    fig.patch.set_facecolor("white")
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.set_facecolor("#f8fafc")
        ax.axis("off")
    for i, (ax, case) in enumerate(zip(axes.ravel(), shown)):
        draw_high_lift_body(ax, case, show_flow=False, show_camber=False)
        ax.set_title(
            f"case {i} | AoA {math.degrees(case.aoa):.1f} deg",
            fontsize=8.2,
            fontweight="bold",
            color="#0f172a",
            pad=2,
        )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995, color="#0f172a")
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=0.75, w_pad=0.4)
    return fig


def airfoil_gallery_figure(cases: list[AirfoilCase], title: str, max_cases: int = 12):
    return high_lift_gallery_figure(cases, title=title, max_cases=max_cases)


def pressure_figure(case: AirfoilCase, prediction: np.ndarray, title: str | None = None):
    """Side-by-side scatter of truth vs predicted pressure for one AirfoilCase."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), dpi=120)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("#f7f9fc")
        ax.set_aspect("equal")
        ax.axis("off")

    truth = case.pressure
    pred = prediction[:, 0]
    vmin = min(np.min(truth), np.min(pred))
    vmax = max(np.max(truth), np.max(pred))

    for ax, values, label in zip(axes, [truth, pred], ["ground truth", "prediction"]):
        sc = ax.scatter(
            case.points[:, 0],
            case.points[:, 1],
            c=values,
            cmap="RdBu_r",
            s=11,
            linewidths=0,
            vmin=vmin,
            vmax=vmax,
            alpha=0.95,
        )
        ax.set_title(label, fontsize=11, fontweight="bold")

    fig.suptitle(title or "Pressure field comparison", fontsize=13, fontweight="bold", y=0.98)
    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.03)
    cbar.ax.set_title("Cp", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def slice_figure(case: AirfoilCase, slice_argmax: np.ndarray, epoch: int):
    fig, ax = plt.subplots(figsize=(3.8, 3.2), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f9fc")
    scatter = ax.scatter(
        case.points[:, 0],
        case.points[:, 1],
        c=slice_argmax,
        cmap="tab20",
        s=14,
        alpha=0.9,
        edgecolor="none",
    )
    ax.set_title(f"Physics slices - epoch {epoch}", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_ticks(np.arange(int(np.max(slice_argmax)) + 1))
    cbar.ax.set_title("slice", fontsize=8)
    fig.tight_layout()
    return fig


def slice_video_frame(case: AirfoilCase, slice_argmax: np.ndarray, epoch: int, n_slices: int) -> np.ndarray:
    """Render one RGB frame for the slice-evolution video."""
    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f9fc")
    scatter = ax.scatter(
        case.points[:, 0],
        case.points[:, 1],
        c=slice_argmax,
        cmap="turbo",
        s=22,
        alpha=0.92,
        edgecolor="none",
        vmin=0,
        vmax=max(n_slices - 1, 1),
    )
    ax.set_title(f"Physics-Attention slice evolution - epoch {epoch}", fontsize=12, fontweight="bold")
    ax.text(
        0.02,
        0.04,
        f"AoA {math.degrees(case.aoa):.1f} deg - Re {case.re:.2e}",
        transform=ax.transAxes,
        fontsize=9,
        color="#334155",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.25", "alpha": 0.88},
    )
    ax.set_aspect("equal")
    ax.axis("off")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.035)
    cbar.ax.set_title("slice", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def write_slice_evolution_gif(frames: list[np.ndarray], output_path: Path, fps: int = 2) -> Path:
    """Write frames as a GIF for wandb.Video."""
    if not frames:
        raise ValueError("slice evolution video needs at least one frame")
    duration_ms = int(1000 / max(fps, 1))
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:], duration=duration_ms, loop=0)
    return output_path


def figure_to_rgb_frame(fig) -> np.ndarray:
    """Convert a Matplotlib figure to an RGB frame and close it."""
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def slice_usage_frame(slice_weights: torch.Tensor, epoch: int) -> np.ndarray:
    """Render the slice-usage bar chart as an RGB video frame."""
    return figure_to_rgb_frame(slice_usage_figure(slice_weights, epoch))


def save_figure_png(fig, output_dir: Path, name: str) -> Path:
    """Persist a Matplotlib figure before wrapping it as wandb.Image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    return path


def save_frame_png(frame: np.ndarray, output_dir: Path, name: str) -> Path:
    """Persist an RGB frame before wrapping it as wandb.Image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    Image.fromarray(frame).save(path)
    return path


def comparison_figure(case: AirfoilCase, model_prediction: np.ndarray, baseline_prediction: np.ndarray, epoch: int, case_id: int):
    """Truth vs TinyTransolver vs baseline pressure, with one shared colorbar."""
    fields = [case.pressure, model_prediction[:, 0], baseline_prediction[:, 0]]
    labels = ["ground truth", "TinyTransolver", "pointwise baseline"]
    vmin = min(float(np.min(f)) for f in fields)
    vmax = max(float(np.max(f)) for f in fields)

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7), dpi=140)
    fig.patch.set_facecolor("white")
    for ax, values, label in zip(axes, fields, labels):
        ax.set_facecolor("#f7f9fc")
        sc = ax.scatter(
            case.points[:, 0],
            case.points[:, 1],
            c=values,
            cmap="RdBu_r",
            s=11,
            linewidths=0,
            vmin=vmin,
            vmax=vmax,
            alpha=0.95,
        )
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(
        f"Pressure field comparison - epoch {epoch} - case {case_id} - AoA {math.degrees(case.aoa):.1f} deg",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    cbar = fig.colorbar(sc, ax=axes, fraction=0.022, pad=0.025)
    cbar.ax.set_title("Cp", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def error_figure(case: AirfoilCase, model_prediction: np.ndarray, baseline_prediction: np.ndarray, epoch: int, case_id: int):
    """Absolute pressure error comparison for TinyTransolver and baseline."""
    model_err = np.abs(model_prediction[:, 0] - case.pressure)
    base_err = np.abs(baseline_prediction[:, 0] - case.pressure)
    vmax = max(float(model_err.max()), float(base_err.max()), 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), dpi=140)
    fig.patch.set_facecolor("white")
    for ax, values, label in zip(axes, [model_err, base_err], ["TinyTransolver |Cp error|", "baseline |Cp error|"]):
        ax.set_facecolor("#f7f9fc")
        sc = ax.scatter(
            case.points[:, 0],
            case.points[:, 1],
            c=values,
            cmap="magma",
            s=12,
            linewidths=0,
            vmin=0.0,
            vmax=vmax,
            alpha=0.96,
        )
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(f"Pressure error map - epoch {epoch} - case {case_id}", fontsize=13, fontweight="bold", y=0.98)
    cbar = fig.colorbar(sc, ax=axes, fraction=0.026, pad=0.03)
    cbar.ax.set_title("abs error", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def scorecard_figure(metrics: dict[str, float], epoch: int):
    """Compact image summary for W&B media panels."""
    rows = [
        ("TinyTransolver score", metrics["model_score"], "#1455d9"),
        ("Baseline score", metrics["baseline_score"], "#6b7280"),
        ("Score margin", metrics["score_margin"], "#15803d" if metrics["score_margin"] >= 0 else "#b91c1c"),
        ("Pressure RMSE", metrics["pressure_rmse"], "#1455d9"),
        ("Baseline pressure RMSE", metrics["baseline_pressure_rmse"], "#6b7280"),
        ("Lift proxy rho", metrics["lift_spearman"], "#7c3aed"),
        ("Effective slices", metrics["effective_slices"], "#0f766e"),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")
    ax.axis("off")
    ax.text(0.02, 0.95, f"Epoch {epoch} scorecard", fontsize=15, fontweight="bold", transform=ax.transAxes)
    ax.text(
        0.02,
        0.87,
        "model selection, baseline guardrail, and routing health",
        fontsize=9.5,
        color="#475569",
        transform=ax.transAxes,
    )
    y = 0.74
    for label, value, color in rows:
        ax.text(0.04, y, label, fontsize=10.5, color="#111827", transform=ax.transAxes, va="center")
        ax.text(
            0.96,
            y,
            f"{value:.4f}" if abs(value) < 100 else f"{value:.1f}",
            fontsize=11,
            fontweight="bold",
            color=color,
            transform=ax.transAxes,
            va="center",
            ha="right",
        )
        ax.plot([0.04, 0.96], [y - 0.045, y - 0.045], color="#e5e7eb", lw=0.9, transform=ax.transAxes)
        y -= 0.095
    fig.tight_layout()
    return fig


def slice_usage_figure(slice_weights: torch.Tensor, epoch: int):
    """Bar chart of batch-averaged slice usage for the last block."""
    usage = slice_weights.mean(dim=(0, 1)).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7.8, 3.0), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")
    ax.bar(np.arange(len(usage)), usage, color="#2563eb", width=0.82)
    ax.axhline(1.0 / len(usage), color="#ef4444", lw=1.2, linestyle="--", label="uniform")
    ax.set_title(f"Last-block slice usage - epoch {epoch}", fontsize=12, fontweight="bold")
    ax.set_xlabel("slice")
    ax.set_ylabel("mean assignment")
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
