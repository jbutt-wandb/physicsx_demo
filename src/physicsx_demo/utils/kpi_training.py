from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
import importlib

from physicsx_demo.utils.data import high_lift_surfaces
from physicsx_demo.utils.hilift_kpi import (
    HILIFT_GEOMETRY_COLUMNS,
    HILIFT_TARGET_COLUMNS,
    HILIFT_TOKEN_POSITIONS,
    HiLiftKpiDataset,
    denormalize_kpi_targets,
    kpi_frame_summary,
    prepare_hilift_kpi_bundle,
)
import physicsx_demo.utils.model as model_utils
from physicsx_demo.utils.viz import figure_to_rgb_frame, save_frame_png, save_figure_png, write_slice_evolution_gif

model_utils = importlib.reload(model_utils)
KpiTokenMLP = model_utils.KpiTokenMLP
TinyTransolverKpi = model_utils.TinyTransolverKpi
count_params = model_utils.count_params


def _rmse_by_target(pred: np.ndarray, true: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}/{name}_rmse": float(np.sqrt(np.mean((pred[:, idx] - true[:, idx]) ** 2)))
        for idx, name in enumerate(HILIFT_TARGET_COLUMNS)
    }


def _evaluate(model: torch.nn.Module, loader: DataLoader, stats, device: torch.device) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            losses.append(F.mse_loss(pred, yb).item())
            preds.append(pred.cpu())
            targets.append(yb.cpu())
    pred_norm = torch.cat(preds, dim=0)
    target_norm = torch.cat(targets, dim=0)
    pred_phys = denormalize_kpi_targets(pred_norm, stats)
    target_phys = denormalize_kpi_targets(target_norm, stats)
    return float(np.mean(losses)), _rmse_by_target(pred_phys, target_phys, "val"), pred_phys, target_phys


def _prediction_table(frame, pred: np.ndarray, true: np.ndarray, max_rows: int = 120) -> wandb.Table:
    rows = []
    view = frame.reset_index(drop=True).iloc[:max_rows]
    for i, row in view.iterrows():
        rows.append(
            [
                row["GeoID"],
                float(row["AoA"]),
                *[float(true[i, j]) for j in range(3)],
                *[float(pred[i, j]) for j in range(3)],
                *[float(abs(pred[i, j] - true[i, j])) for j in range(3)],
            ]
        )
    return wandb.Table(
        data=rows,
        columns=[
            "GeoID",
            "AoA",
            "cl_true",
            "cd_true",
            "cm_true",
            "cl_pred",
            "cd_pred",
            "cm_pred",
            "cl_abs_error",
            "cd_abs_error",
            "cm_abs_error",
        ],
    )


def _comparison_table(frame, pred: np.ndarray, baseline_pred: np.ndarray, true: np.ndarray, max_rows: int = 120) -> wandb.Table:
    rows = []
    view = frame.reset_index(drop=True).iloc[:max_rows]
    for i, row in view.iterrows():
        model_err = np.abs(pred[i] - true[i])
        baseline_err = np.abs(baseline_pred[i] - true[i])
        rows.append(
            [
                row["GeoID"],
                float(row["AoA"]),
                *[float(true[i, j]) for j in range(3)],
                *[float(pred[i, j]) for j in range(3)],
                *[float(baseline_pred[i, j]) for j in range(3)],
                *[float(model_err[j]) for j in range(3)],
                *[float(baseline_err[j]) for j in range(3)],
                float(model_err.mean() - baseline_err.mean()),
            ]
        )
    return wandb.Table(
        data=rows,
        columns=[
            "GeoID",
            "AoA",
            "cl_true",
            "cd_true",
            "cm_true",
            "cl_model",
            "cd_model",
            "cm_model",
            "cl_baseline",
            "cd_baseline",
            "cm_baseline",
            "cl_model_abs_error",
            "cd_model_abs_error",
            "cm_model_abs_error",
            "cl_baseline_abs_error",
            "cd_baseline_abs_error",
            "cm_baseline_abs_error",
            "model_minus_baseline_mean_abs_error",
        ],
    )


def _dataset_preview_table(frame, max_rows: int = 80) -> wandb.Table:
    columns = ["GeoID", "AoA", *HILIFT_GEOMETRY_COLUMNS, *HILIFT_TARGET_COLUMNS]
    rows = []
    for _, row in frame.reset_index(drop=True).head(max_rows).iterrows():
        rows.append([row["GeoID"], float(row["AoA"]), *[float(row[col]) for col in HILIFT_GEOMETRY_COLUMNS], *[float(row[col]) for col in HILIFT_TARGET_COLUMNS]])
    return wandb.Table(data=rows, columns=columns)


def _design_variable_table(frame, max_geometries: int = 80) -> wandb.Table:
    design = frame.drop_duplicates("GeoID").sort_values("GeoID").head(max_geometries)
    columns = ["GeoID", *HILIFT_GEOMETRY_COLUMNS]
    rows = []
    for _, row in design.iterrows():
        rows.append([row["GeoID"], *[float(row[col]) for col in HILIFT_GEOMETRY_COLUMNS]])
    return wandb.Table(data=rows, columns=columns)


def _geometry_summary_table(frame, max_geometries: int = 80) -> wandb.Table:
    grouped = frame.groupby("GeoID", sort=True)
    rows = []
    for geo_id, rows_frame in list(grouped)[:max_geometries]:
        first = rows_frame.iloc[0]
        rows.append(
            [
                geo_id,
                int(len(rows_frame)),
                float(rows_frame["AoA"].min()),
                float(rows_frame["AoA"].max()),
                float(rows_frame["cl"].min()),
                float(rows_frame["cl"].max()),
                float(rows_frame["cd"].min()),
                float(rows_frame["cd"].max()),
                float(rows_frame["cm"].min()),
                float(rows_frame["cm"].max()),
                float(first["IB_Flap_Deflection"]),
                float(first["OB_Flap_Deflection"]),
                float(first["IB_Slat_Deflection"]),
                float(first["OB_Slat_Deflection"]),
            ]
        )
    return wandb.Table(
        data=rows,
        columns=[
            "GeoID",
            "rows",
            "aoa_min",
            "aoa_max",
            "cl_min",
            "cl_max",
            "cd_min",
            "cd_max",
            "cm_min",
            "cm_max",
            "IB_Flap_Deflection",
            "OB_Flap_Deflection",
            "IB_Slat_Deflection",
            "OB_Slat_Deflection",
        ],
    )


def _profile_card_table(frame, media_dir: Path, max_profiles: int = 12) -> tuple[wandb.Table, list[Path]]:
    rows = []
    image_paths = []
    selected = frame.sort_values(["GeoID", "AoA"]).drop_duplicates("GeoID").head(max_profiles).reset_index(drop=True)
    for i, (_, row) in enumerate(selected.iterrows()):
        fig = _profile_card_figure(row, title=f"Profile card - {row['GeoID']}")
        path = save_figure_png(fig, media_dir / "dataset" / "profile_cards", f"profile_card_{i:02d}_{row['GeoID']}")
        plt.close(fig)
        image_paths.append(path)
        rows.append(
            [
                row["GeoID"],
                float(row["AoA"]),
                float(row["cl"]),
                float(row["cd"]),
                float(row["cm"]),
                float(row["IB_Flap_Deflection"]),
                float(row["OB_Flap_Deflection"]),
                float(row["IB_Slat_Deflection"]),
                float(row["OB_Slat_Deflection"]),
                wandb.Image(str(path), caption=f"{row['GeoID']} profile card"),
            ]
        )
    return (
        wandb.Table(
            data=rows,
            columns=[
                "GeoID",
                "AoA",
                "cl",
                "cd",
                "cm",
                "IB_Flap_Deflection",
                "OB_Flap_Deflection",
                "IB_Slat_Deflection",
                "OB_Slat_Deflection",
                "profile_card",
            ],
        ),
        image_paths,
    )


def _profile_flythrough_frames(frame, max_profiles: int = 12) -> list[np.ndarray]:
    frames = []
    selected = frame.sort_values(["GeoID", "AoA"]).drop_duplicates("GeoID").head(max_profiles).reset_index(drop=True)
    for _, row in selected.iterrows():
        fig = _profile_card_figure(row, title=f"Dataset profile flythrough - {row['GeoID']}")
        frames.append(figure_to_rgb_frame(fig))
        plt.close(fig)
    return frames


def _prediction_scatter_figure(pred: np.ndarray, true: np.ndarray, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    for ax, name, idx in zip(axes, HILIFT_TARGET_COLUMNS, range(3)):
        ax.scatter(true[:, idx], pred[:, idx], s=18, alpha=0.72, color="#1764d8", edgecolor="none")
        lo = float(min(true[:, idx].min(), pred[:, idx].min()))
        hi = float(max(true[:, idx].max(), pred[:, idx].max()))
        pad = 0.04 * (hi - lo + 1e-8)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#111827", linewidth=1.0)
        ax.set_title(name.upper())
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        ax.grid(alpha=0.22)
    fig.suptitle(title)
    return fig


def _model_baseline_scatter_figure(pred: np.ndarray, baseline_pred: np.ndarray, true: np.ndarray, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
    for ax, name, idx in zip(axes, HILIFT_TARGET_COLUMNS, range(3)):
        ax.scatter(true[:, idx], baseline_pred[:, idx], s=18, alpha=0.55, color="#6b7280", edgecolor="none", label="baseline")
        ax.scatter(true[:, idx], pred[:, idx], s=18, alpha=0.78, color="#1764d8", edgecolor="none", label="TinyTransolverKpi")
        lo = float(min(true[:, idx].min(), pred[:, idx].min(), baseline_pred[:, idx].min()))
        hi = float(max(true[:, idx].max(), pred[:, idx].max(), baseline_pred[:, idx].max()))
        pad = 0.04 * (hi - lo + 1e-8)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#111827", linewidth=1.0)
        ax.set_title(name.upper())
        ax.set_xlabel("true")
        ax.set_ylabel("prediction")
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(title)
    return fig


def _drag_polar_figure(frame, pred: np.ndarray, true: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(6.3, 4.3), constrained_layout=True)
    ax.scatter(true[:, 1], true[:, 0], s=22, label="true", color="#111827", alpha=0.75)
    ax.scatter(pred[:, 1], pred[:, 0], s=22, label="pred", color="#1764d8", alpha=0.72)
    for geo_id in frame["GeoID"].astype(str).unique()[:8]:
        idx = frame["GeoID"].astype(str).to_numpy() == geo_id
        order = np.argsort(frame.loc[idx, "AoA"].to_numpy())
        ax.plot(true[idx, 1][order], true[idx, 0][order], color="#111827", linewidth=0.8, alpha=0.25)
        ax.plot(pred[idx, 1][order], pred[idx, 0][order], color="#1764d8", linewidth=0.8, alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("CD")
    ax.set_ylabel("CL")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)
    return fig


def _model_baseline_drag_polar_figure(frame, pred: np.ndarray, baseline_pred: np.ndarray, true: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.4), constrained_layout=True)
    ax.scatter(true[:, 1], true[:, 0], s=22, label="true", color="#111827", alpha=0.72)
    ax.scatter(baseline_pred[:, 1], baseline_pred[:, 0], s=22, label="baseline", color="#6b7280", alpha=0.55)
    ax.scatter(pred[:, 1], pred[:, 0], s=22, label="TinyTransolverKpi", color="#1764d8", alpha=0.76)
    for geo_id in frame["GeoID"].astype(str).unique()[:6]:
        idx = frame["GeoID"].astype(str).to_numpy() == geo_id
        order = np.argsort(frame.loc[idx, "AoA"].to_numpy())
        ax.plot(true[idx, 1][order], true[idx, 0][order], color="#111827", linewidth=0.8, alpha=0.20)
        ax.plot(baseline_pred[idx, 1][order], baseline_pred[idx, 0][order], color="#6b7280", linewidth=0.8, alpha=0.20)
        ax.plot(pred[idx, 1][order], pred[idx, 0][order], color="#1764d8", linewidth=0.8, alpha=0.24)
    ax.set_title(title)
    ax.set_xlabel("CD")
    ax.set_ylabel("CL")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)
    return fig


def _coefficient_trends_figure(frame, title: str, max_geometries: int = 12):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    geos = frame["GeoID"].astype(str).drop_duplicates().head(max_geometries)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(geos), 1)))
    for color, geo_id in zip(colors, geos):
        rows = frame[frame["GeoID"].astype(str) == geo_id].sort_values("AoA")
        for ax, target in zip(axes, HILIFT_TARGET_COLUMNS):
            ax.plot(rows["AoA"], rows[target], marker="o", linewidth=1.0, markersize=2.8, alpha=0.72, color=color)
    for ax, target in zip(axes, HILIFT_TARGET_COLUMNS):
        ax.set_title(target.upper())
        ax.set_xlabel("AoA")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("coefficient")
    fig.suptitle(title)
    return fig


def _design_heatmap_figure(frame, title: str, max_geometries: int = 40):
    design = frame.drop_duplicates("GeoID").sort_values("GeoID").head(max_geometries)
    values = design[HILIFT_GEOMETRY_COLUMNS].to_numpy(dtype=np.float32)
    values = (values - values.mean(axis=0, keepdims=True)) / (values.std(axis=0, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(9.8, 5.2), constrained_layout=True)
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-2.4, vmax=2.4)
    ax.set_title(title)
    ax.set_xlabel("design variable")
    ax.set_ylabel("geometry")
    ax.set_xticks(range(len(HILIFT_GEOMETRY_COLUMNS)))
    ax.set_xticklabels([col.replace("_", "\n") for col in HILIFT_GEOMETRY_COLUMNS], fontsize=7)
    ax.set_yticks(range(len(design)))
    ax.set_yticklabels(design["GeoID"].astype(str).to_list(), fontsize=7)
    fig.colorbar(im, ax=ax, label="standardized value")
    return fig


def _token_map_figure(row, title: str):
    values = row[HILIFT_GEOMETRY_COLUMNS].to_numpy(dtype=np.float32)
    fig = plt.figure(figsize=(7.0, 5.3), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    points = HILIFT_TOKEN_POSITIONS
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=values,
        cmap="viridis",
        s=220,
        edgecolor="#111827",
        linewidth=0.7,
    )
    for point, label in zip(points, HILIFT_GEOMETRY_COLUMNS):
        ax.text(point[0], point[1], point[2] + 0.08, label.replace("_", "\n"), fontsize=7, ha="center")
    ax.set_title(title)
    ax.set_xlabel("flap/slat family")
    ax.set_ylabel("span station")
    ax.set_zlabel("deflection/gap")
    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])
    ax.set_zticks([-1, 1])
    fig.colorbar(scatter, ax=ax, shrink=0.68, label="raw design value")
    return fig


def _profile_params_from_row(row) -> dict[str, float]:
    flap = 0.5 * (float(row["IB_Flap_Deflection"]) + float(row["OB_Flap_Deflection"]))
    slat = 0.5 * (float(row["IB_Slat_Deflection"]) + float(row["OB_Slat_Deflection"]))
    flap_gap_multiplier = 0.5 * (float(row["IB_Flap_Gap_Multiplier"]) + float(row["OB_Flap_Gap_Multiplier"]))
    slat_gap_multiplier = 0.5 * (float(row["IB_Slat_Gap_Multiplier"]) + float(row["OB_Slat_Gap_Multiplier"]))
    high_lift_index = (flap - 10.0) / 35.0 + 0.45 * (slat - 10.0) / 25.0
    return {
        "main_camber": float(np.clip(0.028 + 0.012 * high_lift_index, 0.018, 0.052)),
        "main_thickness": float(np.clip(0.125 + 0.01 * (flap_gap_multiplier - 1.0), 0.105, 0.155)),
        "slat_deflection": float(np.deg2rad(slat)),
        "flap_deflection": float(np.deg2rad(flap)),
        "flap_gap": float(np.clip(0.045 * flap_gap_multiplier + 0.012 * (slat_gap_multiplier - 1.0), 0.025, 0.075)),
    }


def _draw_profile(ax, row, *, attribution: np.ndarray | None = None):
    curves = high_lift_surfaces(**_profile_params_from_row(row))
    element_tokens = {
        "slat": [4, 5, 6, 7],
        "main": [],
        "flap": [0, 1, 2, 3],
    }
    colors = {"slat": "#0891b2", "main": "#0f766e", "flap": "#7c3aed"}
    if attribution is not None:
        attribution = np.asarray(attribution, dtype=np.float32)
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
    for name, (upper, lower) in curves.items():
        polygon = np.concatenate([upper, lower[::-1]], axis=0)
        alpha = 0.90
        if attribution is not None and element_tokens[name]:
            alpha = float(0.35 + 0.60 * np.mean(attribution[element_tokens[name]]))
        ax.fill(polygon[:, 0], polygon[:, 1], color=colors[name], alpha=alpha, linewidth=0)
        ax.plot(upper[:, 0], upper[:, 1], color="#082f49", lw=1.0)
        ax.plot(lower[:, 0], lower[:, 1], color="#082f49", lw=1.0)
    all_pts = np.concatenate([np.concatenate([upper, lower], axis=0) for upper, lower in curves.values()], axis=0)
    ax.set_xlim(float(all_pts[:, 0].min()) - 0.04, float(all_pts[:, 0].max()) + 0.04)
    y_mid = float(0.5 * (all_pts[:, 1].min() + all_pts[:, 1].max()))
    span = max(float(all_pts[:, 1].max() - all_pts[:, 1].min()), 0.20)
    ax.set_ylim(y_mid - 0.62 * span, y_mid + 0.62 * span)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def _draw_profile_outline(ax, row, *, color: str, alpha: float = 0.42, linewidth: float = 1.2) -> None:
    curves = high_lift_surfaces(**_profile_params_from_row(row))
    for upper, lower in curves.values():
        ax.plot(upper[:, 0], upper[:, 1], color=color, alpha=alpha, lw=linewidth)
        ax.plot(lower[:, 0], lower[:, 1], color=color, alpha=alpha, lw=linewidth)
    all_pts = np.concatenate([np.concatenate([upper, lower], axis=0) for upper, lower in curves.values()], axis=0)
    ax.set_xlim(float(all_pts[:, 0].min()) - 0.05, float(all_pts[:, 0].max()) + 0.05)
    y_mid = float(0.5 * (all_pts[:, 1].min() + all_pts[:, 1].max()))
    span = max(float(all_pts[:, 1].max() - all_pts[:, 1].min()), 0.22)
    ax.set_ylim(y_mid - 0.68 * span, y_mid + 0.68 * span)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def _profile_card_figure(row, title: str | None = None):
    fig = plt.figure(figsize=(9.4, 5.0), constrained_layout=True)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(2, 3, width_ratios=[2.1, 1.0, 1.0], height_ratios=[1.2, 1.0])
    ax_profile = fig.add_subplot(grid[:, 0])
    ax_coeff = fig.add_subplot(grid[0, 1:])
    ax_design = fig.add_subplot(grid[1, 1:])

    ax_profile.set_facecolor("#f8fafc")
    _draw_profile(ax_profile, row)
    ax_profile.set_title(f"{row['GeoID']} | AoA {float(row['AoA']):.1f}", fontsize=11, color="#0f172a", pad=6)

    coeff_values = [float(row[target]) for target in HILIFT_TARGET_COLUMNS]
    coeff_colors = ["#1764d8", "#0891b2", "#7c3aed"]
    ax_coeff.bar(HILIFT_TARGET_COLUMNS, coeff_values, color=coeff_colors, alpha=0.86)
    ax_coeff.axhline(0.0, color="#111827", lw=0.8)
    ax_coeff.set_title("integrated coefficients", fontsize=10)
    ax_coeff.grid(axis="y", alpha=0.18)
    ax_coeff.tick_params(axis="x", labelsize=8)

    design_values = row[HILIFT_GEOMETRY_COLUMNS].to_numpy(dtype=np.float32)
    design_norm = (design_values - np.nanmean(design_values)) / (np.nanstd(design_values) + 1e-8)
    ax_design.barh(range(len(HILIFT_GEOMETRY_COLUMNS)), design_norm, color="#0f766e", alpha=0.82)
    ax_design.axvline(0.0, color="#111827", lw=0.8)
    ax_design.set_yticks(range(len(HILIFT_GEOMETRY_COLUMNS)))
    ax_design.set_yticklabels([col.replace("_", " ") for col in HILIFT_GEOMETRY_COLUMNS], fontsize=7)
    ax_design.invert_yaxis()
    ax_design.set_title("within-profile design contrast", fontsize=10)
    ax_design.grid(axis="x", alpha=0.18)

    fig.suptitle(title or "High-lift profile card", fontsize=14, fontweight="bold", color="#0f172a")
    return fig


def _profile_gallery_figure(frame, title: str, max_profiles: int = 8):
    rows = frame.drop_duplicates("GeoID").head(max_profiles).reset_index(drop=True)
    cols = 4
    n_rows = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 3.3, n_rows * 1.55), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(n_rows, cols)
    fig.patch.set_facecolor("white")
    for ax in axes.ravel():
        ax.axis("off")
    for i, (_, row) in enumerate(rows.iterrows()):
        ax = axes.ravel()[i]
        _draw_profile(ax, row)
        ax.set_title(str(row["GeoID"]), fontsize=9, color="#0f172a")
    fig.suptitle(title)
    return fig


def _profile_matrix_figure(frame, title: str, max_profiles: int = 12):
    rows = (
        frame.sort_values(["IB_Flap_Deflection", "AoA", "GeoID"])
        .drop_duplicates(["GeoID", "AoA"])
        .head(max_profiles)
        .reset_index(drop=True)
    )
    cols = 4
    n_rows = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 3.25, n_rows * 2.15), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(n_rows, cols)
    cl_values = rows["cl"].to_numpy(dtype=np.float32)
    norm = plt.Normalize(vmin=float(cl_values.min()), vmax=float(cl_values.max()))
    cmap = plt.cm.viridis
    for ax in axes.ravel():
        ax.axis("off")
    for i, (_, row) in enumerate(rows.iterrows()):
        ax = axes.ravel()[i]
        _draw_profile(ax, row)
        color = cmap(norm(float(row["cl"])))
        ax.set_facecolor((*color[:3], 0.10))
        ax.set_title(
            f"{row['GeoID']} | AoA {float(row['AoA']):.0f} | CL {float(row['cl']):.2f}",
            fontsize=8.6,
            color="#0f172a",
        )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.82, label="CL")
    fig.suptitle(title, fontsize=14, fontweight="bold", color="#0f172a")
    return fig


def _profile_overlay_figure(frame, title: str, max_profiles: int = 28):
    rows = frame.sort_values(["AoA", "GeoID"]).drop_duplicates("GeoID").head(max_profiles).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.6, 5.3), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")
    values = rows["IB_Flap_Deflection"].to_numpy(dtype=np.float32)
    norm = plt.Normalize(vmin=float(values.min()), vmax=float(values.max()))
    cmap = plt.cm.plasma
    for _, row in rows.iterrows():
        color = cmap(norm(float(row["IB_Flap_Deflection"])))
        _draw_profile_outline(ax, row, color=color, alpha=0.38, linewidth=1.25)
    ref = rows.iloc[len(rows) // 2]
    _draw_profile_outline(ax, ref, color="#111827", alpha=0.92, linewidth=2.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.78, label="IB flap deflection")
    ax.set_title(title)
    return fig


def _slice_attribution_figure(slice_weights: np.ndarray, row, epoch: int, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), gridspec_kw={"width_ratios": [1.05, 1.55]}, constrained_layout=True)
    token_strength = slice_weights.max(axis=1)
    _draw_profile(axes[0], row, attribution=token_strength)
    axes[0].set_title(f"{row['GeoID']} profile | AoA {float(row['AoA']):.0f}")

    im = axes[1].imshow(slice_weights, aspect="auto", cmap="magma", vmin=0.0, vmax=max(0.35, float(slice_weights.max())))
    axes[1].set_title("token-to-slice attribution")
    axes[1].set_xlabel("learned slice")
    axes[1].set_ylabel("design token")
    axes[1].set_yticks(range(len(HILIFT_GEOMETRY_COLUMNS)))
    axes[1].set_yticklabels([col.replace("_", " ") for col in HILIFT_GEOMETRY_COLUMNS], fontsize=7)
    axes[1].set_xticks(range(slice_weights.shape[1]))
    axes[1].set_xticklabels(range(slice_weights.shape[1]), fontsize=8)
    fig.colorbar(im, ax=axes[1], label="assignment weight")
    fig.suptitle(f"{title} - epoch {epoch}")
    return fig


def _slice_usage_by_token_figure(slice_weights: np.ndarray, epoch: int):
    token_strength = slice_weights.max(axis=1)
    slice_usage = slice_weights.mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8), constrained_layout=True)
    axes[0].barh(range(len(HILIFT_GEOMETRY_COLUMNS)), token_strength, color="#1764d8")
    axes[0].set_yticks(range(len(HILIFT_GEOMETRY_COLUMNS)))
    axes[0].set_yticklabels([col.replace("_", " ") for col in HILIFT_GEOMETRY_COLUMNS], fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("max slice assignment")
    axes[0].set_title("token focus")
    axes[1].bar(range(len(slice_usage)), slice_usage, color="#7c3aed")
    axes[1].set_xlabel("slice")
    axes[1].set_ylabel("mean assignment")
    axes[1].set_title("slice usage")
    fig.suptitle(f"KPI slice routing summary - epoch {epoch}")
    return fig


def _coefficient_curves_figure(frame, pred: np.ndarray, true: np.ndarray, title: str, max_geometries: int = 4):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    frame = frame.reset_index(drop=True)
    geos = frame["GeoID"].astype(str).drop_duplicates().head(max_geometries)
    colors = plt.cm.Dark2(np.linspace(0, 1, max(len(geos), 1)))
    for color, geo_id in zip(colors, geos):
        idx = np.flatnonzero(frame["GeoID"].astype(str).to_numpy() == geo_id)
        order = np.argsort(frame.loc[idx, "AoA"].to_numpy())
        idx = idx[order]
        aoa = frame.loc[idx, "AoA"].to_numpy()
        for ax, target_idx, target in zip(axes, range(3), HILIFT_TARGET_COLUMNS):
            ax.plot(aoa, true[idx, target_idx], color=color, linewidth=1.4, alpha=0.78)
            ax.plot(aoa, pred[idx, target_idx], color=color, linewidth=1.2, linestyle="--", alpha=0.9)
            ax.set_title(target.upper())
            ax.set_xlabel("AoA")
            ax.grid(alpha=0.22)
    axes[0].set_ylabel("coefficient")
    axes[0].plot([], [], color="#111827", label="true")
    axes[0].plot([], [], color="#111827", linestyle="--", label="pred")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(title)
    return fig


def _model_baseline_curves_figure(frame, pred: np.ndarray, baseline_pred: np.ndarray, true: np.ndarray, title: str, max_geometries: int = 4):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), constrained_layout=True)
    frame = frame.reset_index(drop=True)
    geos = frame["GeoID"].astype(str).drop_duplicates().head(max_geometries)
    colors = plt.cm.Dark2(np.linspace(0, 1, max(len(geos), 1)))
    for color, geo_id in zip(colors, geos):
        idx = np.flatnonzero(frame["GeoID"].astype(str).to_numpy() == geo_id)
        order = np.argsort(frame.loc[idx, "AoA"].to_numpy())
        idx = idx[order]
        aoa = frame.loc[idx, "AoA"].to_numpy()
        for ax, target_idx, target in zip(axes, range(3), HILIFT_TARGET_COLUMNS):
            ax.plot(aoa, true[idx, target_idx], color=color, linewidth=1.55, alpha=0.76)
            ax.plot(aoa, pred[idx, target_idx], color=color, linewidth=1.25, linestyle="--", alpha=0.92)
            ax.plot(aoa, baseline_pred[idx, target_idx], color=color, linewidth=1.1, linestyle=":", alpha=0.85)
            ax.set_title(target.upper())
            ax.set_xlabel("AoA")
            ax.grid(alpha=0.22)
    axes[0].set_ylabel("coefficient")
    axes[0].plot([], [], color="#111827", label="true")
    axes[0].plot([], [], color="#111827", linestyle="--", label="TinyTransolverKpi")
    axes[0].plot([], [], color="#111827", linestyle=":", label="baseline")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(title)
    return fig


def _model_baseline_error_delta_figure(frame, pred: np.ndarray, baseline_pred: np.ndarray, true: np.ndarray, title: str):
    frame = frame.reset_index(drop=True)
    model_err = np.abs(pred - true)
    baseline_err = np.abs(baseline_pred - true)
    delta = model_err - baseline_err
    geos = sorted(frame["GeoID"].astype(str).unique())
    aoas = sorted(frame["AoA"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.4), constrained_layout=True)
    vmax = max(1e-6, float(np.nanpercentile(np.abs(delta), 95)))
    for ax, target_idx, target in zip(axes, range(3), HILIFT_TARGET_COLUMNS):
        grid = np.full((len(geos), len(aoas)), np.nan, dtype=np.float32)
        for i, geo_id in enumerate(geos):
            for j, aoa in enumerate(aoas):
                mask = (frame["GeoID"].astype(str).to_numpy() == geo_id) & (frame["AoA"].to_numpy() == aoa)
                if mask.any():
                    grid[i, j] = float(delta[mask, target_idx].mean())
        im = ax.imshow(grid, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{target.upper()} error delta")
        ax.set_xlabel("AoA")
        ax.set_xticks(range(len(aoas)))
        ax.set_xticklabels([str(int(x)) if float(x).is_integer() else str(x) for x in aoas], rotation=45, fontsize=8)
        ax.set_yticks(range(len(geos)))
        ax.set_yticklabels(geos, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    axes[0].set_ylabel("held-out GeoID")
    fig.suptitle(f"{title} | blue means model lower error")
    return fig


def _error_heatmap_figure(frame, pred: np.ndarray, true: np.ndarray, title: str):
    frame = frame.reset_index(drop=True)
    abs_err = np.abs(pred - true)
    geos = sorted(frame["GeoID"].astype(str).unique())
    aoas = sorted(frame["AoA"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
    for ax, target_idx, target in zip(axes, range(3), HILIFT_TARGET_COLUMNS):
        grid = np.full((len(geos), len(aoas)), np.nan, dtype=np.float32)
        for i, geo_id in enumerate(geos):
            for j, aoa in enumerate(aoas):
                mask = (frame["GeoID"].astype(str).to_numpy() == geo_id) & (frame["AoA"].to_numpy() == aoa)
                if mask.any():
                    grid[i, j] = float(abs_err[mask, target_idx].mean())
        im = ax.imshow(grid, aspect="auto", cmap="magma")
        ax.set_title(f"|{target.upper()} error|")
        ax.set_xlabel("AoA")
        ax.set_xticks(range(len(aoas)))
        ax.set_xticklabels([str(int(x)) if float(x).is_integer() else str(x) for x in aoas], rotation=45, fontsize=8)
        ax.set_yticks(range(len(geos)))
        ax.set_yticklabels(geos, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    axes[0].set_ylabel("held-out GeoID")
    fig.suptitle(title)
    return fig


def _residual_hist_figure(pred: np.ndarray, true: np.ndarray, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    residual = pred - true
    for ax, target_idx, target in zip(axes, range(3), HILIFT_TARGET_COLUMNS):
        ax.hist(residual[:, target_idx], bins=26, color="#1764d8", alpha=0.78)
        ax.axvline(0.0, color="#111827", linewidth=1.0)
        ax.set_title(target.upper())
        ax.set_xlabel("pred - true")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("rows")
    fig.suptitle(title)
    return fig


def _make_checkpoint(
    *,
    model: torch.nn.Module,
    config: dict,
    stats,
    epoch: int,
    val_loss: float,
    val_rmse: dict[str, float],
    path: Path,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": dict(config),
            "stats": stats.to_dict(),
            "target_columns": HILIFT_TARGET_COLUMNS,
            "epoch": epoch,
            "val_mse": val_loss,
            "val_rmse": dict(val_rmse),
        },
        path,
    )


def run_kpi_training_experiment(
    *,
    entity: str,
    project: str,
    registry_name: str,
    dataset_registry_path: str,
    dataset_dir: str | Path,
    n_geometries: int,
    seed: int,
    device: torch.device,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 3e-3,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    bundle = prepare_hilift_kpi_bundle(dataset_dir, n_geometries=n_geometries, seed=seed)
    model_config = {
        "in_dim": 7,
        "out_dim": 3,
        "hidden": 64,
        "n_layers": 3,
        "n_slices": 8,
        "n_heads": 4,
        "ffn_expansion": 2,
        "dropout": 0.05,
        "use_global_context": True,
    }
    train_config = {
        **model_config,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "seed": seed,
        "n_geometries": n_geometries,
        "dataset_mode": "real_hiliftaeroml_kpi",
        "target_columns": HILIFT_TARGET_COLUMNS,
        "split_strategy": "held-out GeoID",
    }

    train_loader = DataLoader(HiLiftKpiDataset(bundle.train, bundle.stats), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(HiLiftKpiDataset(bundle.val, bundle.stats), batch_size=batch_size)
    test_loader = DataLoader(HiLiftKpiDataset(bundle.test, bundle.stats), batch_size=batch_size)
    visual_rows = bundle.val.reset_index(drop=True).head(4)
    visual_tokens = torch.stack([HiLiftKpiDataset(visual_rows, bundle.stats)[i][0] for i in range(len(visual_rows))])
    visual_row = visual_rows.iloc[0]
    model = TinyTransolverKpi(**model_config).to(device)
    baseline = KpiTokenMLP().to(device)
    train_config["n_params"] = count_params(model)
    train_config["baseline_n_params"] = count_params(baseline)

    tmp_dir = Path(tempfile.mkdtemp(prefix="physicsx_kpi_", dir=tempfile.gettempdir()))
    media_dir = tmp_dir / "wandb_media"
    media_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = tmp_dir / "tiny_transolver_kpi.pt"
    baseline_path = tmp_dir / "kpi_token_mlp.pt"

    print(f"real KPI frame: {kpi_frame_summary(bundle.frame)}")
    print(f"train: {kpi_frame_summary(bundle.train)}")
    print(f"val:   {kpi_frame_summary(bundle.val)}")
    print(f"test:  {kpi_frame_summary(bundle.test)}")
    print(f"TinyTransolverKpi params: {count_params(model):,}")
    print(f"KpiTokenMLP params       : {count_params(baseline):,}")

    with wandb.init(
        entity=entity,
        project=project,
        job_type="train",
        config=train_config,
        tags=["notebook", "experiment-tracking", "tiny-transolver-kpi", "real-hiliftaeroml"],
        name="tiny-transolver-kpi-train",
    ) as run:
        run.use_artifact(dataset_registry_path)
        run.summary["model/family"] = "TinyTransolverKpi"
        run.summary["dataset/source"] = "nvidia/HiLiftAeroML CSV summaries"

        dataset_table = wandb.Table(data=[[k, v] for k, v in kpi_frame_summary(bundle.frame).items()], columns=["stat", "value"])
        split_table = wandb.Table(
            data=[
                ["train", *kpi_frame_summary(bundle.train).values()],
                ["val", *kpi_frame_summary(bundle.val).values()],
                ["test", *kpi_frame_summary(bundle.test).values()],
            ],
            columns=["split", "rows", "geometries", "aoa_min", "aoa_max"],
        )
        preview_table = _dataset_preview_table(bundle.frame)
        design_table = _design_variable_table(bundle.frame)
        geometry_table = _geometry_summary_table(bundle.frame)
        profile_card_table, profile_card_paths = _profile_card_table(bundle.train, media_dir)
        fig_trends = _coefficient_trends_figure(bundle.frame, "HiLiftAeroML coefficient trends by geometry")
        trends_path = save_figure_png(fig_trends, media_dir / "dataset", "coefficient_trends")
        plt.close(fig_trends)
        fig_design = _design_heatmap_figure(bundle.train, "Held-out-split design variable map")
        design_path = save_figure_png(fig_design, media_dir / "dataset", "design_variable_heatmap")
        plt.close(fig_design)
        fig_tokens = _token_map_figure(bundle.train.iloc[0], "Semantic design tokens for one training geometry")
        token_path = save_figure_png(fig_tokens, media_dir / "dataset", "semantic_token_map")
        plt.close(fig_tokens)
        fig_profiles = _profile_gallery_figure(bundle.train, "Simplified profiles from real HiLift design variables")
        profiles_path = save_figure_png(fig_profiles, media_dir / "dataset", "profile_gallery")
        plt.close(fig_profiles)
        fig_profile_matrix = _profile_matrix_figure(bundle.train, "High-lift profile matrix colored by CL")
        profile_matrix_path = save_figure_png(fig_profile_matrix, media_dir / "dataset", "profile_matrix")
        plt.close(fig_profile_matrix)
        fig_profile_overlay = _profile_overlay_figure(bundle.train, "Profile design-space overlay")
        profile_overlay_path = save_figure_png(fig_profile_overlay, media_dir / "dataset", "profile_overlay")
        plt.close(fig_profile_overlay)
        fig_featured_profile = _profile_card_figure(bundle.train.iloc[0], "Featured real-design profile card")
        featured_profile_path = save_figure_png(fig_featured_profile, media_dir / "dataset", "featured_profile_card")
        plt.close(fig_featured_profile)
        profile_flythrough_frames = _profile_flythrough_frames(bundle.train)
        profile_flythrough_path = write_slice_evolution_gif(
            profile_flythrough_frames,
            tmp_dir / "dataset_profile_flythrough.gif",
            fps=2,
        )

        dataset_media_artifact = wandb.Artifact(
            "workshop_kpi_dataset_media",
            type="media",
            description="Dataset-level HiLiftAeroML KPI visuals for the PhysicsX workshop.",
        )
        dataset_media_artifact.add_file(str(trends_path), name="coefficient_trends.png")
        dataset_media_artifact.add_file(str(design_path), name="design_variable_heatmap.png")
        dataset_media_artifact.add_file(str(token_path), name="semantic_token_map.png")
        dataset_media_artifact.add_file(str(profiles_path), name="profile_gallery.png")
        dataset_media_artifact.add_file(str(profile_matrix_path), name="profile_matrix.png")
        dataset_media_artifact.add_file(str(profile_overlay_path), name="profile_overlay.png")
        dataset_media_artifact.add_file(str(featured_profile_path), name="featured_profile_card.png")
        dataset_media_artifact.add_file(str(profile_flythrough_path), name="profile_flythrough.gif")
        for path in profile_card_paths:
            dataset_media_artifact.add_file(str(path), name=f"profile_cards/{path.name}")
        dataset_media_artifact.add(preview_table, "dataset_preview")
        dataset_media_artifact.add(design_table, "design_variables")
        dataset_media_artifact.add(geometry_table, "geometry_summary")
        dataset_media_artifact.add(profile_card_table, "profile_cards")
        logged_dataset_media = run.log_artifact(dataset_media_artifact, aliases=["latest"])

        run.log(
            {
                "dataset/summary": dataset_table,
                "dataset/splits": split_table,
                "dataset/preview_table": preview_table,
                "dataset/design_variable_table": design_table,
                "dataset/geometry_summary_table": geometry_table,
                "dataset/profile_card_table": profile_card_table,
                "dataset/coefficient_trends": wandb.Image(str(trends_path), caption="CL/CD/CM trends across geometry and AoA"),
                "dataset/design_variable_heatmap": wandb.Image(str(design_path), caption="Standardized flap/slat design variables"),
                "dataset/semantic_token_map": wandb.Image(str(token_path), caption="Eight semantic design tokens used by TinyTransolverKpi"),
                "dataset/high_lift_profile_gallery": wandb.Image(
                    str(profiles_path),
                    caption="Simplified high-lift profiles derived from real flap/slat design variables",
                ),
                "dataset/high_lift_profile_matrix": wandb.Image(
                    str(profile_matrix_path),
                    caption="Profile matrix colored by real CL values",
                ),
                "dataset/high_lift_profile_overlay": wandb.Image(
                    str(profile_overlay_path),
                    caption="Overlaid profile outlines colored by flap deflection",
                ),
                "dataset/featured_profile_card": wandb.Image(
                    str(featured_profile_path),
                    caption="Profile card combining geometry, coefficients, AoA, and design variables",
                ),
                "dataset/profile_flythrough_video": wandb.Video(str(profile_flythrough_path), format="gif"),
            }
        )

        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=train_config["weight_decay"])
        baseline_optim = torch.optim.AdamW(baseline.parameters(), lr=lr, weight_decay=train_config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
        baseline_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(baseline_optim, T_max=epochs)

        best_loss = float("inf")
        best_epoch = -1
        best_artifact = None
        best_baseline_loss = float("inf")
        best_baseline_state = None
        slice_attribution_frames = []
        profile_slice_frames = []
        comparison_curves_frames = []
        error_delta_frames = []
        drag_polar_frames = []
        latest_slice_artifacts = None
        for epoch in range(1, epochs + 1):
            model.train()
            baseline.train()
            train_losses = []
            baseline_losses = []
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = F.mse_loss(pred, yb)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip_norm"])
                optim.step()
                train_losses.append(loss.item())

                baseline_pred = baseline(xb)
                baseline_loss = F.mse_loss(baseline_pred, yb)
                baseline_optim.zero_grad()
                baseline_loss.backward()
                torch.nn.utils.clip_grad_norm_(baseline.parameters(), train_config["grad_clip_norm"])
                baseline_optim.step()
                baseline_losses.append(baseline_loss.item())

            scheduler.step()
            baseline_scheduler.step()

            val_loss, val_rmse, val_pred, val_true = _evaluate(model, val_loader, bundle.stats, device)
            baseline_val_loss, baseline_val_rmse, baseline_val_pred, _ = _evaluate(baseline, val_loader, bundle.stats, device)
            if baseline_val_loss < best_baseline_loss:
                best_baseline_loss = baseline_val_loss
                best_baseline_state = {key: value.detach().cpu().clone() for key, value in baseline.state_dict().items()}

            log_payload = {
                "epoch": epoch,
                "train/loss": float(np.mean(train_losses)),
                "train/lr": scheduler.get_last_lr()[0],
                "val/mse": val_loss,
                "baseline/train_loss": float(np.mean(baseline_losses)),
                "baseline/val_mse": baseline_val_loss,
                "model/selection_score": val_loss,
                "baseline/selection_score": baseline_val_loss,
                "model/selection_margin_vs_baseline": baseline_val_loss - val_loss,
                **val_rmse,
                **{key.replace("val/", "baseline/val_"): value for key, value in baseline_val_rmse.items()},
            }

            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                model.eval()
                with torch.no_grad():
                    _, visual_all_w = model(visual_tokens.to(device), return_slice_weights=True)
                    visual_slice_weights = visual_all_w[-1][0].detach().cpu().numpy()

                fig_scatter = _prediction_scatter_figure(val_pred, val_true, f"Validation predicted vs true - epoch {epoch}")
                scatter_path = save_figure_png(fig_scatter, media_dir / f"epoch_{epoch:03d}", "predicted_vs_true")
                plt.close(fig_scatter)
                fig_comparison_scatter = _model_baseline_scatter_figure(
                    val_pred,
                    baseline_val_pred,
                    val_true,
                    f"Model vs baseline predicted-vs-true - epoch {epoch}",
                )
                comparison_scatter_path = save_figure_png(fig_comparison_scatter, media_dir / f"epoch_{epoch:03d}", "model_baseline_scatter")
                plt.close(fig_comparison_scatter)
                fig_polar = _drag_polar_figure(bundle.val, val_pred, val_true, f"Validation drag polar - epoch {epoch}")
                polar_path = save_figure_png(fig_polar, media_dir / f"epoch_{epoch:03d}", "drag_polar")
                plt.close(fig_polar)
                fig_comparison_polar = _model_baseline_drag_polar_figure(
                    bundle.val,
                    val_pred,
                    baseline_val_pred,
                    val_true,
                    f"Model vs baseline drag polar - epoch {epoch}",
                )
                comparison_polar_path = save_figure_png(fig_comparison_polar, media_dir / f"epoch_{epoch:03d}", "model_baseline_drag_polar")
                drag_polar_frames.append(figure_to_rgb_frame(fig_comparison_polar))
                plt.close(fig_comparison_polar)
                fig_curves = _coefficient_curves_figure(bundle.val, val_pred, val_true, f"Held-out geometry coefficient curves - epoch {epoch}")
                curves_path = save_figure_png(fig_curves, media_dir / f"epoch_{epoch:03d}", "coefficient_curves")
                plt.close(fig_curves)
                fig_comparison_curves = _model_baseline_curves_figure(
                    bundle.val,
                    val_pred,
                    baseline_val_pred,
                    val_true,
                    f"Model vs baseline coefficient curves - epoch {epoch}",
                )
                comparison_curves_path = save_figure_png(fig_comparison_curves, media_dir / f"epoch_{epoch:03d}", "model_baseline_curves")
                comparison_curves_frames.append(figure_to_rgb_frame(fig_comparison_curves))
                plt.close(fig_comparison_curves)
                fig_heatmap = _error_heatmap_figure(bundle.val, val_pred, val_true, f"Held-out geometry error heatmap - epoch {epoch}")
                heatmap_path = save_figure_png(fig_heatmap, media_dir / f"epoch_{epoch:03d}", "error_heatmap")
                plt.close(fig_heatmap)
                fig_error_delta = _model_baseline_error_delta_figure(
                    bundle.val,
                    val_pred,
                    baseline_val_pred,
                    val_true,
                    f"Model minus baseline absolute error - epoch {epoch}",
                )
                error_delta_path = save_figure_png(fig_error_delta, media_dir / f"epoch_{epoch:03d}", "model_baseline_error_delta")
                error_delta_frames.append(figure_to_rgb_frame(fig_error_delta))
                plt.close(fig_error_delta)
                fig_residual = _residual_hist_figure(val_pred, val_true, f"Validation residual distributions - epoch {epoch}")
                residual_path = save_figure_png(fig_residual, media_dir / f"epoch_{epoch:03d}", "residual_hist")
                plt.close(fig_residual)
                fig_slice = _slice_attribution_figure(
                    visual_slice_weights,
                    visual_row,
                    epoch,
                    "KPI token slice attribution",
                )
                slice_path = save_figure_png(fig_slice, media_dir / f"epoch_{epoch:03d}", "slice_attribution")
                slice_frame = figure_to_rgb_frame(fig_slice)
                slice_attribution_frames.append(slice_frame)

                fig_profile_slice = _slice_attribution_figure(
                    visual_slice_weights,
                    visual_row,
                    epoch,
                    "Profile-linked slice attribution",
                )
                profile_slice_path = save_figure_png(fig_profile_slice, media_dir / f"epoch_{epoch:03d}", "profile_slice_attribution")
                profile_slice_frame = figure_to_rgb_frame(fig_profile_slice)
                profile_slice_frames.append(profile_slice_frame)

                fig_usage = _slice_usage_by_token_figure(visual_slice_weights, epoch)
                usage_path = save_figure_png(fig_usage, media_dir / f"epoch_{epoch:03d}", "token_slice_usage")
                plt.close(fig_usage)

                log_payload.update(
                    {
                        f"val/predicted_vs_true_epoch_{epoch}": wandb.Image(str(scatter_path)),
                        f"val/model_baseline_scatter_epoch_{epoch}": wandb.Image(str(comparison_scatter_path)),
                        f"val/drag_polar_epoch_{epoch}": wandb.Image(str(polar_path)),
                        f"val/model_baseline_drag_polar_epoch_{epoch}": wandb.Image(str(comparison_polar_path)),
                        f"val/coefficient_curves_epoch_{epoch}": wandb.Image(str(curves_path)),
                        f"val/model_baseline_curves_epoch_{epoch}": wandb.Image(str(comparison_curves_path)),
                        f"val/error_heatmap_epoch_{epoch}": wandb.Image(str(heatmap_path)),
                        f"val/model_baseline_error_delta_epoch_{epoch}": wandb.Image(str(error_delta_path)),
                        f"val/residual_hist_epoch_{epoch}": wandb.Image(str(residual_path)),
                        f"model/kpi_slice_attribution_epoch_{epoch}": wandb.Image(str(slice_path)),
                        f"model/kpi_profile_slice_epoch_{epoch}": wandb.Image(str(profile_slice_path)),
                        f"model/kpi_token_slice_usage_epoch_{epoch}": wandb.Image(str(usage_path)),
                        "val/predicted_vs_true_latest": wandb.Image(str(scatter_path)),
                        "val/model_baseline_scatter_latest": wandb.Image(str(comparison_scatter_path)),
                        "val/drag_polar_latest": wandb.Image(str(polar_path)),
                        "val/model_baseline_drag_polar_latest": wandb.Image(str(comparison_polar_path)),
                        "val/coefficient_curves_latest": wandb.Image(str(curves_path)),
                        "val/model_baseline_curves_latest": wandb.Image(str(comparison_curves_path)),
                        "val/error_heatmap_latest": wandb.Image(str(heatmap_path)),
                        "val/model_baseline_error_delta_latest": wandb.Image(str(error_delta_path)),
                        "val/residual_hist_latest": wandb.Image(str(residual_path)),
                        "model/kpi_slice_attribution_latest": wandb.Image(str(slice_path)),
                        "model/kpi_profile_slice_latest": wandb.Image(str(profile_slice_path)),
                        "model/kpi_token_slice_usage_latest": wandb.Image(str(usage_path)),
                        "val/predictions_latest": _prediction_table(bundle.val, val_pred, val_true),
                        "val/model_baseline_comparison_latest": _comparison_table(bundle.val, val_pred, baseline_val_pred, val_true),
                    }
                )

            wandb.log(log_payload)

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                _make_checkpoint(
                    model=model,
                    config=train_config,
                    stats=bundle.stats,
                    epoch=epoch,
                    val_loss=val_loss,
                    val_rmse=val_rmse,
                    path=ckpt_path,
                )
                artifact = wandb.Artifact(
                    "high_lift_kpi_surrogate_ckpt",
                    type="model",
                    description="TinyTransolverKpi checkpoint trained on real HiLiftAeroML CL/CD/CM summaries.",
                    metadata={
                        "epoch": epoch,
                        "val_mse": val_loss,
                        "baseline_val_mse": baseline_val_loss,
                        "baseline_guardrail_passed": val_loss <= baseline_val_loss,
                        "n_params": count_params(model),
                        **val_rmse,
                    },
                )
                artifact.add_file(str(ckpt_path))
                best_artifact = run.log_artifact(artifact, aliases=[f"epoch-{epoch}", "best"])

            print(
                f"epoch {epoch:03d} | train {np.mean(train_losses):.5f} | "
                f"val {val_loss:.5f} | baseline {baseline_val_loss:.5f} | "
                f"CL {val_rmse['val/cl_rmse']:.5f} CD {val_rmse['val/cd_rmse']:.5f} CM {val_rmse['val/cm_rmse']:.5f}"
            )

        test_loss, test_rmse, test_pred, test_true = _evaluate(model, test_loader, bundle.stats, device)
        baseline_test_loss, baseline_test_rmse, _, _ = _evaluate(baseline, test_loader, bundle.stats, device)
        run.log(
            {
                "test/mse": test_loss,
                "baseline/test_mse": baseline_test_loss,
                "test/predictions": _prediction_table(bundle.test, test_pred, test_true),
                **{key.replace("val/", "test/"): value for key, value in test_rmse.items()},
                **{key.replace("val/", "baseline/test_"): value for key, value in baseline_test_rmse.items()},
            }
        )

        run.summary["model/best_selection_score"] = best_loss
        run.summary["model/best_epoch"] = best_epoch
        run.summary["baseline/best_selection_score"] = best_baseline_loss
        run.summary["model/baseline_guardrail_passed"] = best_loss <= best_baseline_loss
        run.summary["test/mse"] = test_loss
        for key, value in test_rmse.items():
            run.summary[key.replace("val/", "test/")] = value

        if slice_attribution_frames:
            slice_video_path = write_slice_evolution_gif(
                slice_attribution_frames,
                tmp_dir / "kpi_slice_attribution_evolution.gif",
                fps=2,
            )
            profile_slice_video_path = write_slice_evolution_gif(
                profile_slice_frames,
                tmp_dir / "kpi_profile_slice_evolution.gif",
                fps=2,
            )
            slice_rows = []
            profile_slice_rows = []
            media_epochs = [1, *range(10, epochs + 1, 10)]
            if media_epochs[-1] != epochs:
                media_epochs.append(epochs)
            for media_epoch, frame in zip(media_epochs, slice_attribution_frames):
                frame_path = save_frame_png(frame, media_dir / "slice_attribution_frames", f"slice_epoch_{media_epoch:03d}")
                slice_rows.append([media_epoch, wandb.Image(str(frame_path), caption=f"slice attribution epoch {media_epoch}")])
            for media_epoch, frame in zip(media_epochs, profile_slice_frames):
                frame_path = save_frame_png(frame, media_dir / "profile_slice_frames", f"profile_slice_epoch_{media_epoch:03d}")
                profile_slice_rows.append([media_epoch, wandb.Image(str(frame_path), caption=f"profile slice attribution epoch {media_epoch}")])

            slice_table = wandb.Table(data=slice_rows, columns=["epoch", "slice_attribution"])
            profile_slice_table = wandb.Table(data=profile_slice_rows, columns=["epoch", "profile_slice_attribution"])
            slice_media_artifact = wandb.Artifact(
                "workshop_kpi_slice_attributions",
                type="media-table",
                description="Per-epoch KPI token slice-attribution frames and GIFs.",
            )
            slice_media_artifact.add(slice_table, "slice_attribution_frames")
            slice_media_artifact.add(profile_slice_table, "profile_slice_attribution_frames")
            slice_media_artifact.add_file(str(slice_video_path), name="kpi_slice_attribution_evolution.gif")
            slice_media_artifact.add_file(str(profile_slice_video_path), name="kpi_profile_slice_evolution.gif")
            latest_slice_artifacts = run.log_artifact(slice_media_artifact, aliases=["latest"])
            wandb.log(
                {
                    "model/kpi_slice_attribution_video": wandb.Video(str(slice_video_path), format="gif"),
                    "model/kpi_profile_slice_video": wandb.Video(str(profile_slice_video_path), format="gif"),
                    "model/kpi_slice_attribution_table": slice_table,
                    "model/kpi_profile_slice_table": profile_slice_table,
                }
            )
            run.summary["workshop/kpi_slice_attributions_artifact"] = "workshop_kpi_slice_attributions:latest"

        latest_evolution_artifact = None
        if comparison_curves_frames and error_delta_frames and drag_polar_frames:
            curves_video_path = write_slice_evolution_gif(
                comparison_curves_frames,
                tmp_dir / "kpi_model_baseline_curves_evolution.gif",
                fps=2,
            )
            error_delta_video_path = write_slice_evolution_gif(
                error_delta_frames,
                tmp_dir / "kpi_model_baseline_error_delta_evolution.gif",
                fps=2,
            )
            drag_polar_video_path = write_slice_evolution_gif(
                drag_polar_frames,
                tmp_dir / "kpi_drag_polar_evolution.gif",
                fps=2,
            )
            media_epochs = [1, *range(10, epochs + 1, 10)]
            if media_epochs[-1] != epochs:
                media_epochs.append(epochs)
            curves_rows = []
            delta_rows = []
            polar_rows = []
            for media_epoch, frame in zip(media_epochs, comparison_curves_frames):
                frame_path = save_frame_png(frame, media_dir / "curve_evolution_frames", f"curves_epoch_{media_epoch:03d}")
                curves_rows.append([media_epoch, wandb.Image(str(frame_path), caption=f"model vs baseline curves epoch {media_epoch}")])
            for media_epoch, frame in zip(media_epochs, error_delta_frames):
                frame_path = save_frame_png(frame, media_dir / "error_delta_frames", f"error_delta_epoch_{media_epoch:03d}")
                delta_rows.append([media_epoch, wandb.Image(str(frame_path), caption=f"error delta epoch {media_epoch}")])
            for media_epoch, frame in zip(media_epochs, drag_polar_frames):
                frame_path = save_frame_png(frame, media_dir / "drag_polar_frames", f"drag_polar_epoch_{media_epoch:03d}")
                polar_rows.append([media_epoch, wandb.Image(str(frame_path), caption=f"drag polar epoch {media_epoch}")])

            curves_table = wandb.Table(data=curves_rows, columns=["epoch", "model_baseline_curves"])
            delta_table = wandb.Table(data=delta_rows, columns=["epoch", "model_baseline_error_delta"])
            polar_table = wandb.Table(data=polar_rows, columns=["epoch", "drag_polar"])
            evolution_artifact = wandb.Artifact(
                "workshop_kpi_prediction_evolution",
                type="media-table",
                description="Epoch-time videos for KPI predictions, error deltas, and drag-polar evolution.",
            )
            evolution_artifact.add(curves_table, "model_baseline_curves_frames")
            evolution_artifact.add(delta_table, "model_baseline_error_delta_frames")
            evolution_artifact.add(polar_table, "drag_polar_frames")
            evolution_artifact.add_file(str(curves_video_path), name="kpi_model_baseline_curves_evolution.gif")
            evolution_artifact.add_file(str(error_delta_video_path), name="kpi_model_baseline_error_delta_evolution.gif")
            evolution_artifact.add_file(str(drag_polar_video_path), name="kpi_drag_polar_evolution.gif")
            latest_evolution_artifact = run.log_artifact(evolution_artifact, aliases=["latest"])
            wandb.log(
                {
                    "val/model_baseline_curves_video": wandb.Video(str(curves_video_path), format="gif"),
                    "val/model_baseline_error_delta_video": wandb.Video(str(error_delta_video_path), format="gif"),
                    "val/drag_polar_evolution_video": wandb.Video(str(drag_polar_video_path), format="gif"),
                    "val/model_baseline_curves_table": curves_table,
                    "val/model_baseline_error_delta_table": delta_table,
                    "val/drag_polar_evolution_table": polar_table,
                }
            )
            run.summary["workshop/kpi_prediction_evolution_artifact"] = "workshop_kpi_prediction_evolution:latest"

        if best_artifact is None:
            raise RuntimeError("No model artifact was logged.")
        best_artifact.wait()
        api = wandb.Api(overrides={"entity": entity, "project": project})
        try:
            registry = api.create_registry(name=registry_name, visibility="organization")
            print(f"created registry: {registry.name}")
        except Exception as exc:
            print(f"registry create skipped (may already exist): {exc}")
        target_path = f"wandb-registry-{registry_name}/checkpoints"
        run.link_artifact(best_artifact, target_path=target_path, aliases=["production", f"epoch-{best_epoch}"])
        run.summary["model_registry/path"] = target_path
        run.summary["model_registry/aliases"] = ["production", f"epoch-{best_epoch}"]

        if best_baseline_loss < best_loss and best_baseline_state is not None:
            torch.save(
                {
                    "state_dict": best_baseline_state,
                    "config": {"in_dim": 7, "out_dim": 3, "hidden": 64, "depth": 3, "dropout": 0.05},
                    "stats": bundle.stats.to_dict(),
                    "target_columns": HILIFT_TARGET_COLUMNS,
                    "val_mse": best_baseline_loss,
                },
                baseline_path,
            )
            baseline_artifact = wandb.Artifact(
                "kpi_token_mlp_baseline_ckpt",
                type="model",
                description="Pooled-token MLP baseline checkpoint for real HiLiftAeroML KPI prediction.",
                metadata={"val_mse": best_baseline_loss, "n_params": count_params(baseline)},
            )
            baseline_artifact.add_file(str(baseline_path))
            logged_baseline = run.log_artifact(baseline_artifact, aliases=["baseline-best"])
            logged_baseline.wait()
            run.link_artifact(logged_baseline, target_path=f"wandb-registry-{registry_name}/baselines", aliases=["production", "baseline-best"])

        logged_dataset_media.wait()
        if latest_slice_artifacts is not None:
            latest_slice_artifacts.wait()
        if latest_evolution_artifact is not None:
            latest_evolution_artifact.wait()

    return train_config
