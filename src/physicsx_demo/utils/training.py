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
import torch.nn.functional as F
import wandb

from physicsx_demo.utils.data import stack_cases
from physicsx_demo.utils.metrics import (
    pressure_force_proxy,
    proxy_metrics,
    relative_l2,
    selection_score,
    slice_regularization,
)
from physicsx_demo.utils.model import count_params
from physicsx_demo.utils.viz import (
    airfoil_gallery_figure,
    airfoil_profile_figure,
    comparison_figure,
    error_figure,
    save_figure_png,
    save_frame_png,
    scorecard_figure,
    slice_figure,
    slice_usage_frame,
    slice_usage_figure,
    slice_video_frame,
    write_slice_evolution_gif,
)


def run_training_experiment(
    *,
    entity: str,
    project: str,
    registry_name: str,
    dataset_registry_path: str,
    model_config: dict,
    model: torch.nn.Module,
    baseline: torch.nn.Module,
    train_cases: list,
    val_cases: list,
    n_points: int,
    seed: int,
    device: torch.device,
) -> dict:
    """Run the full workshop training flow and return the W&B training config."""
    ENTITY = entity
    PROJECT = project
    REGISTRY_NAME = registry_name
    DATASET_REGISTRY_PATH = dataset_registry_path
    MODEL_CONFIG = model_config
    N_POINTS = n_points
    SEED = seed
    # ---- Training config and tensors ----
    TRAIN_CONFIG = {
        **MODEL_CONFIG,
        "epochs": 32,
        "batch_size": 8,
        "lr": 2e-3,
        "baseline_lr": 2e-3,
        "weight_decay": 1e-4,
        "lambda_slice_usage": 0.02,
        "lambda_slice_sharpness": 0.04,
        "pressure_channel_weight": 2.0,
        "velocity_channel_weight": 1.0,
        "surface_weight_scale": 2.0,
        "surface_weight_decay": 0.08,
        "grad_clip_norm": 1.0,
        "selection_score": "pressure_rmse + 0.35*field_relative_l2 + 0.20*lift_proxy_mae + 0.10*drag_proxy_mae",
        "training_objective": "TinyTransolver uses pressure/channel + near-surface weighted MSE; baseline keeps plain pointwise MSE",
        "n_cases": len(train_cases),
        "n_points": N_POINTS,
        "seed": SEED,
        "n_params": count_params(model),
        "baseline_n_params": count_params(baseline),
        "slice_video_fps": 2,
    }

    tmp_dir = Path(tempfile.mkdtemp(prefix="physicsx_ckpts_", dir=tempfile.gettempdir()))
    media_dir = tmp_dir / "wandb_media"
    media_dir.mkdir(parents=True, exist_ok=True)
    X_train, Y_train = stack_cases(train_cases)
    X_val, Y_val = stack_cases(val_cases)

    CHECKPOINT_ARTIFACT_NAME = "airfoil_surrogate_ckpt"
    BASELINE_ARTIFACT_NAME = "pointwise_mlp_baseline_ckpt"
    CHECKPOINT_COLLECTION = "checkpoints"
    BASELINE_COLLECTION = "baselines"

    print(f"X_train shape: {tuple(X_train.shape)}    Y_train shape: {tuple(Y_train.shape)}")
    print(f"X_val   shape: {tuple(X_val.shape)}    Y_val   shape: {tuple(Y_val.shape)}")
    print(f"checkpoint scratch dir: {tmp_dir}")

    with wandb.init(
        entity=ENTITY,
        project=PROJECT,
        job_type="train",
        config=TRAIN_CONFIG,
        tags=["notebook", "experiment-tracking", "tiny-transolver", "wandb-models"],
        name="tiny-transolver-train",
    ) as run:
        run.use_artifact(DATASET_REGISTRY_PATH)
        run.summary["model/family"] = "TinyTransolver"
        run.summary["model/showcase"] = "W&B Models: lineage, checkpoints, Registry aliases"

        # Make the physical shape obvious in the run before training metrics arrive.
        # File-backed wandb.Image objects are the most reliable way to render media in Tables.
        gallery_cases = list(val_cases[:8]) + list(train_cases[:4])
        airfoil_columns = ["case_id", "split", "aoa_deg", "re", "airfoil_profile"]
        airfoil_table = wandb.Table(columns=airfoil_columns)
        airfoil_artifact_table = wandb.Table(columns=airfoil_columns)
        for i, c in enumerate(gallery_cases):
            split = "val" if i < min(8, len(val_cases)) else "train"
            fig = airfoil_profile_figure(c, title=f"{split} case {i} airfoil profile")
            profile_path = save_figure_png(fig, media_dir / "airfoil_profiles", f"{split}_case_{i:02d}_airfoil_profile")
            plt.close(fig)
            caption = f"{split} case {i} - AoA {math.degrees(c.aoa):.1f} deg"
            row = [i, split, float(math.degrees(c.aoa)), float(c.re)]
            airfoil_table.add_data(*row, wandb.Image(str(profile_path), caption=caption))
            airfoil_artifact_table.add_data(*row, wandb.Image(str(profile_path), caption=caption))

        fig_gallery = airfoil_gallery_figure(gallery_cases, "Airfoil profile gallery - validation and train samples")
        gallery_path = save_figure_png(fig_gallery, media_dir / "airfoil_profiles", "airfoil_profile_gallery")
        plt.close(fig_gallery)

        fig_reference_profile = airfoil_profile_figure(val_cases[0], title="canonical validation airfoil")
        reference_profile_path = save_figure_png(
            fig_reference_profile,
            media_dir / "airfoil_profiles",
            "canonical_validation_airfoil_profile",
        )
        plt.close(fig_reference_profile)

        airfoil_media_artifact = wandb.Artifact(
            "workshop_airfoil_profiles",
            type="media-table",
            description="Airfoil profile gallery and image table for the PhysicsX workshop.",
        )
        airfoil_media_artifact.add(airfoil_artifact_table, "airfoil_profiles")
        airfoil_media_artifact.add_file(str(gallery_path), name="airfoil_profile_gallery.png")
        airfoil_media_artifact.add_file(str(reference_profile_path), name="canonical_validation_airfoil_profile.png")
        logged_airfoil_media = run.log_artifact(airfoil_media_artifact, aliases=["latest"])

        wandb.log({
            "dataset/canonical_airfoil_profile": wandb.Image(
                str(reference_profile_path),
                caption="Static geometry reference - logged once because the airfoil shape does not change by epoch.",
            ),
            "workshop/airfoil_profile_reference": wandb.Image(
                str(reference_profile_path),
                caption="Static airfoil shape reference",
            ),
            "workshop/airfoil_profile_gallery": wandb.Image(str(gallery_path), caption="Airfoil shape gallery"),
            "workshop/airfoil_profiles_table": airfoil_table,
        })
        run.summary["workshop/airfoil_profiles_artifact"] = "workshop_airfoil_profiles:latest"
        run.summary["workshop/airfoil_profile_note"] = (
            "Airfoil geometry is a fixed dataset input; epoch sliders are reserved for model outputs "
            "such as slice usage, slice assignment, pressure comparisons, and error maps."
        )

        feat_mean = X_train.mean(dim=(0, 1), keepdim=True)
        feat_std = X_train.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
        tgt_mean = Y_train.mean(dim=(0, 1), keepdim=True)
        tgt_std = Y_train.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)

        X_train_n = (X_train - feat_mean) / feat_std
        Y_train_n = (Y_train - tgt_mean) / tgt_std
        X_val_n = (X_val - feat_mean) / feat_std
        Y_val_n = (Y_val - tgt_mean) / tgt_std

        tgt_mean_dev = tgt_mean.to(device)
        tgt_std_dev = tgt_std.to(device)
        Y_val_dev = Y_val.to(device)

        print("target normalisation stats (channels: [pressure, u, v]):")
        print(f"  mean = {tgt_mean.squeeze().tolist()}")
        print(f"  std  = {tgt_std.squeeze().tolist()}")

        channel_weights = torch.tensor(
            [
                TRAIN_CONFIG["pressure_channel_weight"],
                TRAIN_CONFIG["velocity_channel_weight"],
                TRAIN_CONFIG["velocity_channel_weight"],
            ],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, 3)

        def surface_weights_from_features(raw_features: torch.Tensor) -> torch.Tensor:
            sdf = raw_features[..., 2].abs()
            return 1.0 + TRAIN_CONFIG["surface_weight_scale"] * torch.exp(-sdf / TRAIN_CONFIG["surface_weight_decay"])

        def weighted_field_mse(pred: torch.Tensor, target: torch.Tensor, raw_features: torch.Tensor):
            per_channel_error = (pred - target).pow(2)
            surface_weights = surface_weights_from_features(raw_features).unsqueeze(-1)
            weighted_error = per_channel_error * channel_weights * surface_weights
            return weighted_error.mean(), per_channel_error.mean(), surface_weights.mean()

        print("TinyTransolver objective: pressure/channel + near-surface weighted MSE")
        print(f"  channel_weights = {channel_weights.flatten().tolist()}")
        print(f"  surface_weight = 1 + {TRAIN_CONFIG['surface_weight_scale']} * exp(-abs(sdf) / {TRAIN_CONFIG['surface_weight_decay']})")

        optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["lr"], weight_decay=TRAIN_CONFIG["weight_decay"])
        baseline_optimizer = torch.optim.AdamW(
            baseline.parameters(),
            lr=TRAIN_CONFIG["baseline_lr"],
            weight_decay=TRAIN_CONFIG["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG["epochs"])
        baseline_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(baseline_optimizer, T_max=TRAIN_CONFIG["epochs"])

        best_score = float("inf")
        best_epoch = None
        best_ckpt_artifact = None
        best_model_metrics = None
        best_baseline_score = float("inf")
        best_baseline_epoch = None
        best_baseline_metrics = None
        best_baseline_state = None
        slice_video_frames = []
        slice_usage_video_frames = []
        slice_video_case = val_cases[0]
        latest_case_diagnostics_rows = []
        latest_case_diagnostics_epoch = None

        n = X_train_n.size(0)
        batch_size = TRAIN_CONFIG["batch_size"]

        for epoch in range(TRAIN_CONFIG["epochs"]):
            model.train()
            baseline.train()
            perm = torch.randperm(n)
            train_losses = []
            field_mse_losses = []
            weighted_field_mse_losses = []
            surface_weight_means = []
            baseline_losses = []

            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb = X_train_n[idx].to(device)
                x_raw = X_train[idx].to(device)
                yb = Y_train_n[idx].to(device)

                pred, all_w = model(xb, return_slice_weights=True)
                weighted_mse, field_mse, surface_weight_mean = weighted_field_mse(pred, yb, x_raw)
                aux, _ = slice_regularization(
                    all_w,
                    usage_weight=TRAIN_CONFIG["lambda_slice_usage"],
                    sharpness_weight=TRAIN_CONFIG["lambda_slice_sharpness"],
                )
                loss = weighted_mse + aux

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip_norm"])
                optimizer.step()
                train_losses.append(loss.item())
                field_mse_losses.append(field_mse.item())
                weighted_field_mse_losses.append(weighted_mse.item())
                surface_weight_means.append(surface_weight_mean.item())

                base_pred = baseline(xb)
                base_loss = F.mse_loss(base_pred, yb)
                baseline_optimizer.zero_grad()
                base_loss.backward()
                torch.nn.utils.clip_grad_norm_(baseline.parameters(), TRAIN_CONFIG["grad_clip_norm"])
                baseline_optimizer.step()
                baseline_losses.append(base_loss.item())

                wandb.log({
                    "train/loss": loss.item(),
                    "train/mse": field_mse.item(),
                    "train/weighted_mse": weighted_mse.item(),
                    "train/surface_weight_mean_step": surface_weight_mean.item(),
                    "train/aux": aux.item(),
                    "baseline/train_mse": base_loss.item(),
                    "epoch": epoch,
                })

            scheduler.step()
            baseline_scheduler.step()

            model.eval()
            baseline.eval()
            with torch.no_grad():
                xv = X_val_n.to(device)
                yv = Y_val_n.to(device)
                pred_n, all_w = model(xv, return_slice_weights=True)
                baseline_pred_n = baseline(xv)

                val_weighted_mse, val_field_mse, val_surface_weight_mean = weighted_field_mse(pred_n, yv, X_val.to(device))
                val_mse = val_field_mse.item()
                val_aux, slice_diag = slice_regularization(
                    all_w,
                    usage_weight=TRAIN_CONFIG["lambda_slice_usage"],
                    sharpness_weight=TRAIN_CONFIG["lambda_slice_sharpness"],
                )
                val_loss = val_weighted_mse.item() + val_aux.item()

                err = pred_n - yv
                pressure_rmse = err[..., 0].pow(2).mean().sqrt().item()
                u_rmse = err[..., 1].pow(2).mean().sqrt().item()
                v_rmse = err[..., 2].pow(2).mean().sqrt().item()
                velocity_rmse = float(((u_rmse ** 2 + v_rmse ** 2) / 2) ** 0.5)

                baseline_err = baseline_pred_n - yv
                baseline_pressure_rmse = baseline_err[..., 0].pow(2).mean().sqrt().item()
                baseline_u_rmse = baseline_err[..., 1].pow(2).mean().sqrt().item()
                baseline_v_rmse = baseline_err[..., 2].pow(2).mean().sqrt().item()
                baseline_velocity_rmse = float(((baseline_u_rmse ** 2 + baseline_v_rmse ** 2) / 2) ** 0.5)

                val_pred = pred_n * tgt_std_dev + tgt_mean_dev
                baseline_val_pred = baseline_pred_n * tgt_std_dev + tgt_mean_dev
                field_rel_l2 = relative_l2(val_pred, Y_val_dev).item()
                baseline_field_rel_l2 = relative_l2(baseline_val_pred, Y_val_dev).item()

                val_pred_np = val_pred.cpu().numpy()
                baseline_val_pred_np = baseline_val_pred.cpu().numpy()
                proxy = proxy_metrics(val_cases, val_pred_np)
                baseline_proxy = proxy_metrics(val_cases, baseline_val_pred_np)

                last_w = all_w[-1]
                slice_video_argmax = last_w[0].argmax(dim=-1).cpu().numpy()

            slice_video_frames.append(
                slice_video_frame(
                    slice_video_case,
                    slice_video_argmax,
                    epoch,
                    TRAIN_CONFIG["n_slices"],
                )
            )
            slice_usage_video_frames.append(slice_usage_frame(last_w, epoch))

            model_score = selection_score(
                pressure_rmse,
                field_rel_l2,
                proxy["val/lift_proxy_mae"],
                proxy["val/drag_proxy_mae"],
            )
            baseline_score = selection_score(
                baseline_pressure_rmse,
                baseline_field_rel_l2,
                baseline_proxy["val/lift_proxy_mae"],
                baseline_proxy["val/drag_proxy_mae"],
            )
            improvement = (baseline_pressure_rmse - pressure_rmse) / max(baseline_pressure_rmse, 1e-8)
            score_margin = baseline_score - model_score

            wandb.log({
                "epoch": epoch,
                "train/epoch_loss": float(np.mean(train_losses)),
                "train/field_mse": float(np.mean(field_mse_losses)),
                "train/weighted_field_mse": float(np.mean(weighted_field_mse_losses)),
                "train/surface_weight_mean": float(np.mean(surface_weight_means)),
                "train/pressure_channel_weight": TRAIN_CONFIG["pressure_channel_weight"],
                "train/lr": scheduler.get_last_lr()[0],
                "val/loss": val_loss,
                "val/mse": val_mse,
                "val/weighted_mse": val_weighted_mse.item(),
                "val/surface_weight_mean": val_surface_weight_mean.item(),
                "val/aux": val_aux.item(),
                "val/pressure_rmse": pressure_rmse,
                "val/velocity_rmse": velocity_rmse,
                "val/u_rmse": u_rmse,
                "val/v_rmse": v_rmse,
                "val/field_relative_l2": field_rel_l2,
                "model/selection_score": model_score,
                "baseline/epoch_mse": float(np.mean(baseline_losses)),
                "baseline/val_pressure_rmse": baseline_pressure_rmse,
                "baseline/val_velocity_rmse": baseline_velocity_rmse,
                "baseline/field_relative_l2": baseline_field_rel_l2,
                "baseline/selection_score": baseline_score,
                "model/improvement_vs_pointwise_pressure_rmse": improvement,
                "model/selection_margin_vs_baseline": score_margin,
                **slice_diag,
                **proxy,
                "baseline/lift_proxy_mae": baseline_proxy["val/lift_proxy_mae"],
                "baseline/drag_proxy_mae": baseline_proxy["val/drag_proxy_mae"],
                "baseline/lift_proxy_spearman": baseline_proxy["val/lift_proxy_spearman"],
                "baseline/drag_proxy_spearman": baseline_proxy["val/drag_proxy_spearman"],
            })

            ckpt_path = tmp_dir / "tiny_transolver.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": dict(TRAIN_CONFIG),
                    "normalization": {
                        "feature_mean": feat_mean.cpu(),
                        "feature_std": feat_std.cpu(),
                        "target_mean": tgt_mean.cpu(),
                        "target_std": tgt_std.cpu(),
                    },
                    "epoch": epoch,
                    "val_pressure_rmse": pressure_rmse,
                    "val_field_relative_l2": field_rel_l2,
                    "val_lift_proxy_spearman": proxy["val/lift_proxy_spearman"],
                    "selection_score": model_score,
                    "baseline_selection_score": baseline_score,
                },
                ckpt_path,
            )
            ckpt_artifact = wandb.Artifact(
                name=CHECKPOINT_ARTIFACT_NAME,
                type="model",
                description="TinyTransolver airfoil surrogate checkpoint for the PhysicsX W&B Models workshop.",
                metadata={
                    "epoch": epoch,
                    "selection_score": model_score,
                    "baseline_selection_score": baseline_score,
                    "baseline_guardrail_passed": model_score <= baseline_score,
                    "val_pressure_rmse": pressure_rmse,
                    "val_field_relative_l2": field_rel_l2,
                    "val_lift_proxy_spearman": proxy["val/lift_proxy_spearman"],
                    "baseline_val_pressure_rmse": baseline_pressure_rmse,
                    "n_params": count_params(model),
                },
            )
            ckpt_artifact.add_file(str(ckpt_path))
            logged = run.log_artifact(ckpt_artifact, aliases=[f"epoch-{epoch}"])

            if model_score < best_score:
                best_score = model_score
                best_epoch = epoch
                best_ckpt_artifact = logged
                best_model_metrics = {
                    "selection_score": model_score,
                    "pressure_rmse": pressure_rmse,
                    "field_relative_l2": field_rel_l2,
                    "lift_proxy_mae": proxy["val/lift_proxy_mae"],
                    "drag_proxy_mae": proxy["val/drag_proxy_mae"],
                }

            if baseline_score < best_baseline_score:
                best_baseline_score = baseline_score
                best_baseline_epoch = epoch
                best_baseline_metrics = {
                    "selection_score": baseline_score,
                    "pressure_rmse": baseline_pressure_rmse,
                    "field_relative_l2": baseline_field_rel_l2,
                    "lift_proxy_mae": baseline_proxy["val/lift_proxy_mae"],
                    "drag_proxy_mae": baseline_proxy["val/drag_proxy_mae"],
                }
                best_baseline_state = {k: v.detach().cpu().clone() for k, v in baseline.state_dict().items()}

            if epoch % 3 == 0 or epoch == TRAIN_CONFIG["epochs"] - 1:
                pred_n_np = pred_n.cpu().numpy()
                baseline_pred_n_np = baseline_pred_n.cpu().numpy()
                y_val_n_np = Y_val_n.numpy()

                # Rich media: standalone images for W&B media panels plus a Table
                # whose rows bundle metrics and images for case-by-case inspection.
                media_payload = {}
                scorecard_metrics = {
                    "model_score": model_score,
                    "baseline_score": baseline_score,
                    "score_margin": score_margin,
                    "pressure_rmse": pressure_rmse,
                    "baseline_pressure_rmse": baseline_pressure_rmse,
                    "lift_spearman": proxy["val/lift_proxy_spearman"],
                    "effective_slices": slice_diag["slice/effective_slices"],
                }
                epoch_media_dir = media_dir / f"epoch_{epoch:03d}"

                fig_score = scorecard_figure(scorecard_metrics, epoch)
                score_path = save_figure_png(fig_score, epoch_media_dir, f"scorecard_epoch_{epoch:03d}")
                media_payload[f"summary/scorecard_epoch_{epoch}"] = wandb.Image(str(score_path), caption=f"scorecard epoch {epoch}")
                plt.close(fig_score)

                fig_usage = slice_usage_figure(last_w, epoch)
                usage_path = save_figure_png(fig_usage, epoch_media_dir, f"slice_usage_epoch_{epoch:03d}")
                media_payload[f"model/slice_usage_epoch_{epoch}"] = wandb.Image(str(usage_path), caption=f"slice usage epoch {epoch}")
                plt.close(fig_usage)

                diagnostics_columns = [
                    "case_id",
                    "aoa_deg",
                    "re",
                    "model_pressure_rmse_norm",
                    "baseline_pressure_rmse_norm",
                    "model_lift_proxy",
                    "truth_lift_proxy",
                    "model_drag_proxy",
                    "truth_drag_proxy",
                    "airfoil_profile",
                    "comparison_image",
                    "error_image",
                    "slice_image",
                ]
                comparison_table = wandb.Table(columns=diagnostics_columns)
                current_diagnostics_rows = []
                leaderboard_rows = []
                for i in range(min(8, len(val_cases))):
                    c = val_cases[i]
                    truth_field = np.concatenate([c.pressure[:, None], c.velocity], axis=1)
                    truth_proxy = pressure_force_proxy(c, truth_field)
                    model_proxy = pressure_force_proxy(c, val_pred_np[i])
                    model_rmse_i = float(np.sqrt(np.mean((pred_n_np[i, :, 0] - y_val_n_np[i, :, 0]) ** 2)))
                    baseline_rmse_i = float(np.sqrt(np.mean((baseline_pred_n_np[i, :, 0] - y_val_n_np[i, :, 0]) ** 2)))

                    fig_profile_i = airfoil_profile_figure(c, title=f"case {i} profile")
                    fig_cmp = comparison_figure(c, val_pred_np[i], baseline_val_pred_np[i], epoch, i)
                    fig_err = error_figure(c, val_pred_np[i], baseline_val_pred_np[i], epoch, i)
                    slice_argmax_i = last_w[i].argmax(dim=-1).cpu().numpy()
                    fig_slice_i = slice_figure(c, slice_argmax_i, epoch)

                    profile_i_path = save_figure_png(fig_profile_i, epoch_media_dir / "case_diagnostics", f"case_{i:02d}_airfoil_profile")
                    cmp_path = save_figure_png(fig_cmp, epoch_media_dir / "case_diagnostics", f"case_{i:02d}_comparison")
                    err_path = save_figure_png(fig_err, epoch_media_dir / "case_diagnostics", f"case_{i:02d}_error")
                    slice_path = save_figure_png(fig_slice_i, epoch_media_dir / "case_diagnostics", f"case_{i:02d}_slice_assignment")

                    if i == 0:
                        media_payload[f"val/comparison_epoch_{epoch}"] = wandb.Image(str(cmp_path), caption=f"pressure comparison epoch {epoch}")
                        media_payload[f"val/error_map_epoch_{epoch}"] = wandb.Image(str(err_path), caption=f"pressure error epoch {epoch}")
                        media_payload[f"model/slice_assignment_epoch_{epoch}"] = wandb.Image(str(slice_path), caption=f"slice assignment epoch {epoch}")

                    diagnostics_scalar_row = [
                        i,
                        float(math.degrees(c.aoa)),
                        float(c.re),
                        model_rmse_i,
                        baseline_rmse_i,
                        model_proxy["lift_proxy"],
                        truth_proxy["lift_proxy"],
                        model_proxy["drag_proxy"],
                        truth_proxy["drag_proxy"],
                    ]
                    diagnostics_image_paths = [profile_i_path, cmp_path, err_path, slice_path]
                    comparison_table.add_data(
                        *diagnostics_scalar_row,
                        wandb.Image(str(profile_i_path), caption=f"case {i} airfoil profile"),
                        wandb.Image(str(cmp_path), caption=f"case {i} prediction comparison"),
                        wandb.Image(str(err_path), caption=f"case {i} error map"),
                        wandb.Image(str(slice_path), caption=f"case {i} slice assignment"),
                    )
                    current_diagnostics_rows.append((diagnostics_scalar_row, diagnostics_image_paths))
                    leaderboard_rows.append([i, model_rmse_i, baseline_rmse_i, model_rmse_i - baseline_rmse_i])
                    plt.close(fig_profile_i)
                    plt.close(fig_cmp)
                    plt.close(fig_err)
                    plt.close(fig_slice_i)

                media_payload[f"val/case_diagnostics_epoch_{epoch}"] = comparison_table
                media_payload["workshop/case_diagnostics_latest"] = comparison_table
                latest_case_diagnostics_rows = current_diagnostics_rows
                latest_case_diagnostics_epoch = epoch

                leaderboard = wandb.Table(
                    data=leaderboard_rows,
                    columns=["case_id", "model_pressure_rmse_norm", "baseline_pressure_rmse_norm", "model_minus_baseline"],
                )
                media_payload[f"val/case_error_leaderboard_epoch_{epoch}"] = wandb.plot.bar(
                    leaderboard,
                    "case_id",
                    "model_minus_baseline",
                    title=f"Model minus baseline pressure RMSE by case - epoch {epoch}",
                )

                buckets = [("neg", -20.0, 0.0), ("0-5 deg", 0.0, 5.0), ("5-10 deg", 5.0, 10.0), ("10+ deg", 10.0, 20.0)]
                rows = []
                for label, lo, hi in buckets:
                    in_bucket = [
                        float(np.sqrt(np.mean((pred_n_np[i, :, 0] - y_val_n_np[i, :, 0]) ** 2)))
                        for i, c in enumerate(val_cases)
                        if lo <= math.degrees(c.aoa) < hi
                    ]
                    if in_bucket:
                        rows.append([label, float(np.mean(in_bucket))])
                if rows:
                    bucket_table = wandb.Table(data=rows, columns=["aoa_bucket", "mean_rmse_norm"])
                    media_payload[f"val/rmse_by_aoa_epoch_{epoch}"] = wandb.plot.bar(
                        bucket_table,
                        "aoa_bucket",
                        "mean_rmse_norm",
                        title=f"Val pressure RMSE by AoA bucket - epoch {epoch}",
                    )

                wandb.log(media_payload)

            print(
                f"epoch {epoch:2d} | train {np.mean(train_losses):.4f} | weighted {np.mean(weighted_field_mse_losses):.4f} | "
                f"p_rmse {pressure_rmse:.4f} | base {baseline_pressure_rmse:.4f} | "
                f"score {model_score:.4f} | margin {score_margin:+.4f} | "
                f"eff_slices {slice_diag['slice/effective_slices']:.1f}"
            )

        print()
        print(f"best TinyTransolver epoch: {best_epoch}   selection_score: {best_score:.4f}")
        print(f"best pointwise baseline epoch: {best_baseline_epoch}   selection_score: {best_baseline_score:.4f}")

        baseline_guardrail_passed = best_score <= best_baseline_score
        run.summary["model/best_selection_score"] = best_score
        run.summary["baseline/best_selection_score"] = best_baseline_score
        run.summary["baseline/best_epoch"] = best_baseline_epoch
        run.summary["model/baseline_guardrail_passed"] = baseline_guardrail_passed

        slice_video_path = write_slice_evolution_gif(
            slice_video_frames,
            tmp_dir / "slice_assignment_evolution.gif",
            fps=TRAIN_CONFIG["slice_video_fps"],
        )
        slice_usage_video_path = write_slice_evolution_gif(
            slice_usage_video_frames,
            tmp_dir / "slice_usage_evolution.gif",
            fps=TRAIN_CONFIG["slice_video_fps"],
        )
        if latest_case_diagnostics_rows:
            diagnostics_artifact_table = wandb.Table(columns=diagnostics_columns)
            for scalar_row, image_paths in latest_case_diagnostics_rows:
                profile_i_path, cmp_path, err_path, slice_path = image_paths
                case_id = scalar_row[0]
                diagnostics_artifact_table.add_data(
                    *scalar_row,
                    wandb.Image(str(profile_i_path), caption=f"case {case_id} airfoil profile"),
                    wandb.Image(str(cmp_path), caption=f"case {case_id} prediction comparison"),
                    wandb.Image(str(err_path), caption=f"case {case_id} error map"),
                    wandb.Image(str(slice_path), caption=f"case {case_id} slice assignment"),
                )
            diagnostics_artifact = wandb.Artifact(
                "workshop_case_diagnostics",
                type="media-table",
                description="Final validation case diagnostics with airfoil, prediction, error, and slice images.",
                metadata={"epoch": latest_case_diagnostics_epoch},
            )
            diagnostics_artifact.add(diagnostics_artifact_table, "case_diagnostics_latest")
            logged_diagnostics_artifact = run.log_artifact(
                diagnostics_artifact,
                aliases=["latest", f"epoch-{latest_case_diagnostics_epoch}"],
            )
            run.summary["workshop/case_diagnostics_artifact"] = "workshop_case_diagnostics:latest"

        slice_frame_rows = []
        slice_frame_artifact_rows = []
        for i, frame in enumerate(slice_video_frames):
            frame_path = save_frame_png(frame, media_dir / "slice_evolution_frames", f"slice_assignment_epoch_{i:03d}")
            slice_frame_rows.append([i, wandb.Image(str(frame_path), caption=f"slice assignment epoch {i}")])
            slice_frame_artifact_rows.append([i, wandb.Image(str(frame_path), caption=f"slice assignment epoch {i}")])

        slice_evolution_table = wandb.Table(data=slice_frame_rows, columns=["epoch", "slice_frame"])
        slice_evolution_artifact_table = wandb.Table(data=slice_frame_artifact_rows, columns=["epoch", "slice_frame"])
        usage_frame_rows = []
        usage_frame_artifact_rows = []
        for i, frame in enumerate(slice_usage_video_frames):
            frame_path = save_frame_png(frame, media_dir / "slice_usage_evolution_frames", f"slice_usage_epoch_{i:03d}")
            usage_frame_rows.append([i, wandb.Image(str(frame_path), caption=f"slice usage epoch {i}")])
            usage_frame_artifact_rows.append([i, wandb.Image(str(frame_path), caption=f"slice usage epoch {i}")])

        usage_evolution_table = wandb.Table(data=usage_frame_rows, columns=["epoch", "slice_usage_frame"])
        usage_evolution_artifact_table = wandb.Table(data=usage_frame_artifact_rows, columns=["epoch", "slice_usage_frame"])
        slice_media_artifact = wandb.Artifact(
            "workshop_slice_assignment_frames",
            type="media-table",
            description="Per-epoch slice assignment frames backing the W&B video.",
        )
        slice_media_artifact.add(slice_evolution_artifact_table, "slice_assignment_frames")
        slice_media_artifact.add_file(str(slice_video_path), name="slice_assignment_evolution.gif")
        logged_slice_media = run.log_artifact(slice_media_artifact, aliases=["latest"])
        usage_media_artifact = wandb.Artifact(
            "workshop_slice_usage_frames",
            type="media-table",
            description="Per-epoch slice usage frames backing the W&B video.",
        )
        usage_media_artifact.add(usage_evolution_artifact_table, "slice_usage_frames")
        usage_media_artifact.add_file(str(slice_usage_video_path), name="slice_usage_evolution.gif")
        logged_usage_media = run.log_artifact(usage_media_artifact, aliases=["latest"])

        wandb.log({
            "model/slice_usage_evolution_video": wandb.Video(str(slice_usage_video_path), format="gif"),
            "model/slice_usage_evolution_table": usage_evolution_table,
            "model/slice_assignment_evolution_video": wandb.Video(str(slice_video_path), format="gif"),
            "model/slice_assignment_evolution_table": slice_evolution_table,
            "workshop/slice_usage_video": wandb.Video(str(slice_usage_video_path), format="gif"),
            "workshop/slice_usage_frames_table": usage_evolution_table,
            "workshop/slice_assignment_video": wandb.Video(str(slice_video_path), format="gif"),
            "workshop/slice_assignment_frames_table": slice_evolution_table,
        })
        run.summary["workshop/slice_usage_artifact"] = "workshop_slice_usage_frames:latest"
        run.summary["workshop/slice_assignment_artifact"] = "workshop_slice_assignment_frames:latest"

        for artifact_ref in [
            logged_airfoil_media,
            locals().get("logged_diagnostics_artifact"),
            logged_slice_media,
            logged_usage_media,
        ]:
            if artifact_ref is not None:
                artifact_ref.wait()

        api = wandb.Api(overrides={"entity": ENTITY, "project": PROJECT})
        try:
            registry = api.create_registry(name=REGISTRY_NAME, visibility="organization")
            print(f"created registry: {registry.name}")
        except Exception as exc:
            print(f"registry create skipped (may already exist): {exc}")

        best_alias = f"epoch-{best_epoch}"
        best_ckpt_artifact.wait()
        target_path = f"wandb-registry-{REGISTRY_NAME}/{CHECKPOINT_COLLECTION}"
        run.link_artifact(best_ckpt_artifact, target_path=target_path, aliases=["production", best_alias])
        run.summary["model_registry/path"] = target_path
        run.summary["model_registry/aliases"] = ["production", best_alias]
        run.summary["model_registry/promotion_basis"] = "best TinyTransolver composite selection score"
        print(f"linked TinyTransolver to {target_path} with aliases: production, {best_alias}")

        if not baseline_guardrail_passed:
            print("WARNING: pointwise baseline beat TinyTransolver on the composite score; logging baseline artifact as a guardrail champion.")
            baseline_path = tmp_dir / "pointwise_mlp_baseline.pt"
            torch.save(
                {
                    "state_dict": best_baseline_state,
                    "config": {
                        "in_dim": MODEL_CONFIG["in_dim"],
                        "out_dim": MODEL_CONFIG["out_dim"],
                        "hidden": 128,
                        "depth": 4,
                        "dropout": MODEL_CONFIG["dropout"],
                    },
                    "normalization": {
                        "feature_mean": feat_mean.cpu(),
                        "feature_std": feat_std.cpu(),
                        "target_mean": tgt_mean.cpu(),
                        "target_std": tgt_std.cpu(),
                    },
                    "epoch": best_baseline_epoch,
                    **best_baseline_metrics,
                },
                baseline_path,
            )
            baseline_artifact = wandb.Artifact(
                name=BASELINE_ARTIFACT_NAME,
                type="model",
                description="Pointwise MLP baseline guardrail checkpoint for the PhysicsX W&B Models workshop.",
                metadata={"epoch": best_baseline_epoch, **best_baseline_metrics, "n_params": count_params(baseline)},
            )
            baseline_artifact.add_file(str(baseline_path))
            logged_baseline = run.log_artifact(baseline_artifact, aliases=["baseline-best"])
            logged_baseline.wait()
            baseline_target = f"wandb-registry-{REGISTRY_NAME}/{BASELINE_COLLECTION}"
            run.link_artifact(logged_baseline, target_path=baseline_target, aliases=["production", "baseline-best"])
            run.summary["baseline_registry/path"] = baseline_target
            run.summary["baseline_registry/aliases"] = ["production", "baseline-best"]
            print(f"linked baseline guardrail to {baseline_target} with aliases: production, baseline-best")

    # `with` exited -> run.finish() was called automatically.
    return TRAIN_CONFIG
