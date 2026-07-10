from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import wandb


def find_repo_root() -> Path:
    candidates = [Path.cwd(), Path.cwd() / "physicsx_demo", *Path.cwd().parents]
    return next(p for p in candidates if (p / "src" / "physicsx_demo").exists())


REPO_ROOT = find_repo_root()
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from physicsx_demo.utils.env import configure_matplotlib, load_local_env, resolve_wandb_target

configure_matplotlib()

from physicsx_demo.utils.kpi_training import run_kpi_training_experiment


def resolve_dataset_dir(api: wandb.Api, dataset_registry_path: str, local_data_dir: Path) -> Path:
    try:
        artifact = api.artifact(dataset_registry_path)
        return Path(artifact.download())
    except Exception:
        if (local_data_dir / "force_mom_all.csv").exists() and (local_data_dir / "geo_values_all.csv").exists():
            return local_data_dir
        raise


def save_workspace(entity: str, project: str) -> str:
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    workspace = ws.Workspace(
        entity=entity,
        project=project,
        name="PhysicsX Workshop - Real HiLift KPI",
        settings=ws.WorkspaceSettings(
            x_axis="epoch",
            smoothing_type="exponential",
            smoothing_weight=20,
            max_runs=12,
        ),
        runset_settings=ws.RunsetSettings(
            query="tiny-transolver-kpi",
            pinned_columns=[
                "State",
                "Created",
                "summary.model/best_selection_score",
                "summary.baseline/best_selection_score",
                "summary.test/mse",
            ],
        ),
        auto_generate_panels=False,
        sections=[
            ws.Section(
                name="Dataset Visuals",
                is_open=True,
                panels=[
                    wr.MediaBrowser(title="Coefficient trends", media_keys=["dataset/coefficient_trends"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="Design variables", media_keys=["dataset/design_variable_heatmap"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="Semantic tokens", media_keys=["dataset/semantic_token_map"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="High-lift profiles", media_keys=["dataset/high_lift_profile_gallery"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="Profile matrix", media_keys=["dataset/high_lift_profile_matrix"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="Profile overlay", media_keys=["dataset/high_lift_profile_overlay"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="Featured profile card", media_keys=["dataset/featured_profile_card"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="Profile flythrough video", media_keys=["dataset/profile_flythrough_video"], num_columns=1, mode="gallery"),
                ],
            ),
            ws.Section(
                name="Dataset Tables",
                is_open=True,
                panels=[
                    wr.WeavePanelSummaryTable(table_name="dataset/preview_table"),
                    wr.WeavePanelSummaryTable(table_name="dataset/design_variable_table"),
                    wr.WeavePanelSummaryTable(table_name="dataset/geometry_summary_table"),
                    wr.WeavePanelSummaryTable(table_name="dataset/profile_card_table"),
                ],
            ),
            ws.Section(
                name="Model vs Baseline Comparisons",
                is_open=True,
                panels=[
                    wr.MediaBrowser(title="Comparison scatter", media_keys=["val/model_baseline_scatter_latest"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="Comparison drag polar", media_keys=["val/model_baseline_drag_polar_latest"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="Comparison coefficient curves", media_keys=["val/model_baseline_curves_latest"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="Model minus baseline error", media_keys=["val/model_baseline_error_delta_latest"], num_columns=1, mode="gallery"),
                ],
            ),
            ws.Section(
                name="Prediction Evolution Videos",
                is_open=True,
                panels=[
                    wr.MediaBrowser(title="Coefficient curve evolution", media_keys=["val/model_baseline_curves_video"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="Error delta evolution", media_keys=["val/model_baseline_error_delta_video"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="Drag polar evolution", media_keys=["val/drag_polar_evolution_video"], num_columns=1, mode="gallery"),
                ],
            ),
            ws.Section(
                name="KPI RMSE",
                is_open=True,
                panels=[
                    wr.LinePlot(title="CL RMSE", x="epoch", y=["val/cl_rmse", "baseline/val_cl_rmse"]),
                    wr.LinePlot(title="CD RMSE", x="epoch", y=["val/cd_rmse", "baseline/val_cd_rmse"]),
                    wr.LinePlot(title="CM RMSE", x="epoch", y=["val/cm_rmse", "baseline/val_cm_rmse"]),
                ],
            ),
            ws.Section(
                name="Model Selection",
                is_open=True,
                panels=[
                    wr.LinePlot(title="Validation MSE", x="epoch", y=["val/mse", "baseline/val_mse"]),
                    wr.LinePlot(title="Model margin vs baseline", x="epoch", y=["model/selection_margin_vs_baseline"]),
                ],
            ),
            ws.Section(
                name="Profiles and Slice Attribution",
                is_open=True,
                panels=[
                    wr.MediaBrowser(title="Profile-linked slice attribution video", media_keys=["model/kpi_profile_slice_video"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="Slice attribution video", media_keys=["model/kpi_slice_attribution_video"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="Profile-linked attribution latest", media_keys=["model/kpi_profile_slice_latest"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="Token slice usage latest", media_keys=["model/kpi_token_slice_usage_latest"], num_columns=1, mode="gallery"),
                ],
            ),
            ws.Section(
                name="Prediction Media",
                is_open=True,
                panels=[
                    wr.MediaBrowser(title="Predicted vs true", media_keys=["val/predicted_vs_true_latest"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="Drag polar", media_keys=["val/drag_polar_latest"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="Coefficient curves", media_keys=["val/coefficient_curves_latest"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="Error heatmap", media_keys=["val/error_heatmap_latest"], num_columns=1, mode="gallery"),
                    wr.MediaBrowser(title="Residuals", media_keys=["val/residual_hist_latest"], num_columns=1, mode="gallery"),
                ],
            ),
        ],
    )
    workspace.save()
    return workspace.url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--n-geometries", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--registry-name", default="high_lift_surrogates")
    parser.add_argument("--save-workspace", action="store_true")
    parser.add_argument("--local-data-dir", type=Path, default=REPO_ROOT.parent / "data" / "hiliftaeroml_subset")
    args = parser.parse_args()

    os.environ.setdefault("WANDB_SILENT", "true")
    load_local_env()
    entity, project = resolve_wandb_target()
    api = wandb.Api(overrides={"entity": entity, "project": project})
    dataset_registry_path = f"wandb-registry-{args.registry_name}/datasets:production"
    dataset_dir = resolve_dataset_dir(api, dataset_registry_path, args.local_data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_config = run_kpi_training_experiment(
        entity=entity,
        project=project,
        registry_name=args.registry_name,
        dataset_registry_path=dataset_registry_path,
        dataset_dir=dataset_dir,
        n_geometries=args.n_geometries,
        seed=args.seed,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"finished training: best_epoch={train_config.get('best_epoch')} best_val_mse={train_config.get('best_val_mse')}")
    if args.save_workspace:
        print(f"curated workspace: {save_workspace(entity, project)}")


if __name__ == "__main__":
    main()
