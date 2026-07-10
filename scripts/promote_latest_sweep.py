from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import wandb


def find_repo_root() -> Path:
    candidates = [Path.cwd(), Path.cwd() / "physicsx_demo", *Path.cwd().parents]
    return next(p for p in candidates if (p / "src" / "physicsx_demo").exists())


REPO_ROOT = find_repo_root()
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from physicsx_demo.utils.env import load_local_env, resolve_wandb_target


REGISTRY_NAME = "airfoil_surrogates"
TUNED_COLLECTION = "model_hyperparameter_tuned"
CHECKPOINT_COLLECTION = "checkpoints"


def parse_wandb_time(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def finite_summary_float(run, key: str) -> float:
    value = run.summary.get(key, float("inf"))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return value if math.isfinite(value) else float("inf")


def run_sweep_id(run) -> str:
    sweep = getattr(run, "sweep", None)
    if sweep is None:
        return "unknown"
    return getattr(sweep, "id", None) or getattr(sweep, "name", None) or "unknown"


def find_latest_sweep_setup(api: wandb.Api, path: str):
    setup_runs = list(
        api.runs(
            path,
            filters={"display_name": "sweep-setup", "tags": "sweep", "state": "finished"},
            order="-created_at",
            per_page=20,
        )
    )
    return setup_runs[0] if setup_runs else None


def find_best_candidate(api: wandb.Api, path: str, *, since: datetime | None):
    runs = list(
        api.runs(
            path,
            filters={"tags": "sweep", "state": "finished"},
            order="-created_at",
            per_page=100,
        )
    )
    candidates = []
    for run in runs:
        artifact_ref = run.summary.get("model_checkpoint/artifact")
        selection_score = finite_summary_float(run, "val/selection_score")
        pressure_rmse = finite_summary_float(run, "val/pressure_rmse")
        created_at = parse_wandb_time(run.created_at)
        if not artifact_ref or not math.isfinite(selection_score):
            continue
        if since is not None and created_at < since:
            continue
        candidates.append((selection_score, pressure_rmse, created_at, run, artifact_ref))

    if not candidates:
        raise RuntimeError("No viable finished sweep candidate runs found.")
    return min(candidates, key=lambda item: item[0]), candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Promote the best viable checkpoint from the latest sweep batch into the W&B Models Registry, "
            "ranked by val/selection_score."
        )
    )
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--registry-name", default=REGISTRY_NAME)
    parser.add_argument("--tuned-collection", default=TUNED_COLLECTION)
    parser.add_argument(
        "--all-sweeps",
        action="store_true",
        help="Consider all viable sweep candidates instead of only runs created after the latest sweep-setup run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected candidate without linking it into the Registry.",
    )
    args = parser.parse_args()

    os.environ.setdefault("WANDB_SILENT", "true")
    load_local_env()
    default_entity, default_project = resolve_wandb_target()
    entity = args.entity or default_entity
    project = args.project or default_project
    path = f"{entity}/{project}"

    api = wandb.Api(overrides={"entity": entity, "project": project})
    setup = None if args.all_sweeps else find_latest_sweep_setup(api, path)
    since = None if setup is None else parse_wandb_time(setup.created_at)
    (best_score, best_rmse, _, best_run, artifact_ref), candidates = find_best_candidate(api, path, since=since)

    source_desc = "all sweep history" if since is None else f"runs after latest sweep-setup {setup.id}"
    print(f"target project: {path}")
    print(f"candidate window: {source_desc}")
    print(f"viable candidates: {len(candidates)}")
    print(f"winner: {best_run.name} ({best_run.id})")
    print(f"winner sweep: {run_sweep_id(best_run)}")
    print(f"val/selection_score: {best_score:.6f}")
    print(f"val/pressure_rmse: {best_rmse:.6f}")
    print(f"artifact: {artifact_ref}")

    if args.dry_run:
        return

    target_path = f"wandb-registry-{args.registry_name}/{args.tuned_collection}"
    aliases = ["production", "sweep-winner", f"sweep-{run_sweep_id(best_run)}", f"run-{best_run.id}"]

    with wandb.init(
        entity=entity,
        project=project,
        job_type="sweep-promotion",
        name="sweep-winner-promotion",
        tags=["notebook", "sweep", "promotion", "workshop-fix"],
    ) as promo_run:
        source_artifact = promo_run.use_artifact(artifact_ref)
        promo_run.link_artifact(source_artifact, target_path=target_path, aliases=aliases)
        promo_run.summary["sweep/winner_run"] = best_run.name
        promo_run.summary["sweep/winner_run_id"] = best_run.id
        promo_run.summary["sweep/winner_selection_score"] = best_score
        promo_run.summary["sweep/winner_pressure_rmse"] = best_rmse
        promo_run.summary["sweep/winner_artifact"] = artifact_ref
        promo_run.summary["sweep/candidate_window"] = source_desc
        promo_run.summary["model_registry/path"] = target_path
        promo_run.summary["model_registry/aliases"] = aliases

    print(f"linked -> {target_path}")
    print(f"aliases: {', '.join(aliases)}")


if __name__ == "__main__":
    main()
