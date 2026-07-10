from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbformat.validator import normalize
import wandb


def find_repo_root() -> Path:
    candidates = [Path.cwd(), Path.cwd() / "physicsx_demo", *Path.cwd().parents]
    return next(p for p in candidates if (p / "src" / "physicsx_demo").exists())


REPO_ROOT = find_repo_root()
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from physicsx_demo.utils.env import load_local_env, resolve_wandb_target


REGISTRY_REFS = {
    "dataset": "wandb-registry-airfoil_surrogates/datasets:production",
    "checkpoint": "wandb-registry-airfoil_surrogates/checkpoints:production",
    "tuned": "wandb-registry-airfoil_surrogates/model_hyperparameter_tuned:production",
}


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


def validate_notebooks() -> list[str]:
    issues = []
    for nb_path in sorted((REPO_ROOT / "notebooks").glob("*.ipynb")):
        if nb_path.name.endswith(".executed.ipynb"):
            continue
        try:
            nb = nbformat.read(nb_path, as_version=4)
            _, nb = normalize(nb)
            nbformat.validate(nb)
        except Exception as exc:
            issues.append(f"{nb_path.relative_to(REPO_ROOT)} is invalid: {type(exc).__name__}: {exc}")
    return issues


def main() -> None:
    os.environ.setdefault("WANDB_SILENT", "true")
    load_local_env()
    entity, project = resolve_wandb_target()
    path = f"{entity}/{project}"
    api = wandb.Api(overrides={"entity": entity, "project": project})

    blockers: list[str] = []
    warnings: list[str] = []

    print(f"Workshop audit for {path}")
    print()

    notebook_issues = validate_notebooks()
    if notebook_issues:
        blockers.extend(notebook_issues)
    else:
        print("notebooks: valid")

    artifacts = {}
    logged_by_runs = {}
    for label, ref in REGISTRY_REFS.items():
        try:
            art = api.artifact(ref)
            artifacts[label] = art
            logged_by = art.logged_by()
            logged_by_runs[label] = logged_by
            print(
                f"{label}: {art.name} {art.version} updated={art.updated_at} "
                f"logged_by={logged_by.name} ({logged_by.id})"
            )
        except Exception as exc:
            blockers.append(f"missing Registry artifact {ref}: {type(exc).__name__}: {exc}")

    latest_training = next(
        iter(api.runs(path, filters={"display_name": "tiny-transolver-train", "state": "finished"}, order="-created_at", per_page=1)),
        None,
    )
    if latest_training is None:
        blockers.append("no finished tiny-transolver-train run found")
    else:
        model_score = finite_summary_float(latest_training, "model/best_selection_score")
        baseline_score = finite_summary_float(latest_training, "baseline/best_selection_score")
        guardrail = latest_training.summary.get("model/baseline_guardrail_passed")
        print(
            f"latest training: {latest_training.id} created={latest_training.created_at} "
            f"score={model_score:.6f} baseline={baseline_score:.6f} guardrail={guardrail}"
        )
        if guardrail is not True or not model_score <= baseline_score:
            blockers.append("latest training run does not pass the baseline guardrail")

    if latest_training is not None and "tuned" in logged_by_runs:
        latest_training_time = parse_wandb_time(latest_training.created_at)
        tuned_logged_by_time = parse_wandb_time(logged_by_runs["tuned"].created_at)
        if tuned_logged_by_time < latest_training_time:
            blockers.append("tuned production promotion predates the latest successful training run")
        tuned_metadata = dict(artifacts["tuned"].metadata or {})
        if "val_selection_score" not in tuned_metadata:
            blockers.append("tuned production artifact is missing val_selection_score metadata")

    reports = list(api.reports(path=path, per_page=10))
    if not reports:
        warnings.append("no W&B Report found for the target project")
    else:
        newest_report = max(reports, key=lambda report: parse_wandb_time(getattr(report, "updated_at", None)))
        print(f"newest report: updated={newest_report.updated_at} url={newest_report.url}")
        if latest_training is not None:
            report_time = parse_wandb_time(newest_report.updated_at)
            latest_training_time = parse_wandb_time(latest_training.created_at)
            if report_time < latest_training_time:
                warnings.append("newest W&B Report is older than the latest successful training run")

    print()
    if blockers:
        print("VERDICT: NO-GO")
        for item in blockers:
            print(f"- BLOCKER: {item}")
        for item in warnings:
            print(f"- WARNING: {item}")
        raise SystemExit(1)

    if warnings:
        print("VERDICT: GREEN WITH NOTES")
        for item in warnings:
            print(f"- WARNING: {item}")
    else:
        print("VERDICT: GREENLIGHT")


if __name__ == "__main__":
    main()
