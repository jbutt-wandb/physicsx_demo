from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    "dataset": "wandb-registry-high_lift_surrogates/datasets:production",
    "checkpoint": "wandb-registry-high_lift_surrogates/checkpoints:production",
    "tuned": "wandb-registry-high_lift_surrogates/model_hyperparameter_tuned:production",
    "baseline": "wandb-registry-high_lift_surrogates/baselines:production",
}

MEDIA_PREFIXES = ("media/images", "media/videos", "media/tables")
IMPORTANT_SUMMARY_KEYS = [
    "model/best_selection_score",
    "baseline/best_selection_score",
    "model/baseline_guardrail_passed",
    "model/best_epoch",
    "test/mse",
    "test/cl_rmse",
    "test/cd_rmse",
    "test/cm_rmse",
    "baseline/test_mse",
    "baseline/test_cl_rmse",
    "baseline/test_cd_rmse",
    "baseline/test_cm_rmse",
]


def parse_wandb_time(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def file_kind(name: str) -> str:
    for prefix in MEDIA_PREFIXES:
        if name.startswith(prefix):
            return prefix
    return "other"


def summarize_run(run: Any) -> dict[str, Any]:
    files = list(run.files())
    file_counts = Counter(file_kind(file.name) for file in files)
    media_files = [
        file.name
        for file in files
        if file.name.startswith(MEDIA_PREFIXES)
    ]
    summary = dict(run.summary)
    important_summary = {
        key: jsonable(summary.get(key))
        for key in IMPORTANT_SUMMARY_KEYS
        if key in summary
    }
    media_summary_keys = sorted(
        key
        for key in summary
        if key.startswith(("dataset/", "val/", "model/kpi_", "model/token_", "model/profile_"))
    )
    table_summary_keys = sorted(key for key in media_summary_keys if "table" in key.lower())

    logged_artifacts = []
    for artifact in run.logged_artifacts():
        logged_artifacts.append(
            {
                "name": artifact.name,
                "type": artifact.type,
                "version": artifact.version,
                "aliases": sorted(alias for alias in artifact.aliases),
                "metadata": {key: jsonable(value) for key, value in dict(artifact.metadata or {}).items()},
            }
        )

    return {
        "id": run.id,
        "name": run.name,
        "display_name": run.display_name,
        "url": run.url,
        "state": run.state,
        "created_at": run.created_at,
        "updated_at": getattr(run, "updated_at", None),
        "tags": list(run.tags or []),
        "important_summary": important_summary,
        "summary_key_count": len(summary),
        "media_summary_key_count": len(media_summary_keys),
        "media_summary_keys": media_summary_keys,
        "table_summary_key_count": len(table_summary_keys),
        "table_summary_keys": table_summary_keys,
        "file_count": len(files),
        "file_counts": dict(file_counts),
        "media_files": sorted(media_files),
        "logged_artifacts": logged_artifacts,
    }


def summarize_registry(api: wandb.Api) -> list[dict[str, Any]]:
    rows = []
    for label, ref in REGISTRY_REFS.items():
        try:
            artifact = api.artifact(ref)
            logged_by = artifact.logged_by()
            rows.append(
                {
                    "label": label,
                    "ref": ref,
                    "status": "found",
                    "name": artifact.name,
                    "type": artifact.type,
                    "version": artifact.version,
                    "aliases": sorted(alias for alias in artifact.aliases),
                    "updated_at": artifact.updated_at,
                    "metadata": {key: jsonable(value) for key, value in dict(artifact.metadata or {}).items()},
                    "logged_by": {
                        "id": logged_by.id,
                        "name": logged_by.name,
                        "display_name": logged_by.display_name,
                        "created_at": logged_by.created_at,
                        "url": logged_by.url,
                    },
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "label": label,
                    "ref": ref,
                    "status": "missing",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def panel_to_dict(panel: Any) -> dict[str, Any]:
    panel_dict = {
        "type": type(panel).__name__,
        "title": getattr(panel, "title", None),
    }
    for attr in ["x", "y", "metrics", "media_keys", "table_name", "query"]:
        if hasattr(panel, attr):
            panel_dict[attr] = jsonable(getattr(panel, attr))
    return panel_dict


def summarize_workspace(workspace_url: str | None) -> dict[str, Any] | None:
    if not workspace_url:
        return None
    try:
        import wandb_workspaces.workspaces as ws

        workspace = ws.Workspace.from_url(workspace_url)
        sections = []
        for section in getattr(workspace, "sections", []) or []:
            panels = [panel_to_dict(panel) for panel in getattr(section, "panels", []) or []]
            sections.append(
                {
                    "name": getattr(section, "name", None),
                    "panel_count": len(panels),
                    "panels": panels,
                }
            )
        return {
            "url": workspace_url,
            "name": getattr(workspace, "name", None),
            "entity": getattr(workspace, "entity", None),
            "project": getattr(workspace, "project", None),
            "section_count": len(sections),
            "sections": sections,
            "runset_settings": jsonable(getattr(workspace, "runset_settings", None)),
        }
    except Exception as exc:
        return {"url": workspace_url, "error": f"{type(exc).__name__}: {exc}"}


def compute_analysis(snapshot: dict[str, Any]) -> dict[str, Any]:
    runs = snapshot["runs"]
    latest = runs[0] if runs else None
    best = None
    scored_runs = []
    for run in runs:
        score = float_or_none(run["important_summary"].get("model/best_selection_score"))
        if score is not None:
            scored_runs.append((score, run))
    if scored_runs:
        best = min(scored_runs, key=lambda item: item[0])[1]

    media_progression = [
        {
            "id": run["id"],
            "created_at": run["created_at"],
            "images": run["file_counts"].get("media/images", 0),
            "tables": max(run["file_counts"].get("media/tables", 0), run.get("table_summary_key_count", 0)),
            "videos": run["file_counts"].get("media/videos", 0),
            "media_summary_keys": run["media_summary_key_count"],
        }
        for run in sorted(runs, key=lambda item: parse_wandb_time(item["created_at"]))
    ]

    registry_findings = []
    for row in snapshot["registry"]:
        if row["status"] != "found":
            registry_findings.append(f"{row['label']} Registry alias is missing")
            continue
        if row["label"] == "checkpoint" and latest:
            logged_by_id = row.get("logged_by", {}).get("id")
            if logged_by_id != latest["id"]:
                registry_findings.append(
                    "checkpoint production Registry alias resolves to a run other than the latest visual-rich run"
                )
            if best:
                checkpoint_score = float_or_none((row.get("metadata") or {}).get("val_mse"))
                best_score = float_or_none(best["important_summary"].get("model/best_selection_score"))
                if checkpoint_score is not None and best_score is not None and checkpoint_score > best_score:
                    registry_findings.append(
                        "checkpoint production Registry alias is not the historical best validation checkpoint"
                    )
    if latest:
        latest_summary = latest["important_summary"]
        model_mse = float_or_none(latest_summary.get("test/mse"))
        baseline_mse = float_or_none(latest_summary.get("baseline/test_mse"))
        if model_mse is not None and baseline_mse is not None and model_mse < baseline_mse:
            quality = "model beats baseline on test MSE"
        else:
            quality = "model-vs-baseline test result needs attention"
    else:
        quality = "no finished KPI training runs found"

    workspace = snapshot.get("workspace") or {}
    workspace_sections = [section.get("name") for section in workspace.get("sections", [])]
    workspace_has_slice = any(
        "slice" in json.dumps(section).lower() or "profile" in json.dumps(section).lower()
        for section in workspace.get("sections", [])
    )

    return {
        "latest_run_id": latest["id"] if latest else None,
        "best_run_id": best["id"] if best else None,
        "quality": quality,
        "media_progression": media_progression,
        "registry_findings": registry_findings,
        "workspace_sections": workspace_sections,
        "workspace_has_profile_slice_panels": workspace_has_slice,
    }


def write_markdown(snapshot: dict[str, Any], analysis: dict[str, Any], path: Path) -> None:
    latest = snapshot["runs"][0] if snapshot["runs"] else None
    best = next((run for run in snapshot["runs"] if run["id"] == analysis["best_run_id"]), None)

    lines = [
        "# W&B Workshop State Analysis",
        "",
        f"- Generated: `{snapshot['generated_at']}`",
        f"- Project: `{snapshot['project_path']}`",
        f"- Workspace: {snapshot.get('workspace_url') or 'not provided'}",
        "",
        "## Executive Readout",
        "",
        f"- Latest run: `{analysis['latest_run_id']}`",
        f"- Best run by `model/best_selection_score`: `{analysis['best_run_id']}`",
        f"- Quality signal: {analysis['quality']}",
        f"- Workspace has profile/slice panels: `{analysis['workspace_has_profile_slice_panels']}`",
        "",
    ]

    if latest:
        lines.extend(
            [
                "## Latest Run",
                "",
                f"- URL: {latest['url']}",
                f"- Created: `{latest['created_at']}`",
                f"- Files: `{latest['file_count']}` total, "
                f"`{latest['file_counts'].get('media/images', 0)}` images, "
                f"`{max(latest['file_counts'].get('media/tables', 0), latest.get('table_summary_key_count', 0))}` table keys, "
                f"`{latest['file_counts'].get('media/videos', 0)}` videos",
                f"- Media summary keys: `{latest['media_summary_key_count']}`",
                "",
                "| Metric | Value |",
                "|---|---:|",
            ]
        )
        for key, value in latest["important_summary"].items():
            lines.append(f"| `{key}` | `{value}` |")
        lines.append("")

    if best and latest and best["id"] != latest["id"]:
        lines.extend(
            [
                "## Best Run",
                "",
                f"- URL: {best['url']}",
                f"- `model/best_selection_score`: `{best['important_summary'].get('model/best_selection_score')}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Media Progression",
            "",
            "| Run | Created | Images | Tables | Videos | Media Keys |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["media_progression"]:
        lines.append(
            f"| `{row['id']}` | `{row['created_at']}` | {row['images']} | "
            f"{row['tables']} | {row['videos']} | {row['media_summary_keys']} |"
        )
    lines.append("")

    lines.extend(["## Registry", ""])
    for row in snapshot["registry"]:
        if row["status"] == "found":
            logged_by = row.get("logged_by", {})
            lines.append(
                f"- `{row['label']}`: `{row['name']}` `{row['version']}`, "
                f"aliases `{row['aliases']}`, logged by `{logged_by.get('id')}`"
            )
        else:
            lines.append(f"- `{row['label']}`: missing, `{row['error']}`")
    lines.append("")

    if analysis["registry_findings"]:
        lines.extend(["## Issues", ""])
        for finding in analysis["registry_findings"]:
            lines.append(f"- {finding}")
        lines.append("")

    workspace = snapshot.get("workspace") or {}
    lines.extend(["## Workspace", ""])
    if workspace.get("error"):
        lines.append(f"- Could not load workspace: `{workspace['error']}`")
    else:
        lines.append(f"- Name: `{workspace.get('name')}`")
        for section in workspace.get("sections", []):
            lines.append(f"- Section `{section.get('name')}`: `{section.get('panel_count')}` panels")
    lines.append("")

    lines.extend(
        [
            "## Recommendation",
            "",
            "Use the latest visual-rich run for the live workshop. Relink the checkpoint Registry production alias "
            "from that run so Registry lineage, workspace panels, and reports all tell the same story. Keep the "
            "profile/slice visuals, but describe them as design-variable-derived visual proxies rather than exact CAD.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-url", default=os.environ.get("WANDB_WORKSPACE_URL"))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "analysis" / "wandb_state")
    args = parser.parse_args()

    os.environ.setdefault("WANDB_SILENT", "true")
    load_local_env()
    entity, project = resolve_wandb_target()
    project_path = f"{entity}/{project}"
    api = wandb.Api(overrides={"entity": entity, "project": project})

    filters = {"display_name": "tiny-transolver-kpi-train", "state": "finished"}
    runs = [
        summarize_run(run)
        for run in api.runs(project_path, filters=filters, order="-created_at", per_page=args.limit)
    ]
    reports = [
        {
            "display_name": getattr(report, "display_name", None),
            "updated_at": getattr(report, "updated_at", None),
            "url": getattr(report, "url", None),
        }
        for report in api.reports(path=project_path, per_page=20)
    ]
    reports.sort(key=lambda item: parse_wandb_time(item["updated_at"]), reverse=True)

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_path": project_path,
        "workspace_url": args.workspace_url,
        "filters": filters,
        "runs": runs,
        "registry": summarize_registry(api),
        "reports": reports,
        "workspace": summarize_workspace(args.workspace_url),
    }
    analysis = compute_analysis(snapshot)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.out_dir / "wandb_state_snapshot.json"
    analysis_path = args.out_dir / "wandb_state_analysis.json"
    markdown_path = args.out_dir / "wandb_state_analysis.md"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(snapshot, analysis, markdown_path)

    print(f"wrote {snapshot_path}")
    print(f"wrote {analysis_path}")
    print(f"wrote {markdown_path}")
    if analysis["registry_findings"]:
        print("registry findings:")
        for finding in analysis["registry_findings"]:
            print(f"- {finding}")


if __name__ == "__main__":
    main()
