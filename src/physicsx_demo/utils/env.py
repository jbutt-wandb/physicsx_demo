from __future__ import annotations

import os
import tempfile
from pathlib import Path


def load_local_env(filename: str = ".env") -> None:
    """Load simple KEY=VALUE lines from the nearest .env without echoing secrets."""
    for directory in [Path.cwd(), *Path.cwd().parents]:
        env_path = directory / filename
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        return


def configure_matplotlib() -> None:
    """Use a headless Matplotlib backend that works in notebooks and W&B media."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")


def resolve_wandb_target(default_entity: str = "wandb_emea", default_project: str = "jb_physics_demo") -> tuple[str, str]:
    """Resolve W&B entity/project and tolerate the common entity/project footgun."""
    entity = os.getenv("WANDB_ENTITY", default_entity)
    project = os.getenv("WANDB_PROJECT", default_project)
    if "/" in project:
        project = project.split("/")[-1]
    return entity, project

