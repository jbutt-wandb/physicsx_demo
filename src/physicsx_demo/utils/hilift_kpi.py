from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


HILIFT_GEOMETRY_COLUMNS = [
    "IB_Flap_Deflection",
    "OB_Flap_Deflection",
    "IB_Flap_Gap_Multiplier",
    "OB_Flap_Gap_Multiplier",
    "IB_Slat_Deflection",
    "OB_Slat_Deflection",
    "IB_Slat_Gap_Multiplier",
    "OB_Slat_Gap_Multiplier",
]
HILIFT_TARGET_COLUMNS = ["cl", "cd", "cm"]
HILIFT_TOKEN_POSITIONS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class HiLiftKpiStats:
    geom_mean: np.ndarray
    geom_std: np.ndarray
    aoa_mean: float
    aoa_std: float
    target_mean: np.ndarray
    target_std: np.ndarray

    def to_dict(self) -> dict:
        return {
            "geom_mean": self.geom_mean.tolist(),
            "geom_std": self.geom_std.tolist(),
            "aoa_mean": self.aoa_mean,
            "aoa_std": self.aoa_std,
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }


@dataclass(frozen=True)
class HiLiftKpiBundle:
    frame: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    stats: HiLiftKpiStats
    data_dir: Path


def _find_csv(data_dir: Path, filename: str) -> Path:
    direct = data_dir / filename
    if direct.exists():
        return direct
    matches = sorted(data_dir.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {filename} under {data_dir.resolve()}")


def _coerce_target_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for target in HILIFT_TARGET_COLUMNS:
        if target in frame.columns:
            continue
        upper = target.upper()
        title = target.title()
        if upper in frame.columns:
            renamed[upper] = target
        elif title in frame.columns:
            renamed[title] = target
    return frame.rename(columns=renamed)


def _add_geo_num(frame: pd.DataFrame) -> pd.DataFrame:
    if "geo_num" in frame.columns:
        return frame
    if "GeoID" not in frame.columns:
        return frame
    out = frame.copy()
    out["geo_num"] = (
        out["GeoID"].astype(str).str.extract(r"(\d+)", expand=False).astype("float").astype("Int64")
    )
    return out


def load_hilift_kpi_frame(
    data_dir: str | Path,
    *,
    force_file: str | Path | None = None,
    geo_file: str | Path | None = None,
) -> pd.DataFrame:
    """Load and merge HiLiftAeroML coefficient and geometry summary CSVs."""
    root = Path(data_dir)
    force_path = Path(force_file) if force_file is not None else _find_csv(root, "force_mom_all.csv")
    geo_path = Path(geo_file) if geo_file is not None else _find_csv(root, "geo_values_all.csv")

    forces = _coerce_target_columns(pd.read_csv(force_path))
    geometry = pd.read_csv(geo_path)
    if "GeoID" not in forces.columns or "GeoID" not in geometry.columns:
        raise ValueError("HiLiftAeroML summary CSVs must include a GeoID column")
    if "AoA" not in forces.columns:
        raise ValueError("force_mom_all.csv must include an AoA column")

    frame = forces.merge(geometry, on="GeoID", how="inner", suffixes=("", "_geo"))
    frame = _add_geo_num(frame)
    required = ["GeoID", "AoA", *HILIFT_GEOMETRY_COLUMNS, *HILIFT_TARGET_COLUMNS]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"HiLiftAeroML KPI frame is missing columns: {missing}")

    sort_cols = [col for col in ["geo_num", "GeoID", "AoA"] if col in frame.columns]
    frame = frame.sort_values(sort_cols).reset_index(drop=True)
    return frame


def geometry_holdout_split(
    frame: pd.DataFrame,
    *,
    n_geometries: int | None = 80,
    seed: int = 42,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by held-out GeoID, not by rows, so validation/test use unseen variants."""
    if "GeoID" not in frame.columns:
        raise ValueError("frame must include GeoID")
    if not 0.0 < train_fraction < 1.0 or not 0.0 <= val_fraction < 1.0:
        raise ValueError("train_fraction and val_fraction must be in [0, 1]")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")

    geo_order = frame[["GeoID"] + (["geo_num"] if "geo_num" in frame.columns else [])].drop_duplicates()
    if "geo_num" in geo_order.columns:
        geo_order = geo_order.sort_values(["geo_num", "GeoID"])
    else:
        geo_order = geo_order.sort_values("GeoID")
    geos = geo_order["GeoID"].astype(str).to_numpy()
    if n_geometries is not None:
        if n_geometries < 3:
            raise ValueError("n_geometries must be at least 3 for train/val/test splits")
        geos = geos[: min(n_geometries, len(geos))]
    if len(geos) < 3:
        raise ValueError("Need at least three geometries for train/val/test splits")

    shuffled = np.array(geos, copy=True)
    np.random.default_rng(seed).shuffle(shuffled)
    n_train = max(1, int(train_fraction * len(shuffled)))
    n_val = max(1, int(val_fraction * len(shuffled)))
    if n_train + n_val >= len(shuffled):
        n_train = max(1, len(shuffled) - 2)
        n_val = 1

    train_geos = set(shuffled[:n_train])
    val_geos = set(shuffled[n_train : n_train + n_val])
    test_geos = set(shuffled[n_train + n_val :])

    train = frame[frame["GeoID"].astype(str).isin(train_geos)].reset_index(drop=True)
    val = frame[frame["GeoID"].astype(str).isin(val_geos)].reset_index(drop=True)
    test = frame[frame["GeoID"].astype(str).isin(test_geos)].reset_index(drop=True)
    return train, val, test


def fit_kpi_stats(train_frame: pd.DataFrame) -> HiLiftKpiStats:
    geom_std = train_frame[HILIFT_GEOMETRY_COLUMNS].std().replace(0, 1).to_numpy(dtype=np.float32)
    target_std = train_frame[HILIFT_TARGET_COLUMNS].std().replace(0, 1).to_numpy(dtype=np.float32)
    aoa_std = float(train_frame["AoA"].std())
    if not np.isfinite(aoa_std) or aoa_std == 0.0:
        aoa_std = 1.0
    return HiLiftKpiStats(
        geom_mean=train_frame[HILIFT_GEOMETRY_COLUMNS].mean().to_numpy(dtype=np.float32),
        geom_std=geom_std,
        aoa_mean=float(train_frame["AoA"].mean()),
        aoa_std=aoa_std,
        target_mean=train_frame[HILIFT_TARGET_COLUMNS].mean().to_numpy(dtype=np.float32),
        target_std=target_std,
    )


def prepare_hilift_kpi_bundle(
    data_dir: str | Path,
    *,
    n_geometries: int | None = 80,
    seed: int = 42,
) -> HiLiftKpiBundle:
    frame = load_hilift_kpi_frame(data_dir)
    train, val, test = geometry_holdout_split(frame, n_geometries=n_geometries, seed=seed)
    stats = fit_kpi_stats(train)
    return HiLiftKpiBundle(frame=frame, train=train, val=val, test=test, stats=stats, data_dir=Path(data_dir))


class HiLiftKpiDataset(Dataset):
    """Geometry-token dataset for CL/CD/CM prediction from HiLiftAeroML summaries."""

    def __init__(self, frame: pd.DataFrame, stats: HiLiftKpiStats):
        self.frame = frame.reset_index(drop=True)
        self.stats = stats

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[idx]
        geom = row[HILIFT_GEOMETRY_COLUMNS].to_numpy(dtype=np.float32)
        geom_norm = (geom - self.stats.geom_mean) / self.stats.geom_std
        aoa = float(row["AoA"])
        aoa_norm = (aoa - self.stats.aoa_mean) / self.stats.aoa_std
        aoa_rad = np.deg2rad(aoa)
        condition = np.array([aoa_norm, np.sin(aoa_rad), np.cos(aoa_rad)], dtype=np.float32)
        tiled_condition = np.repeat(condition[None, :], len(HILIFT_GEOMETRY_COLUMNS), axis=0)
        tokens = np.concatenate(
            [geom_norm[:, None], HILIFT_TOKEN_POSITIONS, tiled_condition],
            axis=1,
        ).astype(np.float32)
        target = row[HILIFT_TARGET_COLUMNS].to_numpy(dtype=np.float32)
        target_norm = (target - self.stats.target_mean) / self.stats.target_std
        return torch.from_numpy(tokens), torch.from_numpy(target_norm.astype(np.float32))


def denormalize_kpi_targets(values: torch.Tensor | np.ndarray, stats: HiLiftKpiStats) -> np.ndarray:
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    return array * stats.target_std + stats.target_mean


def kpi_frame_summary(frame: pd.DataFrame) -> dict[str, int | float]:
    return {
        "rows": int(len(frame)),
        "geometries": int(frame["GeoID"].nunique()),
        "aoa_min": float(frame["AoA"].min()),
        "aoa_max": float(frame["AoA"].max()),
    }
