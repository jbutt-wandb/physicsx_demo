from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class HighLiftCase:
    points: np.ndarray
    sdf: np.ndarray
    aoa: float
    re: float
    pressure: np.ndarray
    velocity: np.ndarray
    main_camber: float | None = None
    main_thickness: float | None = None
    slat_deflection: float | None = None
    flap_deflection: float | None = None
    flap_gap: float | None = None

    def features(self) -> np.ndarray:
        """Per-point input features: (x, y, sdf, aoa, log_re)."""
        n = self.points.shape[0]
        aoa_col = np.full((n, 1), self.aoa, dtype=np.float32)
        re_col = np.full((n, 1), np.log10(self.re) - 6.0, dtype=np.float32)
        return np.concatenate(
            [
                self.points.astype(np.float32),
                self.sdf[:, None].astype(np.float32),
                aoa_col,
                re_col,
            ],
            axis=1,
        )

    def targets(self) -> np.ndarray:
        """Per-point targets: (pressure, u, v)."""
        return np.concatenate([self.pressure[:, None], self.velocity], axis=1).astype(np.float32)


# Backwards-compatible name for existing notebooks and artifacts.
AirfoilCase = HighLiftCase


def naca4_camber(x: np.ndarray, m: float, p: float) -> np.ndarray:
    yc = np.zeros_like(x)
    if p > 0:
        mask = x < p
        yc[mask] = (m / p**2) * (2 * p * x[mask] - x[mask] ** 2)
        yc[~mask] = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x[~mask] - x[~mask] ** 2)
    return yc


def naca4_thickness(x: np.ndarray, t: float) -> np.ndarray:
    return (t / 0.2) * (
        0.2969 * np.sqrt(np.clip(x, 1e-6, None))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )


def _transform_element(points: np.ndarray, chord: float, origin: tuple[float, float], angle_rad: float) -> np.ndarray:
    scaled = points * np.array([chord, chord], dtype=np.float64)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return scaled @ rot.T + np.array(origin, dtype=np.float64)


def _element_surface(
    *,
    chord: float,
    origin: tuple[float, float],
    angle_rad: float,
    camber: float,
    thickness: float,
    n: int = 220,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, n)
    yc = naca4_camber(x, camber, 0.42)
    yt = naca4_thickness(x, thickness)
    upper_local = np.stack([x, yc + yt], axis=1)
    lower_local = np.stack([x, yc - yt], axis=1)
    upper = _transform_element(upper_local, chord, origin, angle_rad)
    lower = _transform_element(lower_local, chord, origin, angle_rad)
    surface = np.concatenate([upper, lower[::-1]], axis=0)
    return upper.astype(np.float32), lower.astype(np.float32), surface.astype(np.float32)


def high_lift_surfaces(case: HighLiftCase | None = None, **params: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return upper/lower curves for a simplified CRM-HL-style section."""
    main_camber = float(params.get("main_camber", getattr(case, "main_camber", 0.035) or 0.035))
    main_thickness = float(params.get("main_thickness", getattr(case, "main_thickness", 0.125) or 0.125))
    slat_deflection = float(params.get("slat_deflection", getattr(case, "slat_deflection", np.deg2rad(18.0)) or np.deg2rad(18.0)))
    flap_deflection = float(params.get("flap_deflection", getattr(case, "flap_deflection", np.deg2rad(26.0)) or np.deg2rad(26.0)))
    flap_gap = float(params.get("flap_gap", getattr(case, "flap_gap", 0.045) or 0.045))

    slat_upper, slat_lower, _ = _element_surface(
        chord=0.23,
        origin=(-0.15, 0.035 + 0.20 * flap_gap),
        angle_rad=np.deg2rad(7.0) + 0.22 * slat_deflection,
        camber=0.035,
        thickness=0.105,
    )
    main_upper, main_lower, _ = _element_surface(
        chord=0.88,
        origin=(0.02, 0.0),
        angle_rad=0.0,
        camber=main_camber,
        thickness=main_thickness,
    )
    flap_upper, flap_lower, _ = _element_surface(
        chord=0.36,
        origin=(0.73, -0.018 - flap_gap),
        angle_rad=-flap_deflection,
        camber=main_camber + 0.018,
        thickness=max(0.075, main_thickness - 0.025),
    )
    return {
        "slat": (slat_upper, slat_lower),
        "main": (main_upper, main_lower),
        "flap": (flap_upper, flap_lower),
    }


def _stack_surface_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    return np.concatenate([np.concatenate([upper, lower[::-1]], axis=0) for upper, lower in curves.values()], axis=0)


def sample_high_lift_case(rng: np.random.Generator, n_points: int) -> HighLiftCase:
    main_camber = rng.uniform(0.018, 0.052)
    main_thickness = rng.uniform(0.105, 0.155)
    slat_deflection = rng.uniform(12.0, 24.0) * np.pi / 180.0
    flap_deflection = rng.uniform(18.0, 38.0) * np.pi / 180.0
    flap_gap = rng.uniform(0.025, 0.075)
    aoa = rng.uniform(-2.0, 16.0) * np.pi / 180.0
    re = 10 ** rng.uniform(6.0, 7.2)

    pts = rng.uniform([-0.35, -0.85], [1.35, 0.75], size=(n_points, 2)).astype(np.float32)

    curves = high_lift_surfaces(
        main_camber=main_camber,
        main_thickness=main_thickness,
        slat_deflection=slat_deflection,
        flap_deflection=flap_deflection,
        flap_gap=flap_gap,
    )
    surface = _stack_surface_curves(curves)
    dists = np.linalg.norm(pts[:, None, :] - surface[None, :, :], axis=-1)
    sdf = dists.min(axis=1).astype(np.float32)

    x = pts[:, 0]
    y = pts[:, 1]
    rel = pts - np.array([0.48, -0.02], dtype=np.float32)
    r = np.maximum(np.linalg.norm(rel, axis=1), 0.12)
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    u_inf = np.cos(aoa)
    v_inf = np.sin(aoa)
    re_scaled = np.log10(re) - 6.0

    flap_norm = (flap_deflection - np.deg2rad(18.0)) / np.deg2rad(20.0)
    slat_norm = (slat_deflection - np.deg2rad(12.0)) / np.deg2rad(12.0)
    camber_norm = (main_camber - 0.035) / 0.017
    thickness_norm = (main_thickness - 0.13) / 0.025
    gap_norm = (flap_gap - 0.05) / 0.025
    high_lift_index = 0.52 * flap_norm + 0.26 * slat_norm + 0.20 * camber_norm - 0.10 * gap_norm

    circulation = -2 * np.pi * (np.sin(aoa) + 0.12 + 0.12 * high_lift_index) * (1.0 + 0.08 * re_scaled)
    doublet = 0.045 * (1.0 + 0.35 * thickness_norm)
    u = u_inf - doublet * np.cos(2 * theta) / r**2 - circulation * np.sin(theta) / (2 * np.pi * r)
    v = v_inf - doublet * np.sin(2 * theta) / r**2 + circulation * np.cos(theta) / (2 * np.pi * r)

    downstream = 1.0 / (1.0 + np.exp(-12.0 * (x - 0.66)))
    wake_center = -0.05 - 0.15 * flap_norm + 0.08 * np.sin(aoa) + 0.07 * high_lift_index + 0.06 * (x - 0.75)
    wake_width = 0.055 + 0.06 * np.clip(x - 0.55, 0.0, None) + 0.018 * abs(flap_norm)
    wake_strength = np.clip(0.22 + 0.12 * flap_norm + 0.08 * abs(high_lift_index) + 0.04 * re_scaled, 0.08, 0.62)
    wake = downstream * np.exp(-((y - wake_center) ** 2) / (2 * wake_width**2))
    u -= wake_strength * wake
    v += 0.32 * wake_strength * (y - wake_center) / (wake_width + 1e-4) * wake

    slat_suction = np.exp(-(((x + 0.08) / 0.13) ** 2 + ((y - 0.055) / 0.10) ** 2))
    main_suction = np.exp(-(((x - 0.22) / 0.24) ** 2 + ((y - 0.085) / 0.13) ** 2))
    flap_suction = np.exp(-(((x - 0.88) / 0.18) ** 2 + ((y + 0.10 + 0.12 * flap_norm) / 0.13) ** 2))
    recovery = (
        0.11 * high_lift_index * downstream
        - 0.30 * slat_suction * (0.5 + 0.5 * slat_norm)
        - 0.34 * main_suction * (0.55 + 0.45 * np.sin(aoa + 0.1))
        - 0.26 * flap_suction * (0.55 + 0.45 * flap_norm)
    )

    u = np.clip(u, -3.0, 3.0)
    v = np.clip(v, -3.0, 3.0)
    pressure = np.clip(1.0 - (u**2 + v**2) + recovery, -3.2, 1.1)

    return HighLiftCase(
        points=pts,
        sdf=sdf,
        aoa=float(aoa),
        re=float(re),
        pressure=pressure.astype(np.float32),
        velocity=np.stack([u, v], axis=1).astype(np.float32),
        main_camber=float(main_camber),
        main_thickness=float(main_thickness),
        slat_deflection=float(slat_deflection),
        flap_deflection=float(flap_deflection),
        flap_gap=float(flap_gap),
    )


def sample_airfoil_case(rng: np.random.Generator, n_points: int) -> HighLiftCase:
    """Compatibility wrapper; new workshop cases are high-lift sections."""
    return sample_high_lift_case(rng, n_points)


def synthetic_dataset(n_cases: int, n_points: int = 512, seed: int = 0) -> list[HighLiftCase]:
    rng = np.random.default_rng(seed)
    return [sample_high_lift_case(rng, n_points) for _ in range(n_cases)]


def cases_to_json_payload(cases: list[HighLiftCase], *, seed: int, n_points: int) -> dict:
    return {
        "meta": {
            "n_cases": len(cases),
            "n_points": n_points,
            "seed": seed,
            "generator": "synthetic CRM-HL high-lift sections inspired by HiLiftAeroML + hidden wake/circulation terms",
            "generator_version": "v3_high_lift_transolver_workshop",
            "inspired_by": "HiLiftAeroML / NASA CRM-HL high-lift surrogate modeling",
        },
        "cases": [
            {
                "aoa": float(c.aoa),
                "re": float(c.re),
                "shape": {
                    "main_camber": c.main_camber,
                    "main_thickness": c.main_thickness,
                    "slat_deflection": c.slat_deflection,
                    "flap_deflection": c.flap_deflection,
                    "flap_gap": c.flap_gap,
                },
                "points": c.points.tolist(),
                "sdf": c.sdf.tolist(),
                "pressure": c.pressure.tolist(),
                "velocity": c.velocity.tolist(),
            }
            for c in cases
        ],
    }


def cases_from_payload(payload: dict, limit: int | None = None) -> list[HighLiftCase]:
    raw_cases = payload["cases"] if limit is None else payload["cases"][:limit]
    loaded = []
    for c in raw_cases:
        shape = c.get("shape", {})
        loaded.append(
            HighLiftCase(
                points=np.array(c["points"], dtype=np.float32),
                sdf=np.array(c["sdf"], dtype=np.float32),
                aoa=float(c["aoa"]),
                re=float(c["re"]),
                pressure=np.array(c["pressure"], dtype=np.float32),
                velocity=np.array(c["velocity"], dtype=np.float32),
                main_camber=shape.get("main_camber", shape.get("m", c.get("m"))),
                main_thickness=shape.get("main_thickness", shape.get("t", c.get("t"))),
                slat_deflection=shape.get("slat_deflection"),
                flap_deflection=shape.get("flap_deflection"),
                flap_gap=shape.get("flap_gap"),
            )
        )
    return loaded


def split_cases(cases: list[HighLiftCase], val_fraction: float = 0.2) -> tuple[list[HighLiftCase], list[HighLiftCase]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if len(cases) < 2:
        raise ValueError("Need at least two cases to create train/val splits")
    n_val = int(len(cases) * val_fraction)
    n_val = min(max(n_val, 1), len(cases) - 1)
    return cases[:-n_val], cases[-n_val:]


def stack_cases(cases: list[HighLiftCase]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([torch.from_numpy(c.features()).float() for c in cases])
    y = torch.stack([torch.from_numpy(c.targets()).float() for c in cases])
    return x, y
