from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class AirfoilCase:
    points: np.ndarray
    sdf: np.ndarray
    aoa: float
    re: float
    pressure: np.ndarray
    velocity: np.ndarray
    m: float | None = None
    p: float | None = None
    t: float | None = None

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


def sample_airfoil_case(rng: np.random.Generator, n_points: int) -> AirfoilCase:
    m = rng.uniform(0.0, 0.06)
    p = rng.uniform(0.2, 0.6)
    t = rng.uniform(0.08, 0.18)
    aoa = rng.uniform(-8.0, 12.0) * np.pi / 180.0
    re = 10 ** rng.uniform(6.0, 7.0)

    pts = rng.uniform([-0.5, -0.8], [1.5, 0.8], size=(n_points, 2)).astype(np.float32)

    x_chord = np.linspace(0.01, 1.0, 240)
    yc = naca4_camber(x_chord, m, p)
    yt = naca4_thickness(x_chord, t)
    upper = np.stack([x_chord, yc + yt], axis=1)
    lower = np.stack([x_chord, yc - yt], axis=1)
    surface = np.concatenate([upper, lower[::-1]], axis=0)

    dists = np.linalg.norm(pts[:, None, :] - surface[None, :, :], axis=-1)
    sdf = dists.min(axis=1).astype(np.float32)

    center = np.array([0.5, 0.0])
    rel = pts - center
    r = np.maximum(np.linalg.norm(rel, axis=1), 0.1)
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    u_inf = np.cos(aoa)
    v_inf = np.sin(aoa)
    re_scaled = np.log10(re) - 6.0

    camber_bias = (m - 0.03) / 0.03
    thickness_bias = (t - 0.13) / 0.05
    camber_location_bias = (p - 0.4) / 0.2
    shape_lift = 0.55 * camber_bias + 0.25 * thickness_bias - 0.15 * camber_location_bias
    wake_strength = 0.20 + 0.16 * thickness_bias + 0.10 * abs(shape_lift) + 0.06 * re_scaled
    wake_strength = float(np.clip(wake_strength, 0.05, 0.55))

    circulation = -2 * np.pi * (np.sin(aoa) + 0.08 * shape_lift) * (1.0 + m * 5.0)
    doublet = 0.25 * t * (1.0 + 0.35 * re_scaled)
    u = u_inf - doublet * np.cos(2 * theta) / r**2 - circulation * np.sin(theta) / (2 * np.pi * r)
    v = v_inf - doublet * np.sin(2 * theta) / r**2 + circulation * np.cos(theta) / (2 * np.pi * r)

    x = pts[:, 0]
    y = pts[:, 1]
    downstream = 1.0 / (1.0 + np.exp(-12.0 * (x - 0.62)))
    wake_center = 0.06 * np.sin(aoa) + 0.10 * shape_lift + 0.05 * (x - 0.8) * np.tan(aoa)
    wake_width = 0.055 + 0.06 * np.clip(x - 0.6, 0.0, None) + 0.025 * abs(thickness_bias)
    wake = downstream * np.exp(-((y - wake_center) ** 2) / (2 * wake_width**2))
    u -= wake_strength * wake
    v += 0.35 * wake_strength * (y - wake_center) / (wake_width + 1e-4) * wake

    recovery = 0.16 * shape_lift * downstream * np.exp(-((y - wake_center) ** 2) / (2 * (2.2 * wake_width) ** 2))

    u = np.clip(u, -3.0, 3.0)
    v = np.clip(v, -3.0, 3.0)
    pressure = np.clip(1.0 - (u**2 + v**2) + recovery, -3.0, 1.0)

    return AirfoilCase(
        points=pts,
        sdf=sdf,
        aoa=float(aoa),
        re=float(re),
        pressure=pressure.astype(np.float32),
        velocity=np.stack([u, v], axis=1).astype(np.float32),
        m=float(m),
        p=float(p),
        t=float(t),
    )


def synthetic_dataset(n_cases: int, n_points: int = 512, seed: int = 0) -> list[AirfoilCase]:
    rng = np.random.default_rng(seed)
    return [sample_airfoil_case(rng, n_points) for _ in range(n_cases)]


def cases_to_json_payload(cases: list[AirfoilCase], *, seed: int, n_points: int) -> dict:
    return {
        "meta": {
            "n_cases": len(cases),
            "n_points": n_points,
            "seed": seed,
            "generator": "synthetic NACA-4-digit airfoils + hidden wake/circulation terms",
            "generator_version": "v2_global_wake",
        },
        "cases": [
            {
                "aoa": float(c.aoa),
                "re": float(c.re),
                "shape": {"m": c.m, "p": c.p, "t": c.t},
                "points": c.points.tolist(),
                "sdf": c.sdf.tolist(),
                "pressure": c.pressure.tolist(),
                "velocity": c.velocity.tolist(),
            }
            for c in cases
        ],
    }


def cases_from_payload(payload: dict, limit: int | None = None) -> list[AirfoilCase]:
    raw_cases = payload["cases"] if limit is None else payload["cases"][:limit]
    return [
        AirfoilCase(
            points=np.array(c["points"], dtype=np.float32),
            sdf=np.array(c["sdf"], dtype=np.float32),
            aoa=float(c["aoa"]),
            re=float(c["re"]),
            pressure=np.array(c["pressure"], dtype=np.float32),
            velocity=np.array(c["velocity"], dtype=np.float32),
            m=c.get("shape", {}).get("m", c.get("m")),
            p=c.get("shape", {}).get("p", c.get("p")),
            t=c.get("shape", {}).get("t", c.get("t")),
        )
        for c in raw_cases
    ]


def split_cases(cases: list[AirfoilCase], val_fraction: float = 0.2) -> tuple[list[AirfoilCase], list[AirfoilCase]]:
    n_val = int(len(cases) * val_fraction)
    return cases[:-n_val], cases[-n_val:]


def stack_cases(cases: list[AirfoilCase]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([torch.from_numpy(c.features()).float() for c in cases])
    y = torch.stack([torch.from_numpy(c.targets()).float() for c in cases])
    return x, y

