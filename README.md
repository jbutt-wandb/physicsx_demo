# PhysicsX W&B Workshop

Hands-on demo notebooks for a 2.5-hour Weights & Biases workshop
tailored to PhysicsX's neural-field CFD surrogate workflow. Everything
attendees need lives in `notebooks/` — each notebook is self-contained,
runs on a laptop CPU, and demonstrates one W&B capability against a
small `TinyTransolver` model trained on synthetic NACA airfoil data.

**Target W&B project:** `wandb_emea/jb_physics_demo`
**Registry entry point:** `wandb-registry-airfoil_surrogates`

## Quickstart

```bash
uv sync                    # Python 3.11+; installs wandb, torch, matplotlib, ...
uv run wandb login
uv run jupyter lab
```

Then open the notebooks in the order below.

## The notebooks — run in this order

| # | Notebook | W&B capability | What it does |
| - | -------- | -------------- | ------------ |
| 1 | `notebooks/registry.ipynb` | **Registry (datasets)** | Generates the synthetic NACA airfoil dataset and promotes it to `wandb-registry-airfoil_surrogates/datasets:production`. |
| 2 | `notebooks/experiment_tracking.ipynb` | **Runs, artifacts, Tables, media** | Pulls the dataset from the Registry, trains an ~811k-param `TinyTransolver` for 12 epochs, logs per-epoch checkpoints as artifacts, and promotes the best checkpoint to `wandb-registry-airfoil_surrogates/checkpoints:production`. |
| 3 | `notebooks/sweeps.ipynb` | **Sweeps (Bayesian HP search)** | Loads `checkpoints:production`, runs an 8-trial Bayesian sweep over `lr`, `weight_decay`, `lambda_aux`, `batch_size`, and promotes the sweep winner to `wandb-registry-airfoil_surrogates/model_hyperparameter_tuned:production`. |
| 4 | `notebooks/reports.ipynb` | **Reports API** | Programmatically generates a shareable Report over the training run's metrics and media panels using `wandb_workspaces.reports.v2`. |

Every notebook uses `with wandb.init(...) as run:` context managers, so
runs finish cleanly on cell exit. Running notebooks out of sequence
raises a clear "artifact not found" — the pipeline is intentionally
strict about ordering.

## Model + dataset

- **Model.** A downscaled Transolver — 6 Physics-Attention layers,
  hidden dim 128, 32 slices, ~811k parameters. Consumes
  `(x, y, sdf, aoa, log_re)` per point and predicts
  `(pressure, u, v)`. Full description of the Physics-Attention
  mechanism is in `docs/explainers/pxtransolver.md`.
- **Dataset.** Synthetic NACA-4-digit airfoils with analytic
  pressure and velocity fields (clipped near singularities). 80
  cases × 512 points, split 80/20 into train/val. Serialised as
  JSON so the payload is self-describing.

## Docs and explainers

- `docs/explainers/pxtransolver.md` — Physics-Attention (slice →
  attend → deslice) mechanism with a full parameter budget and the
  reasoning behind the sizing.
- `docs/explainers/cfd_surrogate_models.md` — introduction to
  neural-field CFD surrogates, the three families
  (surrogate/PINN/operator), and where Transolver sits.
- `docs/explainers/further_reading.md` — curated reading path from
  reviews to foundational papers to modern work, plus a suggested
  six-week schedule.

## Dependencies

Runtime dependencies (from `pyproject.toml`):

- `wandb>=0.18`, `wandb-workspaces>=0.2`
- `torch>=2.4`, `numpy>=1.26`, `scikit-learn>=1.4`
- `matplotlib>=3.8`, `pandas>=2.2`, `pyyaml>=6.0`
- `ipykernel>=7.3`, `nbformat>=5.10`

The notebooks are CPU-friendly by default. Optional extras:

- `[airfrans]` — the real AirfRANS dataset (Bonnet et al. 2022) if
  you want to swap the synthetic generator.
- `[notebooks]` — installs `jupyter` if you'd rather not rely on your
  existing Jupyter install.

Install any extra with `uv sync --extra <name>`.

## What lives in the Registry after a full run

`wandb-registry-airfoil_surrogates` on `wandb_emea`:

- `datasets/synthetic_airfoil_slice` — aliases `production`, `v1`.
- `checkpoints/airfoil_surrogate_ckpt` — best-val version aliased
  `production` + `epoch-N`.
- `model_hyperparameter_tuned/airfoil_surrogate_ckpt_tuned` —
  sweep-winner aliased `production` + `sweep-winner`.

Downstream consumers (attendees' own code, future notebooks, external
projects) resolve each artifact at
`wandb-registry-airfoil_surrogates/{collection}:production`.

## License

Internal workshop material. Not for redistribution.
