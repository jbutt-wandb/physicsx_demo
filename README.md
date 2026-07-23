# PhysicsX W&B Workshop

Hands-on demo notebooks for a Weights & Biases workshop tailored to
PhysicsX's neural-field CFD surrogate workflow. Every notebook is
self-contained, runs on a laptop CPU, and demonstrates one W&B
capability against a `Transolver++` point-cloud surrogate trained
on a demo-synthetic stand-in for the NASA CRM-HL high-lift wing.


**Target W&B project:** `wandb_emea/jb_physicsx_demo` 

**Registry entry point:** `wandb-registry-hilift`

## Setup

Create a venv, activate it, and synchronise dependencies:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

Authenticate with W&B (one-time):

```bash
wandb login
```

Then open the notebooks in your Jupyter environment of choice.

## Notebooks — run in this order

| # | Notebook | W&B capability | What it does |
| - | -------- | -------------- | ------------ |
| 1 | `notebooks/registry.ipynb` | **Registry (datasets)** | Publishes the demo point-cloud dataset to `wandb-registry-hilift/pointcloud_demo:production` and verifies the round-trip by re-downloading it in a separate run. |
| 2 | `notebooks/experiment_tracking_hilift.ipynb` | **Runs, artifacts, Tables, media** | Consumes the dataset from Registry, trains Transolver++ for 50 epochs, logs per-epoch metrics + prediction triptychs + a per-case Table, promotes the best-val checkpoint to `wandb-registry-hilift/checkpoints:production`. |
| 3 | `notebooks/sweeps.ipynb` | **Sweeps (Bayesian HP search)** | Consumes `checkpoints:production`, runs an 8-trial Bayesian sweep fine-tuning from that checkpoint, promotes the winner to `wandb-registry-hilift/checkpoints_tuned:production`. |
| 4 | `notebooks/reports.ipynb` | **Reports API** | Builds a shareable W&B Report following Justin Hodges' HiLiftAeroML tutorial's section structure, with a live panel of 2D predictions from the training run. |
| 5 | `notebooks/advanced_asynchronous_evals.ipynb` | **Async evaluation** | Fires one eval run per checkpoint of the training run in parallel, aggregates results back into the training run's summary via `group=` + `config.source_training_run_id`. |
| 6 | `notebooks/advanced_multistage_setup.ipynb` | **Multi-Registry publishing** | Platform-team notebook: publishes an untrained Transolver++ base model + three curriculum datasets across two dedicated Registries, each Collection documented with a Markdown card. |
| 7 | `notebooks/advanced_multistage_training_pipeline.ipynb` | **Multi-team Registry handoff** | Three sequential training stages (init → pretrain → posttrain), each an independent job that consumes from + promotes to Registry, with norm-stats propagation via checkpoint metadata. |

## Docs and explainers

Under `docs/explainers/`:

- `hilift_example.md` — the workshop's motivating example: what problem the model solves, dataset scale, why we run against synthetic data, model + training loop shape. **Read this first.**

## References

- Justin Hodges' Substack — [*A Beginner's Tutorial: the new HiLiftAeroML dataset with code to train a ML surrogate*](https://hodgesj.substack.com/p/a-beginners-tutorial-the-new-hiliftaeroml) (May 2026). The tutorial that motivates the workshop's HiLift example; `docs/explainers/hilift_example.md` paraphrases and extends its framing.
- Justin Hodges' Substack — [hodgesj.substack.com](https://hodgesj.substack.com/) — further posts on ML for aerospace and CFD.
