# The HiLift point-cloud surrogate

 Before you
open the notebooks, read this to understand *what* problem the
example solves, *why* the dataset choice matters, and *how* the
W&B story maps onto a real computational fluid dynamics (CFD)
surrogate workflow.

Foundational context for this document is Justin Hodges' May 2026
tutorial [A Beginner's Tutorial: the new HiLiftAeroML dataset with
code to train a ML surrogate](https://hodgesj.substack.com/p/a-beginners-tutorial-the-new-hiliftaeroml). This document
paraphrases and extends that framing for the workshop.

---

## 1. What we're trying to do

We want to train a model that, given an aircraft's high-lift
configuration and its angle of attack, **predicts a surface
pressure field over the wing**. In CFD language: a **surrogate for a
scalar boundary field** conditioned on geometry and flight condition.

A real CFD run for one such configuration takes hours of compute
on many CPU cores and produces gigabytes of output. The model must
learn the map from `(geometry, angle_of_attack) → pressure field`
from a corpus of past CFD runs,.

The workshop uses this problem shape because it matches PhysicsX's
production workflow: same neural-field-on-point-cloud primitive,
same conditioning story, same "train on big CFD, deploy in
production" arc. Only the scale changes.

---

## 2. What the workshop covers, and why it's useful

The workshop is a **seven-notebook end-to-end tour of Weights &
Biases patterns**, all built on the HiLift point-cloud surrogate
example. Every notebook is a self-contained tutorial that adds one
capability to the story you already have.

**The goal of this workshop is not accurate CFD modelling but how Weights & Biases can be used to train and deploy models.**

### The notebooks, in run order

1. **`registry.ipynb`** — publishes the demo dataset as a versioned
   `wandb.Artifact` and promotes it into the `hilift` Registry.
   Teaches the two-step *log-then-link* pattern, Collection aliases
   (`production`, `v1`), and reading artifacts back out via the
   Public API.
2. **`experiment_tracking_hilift.ipynb`** — trains Transolver++,
   logs per-epoch metrics + media panels + tables, saves checkpoint
   artifacts every 10 epochs, promotes the best-val checkpoint into
   `hilift/checkpoints`. The core "one run, everything tracked"
   experience.
3. **`sweeps.ipynb`** — Bayesian hyperparameter sweep that
   fine-tunes from the promoted checkpoint. Teaches `wandb.sweep`
   + `wandb.agent`, cross-trial comparison in the UI, and
   promoting a sweep winner into a new Collection.
4. **`reports.ipynb`** — builds a programmatic W&B Report whose
   panels live-query the training run. Teaches
   `wandb_workspaces.reports.v2` and the "stakeholder-facing
   curated view" story.
5. **`advanced_asynchronous_evals.ipynb`** — fires an eval fleet
   against every checkpoint of the training run in parallel,
   aggregates results back into the training run's summary.
   Teaches the safe multi-run + group pattern, artifact-based
   lineage, and why *not* to reopen a training run for eval
   logging.
6. **`advanced_multistage_setup.ipynb`** — a platform-team notebook
   that publishes a base model + three curriculum datasets across
   two dedicated Registries, with a Markdown Collection card on
   every Collection. Teaches multi-Registry organisation and
   in-Registry documentation.
7. **`advanced_multistage_training_pipeline.ipynb`** — three
   sequential training stages (init → pretrain → posttrain), each
   an independent job that consumes from Registry and promotes to
   Registry. Teaches multi-team Registry-only handoff and
   norm-stats propagation via checkpoint metadata.

### Learning outcomes

By the end of the workshop, attendees can:

- **Ship models via Registry** — promote a checkpoint out of a
  training run into an org-visible Collection with alias semantics
  downstream consumers can rely on.
- **Search hyperparameters at scale** — run a Bayesian sweep that
  respects a fixed pretrained architecture, and promote the winner.
- **Curate stakeholder dashboards** — build Reports
  programmatically that stay in sync with live training runs.
- **Decouple long training from evaluation** — fire eval jobs
  independently and aggregate results.
- **Coordinate multiple teams via Registry alone** — pass models
  between stages / teams with no shared filesystem, messaging
  channel, or hand-off script.
- **Document assets in place** — write Collection cards so the
  Registry itself carries the contract between producer and
  consumer.

### Why this matters for aerospace ML

A CFD surrogate program stresses every one of these capabilities:

- **Team handoff.** Multiple teams (geometry, meshing, CFD, ML,
  aerodynamics) share large, expensive artefacts. Registry is the
  audit-trail-friendly handoff surface — better than filesystems,
  better than ad-hoc S3 buckets, better than DMs.
- **Long training.** Real HiLift training runs go for hours to
  days, so evaluation *must* be decoupled from the training
  process — same async pattern the async-evals notebook
  demonstrates.
- **Hyperparameter sensitivity.** Transolver++ is genuinely
  sensitive to learning rate, dropout, and slice count — sweeps
  are a standard part of the pipeline, not a one-off.
- **Stakeholder communication.** Aerodynamicists and program
  managers need curated views of model performance, not raw run
  tables — that's what Reports are for.

The workshop teaches these patterns on a domain-realistic example
so that when you swap the demo generator for real VTU boundary
data, the tooling scaffold you built here stays byte-identical.

---

## 3. The HiLiftAeroML dataset

HiLiftAeroML is a NASA-released dataset that first went public in
mid-2026. It packages **1,800 high-fidelity Wall-Modelled LES
(WMLES) simulations** across:

- **180 geometry variants** of the NASA **CRM-HL** (Common
  Research Model — High Lift). Each variant is a small parametric
  perturbation of the reference geometry.
- **10 angles of attack** per geometry, sweeping from
  cruise-level through near-stall.

Each simulation ships as several files, each targeting a different
kind of downstream use:

| File | Size | What it is |
| ---- | ---- | ---------- |
| `force_mom_all.csv` | Small (KB) | Integrated lift / drag / moment coefficients per run |
| `geo_values_all.csv` | Small (KB) | Design variables defining each geometry variant |
| `boundary_<run>.vtu.tgz` | **~13 GB compressed** | Time-averaged surface fields (pressure coefficient Cp, wall shear, …) as a VTU (VTK unstructured) mesh |
| `volume_<run>.vtu.tgz` | **~23 GB compressed** | Time-averaged volume fields as a 3D VTU mesh |
| STL / STP geometry files | ~MB | Raw CAD geometry |
| Wall-shear renderings | ~MB | Pre-computed visualisation PNGs |


### The two example paths in the Hodges blog

We will follow the pattern of the following tutorial:
1. **A 3D point-cloud surrogate.** Inputs: sampled surface points
   + normals + AoA condition. Outputs: a scalar field per point
   (typically Cp or wall-shear-magnitude). Trained on samples
   drawn from the real `boundary_*.vtu.tgz` files.

---

## 4. Why we run against synthetic data in the workshop

- **13 GB per case** on disk (compressed) — real mode blows a
  workshop laptop's storage and eats hours in downloads.
- **PyVista + huggingface_hub** — real mode adds those runtime
  dependencies; demo mode needs neither.
- **Cross-notebook coherence** — our workshop needs the whole
  pipeline (setup → train → sweep → report) to finish in under
  10 minutes, which is impossible if any stage reads real
  boundary data.

The demo generator is not a stand-in for the physics — it's a
stand-in for the *data pipeline shape*. Concretely, the synthetic
point clouds have:

- The **same tensor shape** as real HiLift boundary caches:
  `points (N, 3)`, `normals (N, 3)`, `target (N, 1)`.
- The **same file format** (compressed NPZ — NumPy's compressed
  archive format) and per-case metadata (`geo_id`, `aoa`,
  `run_id`, `target_name`, `target_mode`).
- The **same three-component topology** as a real CRM-HL high-lift
  wing (see §5).
- A **physically-motivated Cp surrogate** target (§6) — not real
  Cp, but shaped to have the same spatial patterns.

Because the shape matches, **every line of the training code, the
Transolver++ wrapper, the visualisation triptychs, and the W&B
logging works unchanged against real HiLift boundary data.** Swap
the generator for a VTU reader and the rest of the notebook stays
identical. That's the point.

---

## 5. Anatomy of a demo case: the three CRM-HL components

The CRM-HL geometry is a **three-element high-lift wing**:

- A **slat** — a small aerodynamic element that extends the leading
  edge forward and slightly downward, delaying stall at high AoA.
- The **main wing** — the primary lifting surface, cambered and
  swept.
- A **flap** — a trailing-edge element deflected downward to
  increase camber and thus lift at low speed (takeoff, landing).

The demo generator explicitly builds all three:

```
Slat  ~22% of points ┐        Point sample: 8,192 total
Main  ~56% of points ├────►   Component split matches the surface
Flap  ~22% of points ┘        area proportions of a real CRM-HL
```

Each component is placed in its own region of space:

- **Slat**: `x ∈ [-0.20·chord, 0.05·chord]` (ahead of the leading
  edge, LE), elevated `z ≈ 0.05–0.10` (droops slightly).
- **Main wing**: `x ∈ [0.04·chord, 0.96·chord]` (chord span),
  gentle upper-surface arch in `z`.
- **Flap**: `x ∈ [0.90·chord, 1.25·chord]` (behind main wing),
  deflected downward — the deflection angle is computed as
  `22° + 0.35·AoA + 7°·geo_phase`, so it varies with both the flight
  condition and the per-geometry phase.

Per-geometry variation is seeded by the digits of `geo_id`
(`LHC001`, `LHC002`, …). Each `geo_id` gets a distinct
`geo_phase ∈ [0, 1]` that shifts the flap deflection and the
pressure-recovery term — enough variation that the model must
learn to condition on geometry, not memorise a single wing.

**The visual payoff**: in the training notebook's val_plots panels
you'll see three visually distinct clusters per triptych. That's
not a rendering artefact — that's the slat / main / flap surfaces,
faithfully reproduced from the real CRM-HL topology. When the
model gets it right, all three light up with sensible pressure
patterns.

---

## 6. The synthetic Cp target — what it encodes

Real Cp on a high-lift wing has three physically-important features:

1. **Leading-edge suction.** A sharp negative-pressure peak just
   behind the leading edge, whose magnitude grows with AoA. This
   is the region that carries most of the lift.
2. **Flap loading.** A secondary negative-pressure region behind
   the main wing, on the flap's upper surface.
3. **Pressure recovery.** A gradual increase from suction back
   toward freestream as flow moves toward the trailing edge.

The demo target field is a sum of four analytically-shaped pieces
that mimic these:

```
target(x, y, component) =
      leading_edge_suction     (Gaussian dip near LE, scales with AoA)
    + flap_loading              (Gaussian dip on flap only)
    + slat_loading              (Gaussian dip on slat only)
    + pressure_recovery         (mild linear ramp in x, y)
    + noise                     (Gaussian, σ = 0.015)
```

Values live in roughly `[-1.1, +0.25]` — same order of magnitude as
real Cp on a high-lift wing at moderate AoA. The magnitudes and
positions of each term were tuned to make the training exercise
non-trivial (the model must learn spatial structure, not just
memorise a constant per component) without being so realistic that
it obscures what the workshop is actually teaching (the
tooling).

**Caveat, spelled out clearly:** the synthetic Cp is not
aerodynamically valid. Do not report surrogate-quality numbers
against it. Use it to exercise pipelines, evaluate tooling, and
build intuition — nothing more.

---

## 7. The model — Transolver++

The workshop's model is **Transolver++**, wrapped as
`TransolverPointFieldSurrogate`. Transolver++ is the second
generation of the *Transolver* family from Wu et al. (Tsinghua ML
Lab, `thuml/Transolver_plus`) — a transformer architecture designed
for **neural PDE (partial differential equation) solving on
irregular geometries**.

### Why transformers on point clouds

Most classic ML architectures assume regular data — images live on
grids, sequences live on chains, tabular data lives in rows. CFD
fields live on **meshes** or **point clouds** which have neither a
fixed shape nor a natural ordering. Two design responses to that
reality:

- **Graph neural networks (MeshGraphNet et al.)** — treat the mesh
  as a graph, use message passing.
- **Point-cloud transformers (Transolver, Transolver++, PointNet,
  …)** — treat the points as a set with positional embeddings,
  attend across them.

Transolver++ sits in the second camp. Its distinguishing feature
is the **physics-attention mechanism**: instead of computing full
`N × N` attention (impossible for million-point meshes), it
projects the N points down to `slice_num = 16` learned "slice
tokens", attends within that small set, then projects back. The
result is attention complexity that scales linearly in `N` while
still capturing long-range interactions across the mesh — exactly
what you need for a wing where the wingtip vortex affects the root
pressure distribution.

The deep-dive explainer is
`docs/explainers/pxtransolver.md`. This document assumes you've
read that once or intend to; here we just note the practical
consequences:

- Input shape: `(B, N, 6)` per-point features (3 position + 3
  normal), plus a `(B, 3)` conditioning vector (AoA-encoded).
- Output shape: `(B, N, 1)` — one scalar per point.
- Params: our config uses `n_hidden=96, n_layers=4, n_head=4,
  slice_num=16, mlp_ratio=2` — small enough to train on CPU in
  minutes, faithful enough that the full-scale production
  Transolver++ shares the same code.

### Why we clone the reference implementation

The workshop notebook clones `thuml/Transolver_plus` from GitHub at
runtime (cached under `~/.cache/physicsx_demo/`) and applies a
one-line patch to the `all_reduce` calls that assume a distributed
process group. This keeps us **faithful to the published reference
architecture** — the exact model an aerospace ML team would pick up
if they went to the Transolver GitHub tomorrow — rather than a
locally-reimplemented lookalike.

---

## 8. The training shape — patches, AoA conditioning, normalisation

Three details worth understanding before you open the training
notebook:

### 8.1 Patch-based training

Each case has `POINTS_PER_CASE_CACHE = 8192` points. Rather than
feed all 8k per batch item (expensive on CPU, and unnecessary),
we **sample a random `PATCH_POINTS = 1024`-point patch per batch
item, `PATCHES_PER_CASE = 96` times per epoch per case**. Over an
epoch the model sees the full point cloud implicitly, without ever
processing all N points at once.

This is the reference notebook's own discipline. The technique
scales: for real CRM-HL boundary VTUs with a million+ points, you'd
sample the same 1024-point patches, just from a much larger source
distribution.

### 8.2 AoA conditioning

Angle of attack enters the model as a **3-D conditioning vector**,
not a scalar:

```
condition = [ (aoa - aoa_mean) / aoa_std,   # z-normalised scalar
              sin(aoa in radians),          # periodic feature 1
              cos(aoa in radians) ]         # periodic feature 2
```

The three-way encoding gives the model both a *linear* signal
(what value is this?) and a *periodic* signal (is this near a stall
regime? A cruise regime?) — which lets the model interpolate
between training AoAs more smoothly than a bare scalar would.

### 8.3 Train-set-only normalisation

Every input feature and every target dimension is
standard-normalised (zero mean, unit variance) **using stats from
the training subset only**. Val and test cases are normalised using
the *same* stats — that's what makes them a fair
generalisation test. If you accidentally normalised across the full
dataset you'd leak val/test information into training. The training
notebook computes `stats["point_mean"]`, `stats["point_std"]`,
`stats["target_mean"]`, `stats["target_std"]` explicitly in its
§3a for this reason.

Denormalisation happens exactly once, when reporting metrics in
physical units: the training loop tracks MSE on normalised values
(so all components contribute comparably), then `denorm_target(...)`
maps predictions back to physical Cp so MAE and RMSE read in units
an aerospace engineer expects.

---

## 9. What "success" looks like

The point is the tooling, not the model, but attendees should still
be able to read the metrics. Here's what you should expect:

- **`train/loss` and `val_metrics/loss` both decreasing** across
  the 50 epochs — with our tiny model + 12 cases, val loss should
  drop from ~1.0 at epoch 1 to somewhere in the `0.05–0.20`
  neighbourhood by epoch 50. If it stays near 1.0, the checkpoint
  didn't load or normalisation is broken.
- **`val_metrics/mae` in `[0.15, 0.30]` physical Cp units** at
  convergence. Perfect prediction is ~0.02 (the noise floor);
  attendees are seeing a small model learn a coarse
  approximation, not a benchmark-grade surrogate.
- **Slat error dominates** in the abs-error triptych — the
  leading-edge suction band has the sharpest spatial gradient, and
  small models under-predict its magnitude. This is not a bug; it's
  the standard failure mode and is what more epochs / more cases /
  a bigger backbone would attack.
- **`val_tables/regression_metrics`** should show 15 rows total (5
  cadence epochs × 3 sampled test cases), with `R²` climbing from
  the low-hundreds range at epoch 10 into the low-hundreds range
  at epoch 50. Per-case variance is high with only 8 training
  cases — that's expected.

Don't extrapolate any of these numbers to real HiLiftAeroML
performance. The demo is calibrated for pipeline-visibility, not
surrogate quality.

---

## 10. From demo to real HiLift — the one-line swap

The workshop stops at synthetic data intentionally. To scale to real
data, an ML engineer would:

1. **Replace `create_demo_pointcloud_case`** in `registry.ipynb`
   with a VTU reader:
   - Download `boundary_<run>.vtu.tgz` from Hugging Face.
   - Extract the VTU, read with PyVista.
   - Downsample to `POINTS_PER_CASE_CACHE` points.
   - Extract coordinates, normals, and one scalar target field
     (`Cp` or `wallShearStress`).
   - Save as NPZ with identical schema.
2. **Bump `CASE_SPECS`** from 12 tuples to the full HiLift
   catalogue (potentially all 1,800 runs).
3. **Bump the compute budget** — a real training pass on 1,800
   cases with 4-layer Transolver++ takes hours on a GPU, not
   minutes on CPU.

Everything downstream — the training loop, the W&B logging, the
Registry, the sweep integration, the reports — remains the same.

---

## 11. Related reading

1. [A Beginner’s Tutorial: the new HiLiftAeroML dataset with code to train a ML surrogate](https://hodgesj.substack.com/p/a-beginners-tutorial-the-new-hiliftaeroml)
2. [Scaling Physics AI for Automotive Aerodynamics](https://www.physicsx.ai/newsroom/scaling-physics-ai-for-automotive-aerodynamics)
3. [Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/pdf/2402.02366)