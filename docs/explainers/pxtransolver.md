# PXtransolver — Architecture, Datasets, and Losses

**Audience.** ML engineers and W&B users who want to understand the
model family PhysicsX has productionised, at a level that lets them
have a substantive conversation about it. Assumes you have read
`cfd_surrogate_models.md` for the general background on why CFD
surrogates look the way they do.

**A note on naming.** "PXtransolver" is PhysicsX's productionised
adaptation of the base **Transolver** architecture published by
Wu, Luo, Wang, Wang, Long (2024), *Transolver: A Fast Transformer
Solver for PDEs on General Geometries* (arXiv:2402.02366). PhysicsX
has not published the specifics of their divergences from the base
architecture; this document therefore describes the base Transolver
in detail — everything that the "PX" variant demonstrably builds on —
and flags the axes along which a productionised version is most
likely to differ.

---

## 1. Inputs and outputs

We start here because every other design decision in the architecture
follows from the shape of the data.

### 1.1 What one input point looks like

A Transolver sees the discretised domain (the CFD mesh, or a
subsampled point cloud of it) as an unordered set of `N` points. Each
point `i` carries two kinds of features:

- **Geometry features `g_i` ∈ ℝ^{C_g}.** The point's 3D position, and
  optionally derived geometric quantities: surface normal, local
  curvature, distance to the surface, ...
- **Observed physical quantities `u_i` ∈ ℝ^{C_u}` (optional).** Any
  known field values at the point at inference time — e.g. the
  velocity boundary condition at the inlet, or a coarse-solver
  estimate of the pressure. Empty (`C_u = 0`) for pure "geometry →
  field" surrogates.

The full input is a tensor of shape `(N, C_g + C_u)`. Concretely for
a car with 32 000 sampled surface points, geometry-only features
`(x, y, z, nₓ, n_y, n_z)`, you'd feed the model a `(32000, 6)`
tensor.

### 1.2 What one output looks like

The output has the same "one row per point" shape:

- **Predicted fields `ŷ_i` ∈ ℝ^{C_out}`.** Typically the physical
  quantities the CFD solver would report at that same point.
  Examples:
  - Surface-only surrogate: `(p, τₓ, τ_y, τ_z)` — pressure and wall
    shear stress. `C_out = 4`.
  - Volume surrogate: `(p, u, v, w)` — pressure and 3D velocity.
    `C_out = 4`.
  - Combined: the union of both.

Output tensor: `(N, C_out)`. From the fields you then derive scalar
engineering quantities — drag coefficient, lift coefficient — by
integrating over the surface (see Lecture 7 of the CFD explainer).

### 1.3 Why "one row per point" matters

The model is **permutation-equivariant** in the point index. Shuffle
the rows of the input, the rows of the output shuffle the same way.
That is exactly the right symmetry for a mesh or point cloud: there
is no inherent ordering.

This is also why the architecture is a **transformer** rather than a
CNN. Convolutions assume a rigid grid; attention treats the input as
a set.

---

## 2. Model architecture

The architecture has three parts you should understand: the input
embedding, the Physics-Attention block, and the output head.

### 2.1 Input embedding

A linear layer maps each point's feature vector into the hidden
dimension `C`:

```
x⁰ = Linear(Concat(g, u))       # shape (N, C)
```

Typical hidden sizes reported: `C = 128` for simpler benchmarks,
`C = 256` for industrial cases. That is it — no positional encoding
required, because the raw coordinates are already part of the input
features.

### 2.2 The Physics-Attention block — the key idea

This is where Transolver differs from a vanilla transformer.

A vanilla transformer over `N` mesh points would compute attention
between every pair of points: `O(N²)` scores. For a car body with
`N ≈ 32 000` points that is a billion pairs per layer per head —
intractable.

Transolver's move: don't attend point-to-point. Instead, learn to
group points into `M` "physics slices" (with `M ≪ N`, typically
`M = 32` or `M = 64`), attend at the slice level, then broadcast
back.

The three steps of the block are: **slice**, **attend**, **de-slice**.

**Step A — Slicing.** Each point is softly assigned to each of `M`
slices via a learned projection followed by softmax:

```
w_i = Softmax(Project(x_i))            # w_i ∈ ℝᴹ, sums to 1
```

`w_{i,j}` is the "membership weight" of point `i` in slice `j`. The
softmax makes the distribution low-entropy — most points end up
mostly assigned to one or two slices.

Aggregate points into slice tokens by a weighted average:

```
z_j = ( Σᵢ w_{i,j} · x_i ) / ( Σᵢ w_{i,j} )     # z_j ∈ ℝ^C
```

You now have `M` tokens `{z_1, …, z_M}`, each summarising a region of
the mesh where "points with similar physics" have been pooled
together. What "similar physics" means is not hard-coded — the
`Project` weights are learned, so the model decides for itself which
regions to group. In practice the learned slices tend to correspond
to physically meaningful zones (boundary layers, wakes, stagnation
regions), which is why the mechanism is called "Physics-Attention".

**Step B — Standard attention between slice tokens.** With `M`
tokens in hand, apply plain scaled-dot-product attention:

```
Q, K, V = Linear(z)                          # each (M, C)
z' = Softmax(Q Kᵀ / √C) · V                  # (M, C)
```

Multi-head attention is used in practice — split `C` into `H`
sub-spaces and run `H` independent attention heads. This is the same
mechanism worked out by hand in Lecture 5 of the CFD explainer.

**Step C — De-slicing.** Broadcast each updated slice token back
to the mesh points, weighted by the same membership weights:

```
x_i' = Σⱼ w_{i,j} · z_j'                     # x_i' ∈ ℝ^C
```

You now have per-point updated features `(N, C)` again, ready for the
next block.

**Complexity.** `O(NMC + M²C)` — linear in `N` because `M` is a small
constant. This is what makes the architecture tractable on
million-point meshes where a vanilla attention would OOM.

### 2.3 The full block, in one equation each

Standard pre-norm transformer wrapping around Physics-Attention:

```
x̂ˡ = x^{l−1} + Physics-Attn( LayerNorm( x^{l−1} ) )
xˡ  = x̂ˡ    + FeedForward ( LayerNorm( x̂ˡ ) )
```

`L = 8` layers are stacked in the reported experiments.

### 2.4 Output head

A final linear layer maps the last block's `C`-dim features per point
back to the target field dimension:

```
ŷ = Linear(x^L)                              # (N, C_out)
```

That is the full model. Everything else — schedulers, mixed
precision, gradient checkpointing — is standard transformer plumbing.

### 2.5 Parameter budget sketch

For `L = 8`, `C = 256`, `M = 64`, `H = 8`, and a feed-forward
expansion factor of 4:

- Per block: attention (Q/K/V/O projections + FFN) ≈ 3.9 M params.
- Slicing projection per block: `C · M = 256 · 64 ≈ 16 k` params.
- 8 blocks × ~3.9 M ≈ **31 M parameters** — small by transformer
  standards.

The 300 M-parameter production surrogates PhysicsX and peers train
are the same architecture scaled up: larger `C`, larger `M`, more
layers, and often extra ancillary modules the paper doesn't describe.

### 2.6 Where a productionised "PXtransolver" is most likely to differ

A researcher reading the base paper would expect a production
variant to modify:

- **Multi-scale slicing.** Stack multiple slice granularities (e.g.
  `M = 16` for a global view, `M = 256` for local detail) either
  across layers or within a layer.
- **Geometry-aware initial slicing.** Bias the initial `w_{i,j}`
  distribution using known geometric priors — surface distance,
  curvature buckets, part-segmentation labels — instead of pure
  softmax over learned projection.
- **Rotary or Fourier positional encoding on `x_i`** for
  higher-frequency fields like near-wall pressure.
- **Auxiliary output heads** for drag / lift so those integrated
  quantities enter the loss directly.
- **Loss balancing across quantities** since pressure and velocity
  live at different scales.

The base architecture is the load-bearing structure; PhysicsX's
production stack is that plus engineering.

---

## 3. Loss functions

The paper does not explicitly write the training loss — a common
omission when the answer is "mean-squared error, obviously" — but
the metrics are precise, and standard MSE / relative L2 is implied.
Two categories of loss/metric are worth separating.

### 3.1 Field-wise loss

The workhorse loss for training and reporting is **relative L2**:

```
L_rel_L2 = ‖ŷ − y‖₂ / ‖y‖₂
```

Norms are per-example over all `N` mesh points and all output
channels. Reporting the *relative* form (rather than raw MSE) makes
the number comparable across cases with very different physical
scales — pressures in an aggressive downforce car can be an order of
magnitude larger than pressures in a family SUV, so absolute MSE
would be dominated by the largest-scale case.

For training, the paper's public code uses the same relative L2 as
the loss objective. Straight MSE is a valid alternative.

### 3.2 Integrated-quantity metrics

For industrial cases the paper additionally reports engineering
scalars — drag coefficient `C_d` and lift coefficient `C_l` — since
those are what customers actually care about (see Lecture 7 of the
CFD explainer for the physical derivation of these).

Two metrics per integrated quantity:

- **Relative L2 error** of `C_d` (or `C_l`) predictions across all
  test cases.
- **Spearman rank correlation** — do the *rankings* of designs by
  predicted `C_d` match the rankings by ground-truth `C_d`? This
  matters because a surrogate is often used for *design ranking* in a
  larger optimisation loop; the absolute number can be off by a few
  percent as long as the ordering is preserved.

A well-trained model reports something like `rel-L2(field) = 3%`,
`rel-L2(C_d) = 4%`, `Spearman(C_d) = 0.95`. Reporting all three is
standard practice.

### 3.3 What a productionised loss adds

Not documented for PXtransolver specifically, but standard additions:

- **Direct `C_d` / `C_l` terms** in the loss (`λ_Cd · ‖Ĉ_d − C_d‖²`),
  weighted by hyperparameters. Costs one extra surface integral per
  forward pass but forces the integrated quantities to converge fast.
- **Physics-consistency terms** — soft penalties for violating
  mass conservation (`‖∇ · û‖²`) or the no-slip boundary condition
  at the surface. Cheap to compute, catch qualitative failure
  modes the pure field loss misses.
- **Per-channel scaling** to keep pressure and velocity losses
  balanced when they live at different orders of magnitude.

---

## 4. Training datasets

The paper trains and evaluates on **six academic benchmarks** and
**two industrial-scale datasets**. Both are worth knowing.

### 4.1 The six academic benchmarks (all from FNO, Li et al. 2021)

| Benchmark      | Domain           | Dimension | Mesh size | Physics                    |
| -------------- | ---------------- | --------- | --------- | -------------------------- |
| Elasticity     | Point cloud      | 2D        | 972       | Solid stress               |
| Plasticity     | Structured mesh  | 2D + time | 3 131     | Plastic deformation        |
| Airfoil        | Structured mesh  | 2D        | 11 271    | Aerodynamic flow           |
| Pipe           | Structured mesh  | 2D        | 16 641    | Fluid flow in pipes        |
| Navier–Stokes  | Regular grid     | 2D + time | 4 096     | General fluid dynamics     |
| Darcy          | Regular grid     | 2D        | 7 225     | Porous-media flow          |

These are small, standardised benchmarks — the fluid-dynamics
equivalents of MNIST/CIFAR. Useful for architecture ablations and
apples-to-apples comparison; not representative of production
workloads.

### 4.2 The two industrial-scale datasets in the paper

| Dataset       | Geometry   | Points per case | Purpose                              |
| ------------- | ---------- | --------------- | ------------------------------------ |
| ShapeNet Car  | 3D, unstructured | 32 186          | Estimate surface pressure + volume velocity from car geometry |
| AirfRANS      | 2D, unstructured | 32 000          | Airfoil lift/drag under varying AoA and Reynolds number |

**AirfRANS** is the dataset the workshop repo uses (`data.py` has
the loader). Bonnet et al. (2022) — high-fidelity RANS on NACA
airfoils.

**ShapeNet Car** is a synthetic wind-tunnel dataset over ShapeNet
car geometries, historically used in graph-network surrogate papers.
32 k points per case; drag coefficient available as a scalar
regression target.

### 4.3 Production-scale datasets the community uses

The two datasets you asked about — SUV (SHIFT-SUV on Hugging Face)
and DrivAerNet++ — are the current public frontier for automotive
CFD surrogate research. Neither appears in the Transolver paper (both
were released after or in parallel), but both are the shape of data
a PhysicsX-style workflow trains on. What one sample from each looks
like is worth spelling out in detail.

---

## 5. What a single training instance looks like

### 5.1 SHIFT-SUV — one sample

Dataset URL: `huggingface.co/datasets/luminary-shift/SUV`. 18.3 TB
today, targeting 25 000 samples. Solver: Luminary Cloud running
**Detached Eddy Simulation (DES)** — a transient scale-resolving
compressible solver. Ground-truth validated against 1/4-scale wind
tunnel measurements (FKFS).

One directory (`run_00001/` inside e.g. `AeroSUV_qrtr_scale_estate_transient/`)
contains, roughly:

```
merged_surfaces.stl          Surface triangle mesh (car geometry)
merged_surfaces_filled.stl   Same, with tire/floor patches (closed)
merged_surfaces.vtp          Surface fields (VTK PolyData):
                               - pressure          (instantaneous)
                               - pressure_avg      (time-averaged)
                               - wall_shear_stress (instantaneous)
                               - wall_shear_stress_avg
merged_volumes.vtu           Volume fields (VTK UnstructuredGrid):
                               - pressure          (instantaneous)
                               - pressure_avg
                               - velocity          (instantaneous, 3D)
                               - velocity_avg
DRAG.csv                     20 000-row time history (one row per iteration)
DRAG_COEFFICIENT.csv         time-history of C_d
LIFT.csv, LIFT_COEFFICIENT.csv
residuals.csv                Solver residuals over iterations
residuals_and_forces.png     Diagnostic chart
viz/                         PNG snapshots of geometry and flow slices
```

Variation across samples:
- **Scale.** Quarter-scale (`qrtr_scale`, cheaper) vs full-scale.
- **Rear-end style.** `estate` vs `fastback`.
- **Underbody.** Smooth vs detailed.
- **Design parameters.** A morphing cage around the AeroSUV
  baseline; each sample perturbs a set of design parameters
  (informed by Honda) via a Latin-hypercube sample.

**What "one training instance" looks like to a Transolver.** After
preprocessing the VTP/VTU files into tensors:

- **Input.** A subsampled point cloud of size roughly `N = 100 000`
  from `merged_surfaces.vtp`. Per point: `(x, y, z, nₓ, n_y, n_z)`
  → shape `(100 000, 6)`.
- **Target.** Per-point time-averaged pressure and wall shear stress
  from the same VTP → shape `(100 000, 4)`: one pressure scalar and
  a 3-vector shear.
- **Auxiliary target.** Scalars `C_d`, `C_l` (from the time-averaged
  columns of the CSVs).

That is the "point cloud in, field out" instance. Volume-field
surrogates additionally subsample from the VTU.

### 5.2 DrivAerNet++ — one sample

Dataset: `github.com/Mohamedelrefaie/DrivAerNet`. Data on Harvard
Dataverse. 8 150 car designs, 39 TB total. Solver: **OpenFOAM**.
Ran on MIT Supercloud, ~3 M CPU-hours.

One car (`DrivAer_F_D_WM_WW_3000` — the naming encodes body style,
underbody, wheels, mirrors, run id) has roughly:

```
STL/                         DrivAer_<id>.stl   -- surface triangle mesh
PointClouds/                 Dense + sparse (.npy) point-cloud sampling
SurfacePressureVTK/          <id>.vtk with point_data:
                               - 'p'    scalar pressure per point
                               - 'U'    3-vector velocity per point
VolumeVTK/                   Volume fields (p, U, turbulence quantities)
SurfaceFieldsVTK/            Wall shear stress + coefficient of pressure (Cp)
AeroCoefficients_....csv     Rows keyed by car id, columns:
                               - Average Cd, Average Cl, moment coefficients
                               - Frontal area
Parameters/                  26-column parametric descriptor per car
SemanticLabels/              29 component labels (wheels, mirrors, ...)
Renderings/                  2D photorealistic PNGs
Sketches/                    Canny-edge and CLIPasso stylised sketches
```

Variation across samples:
- **Body style.** Fastback / notchback / estateback.
- **Powertrain.** ICE vs EV (differences show up in underbody and
  cooling ducts).
- **Underbody and wheels.** Smooth / detailed variants.
- **Design parameters.** 26-dim parametric family around the baseline
  DrivAer geometry.

The reference `DrivAerNet_tutorial.py` shows the loading pattern:

```python
import pyvista as pv, numpy as np
mesh = pv.read('../SurfacePressureVTK/DrivAer_F_D_WM_WW_3000.vtk')

# Subsample to 100 000 points for training
n_points = 100_000
indices  = np.random.choice(mesh.n_points, n_points, replace=False)

points     = mesh.points[indices]              # (100 000, 3)
pressures  = mesh.point_data['p'][indices]     # (100 000,)
velocities = mesh.point_data['U'][indices]     # (100 000, 3)
```

**What "one training instance" looks like to a Transolver.**

- **Input.** `points` (or points + surface normals, computed from
  the mesh) → shape `(100 000, 3)` or `(100 000, 6)`.
- **Target.** `pressures` + `velocities` → shape `(100 000, 4)`.
  Optionally `wall_shear` from `SurfaceFieldsVTK` (4-vector).
- **Auxiliary target.** `Cd`, `Cl` from the CSV.

Nearly identical shape to the SUV instance — different solver,
different variation axes, but structurally the same "geometry
tensor in, field tensor out". This shape-uniformity is exactly what
makes a single architecture (Transolver / PXtransolver) applicable
across both.

### 5.3 Side-by-side summary

| Property                 | SHIFT-SUV                       | DrivAerNet++              |
| ------------------------ | ------------------------------- | ------------------------- |
| Solver                   | Luminary Cloud, DES             | OpenFOAM, RANS            |
| Total size               | 18.3 TB (25k target)            | 39 TB                     |
| Samples                  | Thousands, growing              | 8 150                     |
| Time-dependent           | Yes (transient DES)             | No (steady RANS)          |
| Ground truth for surface | `p`, wall shear stress          | `p`, `U`, Cp, wall shear  |
| Ground truth for volume  | `p`, `U`                        | `p`, `U`, turbulence      |
| Geometry format          | STL (open + closed variants)    | STL, point clouds (.npy)  |
| Aero coefficients        | 20k-row time histories          | Time-averaged in CSV      |
| Variation                | Morphing cage + rear/underbody  | 26 params + body style    |
| License                  | CC-BY-NC-4.0                    | CC-BY-NC-4.0              |

### 5.4 Practical training-side implications

Two things to note about how you'd actually train PXtransolver on
this data.

- **You never see the whole mesh.** Both datasets store multi-million
  point meshes; you subsample to `N ≈ 10⁴ – 10⁵` per training step.
  The number `M` of slices in Physics-Attention is fixed and tiny
  by comparison — the whole point of the architecture is that
  training scales linearly in `N`.
- **Data pipelines dominate wall-clock.** A 39 TB dataset with
  per-sample VTK reads is I/O-bound long before it is
  compute-bound. Preprocessing everything to `.npy` or `.h5` tensor
  files up front is standard; PhysicsX's production stack almost
  certainly does this at ingest time.

---

## 6. Putting it together — the workflow shape

The end-to-end shape of training a PXtransolver-class model:

1. **Ingest.** Convert VTU/VTP/STL files into per-sample tensor
   files. Log each dataset version as a W&B artifact (this is why
   the workshop leans hard on the artifact lineage story).
2. **Pretrain.** On a large corpus of cheap simulations (SHIFT-SUV
   at quarter-scale, or an in-house synthetic dataset), for many
   epochs.
3. **Fine-tune.** On the high-fidelity slice of the data — full-scale
   DES from SHIFT-SUV, or a customer-specific run set.
4. **Task-specific fine-tune.** On the operating envelope the
   customer cares about (an aggressive-camber roofline family, a
   near-stall AoA range).
5. **Evaluate.** Relative L2 on fields, absolute error and Spearman
   on `C_d` / `C_l`, physics-consistency checks, side-by-side field
   plots on 20 canonical cases.
6. **Register.** Link the final checkpoint into the W&B Registry
   with `staging` → `production` aliases; automations notify the
   customer team on promotion.

Each stage's dataset, checkpoint, and evaluation report is an
artifact; the artifact graph is the auditable record of what trained
what.

---

## 7. Sources and further reading

- **Transolver (base paper).** Wu, Luo, Wang, Wang, Long (2024),
  *Transolver: A Fast Transformer Solver for PDEs on General
  Geometries*, arXiv:2402.02366.
- **SHIFT-SUV dataset.** Luminary Cloud, on Hugging Face at
  `huggingface.co/datasets/luminary-shift/SUV`.
- **DrivAerNet++.** Elrefaie et al.,
  `github.com/Mohamedelrefaie/DrivAerNet`; data on Harvard Dataverse.
- **AirfRANS.** Bonnet et al. (2022), NeurIPS Datasets track — the
  smaller academic sibling used in the workshop repo.
- **CFD surrogates background.** `docs/explainers/cfd_surrogate_models.md`
  in this repo; broader reading path in `further_reading.md`.
