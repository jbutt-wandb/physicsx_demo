# Further Reading — ML Surrogates for CFD

A structured research path for going deeper than the introductory
lecture notes in `cfd_surrogate_models.md`. Reviews first, then
foundational papers by model family, then modern work, then datasets,
then free courses, then how to keep current. Finish with a suggested
six-week reading plan if you want a schedule.

Nothing here is behind a paywall unless noted; every arXiv link is the
open version.

---

## A. Start with reviews — build the map before picking a paper

The reviews are dense but they give you the vocabulary and the sense of
which papers are load-bearing. Read (or at least skim) two of these
before diving into individual papers.

- **Brunton, Noack, Koumoutsakos (2020)** — *Machine Learning for Fluid
  Mechanics*, Annual Review of Fluid Mechanics 52. The canonical entry
  point. Heavy focus on the fluid-dynamics side of the intersection.
  Widely cited; the vocabulary in this paper is the vocabulary the rest
  of the field uses.
- **Karniadakis et al. (2021)** — *Physics-Informed Machine Learning*,
  Nature Reviews Physics 3. Broader than fluids: covers PINNs, neural
  operators, and hybrid solvers as three different ways to inject
  physics into ML. Best single reference for how the model families
  relate to each other.
- **Vinuesa & Brunton (2022)** — *Enhancing Computational Fluid Dynamics
  with Machine Learning*, Nature Computational Science 2. Framed for
  people who already know CFD and want to understand what ML changes.
  Shorter than the two above.
- **Xie, Takikawa, Saito, et al. (2022)** — *Neural Fields in Visual
  Computing and Beyond*, EUROGRAPHICS state-of-the-art report. Not
  CFD-specific, but this is *the* survey of the neural-field literature
  the CFD surrogate community draws from.

---

## B. Foundational papers, by model family

Pick one anchor paper per family. If you have limited time, the
starred one is the highest-leverage read in its group.

### B.1 Neural fields / implicit neural representations

- **Sitzmann et al. (2020)** — *Implicit Neural Representations with
  Periodic Activation Functions* (SIREN). Shows that sinusoidal
  activations let a plain MLP fit sharp, high-frequency signals — the
  trick that made neural fields practical for physics.
- **Mildenhall et al. (2020)** — *NeRF: Representing Scenes as Neural
  Radiance Fields*. Not CFD, but the paper that demonstrated a
  coordinate-conditioned MLP could represent a rich 3D field. Almost
  every subsequent neural-field paper cites it.
- **Tancik et al. (2020)** — *Fourier Features Let Networks Learn High
  Frequency Functions in Low Dimensional Domains*. Explains why raw
  coordinates fail as input and the standard fix (Fourier / positional
  encoding).

### B.2 Physics-informed neural networks (PINNs)

- **Raissi, Perdikaris, Karniadakis (2019)** — *Physics-Informed Neural
  Networks: A Deep Learning Framework for Solving Forward and Inverse
  Problems Involving Nonlinear Partial Differential Equations*, JCP
  378. The paper that started the modern PINN wave. Read this and you
  understand why PINNs are structurally different from surrogates: a
  PINN *solves* one PDE per training run; a surrogate *predicts across
  many* instances of a PDE.

### B.3 Neural operators (learning the map from input function to solution)

- **★ Li et al. (2021)** — *Fourier Neural Operator for Parametric
  Partial Differential Equations* (FNO), ICLR. Shows how to learn
  operators between function spaces in the Fourier domain. The most
  read paper in this family.
- **Lu et al. (2021)** — *Learning Nonlinear Operators via DeepONet
  Based on the Universal Approximation Theorem of Operators*, Nature
  Machine Intelligence 3. Different flavour of operator learning; the
  theoretical companion to FNO.
- **Kovachki et al. (2023)** — *Neural Operator: Learning Maps Between
  Function Spaces With Applications to PDEs*, JMLR 24. The unifying
  treatment covering FNO, DeepONet, and Graph Neural Operator under
  one framework. Read after FNO to see the general theory.

### B.4 Graph / mesh-based simulators

- **★ Pfaff et al. (2021)** — *Learning Mesh-Based Simulation with
  Graph Networks* (MeshGraphNets), ICLR. DeepMind's line of work; the
  most CFD-like architecture from the graph-networks side. Directly
  handles irregular meshes.
- **Sanchez-Gonzalez et al. (2020)** — *Learning to Simulate Complex
  Physics with Graph Networks*, ICML. Earlier paper that established
  the pattern.

### B.5 Transformer surrogates — where the field is actively moving

- **★ Wu, Luo, Wang, Long (2024)** — *Transolver: A Fast Transformer
  Solver for PDEs on General Geometries*, ICML. Introduces a slicing
  attention mechanism to keep transformer surrogates tractable on
  million-point meshes.
- **Alkin, Fürst, Schmid, et al. (2024)** — *Universal Physics
  Transformers*. Broader ambition — one transformer architecture for
  multiple physical domains. Close in spirit to what PhysicsX is doing.
- **Cao (2021)** — *Choose a Transformer: Fourier or Galerkin*, NeurIPS.
  Earlier work motivating why attention is a good primitive for PDE
  surrogates.

---

## C. Datasets and benchmarks — the "just try it" layer

Reading three papers is useful. Training a small model on one of these
datasets is transformative. In increasing order of scale:

- **AirfRANS** (Bonnet et al., NeurIPS Datasets 2022). 2D RANS
  simulations of NACA airfoils. Few gigabytes; runs on a laptop. The
  dataset the workshop uses.
- **PDEBench** (Takamoto et al., NeurIPS Datasets 2022). Broad
  benchmark across many PDEs (Burgers, Darcy, Navier-Stokes, ...).
  Right for comparing model families on the same tasks. Bigger than
  AirfRANS but still workstation-tractable.
- **DrivAerNet++** (Elrefaie et al., NeurIPS 2024). 3D car aerodynamics
  at production scale. Terabytes total. This is the closest public
  proxy for what PhysicsX actually deals with; expect to need real
  compute.

Companion codebases worth having open while reading the corresponding
papers:

- **Neural Operator (`neuraloperator`)** — reference FNO / DeepONet
  implementations from the original authors.
- **DeepMind Graph Nets** — reference MeshGraphNets code.
- **NVIDIA Modulus** — vendor-flavoured but well-documented framework
  covering PINNs, FNOs, and geometric surrogates. Great for reading
  the code even if you don't use the framework.

---

## D. Free books and courses — best for building intuition

- **Nils Thuerey — *Physics-Based Deep Learning*.** Free online at
  physicsbaseddeeplearning.org. Opinionated, code-heavy, has
  end-to-end notebooks. Read this cover to cover if you want one
  concrete resource.
- **Steven Brunton — YouTube channel.** Playlists on SINDy, POD,
  dimensionality reduction, and data-driven dynamics. Companion
  textbook: Brunton & Kutz, *Data-Driven Science and Engineering* (2nd
  ed. 2022). The lectures are the most accessible on-ramp to the
  applied-math side of this whole area.
- **NVIDIA Modulus documentation and tutorials.** Vendor-flavoured but
  the tutorials walk through PINN and neural-operator implementations
  end-to-end. Fastest way to see the pieces plugged together.
- **CFD Python by Lorena Barba (12 Steps to Navier-Stokes).** Free
  Jupyter notebooks that teach you to write a CFD solver from scratch.
  Not ML — but understanding the physics you're surrogating is
  load-bearing, and this is the friendliest available introduction.

---

## E. Communities and how to keep current

Papers-per-week in this space is high enough that a keyword feed
matters more than trying to read everything.

- **Workshops beat conferences for state-of-the-art.** Track:
  - NeurIPS *Machine Learning for the Physical Sciences* (ML4PS) —
    annually.
  - ICML *AI4Science*.
  - ICLR *Neural Fields Across Fields*.
  Papers accepted to these are usually 6–12 months ahead of what
  shows up in the main-conference tracks.
- **arXiv-sanity or Semantic Scholar feeds.** Subscribe to keywords
  "neural operator", "PDE surrogate", "physics-informed", "Fourier
  neural operator". Categories to follow: `cs.LG`, `physics.flu-dyn`,
  `physics.comp-ph`.
- **Semantic Scholar / Connected Papers.** From any anchor paper in
  section B, follow the "cited by" graph to see the frontier that
  built on it. This gets you to current work faster than keyword
  search does.
- **PhysicsX and DeepMind blogs.** PhysicsX publishes engineering-facing
  posts about their production stack; DeepMind's science blog covers
  MeshGraphNets and its successors.
- **X / Twitter accounts.** The Karniadakis, Anima Anandkumar, Nils
  Thuerey, Steven Brunton, and DeepMind Science accounts announce most
  of the notable work in this niche within a day.

---

## F. Suggested six-week reading plan

If you want a schedule rather than a menu, here's one that starts from
zero and ends able to read and evaluate current papers. Assumes ~5
hours per week.

**Week 1 — Landscape.** Read Vinuesa & Brunton (2022) end to end. Skim
Karniadakis et al. (2021) for the taxonomy. Goal: know the three
families (surrogate, PINN, neural operator) and what problem each one
solves.

**Week 2 — Foundations.** Read Raissi et al. (2019) on PINNs. Read the
first half of Xie et al. (2022) on neural fields. Goal: understand the
two contrasting paradigms — physics as loss vs. data as loss.

**Week 3 — Neural operators.** Read the FNO paper (Li et al. 2021).
Skim DeepONet (Lu et al. 2021). Optionally skim Kovachki et al. (2023)
for the unified view. Goal: understand what an "operator" is and why
learning one is more general than learning a fixed-resolution
surrogate.

**Week 4 — Modern surrogate architectures.** Read MeshGraphNets (Pfaff
et al. 2021) and Transolver (Wu et al. 2024). Goal: see the two
architectural threads — graph networks and transformers — and how they
handle irregular geometry.

**Week 5 — Hands-on.** Pick AirfRANS or one PDEBench task. Train a
small baseline (an MLP, then an FNO). Log everything to W&B so you can
compare. Goal: internalise the training loop and evaluation shape.

**Week 6 — Frontier and application.** Read Alkin et al. (2024) on
Universal Physics Transformers, one recent workshop paper of your
choice from ML4PS 2024 or 2025, and the PhysicsX engineering blog.
Goal: know what's state-of-the-art and where the open problems are.

---

## G. Reading-hygiene tactics

Three tactics that pay for themselves quickly in this literature.

1. **Read the reviews' reference lists, not the abstracts.** The
   reviews' bibliographies are curated — the papers they cite in
   context are the ones worth reading. Skimming just the abstract of a
   random arXiv paper is low signal.
2. **Anchor every model family to a working repo.** Reading a neural
   operator paper without the accompanying `neuraloperator` codebase
   open is fine; doing so without ever running the code leaves you
   with a fuzzy mental model. Every foundational paper in section B
   has an official implementation.
3. **Write a one-paragraph summary per paper as you go.** After each
   read, note: (a) what problem the paper solves, (b) what the key
   idea is in one sentence, (c) what it *doesn't* address. Six weeks
   of these adds up to a personal survey that beats any external one
   for your specific use case.
