
We're using Weights & Biases to build a machine-learning surrogate that predicts surface pressure on a high-lift
aircraft wing — the kind of model that turns a multi-hour CFD simulation into a millisecond prediction and this
workshop teaches the tooling patterns you need to ship one to production.


---
1. The problem we're modelling.

    CFD is expensive. To predict how air flows over a wing at a given angle of attack, engineers run
    computational-fluid-dynamics simulations that take hours on many CPU cores and produce gigabytes of output per
    configuration. What we want is a model that learns from a corpus of past CFD runs and produces the pressure field at
    inference time in milliseconds. That's a "CFD surrogate" — and it's a real production workflow at PhysicsX.

---
2. The dataset behind the example.

    The dataset is called HiLiftAeroML. NASA published it in mid-2026. It contains 1,800 high-fidelity simulations across
    180 geometry variants of the CRM-HL — the standard high-lift wing the industry has been benchmarking against for
    years — sweeping 10 angles of attack per geometry, from cruise up through near-stall. Each simulation ships as
    surface fields and volume fields. But the surface archive alone is around 13 GB per case, compressed.

---
3. Why we're on synthetic data, not the real thing.

    13 GB per case doesn't fit on a workshop laptop. So we're not reading the real HiLift boundary files — we've got a
    demo generator that produces synthetic point clouds with exactly the same shape and file format as the real thing.
    Same tensor sizes, same NPZ schema, same three-component wing topology (slat, main, flap, in the correct CRM-HL
    proportions). The target field is a physically-motivated Cp surrogate — not real CFD, but shaped to have the right
    spatial patterns: leading-edge suction, flap loading, pressure recovery. The important property: swap the demo
    generator for a real-VTU reader, and every notebook downstream stays byte-identical.

---
4. What the model actually predicts — the target variable.

    The target is Cp, the pressure coefficient. It's non-dimensionalised static pressure — local pressure minus
    freestream, divided by dynamic pressure — so it's comparable across flight conditions. Negative Cp = suction (below
    freestream, which is where the lift comes from); positive Cp = compression (usually the stagnation point at the
    leading edge). Real Cp on a high-lift wing runs roughly between −3 and +1; our demo target sits in about [−1, +0.25],
    structurally the same shape but shrunk in magnitude. The model outputs one scalar Cp per point on the wing surface.
    Each case carries 8,192 points, so per case we produce 8,192 predicted numbers.

---
5. The model itself.

    We're using Transolver++, an open-source neural PDE-solver architecture from Tsinghua's ML group. Two things worth
    knowing:
    leading-edge suction, flap loading, pressure recovery. The important property: swap the demo generator for a real-VTU reader, and every notebook downstream stays byte-identical.

---
6. What you should expect to see when it runs.

    Fifty epochs of training. Validation loss starts around 1.0 and should drop to somewhere between 0.05 and 0.20 by the end. Mean absolute error settles around 0.05 to 0.10 in physical Cp units. That's coarse-approximation territory — not
    benchmark-grade — but the model is legitimately learning the spatial pressure structure.

    When you look at the prediction plots you'll see three visually distinct bands with white space between them. Those are the slat, main wing, and flap of the deployed high-lift configuration. The gaps aren't rendering artefacts — they're the
    actual physical slots between the deployed elements. And the hardest region is the leading-edge suction band on the main wing — that's where a small model's biggest error concentrates, and it's exactly the aerodynamically expected failure
    mode.

---

None of this is about beating a benchmark. The workshop uses a domain-realistic example so that when you go back to your own CFD-surrogate work, the Weights & Biases tooling pattern you learn here — Registry, artifacts, sweeps, reports, async evaluation, multistage handoff — maps onto real work byte-identically. The demo is calibrated for pipeline visibility, not surrogate quality. Come away with the tooling; the model is the vehicle, not the destination.

---
Outstanding Questions: 

**Why three separate bands on the wing?**

CRM-HL is a three-element wing: slat, main, flap. At cruise they tuck flush and look continuous. On takeoff and landing they deploy out, creating physical slots that are aerodynamically essential (they re-energise the boundary layer downstream and let the wing generate 2–3× the lift of a plain single-element wing).

**Why Cp instead of raw pressure?** 

Cp non-dimensionalises out flight speed and altitude, so it's comparable across cases and directly interpretable to an aerodynamicist.

**Why patch-based training on 1024 points if the case has 8192?** 

Because of attention cost and memory. They scale with point count. At real scale (million-point meshes) full-case forwards don't fit. Patching is what makes the pattern portable from demo to
production.

**Would this beat classical CFD?** 

no. It's a surrogate — trained on CFD outputs — trading accuracy for latency. A model that fits on a laptop won't beat a proper solver; the goal is milliseconds-vs-hours at acceptable accuracy for
exploration.