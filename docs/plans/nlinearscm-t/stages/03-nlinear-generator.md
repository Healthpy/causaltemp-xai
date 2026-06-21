# Stage 3: NlinearSCMT generator

**Goal**: Add an `NlinearSCMT` generator that produces bounded multivariate time series from an `MLPMechanism` over the same graph machinery as `LinearSCMT`, with balanced binary labels.
**Dependencies**: Stage 1 (`Mechanism`, generator refactor), Stage 2 (`MLPMechanism`).

---

## Background

`NlinearSCMT` mirrors `LinearSCMT`'s structure (same graph sampling, same noise distributions, same label rule) but swaps the linear mechanism for an `MLPMechanism`. The forward simulation becomes:

```
x_t = mechanism.forward_numpy(window_{t-L..t-1}) + eps_t
```

Reuse as much of `LinearSCMT` as possible: the graph sampling (`_build_graph_and_mechanisms` → reuse the graph half), `_sample_noise`, burn-in, and the median-threshold label rule are identical. The only differences are mechanism construction and the per-step forward call (already abstracted in Stage 1).

---

## Steps

1. **Add `NlinearSCMT` to `causaltemp_xai/benchmark/generator.py`** (or a sibling module if the file grows large — keep both generators discoverable from `causaltemp_xai.benchmark`).
   - Constructor: `k, L, sparsity, noise_type, T, N, seed` (same as linear) **plus** nonlinear hyperparams `hidden, gain, decay_range, spectral_cap, init_gain, activation` with research-backed defaults (hidden≈16–32 for k≤10; `gain=0.8`; `decay_range=(0.3,0.8)`; `spectral_cap=0.9`; `init_gain=0.7`).
   - In `__init__`: sample the graph (reuse the linear graph sampler — extract a shared `_sample_graph(k, L, sparsity, rng)` helper so both generators share it and the no-self-loop-at-lag-1 rule), then build `self.mechanism = MLPMechanism.random(graph, hidden, rng, decay_range, gain, spectral_cap, init_gain)`.
   - ⚠️ **RNG-order invariant (guards the frozen linear path):** extracting `_sample_graph` out of `LinearSCMT._build_graph_and_mechanisms` must reproduce `LinearSCMT`'s **exact RNG draw sequence** (currently: per lag, the Bernoulli edge mask is drawn *before* the uniform coefficients — `generator.py:147,153`). Reordering those draws shifts every `LinearSCMT` output and silently un-freezes v0.1. Re-run the Stage-1 golden (now a full-`X` compare) after this extraction to confirm.
   - ⚠️ **Construction-order invariant (Axis-D / `shifted_config`):** build the graph and the `MLPMechanism` in `__init__` *before* any noise is sampled (mirror `LinearSCMT`, where noise is drawn later in `generate()`). This is what makes a noise-only `shifted_config` yield a **bit-identical** graph + mechanism — the Stage-5 Axis-D invariant depends on it. Do not draw noise-type-dependent randomness before mechanism construction.

2. **Implement `generate(burn_in=100)`.**
   - Same skeleton as `LinearSCMT.generate`: init first L steps with small noise, draw all noise up front, simulate forward calling `self.mechanism.forward_numpy` on the batched `(N, L, k)` window, add noise, discard burn-in.
   - **Divergence handling**: after simulation, if any trajectory exceeds a bound (e.g. `max|x| > clip` or non-finite), either (a) resample those trajectories with fresh noise (preferred — preserves N), or (b) clip and log. Given the Stage-2 contractive design this should rarely trigger; treat a high trigger rate as a signal to lower `gain` / tighten `spectral_cap` (record under Decisions).
     - ⚠️ **Resampling must stay seed-deterministic** so the "same seed ⇒ identical `X`" check (Verification) still holds: draw the fresh noise from the **same seeded generator RNG** in a fixed, order-independent-of-wall-clock way (e.g. iterate diverging trajectory indices in sorted order). Don't spin up a fresh unseeded RNG.
     - Because the contractive design means this path rarely fires, it is otherwise untested — add a unit test that forces divergence (e.g. construct/temporarily monkeypatch a high-`gain` mechanism or shrink `clip`) and asserts the resampled output is finite *and* reproducible under a fixed seed.
   - Labels: identical median-threshold rule on `X[:, -1, 0]` → balanced binary `Y`.
   - Return `{"X", "Y", "graph", "mechanism"}` (same contract as the refactored `LinearSCMT`).

3. **Write `tests/test_nlinear_generator.py`** (mirror `tests/test_generator.py` structure).
   - Shapes: `X (N,T,k)`, `Y (N,)`, `graph (k,k,L)`, `mechanism` is an `MLPMechanism`.
   - Labels binary, both classes present, ~balanced.
   - **Boundedness/finiteness** over T=200, N≥200, across seeds (the key stability check).
   - Sparsity within tolerance (shared graph sampler ⇒ same guarantee as linear).
   - **Nonlinearity at the data level**: a linear VAR(L) fit to the generated `X` leaves a clearly larger residual than for `LinearSCMT` data (the data is genuinely nonlinear). Keep the threshold loose to avoid flakiness.
   - Reproducibility: same seed ⇒ identical `X`.
   - KS-test rejects Gaussian for the marginal (non-Gaussian noise propagates) — keep as in linear tests, loosen if propagation through tanh softens it.

---

## Verification

> Run with `uv run --offline …` (see `resources/commands.md`).

- [ ] `uv run --offline pytest tests/test_nlinear_generator.py -v` green.
- [ ] Trajectories finite & bounded over T=200 across ≥20 seeds (success criterion).
- [ ] Linear-VAR-fit residual on nonlinear data >> on linear data (nonlinearity confirmed).
- [ ] Labels balanced; both classes present.
- [ ] Reproducible under fixed seed (including the divergence-resample path).
- [ ] Full suite still green.

---

## Commit

`feat(scm): add NlinearSCMT generator (MLP-mechanism additive-noise SCM)`
