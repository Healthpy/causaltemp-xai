# Benchmark Configurations

Two presets, defined as a `BenchmarkConfig` dataclass in `causaltemp_xai/config.py` (Stage 2).
The `smoke` preset is the default for tests and CI; the `full` preset is the locked
paper configuration run once for results.

## FULL (locked paper config — MVP §WP1)

| Param | Value |
|---|---|
| `k` (latents/vars) | 10 |
| `L` (lag) | 1 |
| `sparsity` (s/k²) | 0.2 |
| `noise_type` | `laplace` |
| `T` (length) | 100 |
| `N` (sequences) | 10_000 |
| `seed` | 42 |
| split | 60 / 20 / 20 train/val/test |

> With `L=1` the lagged graph is **trivially acyclic across time** (edges only go
> backward in time, no contemporaneous edges). The "acyclicity" unit test should
> assert there are no contemporaneous (lag-0) edges, i.e. the VAR has no
> instantaneous self/cross dependence — which holds by construction.
>
> **Stationarity caveat:** `generator._stabilise` rescales *each* lag matrix to
> spectral radius ≤ 0.9 independently. This only guarantees VAR stationarity at
> **`L=1`** (the locked config). If a future config raises `L` to 3, per-matrix
> scaling is insufficient — the companion-form spectral radius must be checked
> instead. Do not bump `L` without revisiting `_stabilise`.

## SMOKE (CI / fast iteration / tests)

| Param | Value |
|---|---|
| `k` | 5 |
| `L` | 1 |
| `sparsity` | 0.2 |
| `noise_type` | `laplace` |
| `T` | 30 |
| `N` | 500 |
| `seed` | 0 |

## Optional H5-hint config (time permitting, Stage 2 optional / Stage 8 descriptive)

Same as FULL but `sparsity = 0.1`. Used only for a descriptive CF-faith comparison
across sparsity levels — **no ablation claim** (underpowered, per MVP H5 = "hint only").

## Shift-VR-lite environments (Stage 7)

- **Base env** `E_base`: the FULL config above.
- **Shifted env** `E_shift`: identical SCM (same `graph`, same `mechanisms`, same `seed`),
  but `noise_type = "uniform"`. This isolates a distributional shift in the innovations
  while holding the causal structure fixed, so a flip in validity/CF-faith reflects
  robustness, not a different ground truth.

**Shift-VR protocol (pinned, Stage 7):** keep the **frozen base classifier** (no
retraining). For each method, **generate fresh CFs for `E_shift`'s test inputs** and
measure validity there; report `Shift-VR = validity(E_shift) / validity(E_base)` per
method. Do **not** track "the same CF arrays remaining valid" — under one frozen
classifier a fixed CF's validity cannot change across environments, so that framing is
incoherent. The signal comes from generating recourse for shifted inputs.

## Intervention-time convention (Stages 4–5)

`derive_intervention_t(x, x_cf, tol=1e-6) -> int`:
the smallest `t` such that `max_j |x_cf[t, j] − x[t, j]| > tol`.
If no timestep differs, return `T-1` (degenerate; CF == original).
This single rule is applied to **every** method's output so CF-faith is comparable
across methods.
