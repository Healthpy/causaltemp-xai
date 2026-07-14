# Method provenance notes

Seed for the eventual M3 method-provenance table (docs/PROJECT_PLAN.md P0
item 5). This is a short disclosure of what the attribution methods in
`causaltemp_xai/methods/attribution/` actually are.

Both attribution methods that were previously acknowledged *proxies* (named
`FDSaliency`/`MCMaskSHAP` after being renamed off the original method names
under Standing Decision #3 / risk R4) have now been **replaced by the official
implementations**. No attribution method in this package is a proxy any more.

## `TimeSHAP` (`causaltemp_xai/methods/attribution/timeshap.py`)

- **Implementation source:** the official `timeshap` library (feedzai/timeshap,
  the paper authors' own package), added as a project dependency. This module
  is a thin adapter, not a reimplementation.
- **What it computes:** genuine TimeSHAP cell-level Shapley values (Bento et
  al., 2021, "TimeSHAP: Explaining Recurrent Models through Sequence
  Perturbations", KDD 2021), pivoted into a dense `(T, k)` map.
- **Adapter choices (disclosed):** the pruning stage is bypassed
  (`pruned_idx = 0`) so the map spans the whole sequence, and all cells are
  requested (`top_x_events = T`, `top_x_feats = k`); the perturbation baseline
  is TimeSHAP's average event (from `fit`, else a zero event). These are
  interface-adaptation choices, not changes to TimeSHAP's estimator. See the
  module docstring for the full detail.
- **Dependency note:** `timeshap` imports `shap.explainers._kernel.Kernel`,
  removed in `shap>=0.44`, so `shap` is pinned to `0.42.1`. Nothing else in the
  repo uses `shap`.
- **Replaces:** the former `MCMaskSHAP` proxy (a flat Monte-Carlo
  random-coalition Shapley sampler with no pruning/hierarchy), now removed.

## `Dynamask` (`causaltemp_xai/methods/attribution/dynamask.py`)

- **Implementation source:** the authors' official Dynamask implementation
  ([`JonathanCrabbe/Dynamask`](https://github.com/JonathanCrabbe/Dynamask)),
  vendored as a git submodule at `third_party/dynamask_repo/` (mirroring
  `third_party/cfts_repo/`). This module is a thin adapter over the vendored
  `attribution.mask.Mask` optimizer.
- **What it computes:** genuine Dynamask (Crabbe & van der Schaar, 2021,
  "Explaining Time Series Predictions with Dynamic Masks", ICML 2021) — a
  learned soft temporal mask `M in [0, 1]^{T x k}` fit by gradient descent
  against Dynamask's perturbation objective (prediction-preservation + size
  regulator + temporal-smoothness penalty), returned as the `(T, k)` map.
- **Adapter choices (disclosed):** the classifier's differentiable path
  (`torch_logits` + softmax) is the black box `f`; the perturbation operator is
  Dynamask's Gaussian blur (default) or fade-moving-average; the loss is
  Dynamask's classification `log_loss`. Dynamask preserves the model's full
  predictive distribution, so the mask is class-agnostic — `target_class` is
  accepted for interface compatibility but does not change the result.
- **Replaces:** the former `FDSaliency` proxy (a per-coordinate
  finite-difference numerical gradient), now removed.

## Explicitly out of scope here

`causaltemp_xai/methods/concept/cbm_t.py` (`CBMT`) has a related but
lower-severity naming concern (a real, if minimal, concept-bottleneck-model
architecture, not an unrelated numerical trick under another paper's name).
Not covered by this note; the PI will scope its provenance separately.
