# Method provenance notes

Seed for the eventual M3 method-provenance table (docs/PROJECT_PLAN.md P0
item 5). This is a short disclosure of what two attribution baselines in
`causaltemp_xai/methods/attribution/` actually are, not the full table.

Both classes below were originally named after the paper they were meant to
approximate (`Dynamask`, `TimeSHAP`). That violated this lab's standing
decision that no proxy implementation ships under an original method's name
(docs/PROJECT_PLAN.md Standing Decision #3) and was tracked as risk R4
("misrepresented baselines... fatal if published"). Both have been renamed
and disclosed; see each class's module docstring for the full technical
diff against the paper it approximates.

## `FDSaliency` (`causaltemp_xai/methods/attribution/fd_saliency.py`)

- **Implementation source:** this codebase (ported from
  `causal_tscf_bench/methods/attribution/dynamask.py`, then renamed here).
  Acknowledged proxy, not a reimplementation.
- **What it computes:** forward finite-difference numerical gradient of the
  target class's predicted probability with respect to each `(t, feature)`
  coordinate, `(P(y=target | x + eps*e_{t,m}) - P(y=target | x)) / eps`.
  Closer in spirit to vanilla gradient saliency than to any mask-learning
  method.
- **Paradigm/paper it was meant to approximate:** Dynamask (Crabbe & van der
  Schaar, 2021, "Explaining Time Series Predictions with Dynamic Masks",
  ICML 2021) — a *learned soft temporal mask* optimized against a
  perturbation objective (prediction-preservation vs. sparsity/entropy),
  with extremal-mask and rate-distortion extensions.
- **What the real thing would need:** the mask-learning optimization loop
  itself (gradient descent over a mask tensor against Dynamask's objective,
  including its blur/baseline perturbation operator and the
  extremal-mask variant) is not implemented anywhere in this codebase. An
  official implementation exists:
  [`JonathanCrabbe/Dynamask`](https://github.com/JonathanCrabbe/Dynamask)
  (the paper authors' own repo). Integrating it would mean vendoring or
  depending on that package and adapting its optimization loop to this
  codebase's `TSClassifier` interface (`predict_proba`) — not yet attempted,
  not scoped for this task.

## `MCMaskSHAP` (`causaltemp_xai/methods/attribution/mc_mask_shap.py`)

- **Implementation source:** this codebase (ported from
  `causal_tscf_bench/methods/attribution/timeshap.py`, then renamed and
  seeded here). Acknowledged proxy, not a reimplementation.
- **What it computes:** a flat Monte-Carlo random-coalition Shapley-value
  estimator — i.i.d. uniformly random binary masks drawn directly over the
  full `(T, k)` grid, no pruning stage, no hierarchy. In the same general
  family as KernelSHAP / permutation-SHAP, not TimeSHAP's method. Its RNG is
  now seeded (`seed: int | None = None` constructor argument; `seed=None`
  falls back to a fixed internal seed of `0`, so the class is
  reproducible-by-default like every other seeded stochastic component in
  this codebase — see the class docstring for the precedent this follows).
- **Paradigm/paper it was meant to approximate:** TimeSHAP (Bento et al.,
  2021, "TimeSHAP: Explaining Recurrent Models through Sequence
  Perturbations", KDD 2021) — a *structured*, hierarchical Shapley estimator
  that first prunes to the relevant event horizon (event-level Shapley
  pruning), then estimates feature-level Shapley values, then optionally
  cell-level (event x feature) values, each level's coalition sampling
  informed by the coarser level above it, with a learned/empirical "average
  event" perturbation operator rather than an arbitrary zero/mean baseline.
- **What the real thing would need:** the event-pruning stage and the
  hierarchical feature/cell coalition scheme are not implemented anywhere in
  this codebase. An official implementation exists:
  [`feedzai/timeshap`](https://github.com/feedzai/timeshap) (also on PyPI as
  `timeshap`, the paper authors' own package). Integrating it would mean
  adding it as a dependency and adapting its pruning/explainer API to this
  codebase's `(N, T, k)` tensor convention and `TSClassifier` interface —
  not yet attempted, not scoped for this task.

## Explicitly out of scope here

`causaltemp_xai/methods/concept/cbm_t.py` (`CBMT`) has a related but
lower-severity naming concern (a real, if minimal, concept-bottleneck-model
architecture, not an unrelated numerical trick under another paper's name).
Not covered by this note; the PI will scope its provenance separately.
