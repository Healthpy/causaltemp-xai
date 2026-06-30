# Stage 4: CF-faith generalization + oracle structural-CF

**Goal**: Make `CFfaith` correct under nonlinear mechanisms by reformulating `pearl_delta` as abduction-action-prediction (backward-compatible with linear), and add an **oracle structural-counterfactual** as the positive control that proves CF-faith works on NlinearSCM-T.
**Dependencies**: Stage 1 (cf_faith routed through `mechanism.forward_numpy`), Stage 2–3 (`MLPMechanism`, `NlinearSCMT`).

---

## Background

After Stage 1, `cf_faith.py`'s forward simulation already calls `mechanism.forward_numpy`, so **`noiseless_rollout` semantics generalize to nonlinear for free** (a CF is faithful iff `x_cf[t0:]` equals its own noiseless mechanism rollout). The remaining problem is **`pearl_delta`**: it currently rolls the *difference* `delta = x_cf - x_orig` through `delta[t] = sum_l A_l @ delta[t-l]`, which assumes **linear superposition** and is wrong for nonlinear `f`.

**Correct generalization — abduction-action-prediction (Pearl's 3 steps), valid for additive noise:**
1. **Abduct** the exogenous noise from the *factual*: `eps[t] = x_orig[t] − mechanism.forward_numpy(x_orig window)` for all `t ≥ L`. Exact because noise is additive (Stage 2/D4).
2. **Action**: hold `x_cf[:t0+1]` (the pre-intervention prefix is unchanged by the retroactive check; `x_cf[t0]` is the intervened value).
3. **Predict**: roll forward `x_pred[t] = mechanism.forward_numpy(x_pred window) + eps[t]` for `t > t0`, reusing the **abducted factual noise**. A CF is Pearl-faithful iff `x_cf[t0:]` matches `x_pred[t0:]` within tol.

For **linear** `f`, this reduces exactly to the current `delta[t] = sum_l A_l @ delta[t-l]` recursion (the factual noise cancels in the difference) — so the v0.1 linear pearl numbers must be **unchanged** (golden test from Stage 1 covers this).

---

## Steps

1. **Reformulate the pearl branch in `causaltemp_xai/metrics/cf_faith.py`.**
   - `score(self, x_original, x_cf, intervention_t, graph, mechanism)`.
   - Keep the retroactive check unchanged.
   - `noiseless_rollout` branch: forward-sim `x_cf` itself via `mechanism.forward_numpy` from `intervention_t` (already done in Stage 1 — verify it generalizes).
   - `pearl_delta` branch: replace the homogeneous-delta rollout with abduction-action-prediction (above): abduct `eps` from `x_original`, then roll `x_pred` from `intervention_t` reusing `eps`, compare `x_cf` to `x_pred`. Compute the same L1 residual → hard/soft.
   - Preserve `tol`, `scale`, the `(scale + 1e-12)` soft denominator, and the retroactive-zero behavior. The two semantics must still be **mutually exclusive** for a given CF (a CF can't be hard=1 under both) — this divergence is a benchmark result (supports H4).

2. **Add the oracle structural-counterfactual.**
   - File: `causaltemp_xai/benchmark/structural_cf.py` (new) — or co-locate in `mechanisms.py`.
   - `structural_counterfactual(x_orig, mechanism, t0, node, value) -> x_cf`: implements abduction-action-prediction to produce the **ground-truth** CF for `do(x[t0, node] = value)`:
     - abduct `eps` from `x_orig`,
     - copy `x_orig[:t0]`, set `x_cf[t0] = x_orig[t0]` then override `x_cf[t0, node] = value`,
     - roll forward reusing `eps`.
   - This is the canonical positive control: `CFfaith(semantics="pearl_delta").score(x_orig, oracle_cf, t0, graph, mechanism)` must give **hard = 1**.
   - Also provide a *noiseless* variant (future `eps = 0`) that yields the deterministic skeleton CF satisfying `noiseless_rollout` hard = 1 — documents the rollout-vs-pearl distinction concretely on nonlinear data.

3. **Write `tests/test_structural_cf.py`** (use `NlinearSCMT` SMOKE-scale data).
   - **Abduction exactness**: re-adding abducted `eps` to the mechanism rollout reconstructs `x_orig` to < 1e-6.
   - **Oracle is pearl-faithful**: `pearl_delta` hard = 1 on the oracle CF; **noiseless oracle is rollout-faithful**: `noiseless_rollout` hard = 1 on the skeleton CF.
   - **Negative controls**: a retroactively-edited CF → both hard = 0; a random-perturbation CF → hard = 0 (with high probability) and soft < 1.
   - **Mutual exclusivity** on nonlinear data: the same CF is not hard = 1 under both semantics.

4. **Extend `tests/test_cf_faith.py`** with nonlinear cases.
   - Mirror the linear `TestCFFaithPearl` / `TestCFFaithCompliant` structure on an `MLPMechanism`.
   - **Critical regression guard**: keep the existing linear pearl/rollout assertions intact (Stage 1 golden test already pins the numbers) — confirm the abduction reformulation did not move them.

---

## Verification

> Run with `uv run --offline …` (see `resources/commands.md`).

- [ ] `uv run --offline pytest tests/test_structural_cf.py tests/test_cf_faith.py -v` green.
- [ ] Golden test (`tests/test_golden_linear.py`) still passes — linear pearl/rollout numbers unchanged after reformulation.
- [ ] Oracle CF on NlinearSCM-T: `pearl_delta` hard = 1; noiseless-skeleton CF: `rollout` hard = 1.
- [ ] Abduction reconstructs the factual trajectory to < 1e-6.
- [ ] Negative-control CFs score hard = 0.
- [ ] Full suite green.

---

## Commit

`feat(metrics): generalize CF-faith to nonlinear mechanisms via abduction + add oracle structural-CF`
