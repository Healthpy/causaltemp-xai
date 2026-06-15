# Stage 5: Metric fixes & batch evaluation

**Goal**: Fix the broken `validity` metric, add a batch evaluation pipeline that computes all Axis-C metrics + CF-faith (hard+soft) for a set of CFs using `derive_intervention_t`, and validate CF-faith against manual SCM simulation on ≥10 examples.
**Dependencies**: Stage 4

---

## Steps

1. Fix `validity`.
   - File: `causaltemp_xai/metrics/axis_c.py`
   - Replace the current half-baked `validity` (it returns raw predictions). New contract: `validity(x_cf, model, target_class) -> float` = fraction of CFs whose predicted class == `target_class`. Accept single `(T,k)` or batch `(N,T,k)`; use `TCNClassifier.predict`.
   - **Sweep callers**: `grep -rn "validity(" causaltemp_xai experiments tests` and update every call site to the new signature (the `metrics/__init__` export and `run_all.py` reference it; `run_all.py` is rewritten in Stage 8, but any other caller must be fixed now). Update the README signature note in Stage 8.
   - Keep `proximity`, `sparsity`, `ood_plausibility` as-is (they're correct), but add batch-mean convenience wrappers if the eval pipeline needs them.

2. Build the evaluation pipeline.
   - File: `causaltemp_xai/eval.py` (new)
   - `evaluate_method(model, X_orig, CFs, X_train, graph, mechanisms, target_class) -> dict` computing, averaged over the batch:
     `validity`, `proximity` (L1 & L2), `sparsity` (and its L0 complement = fraction altered), `ood_plausibility` (IF; LOF optional per MVP), and **both** CF-faith metrics: `cf_faith_rollout_hard`, `cf_faith_rollout_soft`, `cf_faith_pearl_hard`, `cf_faith_pearl_soft`.
     - For each `(x, x_cf)` pair: `t = derive_intervention_t(x, x_cf)`, then score with **both** semantics (see Stage 1 step 7 / index Decisions "keep both CF-faith metrics"): `CFfaith(semantics="noiseless_rollout").score(x, x_cf, t, graph, mechanisms)` → `rollout_*`, and `CFfaith(semantics="pearl_delta").score(...)` → `pearl_*`. Instantiate the two scorers once outside the loop.
   - Return a flat dict per method; the harness (Stage 8) aggregates across methods.

3. Validate CF-faith end-to-end on ≥10 examples (MVP DoD item).
   - File: `tests/test_eval.py` (new)
   - Construct ≥10 hand-built CFs: (a) SCM-faithful ones (intervene at `t`, propagate via mechanisms) → `hard==1`; (b) retroactively-edited ones → `hard==0`; confirm `evaluate_method` aggregates match manual computation. This is the "validated against manual SCM simulation on ≥10 examples" DoD criterion.
   - Add `tests/test_axis_c.py` (new): validity flip-rate on a mock model; proximity/sparsity on known vectors; ood returns finite scores.

4. Export the new helpers in `metrics/__init__.py` (and `eval` at package level if convenient).

5. **Fail-fast phenomenon check (the core scientific guard — do this before building IG/Shift-VR/figures).**
   - File: `experiments/phenomenon_check.py` (new, small) — or a clearly-marked block runnable as `uv run python -m experiments.phenomenon_check --config smoke`.
   - On the smoke config + trained smoke TCN: generate CFs with **Wachter** and **CARLA-causal** for ~10 test instances, run `evaluate_method`, and print: per-method mean of **both** CF-faith metrics (`cf_faith_rollout_*` and `cf_faith_pearl_*`), and the distribution of `derive_intervention_t` per method.
   - Assert/flag the expected gap on the **default rollout metric** (the one CARLA is built to satisfy): `CARLA cf_faith_rollout_hard ≫ Wachter cf_faith_rollout_hard`, and that Wachter's `intervention_t` is **not** clustered at `T-1` (which would let a final-timestep-only edit score faithful — see the index "crux" and `cf_faith.py:108`). Also log the `pearl` metric for both methods — under the current CARLA (noiseless rollout), `cf_faith_pearl_hard` is expected ≈0 for CARLA too; that is the documented rollout-vs-pearl contrast, **not** a phenomenon failure. Only the rollout-metric gap gates the STOP decision below.
   - **If the gap is absent**: STOP, mark the stage with the finding in the index Issues section, and surface it to the user. Per the MVP risk table "H1 not confirmed" is a documentable outcome — but the decision must happen here, not after Stage 8 figures are built. Do not silently proceed.

---

## Verification

- [ ] `uv run pytest tests/test_axis_c.py tests/test_eval.py -q` passes, including the ≥10-example CF-faith validation.
- [ ] `validity` returns a scalar in [0,1] for a batch and matches the manual flip-rate on a mock classifier; no stale callers remain (`grep` clean).
- [ ] `evaluate_method` returns all keys: validity, proximity_l1, proximity_l2, sparsity, frac_altered, ood, cf_faith_rollout_hard, cf_faith_rollout_soft, cf_faith_pearl_hard, cf_faith_pearl_soft.
- [ ] `uv run python -m experiments.phenomenon_check --config smoke` shows CARLA `cf_faith_rollout_hard` ≫ Wachter `cf_faith_rollout_hard` (or the absence is escalated to the user, not ignored); both metrics are printed for both methods.

---

## Commit

`feat(metrics): fix validity, add batch Axis-C + CF-faith eval, add phenomenon fail-fast check`
