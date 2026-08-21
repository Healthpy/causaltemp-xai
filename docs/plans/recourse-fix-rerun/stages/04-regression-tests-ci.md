# Stage 4: Regression tests and CI coverage

**Goal**: Make the delta-collapse impossible to reintroduce silently.
**Dependencies**: Stage 3.

---

## Why this stage exists

This failure was already diagnosed on 2026-07-31 and recorded in
`tests/test_metric_adversarial.py:404-410`, cross-referencing RISK-17 — and it still shipped in
run 1. The reason is a precise coverage gap:

- Every CF-method test in `tests/test_methods.py` uses the `trained` fixture at `:36-53`, which
  is built from `LinearSCMT` only. No test drives `CARLARecourse`, `PearlCARLARecourse` or
  `TSCausalCF` against an `MLPMechanism`.
- `TestVacuousIntervention` (`tests/test_metric_adversarial.py:392-460`) tests the *detector* —
  it constructs synthetic vacuous counterfactuals by hand via `_noiseless_cf`/`_pearl_cf`. It
  never invokes a real optimiser. So the vacuity metric is covered; the method triggering it is not.
- CI (`.github/workflows/ci.yml:70-78`) runs the phased pipeline on `smoke` only.

---

## Steps

1. **Add a nonlinear fixture to the method tests.**
   - File: `tests/test_methods.py`
   - Mirror the existing `trained` fixture at `:36-53`, but build it from `NlinearSCMT` with the
     current `_NL_HYPERPARAMS` (`causaltemp_xai/config.py:178-185`: `hidden=16, gain=1.5,
     decay_range=[0.1, 0.4], spectral_cap=6.0, init_gain=3.0, activation="tanh"`). Keep it small
     — `smoke_nl` scale, `k=5`, `T=30` — so it stays fast enough for CI.

2. **Add the delta-magnitude regression test.**
   - For both `CARLARecourse` and `PearlCARLARecourse` on the nonlinear fixture, assert that
     `max|x_cf - x|` at the derived intervention row exceeds `INTERVENTION_TOL`
     (`causaltemp_xai/scm/intervention.py:36`).
   - This is the assertion that would have caught run 1 before it ran. Make its failure message
     name the bug: the optimiser converged to its objective's optimum and that optimum is the
     do-nothing point.

3. **Add the zero-delta control test.**
   - Assert the two semantics separately for a counterfactual built with `delta` exactly zero
     under noiseless rollout on the linear fixture:
     - raw classifier `validity` may be `1.0`; preserve this existing metric unchanged;
     - `no_cf_found == 1`, `vacuous == 1`, and `recourse_validity == 0.0`.
   - This pins the finding that CARLA's raw `validity = 1.000` on the linear tier was
     noise-deletion: with delta exactly zero, 100/100 `full` instances flipped. The regression
     passes only when that classifier outcome is still reported honestly but excluded from
     `recourse_validity`.
   - This test pins the decision that classifier validity and successful recourse are distinct:
     noise deletion can flip the classifier, but it never earns recourse credit.

4. **Add the no-op guard and persistence tests.** Assert that when a method cannot find a
   counterfactual, `generate_batch_with_status()` sets its Boolean flag; Phase 03 writes the
   matching `no_cf_found_<Method>.npy`; Phase 04 rejects a missing or wrong-length CARLA sidecar;
   and the summary reports `n_no_cf_found`, `frac_no_cf_found`, and `recourse_validity`.

5. **Extend CI to the nonlinear path.**
   - File: `.github/workflows/ci.yml:70-78`
   - Add `smoke_nl` alongside `smoke` in the pipeline smoke harness. Keep `n_cf` small.
   - Measure the added CI wall-clock and record it. If it pushes the job past a sensible budget,
     run the nonlinear harness on a schedule or on changes to `causaltemp_xai/methods/**` rather
     than dropping it.

6. **Close RISK-17.** File: `docs/risk_register.md`. Mark it resolved, referencing the fix commit
   from stage 3 and the tests from this stage.

---

## Verification

- [ ] `uv run pytest tests/test_methods.py -q` passes, including the new nonlinear tests.
- [ ] Reverting the stage-3 change to `carla.py` makes the new delta-magnitude test **fail** —
      confirm this explicitly with `git stash`, run, `git stash pop`. A regression test that
      passes against the broken code is worthless.
- [ ] `uv run pytest tests/ -q` passes in full (577 tests at last count).
- [ ] The CI workflow file is valid and the nonlinear harness step is present.
- [ ] `docs/risk_register.md` shows RISK-17 closed.

---

## Commit

`[test] cover the nonlinear recourse path and pin the delta-collapse regression`
