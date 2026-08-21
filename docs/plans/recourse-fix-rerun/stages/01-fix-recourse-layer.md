# Stage 1: Fix the recourse layer

**Goal**: Make `CARLARecourse` and `PearlCARLARecourse` produce real interventions above
`INTERVENTION_TOL` on linear and nonlinear mechanisms while preserving classifier-only validity.
**Dependencies**: None. The existing regenerated `smoke_nl` sweep in the index establishes that
the target boundary is reachable; the separate full-scale reachability stage remains removed.

---

## Steps

1. **Fix the candidate tie-break.**
   - File: `causaltemp_xai/methods/counterfactual/carla.py`, in both candidate loops.
   - Today `(flipped, -proximity)` selects the smallest delta when no candidate flips. Rank
     failed candidates by lowest cross-entropy instead. Preserve lowest proximity among
     candidates that do flip.

2. **Rescale the proximity penalty with a bounded prediction-only fallback.**
   - Run the current objective first. If no genuine candidate flips, retry once with
     `lam_prox=0` and stop at the first above-tolerance target flip.
   - Preserve current behavior and cost for cases that succeed in round one.
   - Record both attempted lambdas and the selected round for diagnostics.
   - Sanity-check against the measured `smoke_nl` reachable scale (median `||delta|| ~= 20`).
   - Execution finding: six successive halvings cost 7x but produced 0/3 CARLA and 0/3
     PearlCARLA flips on the current substrate. The single prediction-only fallback is the
     bounded corrective path; it avoids repeating the same local optimum.

3. **Add an explicit no-CF status without changing validity.**
   - Factor each class into `_generate_one(...) -> (cf, found)`.
   - Keep public `generate()` and `generate_batch()` ndarray-compatible.
   - Add `generate_batch_with_status(...) -> (CFs, no_cf_found)` with a Boolean vector. Mark a
     result when no candidate flips or `max|delta| <= INTERVENTION_TOL`; do not abort the batch.
   - Update `experiments/03_run_cf_methods.py` to persist `no_cf_found_<Method>.npy` beside each
     `X_cf_<Method>.npy`. Methods without a status API receive an all-false inferred vector and
     explicit provenance stating that it was inferred.
   - Update Phase 04 to require and validate CARLA/PearlCARLA sidecars and emit per-instance
     `no_cf_found`, `n_no_cf_found`, and `frac_no_cf_found`.
   - Keep `validity` solely equal to the classifier target-class rate. Do not add
     `recourse_validity` or any composite validity metric; `no_cf_found`, `vacuous`, and
     `frac_degenerate` remain separate diagnostics.

4. **Expose recourse hyperparameters only if required.**
   - Prefer keeping the bounded fallback internal. If configuration is required for reproducible
     tuning, add explicit defaults in `causaltemp_xai/config.py` and route them only through
     `experiments/03_run_cf_methods.py::build_methods()`.
   - Preserve registry tests and document every changed default.

5. **Update documentation in `carla.py`.**
   - Remove stale arguments for fixed `lam_prox` values if the implementation no longer follows
     them.
   - Document fallback bounds, tie-breaking, no-CF status, and classifier-only validity.

---

## Verification

Use freshly regenerated `smoke` and `smoke_nl` substrates from `resources/commands.md`.

- [x] CARLA intervention-row delta exceeds `INTERVENTION_TOL` on at least 90% of `smoke_nl`
      instances; `frac_vacuous <= 0.20` and `frac_no_cf_found <= 0.20`.
- [x] PearlCARLA `frac_degenerate <= 0.10` on `smoke_nl`.
- [x] CARLA classifier `validity > 0.30` on `smoke_nl`; validity is not filtered by no-CF or
      vacuity status.
- [x] A forced zero-delta control may have `validity == 1`, while `no_cf_found == 1` and
      `vacuous == 1` are reported separately.
- [x] Legacy `generate()` and `generate_batch()` callers remain compatible.
- [x] `uv run pytest tests/ -q` passes.
- [x] Record Phase-03 wall-clock before and after for Stage 5 walltime sizing.

Execution evidence (2026-08-21): deterministic regenerated `smoke_nl`, n=20,
train/val/test accuracy 0.96/0.98/0.96. CARLA took 65.0 s and PearlCARLA 103.9 s; both
reported validity 1.00, `frac_no_cf_found=0.00`, `frac_vacuous=0.00`,
`frac_degenerate=0.00`, and 20/20 intervention deltas above tolerance. The pre-fix run-1
timings remain in `resources/commands.md`; Stage 5 will measure the complete phase-03 delta.

---

## Commit

`[fix] rescale CARLA recourse and expose failed searches`
