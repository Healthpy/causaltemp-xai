# Stage 3: Fix the recourse layer

**Goal**: Make `CARLARecourse` and `PearlCARLARecourse` produce counterfactuals whose delta
exceeds `INTERVENTION_TOL`, on both linear and nonlinear mechanisms.
**Dependencies**: Stage 2 returned **GO**.

---

## Steps

1. **Fix the candidate tie-break — do this first, it is independent and cheap.**
   - File: `causaltemp_xai/methods/counterfactual/carla.py:302` and `:525`
   - Current: `if best is None or (cand[0], -cand[1]) > (best[0], -best[1]):`
   - The tuple is `(flipped, proximity)`. When **no** candidate flips, `cand[0]` is equal for
     all of them and the comparison falls through to `-proximity`, which selects the
     **smallest** delta — that is, it actively prefers the most degenerate no-op. Measured on
     `full_nl`: the `t0=50` candidate reached `delta=0.43` and was discarded in favour of the
     `t0=25` candidate at `delta=6e-8`.
   - Fix: when no candidate flips, rank by lowest cross-entropy instead of lowest proximity.
     Keep the existing behaviour unchanged when at least one candidate flips.
   - ~5 lines per class. This alone will change `full_nl` numbers.

2. **Rescale the proximity penalty — this is the root cause.**
   - File: `causaltemp_xai/methods/counterfactual/carla.py:288` and `:511`
   - Current: `loss = self.lam_pred * pred_loss + self.lam_prox * prox`
   - The stationary point is `delta* = -(lam_pred / (2*lam_prox)) * dCE/ddelta`. With
     `dCE/ddelta ~ 1e-5` on nonlinear data, any fixed `lam_prox` of order 0.1-0.5 puts the
     optimum at delta ~ 1e-5, five orders below the 1e-3 tolerance.
   - Two acceptable approaches; pick one, record the choice in the index's Decisions:
     - **(a) Scale-aware penalty.** Normalise `prox` by the per-channel variance of the factual
       data, so the penalty is measured in standardised units rather than raw ones. Makes
       `lam_prox` dimensionless and transferable across configs.
     - **(b) Lambda backoff.** Run the loop; if no candidate flips, halve `lam_prox` and retry,
       up to a bounded number of rounds (e.g. 6, spanning two orders of magnitude). Terminate
       on the first flip. This directly targets the measured problem — the penalty at the
       flipping magnitude (`0.5 * 20^2 = 200`) outweighs the maximum CE reward (3-5) by 40-70x.
   - **(b) is the recommended default**: it is self-calibrating, it cannot make a currently-
     working case worse (the first round is today's behaviour), and its cost is bounded and
     only paid when the method would otherwise fail. Its downside is up to 6x the optimiser
     steps on hard instances — budget for it in stage 7's timing measurement.
   - Use the target delta magnitude measured in stage 2 to sanity-check that the retuned
     objective actually makes that magnitude attractive.

3. **Add a sub-tolerance no-op guard.**
   - File: `causaltemp_xai/methods/counterfactual/carla.py`, after each optimisation loop.
   - After selecting the best candidate, if `max|delta| <= INTERVENTION_TOL`
     (`causaltemp_xai/scm/intervention.py:36`), the method has not produced a counterfactual.
     Emit an explicit marker rather than returning a silent no-op that the metric layer then
     scores as though it were a real counterfactual.
   - This is what turned a total failure into a plausible-looking `proximity_l1 = 0.0001` in
     run 1. Making it loud is worth more than the fix itself for future runs.
   - Use this exact compatibility-preserving contract:
     - Factor each class's implementation into a private `_generate_one(...) -> (cf, found)`.
       Public `generate()` continues to return only `cf`, and `generate_batch()` continues to
       return only the stacked ndarray, so existing callers do not break.
     - Add `generate_batch_with_status(...) -> (CFs, no_cf_found)`, where `no_cf_found` is a
       Boolean array of length `N`. It calls `_generate_one` once per instance and sets the flag
       when `max|delta| <= INTERVENTION_TOL`.
     - In `experiments/03_run_cf_methods.py`, make `generate_cfs()` use
       `generate_batch_with_status` when available; otherwise pair the returned CF array with an
       all-false status array. Persist both `X_cf_<Method>.npy` and
       `no_cf_found_<Method>.npy` under the same `cf/` directory.
     - In `experiments/04_evaluate_axes.py`, require the sidecar for CARLA/PearlCARLA, validate
       its dtype and length, and pass it to both `evaluate_method()` and `score_and_collect()`.
       For old artifacts or methods without a status API, use an all-false array and record that
       the status was inferred rather than generated.
     - In `causaltemp_xai/eval.py` and `experiments/_common.py`, emit per-instance
       `no_cf_found`, plus `n_no_cf_found`, `frac_no_cf_found`, and
       `recourse_validity = mean(valid_i AND NOT no_cf_found_i AND NOT vacuous_i)`.
       Preserve existing raw `validity` unchanged.
   - Do not raise — one failed instance must not abort a 100-instance run.

4. **Surface the recourse hyperparameters into config.** *(optional — do it only if steps 1-3
   leave time; otherwise defer to Backlog.)*
   - `lam_pred`, `lam_prox` and `lr` are hardcoded class defaults at `carla.py:181-184` and
     `:394-397`, and are set at **no call site in the repo** — `build_methods()`
     (`experiments/03_run_cf_methods.py:86-109`) overrides only CARLA's `n_steps=300` and both
     classes' `t0_fractions`. There is no method-hyperparameter registry.
   - `lam_prox=0.1` for PearlCARLA was tuned at smoke scale on the **pre-retuning** substrate
     (see the module docstring, `carla.py:56-78`, dated 2026-07-08), so it is calibrated against
     a mechanism that no longer exists.
   - If changing constructor signatures, note that `tests/test_experiment_registry.py:75-83`
     pins PearlCARLA's `n_steps=500` and asserts `CF_METHOD_KEYS == build_methods().keys()`.

5. **Keep the module docstring truthful.** `carla.py:39-78` and `:361-386` contain a detailed
   rationale for the current `lam_prox` values that this stage invalidates. Update it; do not
   leave a docstring arguing for a default the code no longer has.

---

## Verification

Run against a freshly regenerated `smoke_nl` substrate (see `resources/commands.md`) — the
laptop's checked-in data is stale and the bug does not reproduce on it.

- [ ] CARLA `delta` at `t0` on `smoke_nl` exceeds `1e-3` on at least 90% of instances
      (was 3.3e-05).
- [ ] PearlCARLA `frac_degenerate` on `smoke_nl` is <= 0.10 (was 1.00).
- [ ] CARLA `validity` on `smoke_nl` is > 0.30 (was 0.000) and `frac_vacuous` <= 0.20 (was 1.00).
- [ ] CARLA `recourse_validity` on `smoke_nl` is > 0.30, and the sidecar length equals `n_cf`.
- [ ] On the linear `smoke` config, CARLA still produces valid counterfactuals — but confirm
      the delta is now a real intervention, not the noise-deletion free pass. Compare against a
      zero-delta control: raw `validity` may be 1, but `recourse_validity` must be 0 and
      `no_cf_found` must be true for that control.
- [ ] `uv run pytest tests/ -q` passes.
- [ ] Record the wall-clock cost of phase 03 on `smoke_nl` before and after, so stage 7 can
      extrapolate whether the lambda backoff changes the 26 GPU-hour estimate.

---

## Commit

`[fix] rescale the CARLA proximity penalty and stop selecting degenerate no-ops`
