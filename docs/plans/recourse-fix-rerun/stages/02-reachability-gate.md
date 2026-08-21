# Stage 2: Reachability gate (GO/NO-GO)

**Goal**: Measure whether the decision boundary is reachable at `full_nl` scale, and stop the
plan if it is not.
**Dependencies**: Stage 1.

---

## Why this stage exists

The sweep that proved the boundary is crossable ran only at `smoke_nl` scale: `T=30`, `t0=8`,
so 22 contraction steps between the intervention and the readout. It found 20/20 flips at a
median `||delta||` of 20.

`full_nl` has `T=100` and `t0=25`, so **75** contraction steps. The MLP mechanism contracts at
0.15-0.33 per step and the LSTM reads only `hn[-1]`. If the required delta grows with the
contraction distance, the value needed at full scale may be so large that no proximity-penalised
optimiser should ever select it — the counterfactual would be far off-manifold, and
`scm_noise_plausibility` and `ood` would correctly reject it.

If that is the case, rescaling `lam_prox` cannot fix `full_nl`. The benchmark config itself
needs redesign (a later `t0`, a shallower decay range, or an interior label site). Measuring
this costs minutes. Discovering it after a 26 GPU-hour run costs a day.

---

## Steps

1. **Regenerate the `full_nl` substrate locally.** The laptop's `data/scm_t/` is stale
   (dated 2026-07-07/08, pre-retune `gain=0.8`), and the bug does **not** reproduce on it.
   Use a scratch output directory and expect phases 01/02 to also write into the tracked
   `results/` regardless of `--out-dir` — see `resources/commands.md`.

2. **Write and validate a probe script** under the scratchpad; after the smoke reproduction
   passes, promote the exact script into this plan's `resources/` directory. For each of
   `smoke_nl` and `full_nl`, and for each `t0` in `t0_fractions=(0.25, 0.5)`:
   - Load the trained classifier and the ground-truth mechanism.
   - For 20 factual instances, run two complementary reachability probes:
     1. **Coordinate sweep:** sweep `||delta||` over a geometric grid spanning `[0.1, 500]`,
        across every channel and both signs. This preserves the prior smoke diagnostic but is
        not sufficient for a NO-GO verdict.
     2. **Joint search:** optimise one unconstrained `k`-dimensional `delta` for prediction loss
        only (`lam_prox=0`) from at least 10 starts: zero, both normalised gradient directions,
        and seeded random directions at multiple initial radii. Use the same actionable mask as
        CARLA, stop on the first flip, and cap `max|delta|` at 500. Record the smallest flipping
        norm across all starts.
   - Roll forward under both semantics (`noiseless_rollout` and `pearl_delta`) using the same
     code path the method uses, and record the smallest `||delta||` that flips the label.
   - Record, alongside it, `scm_noise_plausibility` and `ood` for the flipping counterfactual,
     so an unreachable-in-practice case is distinguishable from an off-manifold one.

3. **Record the measurement** in the index's Decisions section as a table:
   config x `t0` x semantics x probe (`coordinate`, `joint`) -> median minimum flipping
   `||delta||`, flip rate, plausibility at the flip point. Preserve the probe script and its
   machine-readable output under the stage-2 commit (for example
   `docs/plans/recourse-fix-rerun/resources/reachability_probe.py` and
   `resources/reachability_results.json`) so the gate is independently reproducible.

4. **Apply the gate.**
   - **GO** if the **joint search** flips `full_nl` at a median `||delta||` within about one
     order of magnitude of `smoke_nl`'s measured joint-search baseline, at both `t0` values or
     at least at `t0=0.5`, and the flipping
     counterfactual is not grossly off-manifold. Proceed to stage 3, and carry the measured
     delta magnitude forward as the target the retuned `lam_prox` must make attractive.
   - **NO-GO** only if neither the coordinate sweep nor any joint-search restart flips
     `full_nl` up to the cap, or every joint-search flip has plausibility worse than TSCausal's
     saturated negative-control range (0.062-0.110). A coordinate-only failure is never
     sufficient for NO-GO.
     Then: mark stages 3-9 BLOCKED, write a Backlog entry describing the measurement and the
     config-redesign options, update the index, commit, and **stop**. Do not proceed
     autonomously — a config redesign is not a decision for an unattended run.

---

## Verification

- [ ] The coordinate probe reproduces the known `smoke_nl` result (20/20 flips, median
      `||delta||` ~20), and the joint probe is at least as successful.
- [ ] A results table for both configs, both `t0` values, both semantics and both probe types is
      written into the index's Decisions section.
- [ ] The probe script and machine-readable results are committed under `resources/`.
- [ ] The GO/NO-GO verdict is stated explicitly in the index, with the number that justifies it.

---

## Commit

`[plan] record the full_nl reachability measurement and gate verdict`
