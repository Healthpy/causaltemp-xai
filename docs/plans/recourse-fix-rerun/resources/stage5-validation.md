# Stage 5 smoke-tier validation

**Date:** 2026-08-21
**Validated source:** `6058400843388eda8d92209d1b9815b9c170e130`
**CFTS revision:** `f925d1ab7821fc1ae5d10bfdc1831daa26d55ba2`

The validation ran outside the main checkout. All three configs regenerated from source and
completed phases 01-05 plus phase 07 with the canonical seven-method roster. The diagnostic
table contains only `smoke`, `smoke_nl`, and `smoke_spring`. Its provenance records
`git_dirty=false` and `git_worktree_dirty=true`, as expected for generated result files in a
source-clean worktree.

## Regeneration and primary gates

| config | train / val / test accuracy | Noiseless delta > tol | Noiseless validity / vacuous / no-CF | Pearl validity / degenerate / scorable | fallback uses, Noiseless / Pearl |
|---|---|---:|---|---|---:|
| `smoke` | 0.990 / 0.950 / 0.920 | 20/20 | 1.00 / 0.00 / 0.00 | 1.00 / 0.00 / 20 | 0 / 1 |
| `smoke_nl` | 0.960 / 0.980 / 0.960 | 20/20 | 1.00 / 0.00 / 0.00 | 1.00 / 0.00 / 20 | 20 / 20 |
| `smoke_spring` | 0.997 / 0.950 / 0.980 | 20/20 | 1.00 / 0.00 / 0.00 | 1.00 / 0.00 / 20 | 16 / 12 |

The required `smoke_nl` accuracy reproduced exactly at 0.96 / 0.98 / 0.96. Classifier
`validity` remains independent of `no_cf_found`, vacuity, and degeneracy; the latter are
separate fields in both per-instance and summary schemas.

## Metric and oracle gates

- All 27 LSTM/oracle method rows satisfy
  `mean_all = mean_pearl_scorable * n_do_scorable / n`.
- All 27 rows satisfy
  `n_cf_faith_scorable = n * (1 - frac_degenerate)`.
- The zero-validity `smoke_nl/CftsCels` row keeps both joint hard-valid rates at 0.0. Its
  conditional hard-given-valid rates are NaN and both suppressions are recorded in final-table
  provenance.
- All six oracle rows contain classifier validity, both do-complexity means, and denominator
  diagnostics. `OracleCF-Pearl` has Pearl-scorable do-complexity 1.0 on every config.
- `OracleCF-Rollout` leads rollout-soft faithfulness and `OracleCF-Pearl` leads Pearl-soft
  faithfulness on every config.
- The Axis-C table has 27 rows and no duplicate `(benchmark, classifier, method)` key.
- Every intended dataset/method pair is present. Null Shift-VR cells are enumerated with
  `validity_base`: `smoke_nl/CftsCels=0.0` and `smoke_spring/CftsConfeti=0.15`.

Oracle controls are not expected to lead proximity: they use fixed structural shifts and do not
optimize `proximity_l1`.

## Runtime and walltime decision

| phase 03 | seconds |
|---|---:|
| full roster, `smoke` | 383 |
| full roster, `smoke_nl` | 439 |
| full roster, `smoke_spring` | 537 |
| SCM controls only, `smoke_nl`, fallback disabled | 293.96 |
| SCM controls only, `smoke_nl`, fallback enabled | 323.11 |

The bounded fallback adds 29.15 seconds, or 9.9%, in the controlled comparison. The run-1 full
job took about 12.1 hours; even a conservative 10% increase remains far below the committed
30-hour allocation. `slurm/helios_full_run.sbatch` therefore needs no walltime change.

## Run-1 comparison

The complete cell-level comparison is in
[`stage5-smoke-diff.csv`](stage5-smoke-diff.csv). Historical names are mapped only for the
comparison: `CARLA` to `NoiselessSCMRecourse` and `PearlCARLA` to `PearlSCMRecourse`.

| category | cells |
|---|---:|
| present in both and unchanged | 356 |
| present in both and changed | 339 |
| new run-2 metric cells | 363 |

Every changed or added row has an attribution. The targeted SCM-control changes trace to the
stage-1 proximity/fallback fix; new denominator, conditional-faithfulness, and oracle fields
trace to stage 2. The user-requested CFTS pull from `2fb4921` to `f925d1a` changed the external
method implementations and is identified separately. Run-1 `smoke` used `n_cf=5`, while this
validation uses 20. The old GH200 checkpoints were not retained, so non-target deltas from fresh
local retraining are diagnostic only and are not interpreted as fix effects.

## Defect-checklist disposition

- **Pass:** C1-C18 at smoke scope. C14 is satisfied by the proximity qualification above.
- **Deferred by design:** C19-C20 require the Stage-7 `full` + `full_nl` publication artifacts.
  The analogous smoke provenance gate passed with exactly the three diagnostic configs and
  commit `6058400`.
- **Qualified claims:** C21-C23. This remains a one-seed diagnostic and establishes no method
  ranking. Axis A still compares the graph to itself; CF-faith is a semantics-admission gate.
  Axis B remains uncontrolled at one seed, and noise plausibility has no calibrated factual
  ceiling.

The literal main-worktree cleanliness check is waived because unrelated untracked audit and
historical-result artifacts predated Stage 5. They were left untouched. Validation outputs stayed
in the disposable scratch worktrees, except that an interrupted timing command deterministically
rewrote the ignored local file `results/smoke_nl/lstm/cf/X_sel.npy` before any method output was
saved. This file is outside Git and is not part of the validation evidence. Only this report and
its diff CSV were added to the repository.

The full repository suite passes: **614 passed, 1 expected xfail**.
