# Stage 2: Metric reporting fixes

**Goal**: Make every column of the shipped table mean exactly one thing.
**Dependencies**: Stage 1 (implementation is independent, but local validation exercises both).

---

## D4 — `do_complexity` ships two different values under near-identical names

`summary.json` averages Pearl-semantic intervention-schedule lengths over every instance;
`pns.json` averages only over instances with a non-empty Pearl schedule. These statistics use
different denominators but ship under near-identical names.

| dataset | method | `do_complexity_mean` | `pns_PS_do_complexity_mean` | `frac_vacuous` |
|---|---|---|---|---|
| `full` | CftsCels | 3.57 | 6.736 | 0.47 |
| `full` | CftsConfeti | 56.00 | 100.00 | 0.44 |
| `smoke_spring` | CftsConfeti | 4.50 | 30.00 (= T) | 0.85 |

This corrupts the run-1 headline directly. CftsConfeti reads 56.0 on `full`, which looks like
mid-range parsimony; it actually intervenes on **all 100** timesteps whenever it intervenes at
all — identical to TSCausal, the stated worst case. On `smoke_spring` it reads 4.50 with
`do_complexity_median = 0.000`: a median counterfactual that is literally vacuous, presented as
best-in-class parsimony.

The earlier diagnosis incorrectly equated the PNS-scorable fraction with
`1 - frac_vacuous`. Vacuity is evaluated under the CF's noiseless continuation, while
do-complexity and PNS use Pearl noise reinjection. The counterexample is `full_nl/NoiselessSCMRecourse`:
`do_complexity_mean=74`, `pns_PS_do_complexity_mean=74`, and `frac_vacuous=1.0`.

### Steps

1. In `causaltemp_xai/eval.py`, compute the Pearl-semantic schedule length `D_i` once per
   instance. Define `do_scorable_i = (D_i > 0)`; this is deliberately independent of the
   noiseless-semantic `vacuous_i` predicate.
2. Emit these unambiguous keys:
   - `do_complexity_mean_all = mean(D_i)` over all instances (current summary behaviour);
   - `do_complexity_mean_pearl_scorable = mean(D_i | D_i > 0)`, or NaN when none are scorable;
   - `n_do_scorable` and `frac_no_do_schedule = 1 - n_do_scorable/n`.
   Preserve the legacy `do_complexity_mean` only as an explicitly documented alias of
   `do_complexity_mean_all` during this migration; do not reuse it for the conditional value.
3. In `experiments/make_final_table.py`, emit the two explicit means and their denominator
   diagnostics. Keep `frac_vacuous` beside them as a separate semantic diagnostic, never as the
   denominator of `mean_pearl_scorable`.
4. Update `tests/test_do_complexity.py` to pin
   `mean_all == mean_pearl_scorable * (n_do_scorable / n)` when `n_do_scorable > 0`, and assert
   NaN for `mean_pearl_scorable` when it is zero. Add the `full_nl/NoiselessSCMRecourse` semantic counterexample
   as a regression: a row may be noiseless-vacuous while Pearl do-complexity is non-zero.

---

## D5 — a conditional faithfulness metric is missing

`cf_faith_rollout_hard_valid` and `cf_faith_pearl_hard_valid` are existing, tested **joint**
rates: `P(hard-faithful AND valid)`. Their value is correctly `0.0` when no counterfactual is
valid. Preserve these keys and their semantics unchanged.

The missing statistic is the conditional rate `P(hard-faithful | valid)`. Without it, a reader
cannot ask "among the counterfactuals that flipped, how many were faithful?" The sharp case is
an empty valid set: the conditional rate is undefined and must be NaN, while the existing joint
rate remains exactly `0.0`.

### Steps

1. Preserve `cf_faith_{rollout,pearl}_hard_valid = mean(hard_i * valid_i)` in both
   `causaltemp_xai/eval.py` and `experiments/_common.py`.
2. Add `cf_faith_{rollout,pearl}_hard_given_valid` to both scoring paths:
   `sum(hard_i * valid_i) / sum(valid_i)`, treating a NaN hard score as no conditional credit.
   Return NaN only when `sum(valid_i) == 0`.
3. In `experiments/make_final_table.py`, emit the joint and conditional keys separately,
   suppress conditional NaNs, and record suppressed `(dataset, classifier, method, metric)` keys
   in `final_table_provenance.json` so suppression differs from an unrun metric.
4. Update `tests/test_metric_adversarial.py` to preserve the existing joint assertions and add
   conditional cases for zero, partial, and full validity. Add table-builder coverage for the
   suppression provenance.

---

## D8 — the oracles carry no `validity`, `do_complexity`, `ood`, `shift_vr` or PNS

Oracle coverage is 140/350 = 40% of the metric space. The costly gap is do-complexity:
`OracleCF-Pearl` performs exactly one `do()` at `t0 = T//2` on one node
(`experiments/_common.py:613-632`, `ORACLE_SHIFT = 1.5`), so its do-complexity is **1.0 by
construction** — the natural calibration anchor for the axis carrying the paper's headline
spread. It is absent, and the run-1 table shows `n/a`.

### Steps

1. In `experiments/05_run_oracle_control.py`, load the LSTM checkpoint from
   `Path(out_dir) / cfg.name / "lstm.pt"` using the same pattern as
   `experiments/04_evaluate_axes.py:83-86`. Fail loudly if it is absent. Pass the classifier to
   `score_and_collect()` and run the oracle arrays through `evaluate_method()` with
   `data["X_train"]`, so both oracle rows carry classifier validity, do-complexity, OOD, joint
   faithfulness-validity, and conditional faithfulness-given-validity.
   Oracle **construction and faithfulness remain classifier-independent**; loading the model
   adds outcome-quality metrics and does not redefine the positive control.
2. Assert in a test that `OracleCF-Pearl`'s
   `do_complexity_mean_pearl_scorable == 1.0`. If it does not
   come out at 1.0, that is a bug in either the oracle or the do-complexity metric, and it is
   worth more than the rest of this stage — record it prominently.
3. Keep the shape of `summary.json` intact: `experiments/04_evaluate_axes.py:151-157` writes
   `methods` as a **list of dicts** each carrying a `"method"` key, and
   `experiments/make_final_table.py` parses exactly that shape. Adding keys is safe; changing
   the container is not.

---

## Verification

- [ ] Regenerate the run-1 table from the archived `results/` and diff it against the committed
      run-1 table. Every changed cell must be explainable by exactly one of D4, D5 or D8 —
      no unexplained drift.
- [ ] Existing `*_hard_valid` remains `0.0` where `validity == 0`; the new
      `*_hard_given_valid` cell is NaN and is listed in suppression provenance.
- [ ] Both explicit do-complexity variants and `n_do_scorable` are present; their relation uses
      `n_do_scorable/n`, never `1 - frac_vacuous`.
- [ ] `OracleCF-Pearl` has `do_complexity_mean_pearl_scorable == 1.0` and both oracle rows carry
      classifier `validity`.
- [ ] `uv run pytest tests/ -q` passes.

---

## Commit

`[fix] disambiguate do-complexity, add conditional faithfulness, score the oracles`
