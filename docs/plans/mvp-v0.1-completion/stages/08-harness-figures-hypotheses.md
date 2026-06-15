# Stage 8: Harness, figures, hypotheses, docs

**Goal**: Rewrite the experiment harness end-to-end against the real API, emit a reproducible `results.json`, produce the 3 publication figures, write the H1/H3/H4 assessment + go/no-go, and refresh the README reproduction section — closing every remaining DoD item.
**Dependencies**: Stages 1–7

---

## Steps

1. Rewrite the harness.
   - File: `experiments/run_all.py` (full rewrite — discard the current broken API)
   - Flow: parse `--config {smoke,full}`; load dataset + frozen `tcn.pt` (Stages 2–3); select a CF subset of the test set (`--n_cf`, default e.g. 100 for full); for each method (Wachter, DiCE, CARLA-causal) generate CFs and call `eval.evaluate_method`; add the IG attribution row (deletion/insertion AUC) as the attribution foil; run `shift_vr` (Stage 7) for each method.
   - Write `experiments/results.json`: one record per method with all Axis-C metrics, **both** CF-faith metrics (`cf_faith_rollout_{hard,soft}` and `cf_faith_pearl_{hard,soft}` — see index Decisions "keep both CF-faith metrics"), Shift-VR, plus an `attribution` block. Include config + classifier accuracy + seed for provenance.
   - Print a formatted results table to stdout.

2. Figures.
   - File: `experiments/figures.py` (new)
   - **Fig 1** — validity vs CF-faith scatter (one point per method/instance; methods coloured by family). Use the **rollout** CF-faith on the y-axis (the default/headline metric); the per-instance dump carries both so a pearl variant can be regenerated.
   - **Fig 2** — CF-faith distribution (violin/hist of hard+soft across methods), drawn for **both** metrics — either side-by-side panels (`rollout` | `pearl`) or overlaid — to make the rollout-vs-pearl contrast visible.
   - **Fig 3** — rank correlation: Spearman ρ between each traditional metric (validity, proximity, sparsity) and CF-faith across methods. Compute against the **rollout** metric (headline); optionally add a `pearl` column.
   - Save publication-quality PNGs to `experiments/figures/`. Reads `results.json` (and any per-instance dump needed for the scatter — have `run_all.py` also write a per-instance CSV/parquet).

3. Hypothesis assessment.
   - File: `docs/hypotheses_assessment.md` (new)
   - Pre-register then assess with point estimates (no CIs required for MVP):
     - **H1**: Wachter/DiCE validity >0.9 AND CF-faith(rollout, hard) <0.3 → expected Confirmed. (Report pearl alongside; expected also low.)
     - **H3**: CARLA-causal CF-faith(**rollout**, hard) >0.7 with moderate validity/proximity degradation → expected Confirmed. Report `cf_faith_pearl` for CARLA too: under the current noiseless-rollout CARLA it is expected ≈0 (off-manifold) — this is the documented rollout-vs-pearl contrast, and the place to note that an on-manifold Pearl-CARLA variant is deferred future work.
     - **H4 (partial)**: Spearman ρ(traditional, CF-faith) <0.5 across 3–4 methods → report as preliminary/underpowered (do NOT claim confirmed).
     - **H5 (hint only)**: if `full_sparse` was generated, descriptive CF-faith comparison across sparsity — no claim.
   - End with the **go/no-go decision** section (freeze v0.1 vs extend to v1.0), with rationale referencing the results.

4. README reproduction.
   - File: `README.md`
   - Replace/extend the quickstart with a "Reproduce v0.1" section: `uv sync --extra dev` → `data_io --config full` → train TCN → `run_all.py --config full` → `figures.py`. Fix the stale CARLA/method-interface description to match the implemented API, **including the new `validity(x_cf, model, target_class)` signature** (the README currently documents `validity(x_cf, model)`). Update project-structure tree (config.py, data_io.py, eval.py, attribution/, figures.py, hypotheses_assessment.md).

5. CI smoke (if not done in Stage 1): add `uv run python experiments/run_all.py --config smoke` to CI so the full pipeline stays runnable.

---

## Verification

- [ ] `uv run python experiments/run_all.py --config smoke` runs end-to-end and writes `results.json` + per-instance dump.
- [ ] `uv run python experiments/figures.py` produces 3 non-empty PNGs in `experiments/figures/`.
- [ ] `docs/hypotheses_assessment.md` contains H1/H3/H4 verdicts with point estimates and a go/no-go section.
- [ ] README "Reproduce v0.1" steps are accurate and reference the real API (no `TCNClassifier(predict_fn)` / `cf_faith_hard` ghosts).
- [ ] `uv run pytest tests/ -q` — full suite green.
- [ ] (full run) Every DoD checkbox in `docs/mvp_plan.md §7` is satisfiable from the produced artifacts.

---

## Commit

`feat(experiments): end-to-end harness, 3 figures, hypothesis assessment, repro docs`
