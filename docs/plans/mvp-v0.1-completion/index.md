# Plan: CausalTemp-XAI v0.1 — MVP Completion

**Date**: 2026-05-30
**Branch**: main (create `mvp-v0.1` feature branch before Stage 1)
**Predecessors**: None (builds on scaffold commit `f46acce`)
**Goal**: Take the current scaffold to a fully reproducible v0.1 benchmark that demonstrates the key phenomenon — traditional temporal-XAI CF metrics (validity/proximity/sparsity) disagree with causal metrics (CF-faith) on LinearSCM-T — satisfying every item in the MVP Definition of Done.

---

## Context

The repo (`docs/mvp_plan.md`, `docs/general_plan.md`) targets a 4-week MVP: one synthetic SCM benchmark, a TCN classifier, 3 CF methods, Axis-C + Shift-VR-lite metrics, and 3 figures showing validity ranks ≠ CF-faith ranks (hypotheses H1, H3, H4-partial).

**State of the scaffold after exploration:**

| Component | File | Status |
|---|---|---|
| LinearSCM-T generator | `benchmark/generator.py` | ✅ implemented + tested (`LinearSCMT(k,L,sparsity,noise_type,T,N,seed).generate()` → `{X,Y,graph,mechanisms}`) |
| CF-faith scorer | `metrics/cf_faith.py` | ✅ implemented + tested (`CFfaith.score(x,x_cf,intervention_t,graph,mechanisms)` → `{hard,soft}`) |
| TCN module | `classifiers/tcn.py` | ◑ `TCN` + `train_tcn()` exist; no val/test split, no checkpoint, no predict/proba interface, no tests |
| Axis-C metrics | `metrics/axis_c.py` | ◑ `proximity`/`sparsity`/`ood_plausibility` ok; **`validity` is broken** (returns raw preds, not a flip-rate); no tests |
| Wachter / DiCE / CARLA | `methods/*.py` | ❌ all three raise `NotImplementedError` |
| Experiment harness | `experiments/run_all.py` | ❌ written against a non-existent API (`n_vars/lag`, `sample_with_labels`, `TCNClassifier`, `cf_faith_hard/soft`, `CARLACF`, `generate_batch`) — does not import |
| Packaging | `pyproject.toml` | ❌ `build-backend = "setuptools.backends.legacy:build"` is invalid → `pip install -e .` and CI fail |
| Environment | — | ❌ no deps installed locally; will use **uv** |

**The crux (an unresolved design gap):** `CFfaith` requires an `intervention_t`, but Wachter/DiCE return a fully-perturbed trajectory with no declared intervention point. The benchmark needs a single, documented rule that **derives `intervention_t` from any CF** (first timestep where `|cf−x| > tol`). This derivation is what makes standard methods score low CF-faith (they edit retroactively and don't propagate through the SCM) while causal recourse scores high — it is the hinge of the entire experiment and is implemented nowhere yet.

**Treat the tested modules (`LinearSCMT`, `CFfaith`) as the source of truth.** `run_all.py` is aspirational and will be rewritten; do not preserve its API.

---

## Strategy

Eight stages in three phases. Each stage is independently testable, leaves the tree green, and is one commit.

- **Phase A — Foundation (Stages 1–3):** uv environment + packaging fix + green baseline; lock the benchmark config (two-tier smoke/full) and persist datasets; add a sklearn-style `TCNClassifier` wrapper trained to >90% with a frozen checkpoint.
- **Phase B — Methods & Metrics (Stages 4–6):** implement the 3 CF methods + the `intervention_t` derivation rule; fix/extend the metric layer and wire batch Axis-C + CF-faith evaluation; add the Integrated-Gradients attribution foil (WP3).
- **Phase C — Robustness & Synthesis (Stages 7–8):** Shift-VR-lite second environment (Axis D); rewrite the experiment harness end-to-end, emit `results.json`, the 3 publication figures, and a formal H1/H3/H4 assessment; refresh README reproduction steps.

**Key decisions (confirmed with user):**
- **Paper:** code + figures + results + hypothesis-assessment doc only. No prose draft (authors write it).
- **DiCE:** wrap the official `dice-ml` library with flatten/reshape adapters; if 3D/torch integration fails, fall back to the from-scratch DPP design already documented in the `DiCECF` stub (the MVP risk table sanctions this). Document whichever path is taken.
- **Compute:** two-tier. A `smoke` config (k=5, T=30, N=500) runs in CI/tests; the `full` locked config (k=10, L=1, s=0.2, Laplace, T=100, N=10 000) is run once for the paper and documented. CPU-only is the assumed target.
- **Dependency management:** `uv` exclusively (no pip/pipx).

See [resources/configs.md](resources/configs.md) for the locked configs and [resources/commands.md](resources/commands.md) for all uv/test/run commands.

---

## Success Criteria

| Criterion (from MVP DoD) | Baseline | Target | Verify in |
|---|---|---|---|
| Existing tests green under uv | not runnable (no env, bad backend) | `pytest` all green | Stage 1 |
| LinearSCM-T config locked, tested, persisted | generator exists, no config/persistence | full+smoke configs saved with X/Y/graph/mechanisms + 60/20/20 splits | Stage 2 |
| TCN >90% test accuracy, frozen checkpoint | `train_tcn` only, no eval/checkpoint | ≥0.90 (≥0.85 documented fallback) test acc; checkpoint on disk | Stage 3 |
| Wachter, DiCE, CARLA-causal produce valid output | all stubs | each returns correctly-shaped CFs on smoke config | Stage 4 |
| CF-faith validated vs manual SCM sim on ≥10 examples | scorer tested in isolation | batch eval + `derive_intervention_t` validated on ≥10 cases | Stages 4–5 |
| All Axis-C + Shift-VR-lite computed, stored reproducibly | `validity` broken, no harness | `results.json` with every metric for every method, both envs | Stages 5,7,8 |
| 3 publication-quality figures | none | scatter, CF-faith distribution, rank-correlation PNGs | Stage 8 |
| H1 & H3 assessed with point estimates | none | `docs/hypotheses_assessment.md` with H1/H3/H4 verdicts | Stage 8 |
| README reproduces end-to-end | quickstart only, broken harness | documented uv → data → train → eval → figures pipeline | Stage 8 |
| Go/no-go decision documented | none | section appended to assessment doc | Stage 8 |

---

## Files That May Be Changed

### Packaging / infra
- `pyproject.toml` — fix build-backend; add `dice-ml`, `captum` (or hand-rolled IG), tighten dev extras
- `requirements.txt` — keep in sync or deprecate in favour of uv lock
- `.github/workflows/ci.yml` — install via uv; run smoke config + tests
- `uv.lock` (new) — committed lockfile

### Source
- `causaltemp_xai/config.py` (new) — `BenchmarkConfig` dataclass, `SMOKE`/`FULL` presets
- `causaltemp_xai/benchmark/generator.py` — minor: dataset save/load helpers (optional)
- `causaltemp_xai/data_io.py` (new) — split + persist datasets
- `causaltemp_xai/classifiers/tcn.py` + `classifiers/__init__.py` — add `TCNClassifier` wrapper
- `causaltemp_xai/methods/{wachter,dice,carla}.py` + `methods/__init__.py` — implement; add `methods/intervention.py` (derive_intervention_t)
- `causaltemp_xai/metrics/axis_c.py` — fix `validity`; add batch helpers
- `causaltemp_xai/metrics/__init__.py` — export new helpers
- `causaltemp_xai/attribution/integrated_gradients.py` (new) + deletion/insertion curves
- `causaltemp_xai/eval.py` (new) — batch evaluation pipeline (Axis-C + CF-faith + Shift-VR)

### Experiments / docs
- `experiments/run_all.py` — full rewrite against the real API
- `experiments/figures.py` (new) — 3 figures
- `docs/hypotheses_assessment.md` (new)
- `README.md` — reproduction section
- `tests/test_*.py` — new tests for classifier, methods, metrics, intervention, attribution

---

## Progress Tracker

| # | Stage | Status | Notes | Commit |
|---|-------|--------|-------|--------|
| 1 | [Environment & packaging fix](stages/01-env-and-packaging.md) | DONE | uv env (py3.13, torch 2.12); fixed cf_faith docstring syntax (module never imported before); CI→uv; requirements.txt→pointer; resolved CF-faith semantics fork (Option #1) → 21/21 green | `chore(infra): fix build backend, set up uv environment, green test baseline` |
| 2 | [Benchmark config & dataset persistence](stages/02-benchmark-config-and-data.md) | PENDING | | |
| 3 | [TCNClassifier wrapper & training](stages/03-classifier-and-training.md) | PENDING | | |
| 4 | [CF methods & intervention-time rule](stages/04-cf-methods-and-intervention.md) | PENDING | | |
| 5 | [Metric fixes & batch evaluation](stages/05-metrics-integration.md) | PENDING | | |
| 6 | [Integrated-Gradients attribution foil](stages/06-attribution-baseline.md) | PENDING | | |
| 7 | [Shift-VR-lite (Axis D)](stages/07-shift-vr-lite.md) | PENDING | | |
| 8 | [Harness, figures, hypotheses, docs](stages/08-harness-figures-hypotheses.md) | PENDING | | |

Statuses: `PENDING` -> `IN_PROGRESS` -> `DONE` | `BLOCKED` | `SKIPPED`

Phases: A = Stages 1–3 (Foundation) · B = Stages 4–6 (Methods & Metrics) · C = Stages 7–8 (Robustness & Synthesis)

---

## Execution Protocol

To execute this plan, follow this loop for each stage:

1. **Read the progress tracker** above and find the first stage that is not DONE
2. **Read the stage file** -- follow the link in the tracker to the stage's .md file
3. **Read resources** -- check `resources/configs.md` and `resources/commands.md`
4. **Clarify ambiguities** -- if anything is unclear or multiple approaches exist, ask the user before implementing. Do not guess.
5. **Implement** -- execute the steps described in the stage. Use `uv` for every Python invocation (`uv run …`, `uv add …`). Never call pip.
6. **Validate** -- run the verification checks listed in the stage. If validation fails, fix the issue before proceeding. Do not skip verification.
7. **Update this index** -- mark the stage as DONE in the progress tracker, add brief notes about what was done and any deviations
8. **Commit** -- atomic commit with the message specified in the stage. Include all changed files (code, config, docs, and this plan's index.md).

Repeat until all stages are DONE or a stage is BLOCKED.

**If a stage cannot be completed**: mark it BLOCKED with a note explaining why, and stop.

**If assumptions are wrong**: stop, document in the Issues section, revise affected stages, and get user confirmation before continuing.

---

## Issues

- **2026-05-30 [BLOCKING — Stage 1]** `cf_faith.py` had an unterminated docstring (line 85 closed with escaped `\"\"\"` instead of `"""`) → the module never imported, so `test_cf_faith.py` had **never actually run** despite the index marking CFfaith "✅ tested". Syntax fixed; 20/21 now pass.
- **2026-05-30 [BLOCKING — metric semantics, entangled with Stage 4]** With the syntax fixed, `test_identical_cf_hard_is_one` fails: an identical CF (`x_cf == x_orig`, null intervention) scores `hard=0.0`, the test asserts `1.0`. Root cause is a genuine design fork in what the forward-sim residual compares against:
  - **Current code (noiseless-rollout):** `x_cf[t0:]` is compared to a *noiseless* VAR rollout seeded from `x_cf`. This penalises the factual trajectory's **own innovation noise** (measured residual ≈ 0.0098 vs `tol=1e-3`), so even a null intervention is judged "unfaithful." Internally consistent with the **Stage 4 CARLA spec** (CARLA emits a noiseless rollout → `hard=1` by construction; Wachter doesn't → `hard=0`), so the H1/H3 phenomenon still holds. But the factual scoring as unfaithful hurts metric interpretability and breaks the scaffold test.
  - **Pearl delta-recursion (principled):** faithful iff `delta = x_cf − x_orig` follows the homogeneous recursion `delta[t] = Σ_l A_l·delta[t−l]` for `t>t0` (noise cancels via abduction). Identical CF → `delta=0` → `hard=1` (test passes). But this requires **Stage 4 CARLA to abduct + re-inject the original innovations** (`x_cf[t] = Σ_l A_l·x_cf[t−l] + e_orig[t]`), not the currently-specified pure noiseless rollout.
  - The three scaffold compliance tests are **provably contradictory**: `test_hard_score_is_one` / `test_soft_score_near_one` build a *noiseless* CF (pass only under noiseless-rollout) while `test_identical_cf_hard_is_one` expects an identical CF to be faithful (passes only under Pearl delta). Verified on the real generator (noise std ≈ 0.128): noiseless-built CF → `#1` residual 0 / `#3` residual 0.272; identical CF → `#1` hard=0 / `#3` hard=0. So "keep all scaffold tests green as-is" was never achievable.
  - **RESOLVED (user decision, 2026-05-30):** keep **Option #1 (noiseless-rollout)** — `cf_faith.py` logic unchanged; corrected the single outlier test (`test_identical_cf_hard_is_one` → `test_identical_cf_hard_is_zero`, asserts `hard=0` with rationale). Stage 4 CARLA spec is **unchanged** (pure noiseless rollout → `hard=1` by construction). 21/21 green. Follow-on watch item for Stage 6: CARLA's noiseless CFs are off the noisy data manifold, so they may score lower on **OOD-plausibility** — expected under #1, report as an honest tension rather than a bug.

---

## Decisions

- **2026-05-30** — Tested modules (`LinearSCMT`, `CFfaith`) are the source-of-truth API; `run_all.py`'s API is discarded and the harness rewritten.
- **2026-05-30** — Paper deliverable limited to code + figures + results + hypothesis assessment (no prose).
- **2026-05-30** — DiCE via official `dice-ml` with adapters; from-scratch DPP fallback permitted if integration fails (per MVP risk table).
- **2026-05-30** — Two-tier `smoke`/`full` configs; CPU target; uv for all dependency management.
- **2026-05-30** — `intervention_t` derived as the first timestep where `|cf − x| > tol`, applied uniformly across all CF methods so CF-faith is comparable.
- **2026-05-30** (pre-review) — Added a **fail-fast phenomenon check** at end of Stage 5: the Wachter-vs-CARLA CF-faith gap must be confirmed on the smoke config before building IG/Shift-VR/figures. Guards against the `intervention_t`→`T-1` edge case (`cf_faith.py:108`) where a final-timestep-only CF would score faithful and silently invert H1.
- **2026-05-30** (pre-review) — **CARLA-causal** fully specified in Stage 4: fixed early intervention timesteps, only `x[t0]` free, differentiable noiseless VAR rollout for `t>t0`, classifier-flip + proximity objective → hard-faith=1 by construction.
- **2026-05-30** (pre-review) — **Shift-VR** pinned: frozen base classifier, regenerate CFs on `E_shift` inputs, `Shift-VR = validity(E_shift)/validity(E_base)`. The "same CFs remain valid" framing is rejected as incoherent under a single classifier.
- **2026-05-30** (pre-review) — **dice-ml** uses the `backend="PYT"` gradient path with a flatten/reshape adapter; a one-attempt decision gate on the smoke config triggers the from-scratch DPP fallback if it fails.
- **2026-05-30** (pre-review) — Datasets are **regenerated in CI** (deterministic via seed), never committed. Shape contract standardised on `(T,k)` at public boundaries (permute lives only inside `TCNClassifier`).
- **2026-05-30** (pre-review) — Propagation convention (`x @ A.T` ≡ `A @ x`) is consistent across generator/`CFfaith`/CARLA — flagged in Stage 4 so it is not "fixed" into a real bug. `_stabilise` stationarity holds only at `L=1`; noted in configs.
- **2026-05-30** (Stage 1, user decision) — **CF-faith semantics = Option #1 (noiseless-rollout):** a CF is faithful iff `x_cf[t0:]` equals the deterministic noiseless VAR rollout of itself. `cf_faith.py` is kept unchanged; Stage 4 CARLA emits a pure noiseless rollout → `hard=1` by construction; Wachter/DiCE don't propagate → `hard=0`. Rejected the alternative (Pearl delta-recursion `delta[t]=Σ A_l·delta[t−l]` with noise abduction) despite it being the textbook counterfactual, to minimise blast radius and keep Stage 4 as written. **Consequence to honour downstream:** CARLA CFs are off the noisy data manifold, so expect lower OOD-plausibility for CARLA in Axis C (Stage 6) — report as an honest metric tension, not a bug. The scaffold's `test_identical_cf_hard_is_one` was the lone outlier (only Pearl could satisfy it) and was corrected to assert `hard=0`.
