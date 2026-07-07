# CausalTemp-XAI — Project Plan (Principal Investigator)

**Date:** 2026-07-07
**Branch at time of writing:** `refactor/causaltemp-xai-restructure`
**Prepared by:** PhD Research Director (PI)
**Evidence base:** direct reading of `README.md`, `docs/` (all plan documents, `hypotheses_assessment.md`, `axis_metrics_report.md`, `BenchmarkingTSCFEs.md`), the `causaltemp_xai/` package, the phased pipeline `experiments/01–06` + `_common.py`, `results/`, the test suite (147 tests executed on 2026-07-07: **144 pass, 3 fail**), and git history/status.

---

## 1. Problem Statement

No existing public benchmark jointly provides (i) ground-truth causal variables, (ii) a ground-truth lagged causal graph, and (iii) analytically exact structural counterfactuals in a **temporal classification** setting. Consequently, the field cannot measure whether temporal counterfactual (CF) explainers respect the causal mechanisms of the data-generating process. Temporal-XAI benchmarks measure validity/proximity/sparsity/plausibility but not causal faithfulness; temporal causal-discovery benchmarks measure graph recovery but have no explainer in the loop.

**CausalTemp-XAI** closes this gap: synthetic temporal SCM benchmarks (linear VAR and additive-noise nonlinear MLP transitions) with exact Pearl abduction, a causal-faithfulness metric (CF-faith, two semantics), a four-axis evaluation protocol (concept, graph, counterfactual, robustness), and an empirical demonstration that **rankings by traditional CF metrics and rankings by causal faithfulness disagree — up to complete rank inversion** (v0.1 result: validity-vs-CF-faith Spearman ρ = −0.87 on the locked linear config; reproduced qualitatively on the nonlinear config with 6 methods).

**Confidence that the core phenomenon is real: High** — reproduced on two mechanism families, two classifiers, and two pipeline generations, with a fail-fast guard (`experiments/phenomenon_check.py`).
**Confidence that it is currently publication-ready: Low** — sample sizes (n_cf = 10 on current runs), single seed, no confidence intervals, two documented metric bugs, and unverified references.

---

## 2. Research Objectives (measurable / falsifiable)

- **O1 — Metric validity.** Every reported metric measures what the paper claims it measures. Falsifiable check: zero open rows in the `axis_metrics_report.md` "fix recommended" table; adversarial unit tests for each metric (a metric must *fail* a constructed violating CF and *pass* the oracle CF). Currently failing: IVR/CF-faith retroactive tolerance mismatch (penalizes CELS with `IVR=1.0`, `soft=0.0` artifacts), `MCC_coverage` ceiling at 1.0.
- **O2 — Powered core claim (H1/H3/H4).** Confirm on ≥ 2 mechanism families × ≥ 2 classifiers × ≥ 6 CF methods × ≥ 5 seeds × n_cf ≥ 100, with bootstrap 95% CIs: standard CF methods reach validity > 0.9 with CF-faith(hard) < 0.3 (H1); the causal recourse baseline reaches CF-faith(rollout, hard) > 0.7 (H3); rank correlation between traditional metrics and CF-faith has ρ < 0.5, with CIs excluding 0.5 (H4, currently underpowered at 3–6 methods).
- **O3 — Coverage of the method-paradigm space.** Evaluate ≥ 8 explainers spanning the four CF-architectural paradigms plus attribution and concept baselines (per the paradigm table in `updated_general_plan.md`), each either an official implementation or an honestly renamed proxy — no proxy may carry an original method's name. Falsifiable check: per-method provenance table in the paper (implementation source, version, validation test).
- **O4 — Generalization axes (H5–H7).** At least two of: Gaussian-noise negative control (H5), non-monotonic mechanism ablation (H6), regime-switching ablation (H7), each with a pre-registered expected direction and a documented verdict — including negative verdicts.
- **O5 — Clinical anchor (go/no-go gated).** Either a SepsisSim semi-synthetic benchmark (MIMIC-IV sepsis cohort, clinician-validated DAG) with the full axis protocol, or a documented descope decision by the M4 gate with the paper reframed as a synthetic-benchmark contribution. Falsifiable check: a written go/no-go memo with data-access evidence.

**Publication target (judgment, Medium confidence — deadlines must be verified by Literature Intelligence before commitment):** primary — NeurIPS Datasets & Benchmarks 2027 or ICLR 2027; secondary — AAAI 2027; journal fallback — Pattern Recognition or Artificial Intelligence in Medicine (if SepsisSim lands). The v0.1 freeze is already workshop-grade; a workshop submission of the frozen phenomenon is a low-cost option while v1.0 matures.

---

## 3. State of the Code (audited 2026-07-07)

### Working and tested
- **Generators:** `LinearSCMT` (VAR(1), non-Gaussian noise) and `NlinearSCMT` (additive-noise per-node MLP, spectral-norm-capped) with ground-truth graph + serializable mechanisms; seeded and persisted (`causaltemp_xai/benchmarks/`, `data_io.py`). Well covered by tests.
- **Oracle structural CF:** abduction–action–prediction, Pearl (noise-reinjecting) and rollout (noiseless) variants; mechanism-generic; mutual-exclusivity property verified for the oracle construction (`benchmarks/structural_cf.py`, `scm/`).
- **CF-faith:** two semantics (`noiseless_rollout`, `pearl_delta`), hard + soft, both always reported (`metrics/cf_faith.py`).
- **Axis C:** validity, proximity (L1/L2/DTW), sparsity, OOD (IsolationForest, Mahalanobis), TRSI, IVR (`metrics/axis_c.py`; tested).
- **CF methods wired (6):** `CARLARecourse` + cfts-backed Wachter/COMTE/CONFETI/CounTS/CELS via the vendored `third_party/cfts_repo` submodule. Native `WachterCF`/`DiCECF` exist but are not in the phased pipeline.
- **Classifier:** LSTM wrapper with train CLI (92–95% test accuracy on smoke configs).
- **Phased pipeline `experiments/01–06` + `_common.py`:** clean per-phase artifacts under `results/<config>/<classifier>/…`, honest axis routing (Axis B explicitly labeled tautological without a discovery method; Axis A LD/MCC reported NaN rather than faked).
- **Results produced:** `smoke` + `smoke_nl` runs (LSTM, n_cf = 10, 6 methods + 2 oracles), accumulated table, 3 figures. v0.1 `full`-config results (TCN, n_cf = 100, 3 methods) with a pre-registered, documented GO decision (`docs/hypotheses_assessment.md`).

### Broken / failing (verified by running the suite today)
1. `tests/test_golden_linear.py::test_full_X_bit_identical` — golden value differs at the 14th decimal (−1.4410774479072699 vs …68). Bit-identity across platforms/BLAS builds is not achievable; the golden policy needs a tolerance decision, not a silent re-lock.
2. `tests/test_golden_linear.py::test_cf_faith_scores` — CF-faith golden soft score fails (same root cause family; must be diagnosed, not assumed).
3. `tests/test_methods.py::test_dice_ml_backend_returns_cfs` — dice-ml backend resolves to `fallback` in the current environment; either the env is missing the extra or the backend broke.

### Known metric defects (documented in `docs/axis_metrics_report.md`, confirmed root-caused)
- **Tolerance mismatch:** `derive_intervention_t` uses per-element tol 1e-3; `CFfaith` retroactive gate and `axis_c.ivr` use a *summed* 1e-4 — any full-trajectory reconstruction method (CELS) is falsely flagged (`IVR=1.0`, `soft=0.0` on every instance). Current CELS numbers are artifacts.
- **`MCC_coverage` ceiling:** default threshold 1e-3 is trivially cleared; metric has near-zero discriminative power.
- **`shift_vr` instability:** ratio explodes when `validity_base` is small at n_cf = 10 (CftsConfeti 2.67× artifact).

### Stubbed / dishonestly named (untracked files)
- `methods/attribution/dynamask.py` — **is finite-difference saliency, not Dynamask** (docstring admits "proxy").
- `methods/attribution/timeshap.py` — crude Monte-Carlo masking, **unseeded RNG**, not TimeSHAP's pruning/coalition scheme.
- `methods/concept/cbm_t.py` — 2-feature logistic probes, time-uniform maps; a very loose CBM reading.
- `methods/concept/ivae.py` — a real (small) iVAE; plausible, but unwired and untested.
- **None of these four is wired into any experiment script or covered by any test.** They are exported from `causaltemp_xai.methods.__init__` under the original method names — an integrity problem if ever reported as those methods.

### Missing entirely (relative to `docs/updated_general_plan.md`)
- SepsisSim (contribution #3); regime-switching and non-monotonic NlinearSCM-T ablations; Gaussian-noise negative control preset; invertible operator-dictionary mechanisms (plan Eq. 1 — the code implements MLP transitions instead: a **spec–code divergence** to reconcile); TCN (dropped in the restructure; README still advertises it) and Transformer classifiers; TSEvo, Glacier, CITRIS/iCITRIS, DoFlow; graph-discovery for a non-tautological Axis B; multi-seed/CI statistics; manuscript skeleton.
- **Definition divergences:** the plan defines CF-faith as normalized DTW vs the analytical CF and ICC as a latent-traversal flip-rate; the code implements residual-based CF-faith and attribution-mass ICC. The paper must describe what the code does, or the code must change — currently they contradict each other.

### Housekeeping debt
- The entire phased pipeline, `results/`, the concept/attribution additions and `axis_metrics_report.md` are **uncommitted** (untracked) on the refactor branch; `docs/references_verified.md` was present in the branch status snapshot and is now **missing from disk** — the reference-verification pass has been lost or was never persisted.
- Dead duplicate: `causaltemp_xai/methods/cfts_methods.py` (497 lines) duplicates `methods/counterfactual/cfts_methods.py` byte-for-line.
- README is stale: references `causaltemp_xai/benchmark/` (now `benchmarks/`), `classifiers/tcn.py` (gone), `attribution/` at package top level (moved), and a "3 CF methods" story superseded by the 6-method pipeline.
- Legacy harness (`run_all.py`, `run_cfts.py`, `figures.py`, root-level result JSON/CSVs in `experiments/`) coexists with the phased pipeline — a provenance hazard.

---

## 4. Milestones

Effort estimates assume the current lab cadence (one methodology-focused contributor plus specialist agents). Dependencies are strict: do not start a milestone whose upstream DoD is open.

### M0 — Consolidate the working tree *(effort: 2–3 days; owner: Methodology & Coding; review: PI)*
The current scientific core exists only as uncommitted files on a refactor branch inside a OneDrive-synced folder. This is the single largest avoidable risk to the project.
- Commit the phased pipeline, `_common.py`, `results/README.md`, metric additions, and docs in reviewable commits; decide and document a `results/` versioning policy (recommend: commit `results/tables` + `figures` + JSON summaries, ignore `.npy` blobs).
- Delete the dead `methods/cfts_methods.py` duplicate; fix README layout drift; decide the fate of the legacy harness (keep as `--legacy` or delete after parity check).
- Resolve the 3 failing tests: adopt a tolerance-based golden policy (documented rationale, not a silent re-lock), root-cause the CF-faith golden failure, fix or explicitly gate the dice-ml backend test on the extra being installed.
- Merge `refactor/causaltemp-xai-restructure` → `main` with CI green.
- **DoD:** `git status` clean; `pytest` 147/147 (or documented skips); CI green on main; README reproduces the phased pipeline from a clean checkout.

### M1 — Metric integrity (O1) *(effort: 1 week; owner: Methodology & Coding; review: Results & Writing challenges each metric with an adversarial case)*
- Fix the retroactive-tolerance mismatch (pick one philosophy from `axis_metrics_report.md` §6, validate against the golden tests, re-score CELS).
- Fix `MCC_coverage` (scale-relative threshold or continuous mass), guard `shift_vr` (report ratio only when `validity_base ≥ 0.3` or report the pair).
- Add adversarial metric tests: each Axis-C/CF-faith metric must flag a constructed violation and pass the oracle.
- Add a **joint faithfulness–validity criterion** (or Pareto reporting) so tiny-edit methods (Confeti) cannot be credited as "faithful" without clearing validity/proximity floors — this closes the documented gameability loophole.
- Re-run `smoke`/`smoke_nl`, regenerate `results/`, update `axis_metrics_report.md` to close its own "fix recommended" rows.
- **DoD:** zero open "fix recommended" rows; CELS scores no longer artifacts; adversarial tests in CI.

### M2 — Statistical rigor and scale (O2) *(effort: 2–3 weeks; depends on M1; owner: Methodology & Coding; review: Results & Writing for the statistics)*
- Multi-seed protocol (≥ 5 seeds through generator → classifier → CF selection), n_cf ≥ 100, bootstrap 95% CIs on every reported mean (plan principle P5).
- Restore TCN (with `pooling='last'` to fix the documented 0.79-accuracy ceiling) and add a small Transformer; classifier-generalization table.
- Run `full` and `full_nl` with the phased pipeline; verify the smoke-scale findings survive (esp. IG input-sensitivity 0.02→0.13 linear→nonlinear, and CARLA's long-horizon validity collapse — the Pearl-CARLA noise-reinjection variant flagged in v0.1 as the priority fix belongs here).
- Formal re-assessment of H1/H3/H4 with CIs; pre-register thresholds before running (as v0.1 did).
- **DoD:** results tables with CIs; H1/H3 verdicts at scale; H4 powered by ≥ 6 methods × 5 seeds; frozen classifier checkpoints for all three architectures.

### M3 — Method zoo completion (O3) *(effort: 3–4 weeks; partially parallel with M2; owner: Methodology & Coding; Literature Intelligence supplies official implementations and citations)*
- Integrate official TSEvo and Glacier (or document substitution with evidence of unavailability).
- Replace or rename the proxy TimeSHAP/Dynamask: official libraries if feasible; otherwise rename (e.g., `FDSaliency`, `MCMaskSHAP`), seed the RNG, and disclose — **no proxy ships under an original method's name.**
- Wire CBM-T and iVAE into an Axis-A phase (enables H2 and the currently-NaN LD/MCC_disent metrics; `concept_stability` in Axis D becomes usable).
- Stretch (may slip to M5): one graph-discovery/self-graphing method (CITRIS or a simple Granger/PCMCI baseline) so Axis B stops being tautological; the `graph_error_decomposition` code already exists.
- Every method: a smoke test + a provenance row (source, version, adapter).
- **DoD:** ≥ 8 explainers spanning the paradigm table, each tested; H2 assessed; method-provenance table drafted.

### M4 — Benchmark extensions + scope gate (O4, O5 gate) *(effort: 2–3 weeks; depends on M1; owner: Methodology & Coding)*
- Gaussian-noise negative-control preset (H5), sparsity sweep beyond `full_sparse`.
- Regime-switching NlinearSCM-T (HMM structural breaks) → H7 via Shift-VR; non-monotonic ablation (H6).
- Reconcile the operator-dictionary spec vs the MLP implementation: either implement the invertible-operator mechanism family as a third generator or amend the plan/paper to the MLP formulation with its (weaker) identifiability story. **The paper cannot cite Eq. 1 of the plan while shipping MLP transitions.**
- **SepsisSim go/no-go gate (PI decision):** evidence required — MIMIC-IV credentialed access confirmed, clinician DAG collaborator named, feasibility spike done. No-go → descope to future work and reframe contribution list; Go → M5.
- **DoD:** ablation presets generated + tested + oracle-validated; H5–H7 verdicts (including negative ones) documented; signed go/no-go memo.

### M5 — SepsisSim (conditional on M4 Go) *(effort: 3–4 weeks; owner: Methodology & Coding + external clinical collaborator; review: Literature Intelligence for clinical-DAG grounding)*
- Cohort extraction, clinician-validated DAG, time-varying treatment simulation, label from clinical threshold (e.g., MAP < 65 mmHg), full axis protocol.
- **DoD:** reproducible cohort pipeline; DAG sign-off documented; benchmark results table.

### M6 — Reference and novelty verification *(effort: 1–2 weeks; parallel with M2–M4; owner: Literature Intelligence)*
- Verify all 46 web-sourced references in `docs/BenchmarkingTSCFEs.md` (several look machine-collected; some venues are non-standard). Recreate the lost `references_verified.md` as a persisted, versioned artifact with BibTeX.
- **Novelty verification pass** against the closest existing work: CausalTime, CausalDynamics, TCS (temporal causal simulation), ExplainTS, XTSC-Bench, CFTS/AMEE, DoFlow, "A Causal DAG Prior for Synthetic Time-Series Classification Datasets", and any 2025–2026 successor. The claim "no benchmark jointly provides causal variables + lagged graph + analytical CFs in a temporal classification setting" must survive this pass or be weakened *before* the manuscript is framed around it.
- **DoD:** verified bibliography committed; a novelty ledger (claim → nearest prior work → surviving delta) approved by PI.

### M7 — Manuscript *(effort: 4 weeks; overlaps M2–M5 tail; owner: Results & Writing; Literature Intelligence verifies citations; Methodology reproduces every number)*
- Skeleton early (intro/related work can start now from `BenchmarkingTSCFEs.md` — after M6 verification).
- Results sections only from frozen, CI'd numbers; per-axis figures beyond the current 3 (the 4-axis protocol needs 4-axis figures); limitations section carries the honest caveats already documented (CARLA circularity as positive control, gameability, synthetic-only if M5 no-go).
- Internal reviewer simulation (R2-style attack: circularity of CF-faith for CARLA; "synthetic toy" objection; metric-validity objections; H4 power).
- **DoD:** full draft; every number traceable to a `results/` artifact; reviewer-simulation issues resolved or acknowledged.

### M8 — Final quality audit + submission *(effort: 1 week; owner: PI)*
- Full pre-submission checklist (see §7); clean-checkout reproduction of the entire pipeline by someone other than the author of the code.
- **DoD:** every checklist item pass; submission package archived with tagged release.

**Critical path:** M0 → M1 → M2 → M7 → M8, with M3/M4/M6 in parallel and M5 conditional. Rough envelope: **12–16 weeks** to submission-ready without SepsisSim; +4 weeks with it.

---

## 5. Prioritized TODO List

**P0 = blocking (nothing downstream is trustworthy until done). P1 = required for submission. P2 = valuable, cut first under pressure.**
Owners: **[M&C]** Methodology & Coding Scientist, **[Lit]** Literature Intelligence, **[R&W]** Results & Writing Scientist.

### P0 — Blocking
1. **[M&C]** Commit the untracked pipeline (`experiments/01–06`, `_common.py`), `results/` policy, new method files, and docs; merge refactor branch to main with CI green. (M0)
2. **[M&C]** Resolve the 3 failing tests: tolerance-based golden policy + root-cause of the CF-faith golden failure + dice-ml backend gate. (M0)
3. **[M&C]** Fix the IVR/CF-faith retroactive tolerance mismatch; re-score CELS; re-run smoke pipelines. Current CELS rows in `results/tables/table_axis_c_cf_faith.csv` are artifacts and must not survive into any draft. (M1)
4. **[M&C]** Fix `MCC_coverage` threshold ceiling; add adversarial metric tests. (M1)
5. **[M&C]** Rename-or-replace decision executed for proxy Dynamask/TimeSHAP/CBM-T (no original names on proxies); seed the TimeSHAP RNG; remove the dead `methods/cfts_methods.py` duplicate; fix stale README. (M0/M3)
6. **[Lit]** Recreate `docs/references_verified.md`: verify all 46 references in `BenchmarkingTSCFEs.md`; flag fabricated/uncheckable entries. **No writing task may cite an unverified reference.** (M6)
7. **[Lit]** Novelty verification pass on the central "no such benchmark exists" claim against CausalTime / TCS / ExplainTS / XTSC-Bench / DoFlow and 2025–26 successors. (M6)

### P1 — Required for submission
8. **[M&C]** Multi-seed (≥5) + bootstrap CI protocol wired into `_common.py` aggregation; n_cf ≥ 100. (M2)
9. **[M&C]** Restore TCN (`pooling='last'`) and add Transformer; classifier table at >90% or documented gap. (M2)
10. **[M&C]** Run `full` + `full_nl` through the phased pipeline; verify smoke-scale findings at scale. (M2)
11. **[M&C]** Pearl-CARLA (noise-reinjecting on-manifold recourse) — the v0.1 priority item; expected to recover validity while `pearl_hard=1`. (M2)
12. **[M&C]** Integrate TSEvo + Glacier (official code) or document substitution. (M3)
13. **[M&C]** Wire CBM-T/iVAE into an Axis-A experiment phase → H2 verdict; enable `concept_stability`. (M3)
14. **[M&C]** Joint faithfulness–validity score / Pareto reporting to close the tiny-edit gameability loophole. (M1)
15. **[M&C]** Gaussian-noise negative control + sparsity sweep (H5); regime-switching + non-monotonic ablations (H6/H7). (M4)
16. **[PI + M&C]** Reconcile plan-vs-code divergences: CF-faith definition (DTW vs residual), ICC definition (latent flip-rate vs attribution mass), mechanism family (operator dictionary vs MLP). Update `updated_general_plan.md` or the code — they currently contradict. (M4/M6)
17. **[PI]** SepsisSim go/no-go memo with data-access evidence. (M4 gate)
18. **[R&W]** Manuscript skeleton + related-work draft (post-verification); per-axis figures; statistics review of the CI protocol. (M7)
19. **[R&W]** Reviewer simulation: circularity, synthetic-only, metric-validity, H4-power attacks. (M7)
20. **[M&C]** `shift_vr` guard (denominator floor or pair-reporting). (M1)
21. **[Lit]** Venue deadline verification (NeurIPS D&B 2027 / ICLR 2027 / AAAI 2027) and format constraints. (M6)

### P2 — Valuable, cut first
22. **[M&C]** SepsisSim implementation (if Go). (M5)
23. **[M&C]** One graph-discovery baseline (PCMCI/Granger or CITRIS) to de-tautologize Axis B and use `graph_error_decomposition`. (M3 stretch)
24. **[M&C]** DoFlow integration for the H6 non-monotonic story.
25. **[M&C]** LOF as second OOD scorer (plan lists IF + LOF; only IF wired).
26. **[M&C]** DTW-based CF-faith variant reported alongside residual-based (bridges plan definition and implementation).
27. **[M&C]** Topological-Causality sub-study on NlinearSCM-T (explicitly deferred in the plan — keep deferred).
28. **[R&W]** Workshop-paper spin-off of the frozen v0.1 phenomenon while v1.0 matures.

---

## 6. Risks and Quality Concerns (PI audit)

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | **Loss/corruption of uncommitted work** — the entire current pipeline is untracked, in a OneDrive-synced tree; `references_verified.md` has already vanished this way | High | Severe | M0 immediately; OneDrive + git is a known hazard — commit early, push often |
| R2 | **Metric-validity attacks** — reviewers find the CELS artifact, the MCC ceiling, or the gameability loophole before we fix them | High if unfixed | Fatal to the paper | M1 is gated *before* any at-scale runs; adversarial metric tests in CI |
| R3 | **Circularity criticism** — CARLA scores `rollout_hard=1` *by construction* against the metric it optimizes; a reviewer will say the headline contrast is rigged | High | Major | Frame CARLA/oracles explicitly as positive controls; the scientific claim rests on the *standard* methods' failure, not CARLA's success; add Pearl-CARLA so faithfulness isn't definitionally tied to one method |
| R4 | **Misrepresented baselines** — proxy code named Dynamask/TimeSHAP/CBM-T; unseeded RNG in TimeSHAP | Certain (code exists) | Fatal if published | P0 item 5: official implementations or renamed proxies with disclosure |
| R5 | **Underpowered claims** — n_cf=10, single seed, 3-method ρ in v0.1; H4 is the headline and currently descriptive only | Certain today | Major | M2; do not allow any draft to promote H4 beyond "preliminary" until powered |
| R6 | **Novelty overrun** — the field moves fast; the "no such benchmark" claim is from mid-2026 sources and TCS/ExplainTS-style work is adjacent | Medium | Major | M6 novelty ledger before manuscript framing; weaken to "first with analytical structural-CF ground truth for classification" if needed |
| R7 | **Scope creep** — 4 axes × 3 benchmarks × 10+ methods × 3 classifiers × 7 hypotheses is a 2-paper program; the plan already once collapsed to an MVP for good reason | High | Schedule | Milestone gates; SepsisSim explicitly conditional; H6/H7 cut before H1–H5; P2 list is the pre-agreed cut line |
| R8 | **Synthetic-only realism objection** | High if M5 no-go | Moderate | SepsisSim if feasible; otherwise foreground identifiability rationale and position as a *diagnostic* benchmark (like synthetic causal-discovery suites), plus the workshop precedent |
| R9 | **Reproducibility drift** — bit-identical golden tests already fail cross-platform; label design once silently capped classifier accuracy (avg-pool vs endpoint label) | Medium | Moderate | Tolerance-based goldens with documented bounds; seed audit (TimeSHAP RNG is a live example); clean-checkout reproduction by a non-author at M8 |
| R10 | **Spec–code divergence** — plan math (operator dictionary, DTW CF-faith, latent ICC) ≠ shipped code; a paper written from the plan would misdescribe the artifact | Certain today | Major | P1 item 16: one source of truth before any methods section is written |

---

## 7. Pre-Submission Quality Gate (M8 checklist)

Every item must pass before the PI approves submission:
- ☐ Research question and contribution list match what the code actually does (R10 closed)
- ☐ Literature review current; novelty ledger signed off (M6)
- ☐ All references verified authentic (`references_verified.md` regenerated and versioned)
- ☐ All metrics pass adversarial tests; no open rows in `axis_metrics_report.md`
- ☐ Method provenance table complete; no proxy under an original name
- ☐ Multi-seed CIs on every headline number; H-verdicts pre-registered
- ☐ Clean-checkout reproduction by a non-author (documented)
- ☐ Limitations honest: positive-control framing, gameability caveat, synthetic scope
- ☐ Figures publication-quality, one per evaluation axis minimum
- ☐ Reviewer-simulation objections addressed in-text
- ☐ Code, data generation, and checkpoints released under a tagged version

---

## 8. Standing Decisions (PI)

1. **v0.1 remains frozen.** No retro-editing of the v0.1 assessment; v1.0 supersedes, never rewrites.
2. **Nothing cites an unverified reference.** Effective immediately.
3. **No proxy ships under an original method's name.** Effective immediately.
4. **Every headline number carries a CI** from M2 onward.
5. **SepsisSim is gated, not assumed.** The paper plan must remain publishable under a no-go.

*Plan owner: PhD Research Director. Next scheduled review: end of M1 (metric integrity), or immediately upon any novelty-ledger surprise from M6.*
