# CausalTemp-XAI — Project Plan (Principal Investigator)

**Date:** 2026-07-07
**Last PI review:** 2026-07-08 — M0/M1 audited; M2 scope cut twice same day (TCN, then Transformer — now LSTM-only, O2 reassessed); M6 dispatched and reviewed on return (see §9)
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
- **O2 — Powered core claim (H1/H3/H4).** Confirm on ≥ 2 mechanism families × the LSTM classifier (single architecture — classifier-architecture generalization is descoped from M2 as of 2026-07-08, see the M2 scope decision and §9.2) × ≥ 6 CF methods × ≥ 5 seeds × n_cf ≥ 100, with bootstrap 95% CIs: standard CF methods reach validity > 0.9 with CF-faith(hard) < 0.3 (H1); the causal recourse baseline reaches CF-faith(rollout, hard) > 0.7 (H3); rank correlation between traditional metrics and CF-faith has ρ < 0.5, with CIs excluding 0.5 (H4, currently underpowered at 3–6 methods).
- **O3 — Coverage of the method-paradigm space.** Evaluate ≥ 8 explainers spanning the four CF-architectural paradigms plus attribution and concept baselines (per the paradigm table in `updated_general_plan.md`), each either an official implementation or an honestly renamed proxy — no proxy may carry an original method's name. Falsifiable check: per-method provenance table in the paper (implementation source, version, validation test).
- **O4 — Generalization axes (H5–H7).** At least two of: Gaussian-noise negative control (H5), non-monotonic mechanism ablation (H6), regime-switching ablation (H7), each with a pre-registered expected direction and a documented verdict — including negative verdicts.
- **O5 — Clinical anchor (go/no-go gated).** Either a SepsisSim semi-synthetic benchmark (MIMIC-IV sepsis cohort, clinician-validated DAG) with the full axis protocol, or a documented descope decision by the M4 gate with the paper reframed as a synthetic-benchmark contribution. Falsifiable check: a written go/no-go memo with data-access evidence.

**Publication target (updated 2026-07-08 per Lit's M6 venue verification, `docs/references_verified.md` Deliverable 3 — confidence tagged per venue, no longer a single blended "Medium"):** primary — **NeurIPS 2027** "Datasets & Benchmarks" or its renamed successor (officially rebranded "Evaluations & Datasets" for the 2026 cycle per the official NeurIPS blog; High confidence on the rename, but the 2027 CFP is not yet published — Low confidence on specific 2027 dates); secondary — **ICLR 2027**, stretch target only (official `iclr.cc` 2027 pages return 404 as of this check; a third-party aggregator estimate of ~Sept 2026 abstract deadline is unofficial — Low confidence, re-verify monthly); **AAAI 2027 dropped as a live target** (High confidence: abstract/full-paper deadlines are 21/28 July 2026, verified directly against `aaai.org` — roughly 2–3 weeks from today, unreachable with the project still mid-M2). Journal fallback unchanged: Pattern Recognition or Artificial Intelligence in Medicine (if SepsisSim lands). The v0.1 freeze is already workshop-grade; a workshop submission of the frozen phenomenon remains a low-cost option — and is now the more natural home for anything on an AAAI-2027-shaped timeline (P2 item 28), since AAAI itself is off the table.

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
> **Scope decision (PI, 2026-07-08): TCN dropped, not restored.** ~~The plan
> previously called for restoring TCN alongside adding a Transformer. TCN
> restoration is cut from M2 scope entirely — the classifier-generalization
> table is **LSTM vs. Transformer** (two architectures, not three).~~ **Superseded
> same day (PI, second pass, 2026-07-08): Transformer is also cut.** M2's
> classifier scope is now **LSTM only** — a single architecture, not two. The
> classifier-generalization table is removed from M2 entirely; there is no
> cross-architecture comparison in this milestone. If TCN and/or a second
> classifier (Transformer or otherwise) are wanted later, each re-enters as
> its own scoped task, not bundled into M2. See §9.2 for the reassessment of
> what this means for O2 as an objective — the multi-classifier
> generalization axis is descoped, not merely delayed, until the PI/user
> revisits it.
- Multi-seed protocol (≥ 5 seeds through generator → classifier → CF selection), n_cf ≥ 100, bootstrap 95% CIs on every reported mean (plan principle P5).
- ~~Add a small Transformer classifier alongside the existing LSTM; classifier-generalization table (LSTM vs. Transformer).~~ **Cut 2026-07-08** — no second classifier in M2; confirm the existing LSTM checkpoint stays frozen and documented as the sole classifier for all M2 statistics.
- Run `full` and `full_nl` with the phased pipeline; verify the smoke-scale findings survive (esp. IG input-sensitivity 0.02→0.13 linear→nonlinear, and CARLA's long-horizon validity collapse — the Pearl-CARLA noise-reinjection variant flagged in v0.1 as the priority fix belongs here).
- Formal re-assessment of H1/H3/H4 with CIs; pre-register thresholds before running (as v0.1 did).
- **DoD:** results tables with CIs; H1/H3 verdicts at scale; H4 powered by ≥ 6 methods × 5 seeds; frozen LSTM classifier checkpoint (single architecture — classifier-generalization axis explicitly out of scope for M2, see §9.2).

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
- **Classifier-diversity checkpoint (PI/user decision, deferred from M2 on 2026-07-08 — see §9.2):** at this gate, re-examine R11 (single-classifier evidence) against the actual M2 results in hand — specifically, does the validity/CF-faith divergence and the H4 rank-inversion hold with the same qualitative shape across the ≥2 mechanism families on LSTM alone, or is there any early sign (e.g. an unexpectedly clean separation, or a borderline H1/H3/H4 verdict) that would make a reviewer's "is this an LSTM artifact?" objection more dangerous than assumed? Decide **go/no-go on adding a second, non-Transformer classifier** (candidates: a small 1D-CNN, or an MLP over the flattened `(T,k)` window — deliberately cheap, chosen for architecture diversity rather than capability). If not resolved here, it carries to the M7 reviewer-simulation pass as the last checkpoint before manuscript freeze — it may not be deferred past M7.
- **DoD:** ablation presets generated + tested + oracle-validated; H5–H7 verdicts (including negative ones) documented; signed go/no-go memo; **classifier-diversity go/no-go decision recorded** (see checkpoint above).

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
- Internal reviewer simulation (R2-style attack: circularity of CF-faith for CARLA; "synthetic toy" objection; metric-validity objections; H4 power; **R11 single-classifier-architecture objection — last call on the M4 classifier-diversity checkpoint if not already resolved there**).
- **DoD:** full draft; every number traceable to a `results/` artifact; reviewer-simulation issues resolved or acknowledged; **R11 (single-classifier evidence) explicitly closed one way or the other — either a second classifier's results are in, or the manuscript states the single-architecture scope as a stated limitation, not a silent gap**.

### M8 — Final quality audit + submission *(effort: 1 week; owner: PI)*
- Full pre-submission checklist (see §7); clean-checkout reproduction of the entire pipeline by someone other than the author of the code.
- **DoD:** every checklist item pass; submission package archived with tagged release.

**Critical path:** M0 → M1 → M2 → M7 → M8, with M3/M4/M6 in parallel and M5 conditional. Rough envelope: **12–16 weeks** to submission-ready without SepsisSim; +4 weeks with it.

---

## 5. Prioritized TODO List

**P0 = blocking (nothing downstream is trustworthy until done). P1 = required for submission. P2 = valuable, cut first under pressure.**
Owners: **[M&C]** Methodology & Coding Scientist, **[Lit]** Literature Intelligence, **[R&W]** Results & Writing Scientist.

### P0 — Blocking
0. **[M&C] NEW 2026-07-08 (§9.5) — Close the M2 method-roster + smoke_nl pipeline-validation gap.**
   Verified directly (not taken on trust): `PearlCARLARecourse` (commit `145c3f3`) is fully
   implemented and unit-tested but is **absent from `experiments/03_run_cf_methods.py::build_methods()`**
   — the one registry every phase-03/04/07 run reads — so it has never flowed through the actual
   pipeline in any run, smoke or full. Independently, `results/smoke_nl/lstm/` contains only
   `train_report.json` and `table_axis_c_cf_faith.csv` has zero `smoke_nl,lstm,*` rows: phases
   03–04 of the current phased pipeline have **never been run on `smoke_nl` at all**, with any
   method roster. Dispatched to Methodology & Coding this review; see §9.5. Smoke-scale only.
1. **[M&C]** Commit the untracked pipeline (`experiments/01–06`, `_common.py`), `results/` policy, new method files, and docs; merge refactor branch to main with CI green. (M0)
2. **[M&C]** Resolve the 3 failing tests: tolerance-based golden policy + root-cause of the CF-faith golden failure + dice-ml backend gate. (M0)
3. **[M&C]** Fix the IVR/CF-faith retroactive tolerance mismatch; re-score CELS; re-run smoke pipelines. Current CELS rows in `results/tables/table_axis_c_cf_faith.csv` are artifacts and must not survive into any draft. (M1)
4. **[M&C]** Fix `MCC_coverage` threshold ceiling; add adversarial metric tests. (M1)
5. **[M&C]** Rename-or-replace decision executed for proxy Dynamask/TimeSHAP/CBM-T (no original names on proxies); seed the TimeSHAP RNG; remove the dead `methods/cfts_methods.py` duplicate; fix stale README. (M0/M3) — **Dynamask/TimeSHAP half dispatched 2026-07-08 (§9.5)**: rename + seed + test for the two attribution proxies only (CBM-T explicitly held out, separately scoped). `causaltemp_xai/methods/cfts_methods.py` dead-duplicate removal already landed (commit `88564c5`, pre-dates this review); README drift not yet re-audited this pass.
6. **[Lit]** Recreate `docs/references_verified.md`: verify all 46 references in `BenchmarkingTSCFEs.md`; flag fabricated/uncheckable entries. **No writing task may cite an unverified reference.** (M6) — **Done 2026-07-08.** See `docs/references_verified.md`: 28 verified, 14 verified-with-corrections, 3 unverifiable, 1 confirmed not a citable reference (#24, a Hugging Face search-results page). **Highest-severity finding: reference #4 (hosted on oaji.net) could not be found anywhere independently — not in the journal's own site, not in its own archive index, zero web corroboration — and is the sole support for a specific, load-bearing empirical claim repeated ~4 times in the draft. Standing rule: no number from ref #4 may appear in any manuscript without independent human in-browser confirmation.** Also: refs #1/#33 and #17/#18 are each the same paper double-cited under two URL formats — collapse before any manuscript use. See §9.3 for the PI's review of this deliverable.
7. **[Lit]** Novelty verification pass on the central "no such benchmark exists" claim against CausalTime / TCS / ExplainTS / XTSC-Bench / DoFlow and 2025–26 successors. (M6) — **Done 2026-07-08.** Verdict (see `docs/references_verified.md` Deliverable 2): the claim does **not** survive unmodified but survives narrowed to require the exact structural-CF ground truth be used **as an evaluation oracle for independently-produced, post-hoc CF-explainer methods** — CAUKER (ICLR 2026 Oral) and "A Causal DAG Prior for Synthetic TS Classification Datasets" (arXiv 2606.21776) already jointly cover ground-truth-DAG + classification labels; DoFlow (arXiv 2511.02137) already covers analytical structural counterfactuals (for forecasting). Neither builds the explainer-evaluation harness that is CausalTemp-XAI's actual contribution. **Adopt this narrowed wording for M7 framing** — see §9.3. **Not yet closed — open before M7 manuscript freeze:** (a) confirm whether CAUKER's/O'Rourke's causal DAG is genuinely *time-lagged* (edges spanning explicit lags τ, matching this project's own TSCM formalism) or a coarser per-instance generative DAG that gets temporalized some other way — the ledger doesn't resolve this and it changes how much of pillar (ii) they really cover; (b) add CausalDynamics (NeurIPS 2025), TimeGraph (KDD 2025/26), XTSC-Bench, and AMEE to related-work (found by Lit, omitted by the draft; none threaten the claim); (c) CausalProfiler (arXiv 2511.22842) needs a full read, not abstract-only, before submission — closest-sounding title match found, not yet ruled out.
7a. **[PI/M&C] Reopened 2026-07-08 — M0's "git status clean" DoD is not actually met.** Independent audit found 8 test files (`tests/test_attribution.py`, `test_cf_faith.py`, `test_classifier.py`, `test_eval.py`, `test_generator.py`, `test_mechanisms.py`, `test_nlinear_generator.py`, `test_structural_cf.py`) whose package-restructure import-path fixes (`causaltemp_xai.benchmark`→`benchmarks`, `causaltemp_xai.attribution`→`methods.attribution`, `causaltemp_xai.methods.intervention`→`scm.intervention`) exist **only in the uncommitted working tree**, not in any commit. Verified directly: `causaltemp_xai/benchmark/` (singular) no longer contains real code in the committed tree (only an empty, gitignored leftover directory), and no top-level `causaltemp_xai/attribution` package exists post-restructure — so a genuine clean checkout of `HEAD` right now fails test **collection** on these 8 files (`ModuleNotFoundError`), independent of anything M1 fixed. This is a trivial fix (the correct content already sits on disk, uncommitted) but it means M0's own DoD — "`git status` clean … README reproduces the phased pipeline from a clean checkout" — is not yet satisfied despite M0 being reported closed, and "CI green on main" cannot be claimed until this lands. **Do not merge to main, and do not cite "147/147"-style pass counts as a property of the repository, until these 8 files are committed and a real clean-checkout run is verified green.** See §9 for the full PI review. (M0, reopened)

### P1 — Required for submission
8. **[M&C]** Multi-seed (≥5) + bootstrap CI protocol wired into `_common.py` aggregation; n_cf ≥ 100. (M2)
9. ~~Add a Transformer classifier alongside the existing LSTM; classifier-generalization table.~~ **[CUT 2026-07-08, second scope pass]** — both TCN and Transformer are now out of M2 scope entirely; M2 is LSTM-only, no classifier-generalization table. Remaining classifier-related M2 work is just confirming the existing LSTM checkpoint (already >90% on smoke configs) stays frozen and documented as the sole classifier for all M2-scale statistics — folded into item 8, not a standalone task. **[PI]** If/when a second classifier is wanted, scope it as its own task and revisit O2's "≥2 classifiers" framing (now single-classifier, see §9.2). (M2)
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
29. **[M&C] NEW 2026-07-08 (§9.5)** — M4 ablation-preset **code + smoke-scale-only validation**
    (Gaussian-noise negative control H5, non-monotonic mechanism H6, regime-switching H7).
    Full-scale runs of these ablations are explicitly **not** in scope for this item — held
    pending the pipeline-clearance gate, same as item 10. Dispatched this review; see §9.5.
30. **[Lit] NEW 2026-07-08 (§9.5)** — the three open M6 follow-ups from §9.3 (lagged-graph
    question for CAUKER/Causal-DAG-Prior, four omitted related-work benchmarks, full
    CausalProfiler read). No pipeline dependency. Dispatched this review; see §9.5. (This is the
    work the coordinating session's brief referred to as "Task #16" — see §9.5's note on task
    numbering.)

---

## 6. Risks and Quality Concerns (PI audit)

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | **Loss/corruption of uncommitted work** — the entire current pipeline is untracked, in a OneDrive-synced tree; `references_verified.md` has already vanished this way | High | Severe | M0 immediately; OneDrive + git is a known hazard — commit early, push often |
| R2 | **Metric-validity attacks** — reviewers find the CELS artifact, the MCC ceiling, or the gameability loophole before we fix them | High if unfixed | Fatal to the paper | M1 is gated *before* any at-scale runs; adversarial metric tests in CI |
| R3 | **Circularity criticism** — CARLA scores `rollout_hard=1` *by construction* against the metric it optimizes; a reviewer will say the headline contrast is rigged | High | Major | Frame CARLA/oracles explicitly as positive controls; the scientific claim rests on the *standard* methods' failure, not CARLA's success; add Pearl-CARLA so faithfulness isn't definitionally tied to one method |
| R4 | **Misrepresented baselines** — proxy code named Dynamask/TimeSHAP/CBM-T; unseeded RNG in TimeSHAP | Certain (code exists) | Fatal if published | P0 item 5: official implementations or renamed proxies with disclosure |
| R5 | **Underpowered claims** — n_cf=10, single seed, 3-method ρ in v0.1; H4 is the headline and currently descriptive only | Certain today | Major | M2; do not allow any draft to promote H4 beyond "preliminary" until powered |
| R6 | **Novelty overrun** — confirmed materializing, not just a theoretical risk: Lit's M6 pass (2026-07-08) found CAUKER (ICLR 2026 Oral) and "A Causal DAG Prior for Synthetic TS Classification Datasets" (arXiv 2606.21776) already jointly provide ground-truth-DAG + classification labels, and DoFlow already provides analytical structural CFs (for forecasting) | Realized (no longer just "Medium") | Major, not fatal | Narrow the claim per P1 item 7 / §9.3: require the structural-CF ground truth be used **as an evaluation oracle for independently-produced, post-hoc CF-explainer methods** — this delta survives the M6 pass. Do not frame the manuscript around the unnarrowed claim |
| R7 | **Scope creep** — 4 axes × 3 benchmarks × 10+ methods × 1 classifier × 7 hypotheses is a 2-paper program; the plan already once collapsed to an MVP for good reason | High | Schedule | Milestone gates; SepsisSim explicitly conditional; H6/H7 cut before H1–H5; P2 list is the pre-agreed cut line |
| R8 | **Synthetic-only realism objection** | High if M5 no-go | Moderate | SepsisSim if feasible; otherwise foreground identifiability rationale and position as a *diagnostic* benchmark (like synthetic causal-discovery suites), plus the workshop precedent |
| R9 | **Reproducibility drift** — bit-identical golden tests already fail cross-platform; label design once silently capped classifier accuracy (avg-pool vs endpoint label) | Medium | Moderate | Tolerance-based goldens with documented bounds; seed audit (TimeSHAP RNG is a live example); clean-checkout reproduction by a non-author at M8 |
| R10 | **Spec–code divergence** — plan math (operator dictionary, DTW CF-faith, latent ICC) ≠ shipped code; a paper written from the plan would misdescribe the artifact | Certain today | Major | P1 item 16: one source of truth before any methods section is written |
| R11 | **Single-classifier evidence** — with TCN and Transformer both cut (2026-07-08), H1/H3/H4 are demonstrated on LSTM only; a reviewer may argue the validity/CF-faith divergence is an architecture-specific artifact rather than a general property of standard CF methods vs. causal faithfulness | Medium–High (near-certain to be asked if unaddressed) | Major to the generalization claim, not fatal to the core phenomenon | **Decision (PI/user, 2026-07-08, deferred not open):** proceed LSTM-only through M2; classifier-diversity go/no-go re-decided at the **M4 gate** against the real M2 results, with a final backstop at **M7 reviewer-simulation** — not left unresolved past manuscript freeze. Candidates if revisited: a small 1D-CNN or a flattened-window MLP (deliberately **not** Transformer — cheap architecture-diversity check, not a capability upgrade). Until then, the ≥2-mechanism-family axis (linear VAR vs. nonlinear MLP) is the stated generalization evidence, and the single-architecture scope is stated honestly, not hidden. |

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

---

## 9. PI Review Log

### 9.1 — 2026-07-08: M0/M1 audit, M2 scope-cut consistency check, M6 dispatch

**Scope of this review:** independent re-verification of the M0 (consolidate) and M1 (metric integrity)
milestones after 8 new commits landed, a consistency check of the M2 TCN-drop scope decision against
the rest of this document, and a decision on whether to start M6 (reference/novelty verification) while
M2's coding tasks (#8–#11: multi-seed/CI infrastructure, Transformer classifier, Pearl-CARLA,
classifier-generalization table) run in the background under Methodology & Coding.

**M1 (metric integrity) — assessed as sound. Confidence: High.** Read `docs/axis_metrics_report.md` §9
in full (not just the summary handed to me) and independently re-ran the full test suite on the current
working tree: **198 passed, 1 skipped** (dice-ml, gated on Python ≥3.9; this env is 3.8 — correctly
documented, not a hidden failure). All four M1 fixes are internally consistent, each rejects at least one
plausible alternative with a stated reason (e.g. per-element-max `INTERVENTION_TOL` over a rescaled sum;
continuous chance-normalized `MCC_coverage` over a scale-relative threshold), and each is backed by
targeted regression tests in `tests/test_metric_adversarial.py` (40 tests, confirmed passing) in addition
to the pre-existing golden tests (`tests/test_golden_linear.py`, confirmed passing, "no constant
re-blessed" claim holds under my own run). The before/after numbers quoted in §9.6 (CELS IVR 1.00→0.00,
soft 0.0→0.784/0.982; MCC_coverage 1.00→1.278; CftsConfeti shift_vr 2.67→1.09) are plausible and
consistent with the stated root causes. **One forward-looking, non-blocking observation**: the joint
faithfulness–validity criterion (§9.4) requires binary validity but no confidence margin — a CF that just
barely crosses the decision boundary with a near-zero edit would still earn full joint credit. This is not
a defect (the M1 rationale for omitting a proximity floor is sound), but it is a place a sharp reviewer
could still probe; worth a one-line acknowledgment in the M7 limitations section, not a re-open.

**M0 (consolidate) — DoD not actually met. This is a real finding, not a nitpick.** The four M0 commits
did land, and the working tree today is scientifically sound and fully tested (198/1 skip, above). But
independent audit (`git status`, `git diff <commit>`, `git ls-tree`) found that **8 test files still carry
uncommitted import-path fixes** required for the package restructure (`causaltemp_xai.benchmark` →
`benchmarks`, `causaltemp_xai.attribution` → `methods.attribution`, `causaltemp_xai.methods.intervention`
→ `scm.intervention`). Verified directly: `causaltemp_xai/benchmark/` (singular) contains no real code in
the committed tree — only an empty, gitignored leftover directory — and no top-level
`causaltemp_xai/attribution` package exists post-restructure. **A clean checkout of `HEAD` today would
fail test collection on these 8 files.** This means M0's own DoD ("`git status` clean … README reproduces
the phased pipeline from a clean checkout") is not satisfied, and any claim of "CI green on main" is
unverified until this lands — I could not confirm CI status directly (`gh run list` returned nothing in
this environment). Filed as new P0 item **7a**. The fix itself is trivial — the correct content already
exists on disk, uncommitted — but per my standing instruction not to run git commit myself, I am not
landing it in this session; flagging it as the top action item for whoever has commit access next.
This is, ironically, R1 (loss/corruption of uncommitted work in a OneDrive-synced tree) recurring in a
smaller form even after M0 was supposed to have closed it — a reminder that "M0 done" should be
re-verified against a literal clean checkout before being trusted again, not just against `git log`.

**M2 scope-cut (TCN dropped) — sound, no objection.** The standing decision (§4, M2 section) is
internally defensible: O2's own falsifiable threshold is "≥2 classifiers," which LSTM+Transformer clears;
TCN carried a known, documented accuracy ceiling (0.79, avg-pool/endpoint-label cap — see
`docs/hypotheses_assessment.md`) that this decision avoids re-litigating. **However, the M2 milestone text
and the P1 TODO list had drifted out of sync**: §4's DoD line had already been corrected (uncommitted) to
"both architectures (LSTM, Transformer)," but P1 TODO item **9** still read "Restore TCN … and add
Transformer," flatly contradicting the scope decision two lines above it in the same document. **Fixed in
this review**: item 9 reworded to match the standing decision; R7's risk illustration ("3 classifiers")
corrected to "2 classifiers" for the same reason. No other stale "three architectures" references found
in this document (a few older, superseded planning docs — `docs/mvp_plan.md`, `docs/general_plan.md`,
`docs/updated_general_plan.md` — still describe the original TCN/LSTM/Transformer ambition; these are
historical planning artifacts this document supersedes and were left untouched, per scope, but are noted
here as a minor known inconsistency if anyone reads them instead of this plan).

**M6 dispatched now (Literature Intelligence), not held.** M6 (reference + novelty verification) has no
code dependency and touches no file the in-flight M&C background agent is using (`docs/references_verified.md`
is a new file; the agent only reads `docs/BenchmarkingTSCFEs.md` and searches external literature) — it is
exactly the "parallel with M2–M4" work this plan already designated it as. Dispatched via the Agent tool
(PhD Literature Intelligence) for P0 items 6, 7, and P1 item 21 (venue deadlines, bundled since same owner,
low effort, no dependency). Before dispatch I spot-checked `docs/BenchmarkingTSCFEs.md`'s 46 references
myself and found concrete, mechanically-verifiable leads to prioritize (handed to the agent as a running
start, not a substitute for its own pass): **references #1 and #33 are the same paper cited under two
numbers** (arXiv 2603.27792, two different URL forms); **reference #24 is a Hugging Face search-results
URL, not a citable paper**; **reference #4 is hosted on oaji.net** (a low-quality aggregator, not a
recognizable publisher or arXiv) yet is cited as load-bearing support for a specific, central empirical
claim (rank-correlation and OOD-AUROC numbers on MIMIC-IV/PhysioNet-Sepsis) an unusually large number of
times through the draft — this single source deserves the most scrutiny of the 46; and the draft's prose
references "Table 1/2/3" that do not exist anywhere in the file. These findings independently corroborate
this plan's pre-existing suspicion ("several look machine-collected," §3) — the P0 gate on unverified
references was the right call and remains in force. Await the agent's `docs/references_verified.md` and
novelty ledger before M6 is marked closed or before any manuscript framing (M7) proceeds.

**Nothing found in this review blocks the in-flight M2 work (#8, #10)** — the M0 gap (7a) is a git-hygiene
issue orthogonal to the Pearl-CARLA/multi-seed code, and the M2 TCN scope-cut is sound. The one
item requiring a human/PI-with-commit-access decision is **7a**: someone needs to commit the 8 pending
test-file fixes and re-verify a real clean checkout before main is touched or CI-green is claimed.

### 9.2 — 2026-07-08 (same day, second pass): Transformer also cut — LSTM-only for M2; O2 reassessed

**New instruction received mid-review, effective immediately: no Transformer classifier either.** M2's (and
the project's, until/unless revisited) classifier scope is now **LSTM only** — a single architecture, not
two. Coordinator has already told the M&C background agent to drop task #9 (Transformer classifier) and
task #11 (classifier-generalization table) if started; those are removed from the shared task list. I did
not dispatch anything Transformer-related myself this session (the Literature Intelligence dispatch in
§9.1 is reference/novelty verification only, no code, no classifiers) — there is nothing of mine to cancel.

**Document updates made in this pass:**
- **O2** (§2): reworded from "≥ 2 classifiers" to "the LSTM classifier (single architecture …)", cross-
  referenced to this section.
- **M2 scope-decision callout** (§4): superseded in place (struck through, not deleted, so the decision
  history stays legible) — TCN and Transformer are both cut; no classifier-generalization table in M2.
- **M2 bullet list and DoD** (§4): Transformer bullet struck through and replaced with a note to confirm
  the existing LSTM checkpoint stays frozen; DoD now asks for one frozen checkpoint, not two.
- **P1 item 9**: struck through and marked cut a second time, with a pointer back to this section.
- **R7** (risk table): "2 classifiers" → "1 classifier" for consistency.
- **New risk R11 added** (see §6): single-classifier evidence as a named, tracked risk to the
  generalization strength of H1/H3/H4 — not swept under the rug.

**My reassessment of O2 as an objective (the actual judgment call this correction asked for), option (a)
chosen — descope and reframe, not block, with an explicit question back to the user:**

O2's original "≥ 2 classifiers" clause existed to demonstrate that the validity/CF-faith divergence is a
property of *standard CF methods vs. causal faithfulness in general*, not an artifact of one classifier's
inductive bias (e.g., an LSTM's smoothness assumptions happening to produce this particular gap). With
only LSTM in scope, **that specific generalization axis is gone from M2**, not delayed — there is no
"second classifier, later" plan currently active. I judged this **does not invalidate H1/H3/H4 as
falsifiable claims** (they can be fully confirmed or refuted on LSTM alone, and the ≥2-mechanism-family
axis — linear VAR vs. nonlinear MLP — still provides real, independent generalization evidence along a
different dimension), **but it does weaken the strongest version of the paper's generalization story**,
and a reviewer is likely to ask "is this an LSTM artifact?" almost immediately (R11).

Decision: **(a) descope, don't block.** M2's remaining coding work (#8 multi-seed/CI infrastructure, #10
Pearl-CARLA, and eventually #12/#13) does not require a second classifier to proceed, so I am not stopping
M2 pending a reply — freezing statistical infrastructure work on a classifier-count question would be a
worse use of the schedule than proceeding and stating the scope honestly. O2 and the M2 DoD are reworded
to be single-classifier-correct effective immediately.

**But I am also exercising option (b) — flagging this explicitly for you to decide, because it is a real
reduction in evidentiary strength for the paper's core claim, not just a wording change:** do you want (i)
single-architecture (LSTM-only) evidence to stand as-is for the submission this project is aiming at, with
the limitation stated plainly in M7; (ii) a cheap second classifier later as a robustness check — not
necessarily a full Transformer, something lighter (e.g. a small 1D-CNN or an MLP-over-flattened-window
baseline) if the concern is architecture-diversity rather than Transformer specifically; or (iii) something
else. No action is blocked on your answer — M2 proceeds LSTM-only under (i) by default until you say
otherwise, and R11 stays open in the risk register as a live reminder either way.

**Resolved, same day (user, 2026-07-08): deferred, not (i)/(ii)/(iii) as originally framed — a fourth
option.** The user's answer is to make **no classifier-diversity decision now**, and instead re-decide it
at a named checkpoint once there is real evidence to decide it with: the **M4 scope gate**, against the
actual M2 results (does the validity/CF-faith divergence and H4 rank-inversion hold up cleanly on LSTM
alone, or does something in the real numbers make the single-architecture objection sharper than assumed?)
— with the **M7 reviewer-simulation pass** as a hard backstop if M4 doesn't resolve it, so this cannot
silently ride unresolved all the way to submission. Candidates if a second classifier is added: a small
1D-CNN or a flattened-window MLP, **not** Transformer (the point would be architecture diversity, not
capability). This is now recorded as a concrete, dated decision — not an open question sitting in the risk
table — in **R11's mitigation column** and as an explicit **DoD line item added to both M4 and M7** (§4),
so a future reader hits a checkpoint with a decision procedure attached, not a dangling "revisit later."

### 9.3 — 2026-07-08: M6 deliverable reviewed (`docs/references_verified.md`) — accepted with open items, not fully closed

The Literature Intelligence agent dispatched in §9.1 returned. I read the full deliverable myself — all
46 reference rows, the novelty ledger, and the venue table — rather than taking its own summary at face
value, per this lab's standing rule that no specialist output is accepted on first pass.

**Assessment of the work: High confidence it is methodologically sound.** It performed live retrieval
(WebFetch/WebSearch) against real, current sources rather than answering from parametric memory — the
right method here, since several cited works postdate any model's training cutoff — and it consistently
distinguished "confirmed fabricated," "unverifiable" (could not access, could not corroborate), and
"verified with corrections" rather than collapsing that distinction, which is exactly the discipline this
lab requires. The ref #4 investigation in particular went well beyond "the PDF 502'd": it checked the
hosting journal's own archive index (found the specific article ID absent while neighboring IDs from the
same issue are indexed) and the journal's own site listing, and ran an independent corroboration search on
the claimed numbers themselves — that is the standard of evidence I'd want from a PhD student's due
diligence, not a shrug. The two duplicate-reference findings (#1/#33, #17/#18) and the not-a-reference
finding (#24) confirm my own spot-check exactly. New findings beyond my spot-check that I did not catch
myself: ref #31's live arXiv page has been retitled between versions (TCS → "Adversarial Causal Tuning");
two aggregator "references" (#10, #22, Emergent Mind) are AI-synthesis pages, not primary sources; four
2025–2026 benchmarks adjacent to the claim (CausalDynamics, TimeGraph, XTSC-Bench, AMEE) are omitted from
the draft's own related work entirely.

**Novelty verdict accepted, with one specific technical gap I am flagging back rather than letting slide.**
The recommended narrowing — require the structural-CF ground truth to serve as an **evaluation oracle for
independently-produced, post-hoc explainer methods**, not merely a data-generation recipe — is the correct
move and matches exactly the direction this plan already anticipated (R6's original mitigation text, now
updated). *However*, the ledger's row on CAUKER / the Causal-DAG-Prior paper asserts they "jointly cover
(i)+(ii)+classification-labels" without confirming a detail that matters for how much of pillar (ii) they
actually take: **is either paper's causal DAG genuinely *time-lagged*** (edges spanning explicit lags τ,
in the same formal sense — $X_t^{(j)} := f_j(pa(X_{t-\tau}^{(i)}), U_t^{(j)})$ — that this project's own
TSCM formalism uses), **or is it a coarser, single per-instance generative DAG** that gets turned into a
time series by some other (possibly non-causal-in-time) mechanism? If the latter, this project's
"ground-truth *lagged*" claim is less threatened than the ledger currently implies, and the delta is
larger than stated. This is not something I will re-dispatch an agent for right now (it doesn't block
anything currently in flight and is correctly scoped by Lit's own recommendation to happen "before M7
manuscript freeze," which is far off) — logged as an explicit open item on P1 item 7 above, to be resolved
before the related-work section is drafted, not silently absorbed into "novelty verified."

**M6 status: substantially done, not fully closed.** P0 items 6 and 7 are marked done with the verdicts
above, but three concrete follow-ups remain open (the lagged-graph question just raised; the omitted
four benchmarks to add to related work; a full, non-abstract-only read of CausalProfiler) — M6's own DoD
("a novelty ledger... approved by PI") is not yet fully signed off; it is accepted-with-conditions. None
of these three block M2, M3, or any currently in-flight work; all three are M7-gating, not now-gating.

**Plan-document actions taken in this pass:** publication-target line (§2) rewritten with per-venue
confidence tags reflecting Lit's verified AAAI/ICLR/NeurIPS findings (AAAI 2027 dropped as unreachable —
deadline is ~2–3 weeks away — ICLR 2027 downgraded to an unverified stretch target, NeurIPS 2027 confirmed
as the sound primary choice modulo an unpublished 2027 CFP and a 2026 track rename); P0 items 6/7 marked
done with verdicts and open sub-items; R6 updated from a theoretical "Medium" risk to a realized one with
the actual surviving-delta wording as its mitigation. Nothing about this M6 review touches or is touched by
the classifier-scope correction in §9.2 — they are independent threads that happened to land the same day.

### 9.4 — 2026-07-08 (same day, third pass): two disputed references corrected by the author

The author supplied direct corrections to the two references flagged most seriously in §9.3's review,
same day. Not re-dispatched to Literature Intelligence — these are author-supplied facts (a working URL
the author personally confirmed; the identity of an intended-but-broken citation), not something requiring
further web research, and both are narrow enough to apply directly.

- **Reference #4** (the load-bearing MIMIC-IV/PhysioNet-Sepsis empirical claim, previously "Unverifiable —
  high suspicion" per Lit's 502s and failed archive-index/search corroboration): the author confirmed the
  article is live at the exact URL the draft already cites (`inass.org`, the journal's own domain).
  Upgraded to **Verified** in `docs/references_verified.md`. Lit's original finding stands as correctly
  documented process (the 502s were real at the time of checking) — the correction is new evidence, not a
  process failure.
- **Reference #24** (the confirmed-broken Hugging Face search-link, "Not a citable reference"): the author
  identified the intended paper as **CausalTime** (arXiv 2310.01753) — already independently found and
  analyzed in the Novelty Ledger's own row 1 before this correction arrived. `docs/BenchmarkingTSCFEs.md`'s
  reference #24 corrected in place; no new novelty analysis needed since Lit had already covered this paper.

**Effect on standing conclusions: none of substance.** The novelty verdict (§9.3, Deliverable 2) and venue
findings are unaffected — ref #24 turning out to be CausalTime *confirms* Lit's independent novelty
analysis of CausalTime rather than changing it, and ref #4's upgrade removes an integrity concern rather
than introducing a new empirical claim into scope. **Updated:** reference-verification summary counts
(Verified 28→29, Verified-with-corrections 14→15, Unverifiable 3→2, Not-a-citable-reference 1→0) in
`docs/references_verified.md`. P0 items 6/7 remain **done**; no change to their status. M6's three open
follow-ups from §9.3 (lagged-graph question, four omitted benchmarks, full CausalProfiler read) are
unaffected and still outstanding, still correctly scoped to before M7, not now.

### 9.5 — 2026-07-08 (fifth pass): independent M2 review; smoke-scale pipeline-clearance gate;
dispatch of pipeline-gap closure (M&C), method-zoo rename (M&C), M4 ablation-preset code (M&C),
and the three remaining M6 follow-ups (Lit)

**Binding context for this review:** the user issued a hard constraint (2026-07-08) — no
`full`/`full_nl`-scale run may be assigned, started, or recommended as imminent until the entire
experiment pipeline has been validated end-to-end **at smoke scale**, across every phase and every
CF method, on both linear and nonlinear configs. Task-list item 10 (full-scale run) stays **on
hold**; nothing in this review un-holds it or assigns work that depends on it.

**M2 core code — reviewed independently (read the actual diffs, not the design doc's summary of
itself). Assessed as sound. Confidence: High.**

- `causaltemp_xai/stats.py` (`git show 145c3f3`): read in full. `bootstrap_ci` (flat percentile
  bootstrap) and `hierarchical_bootstrap_ci` (two-level seed→instance cluster bootstrap) are both
  correctly implemented — NaN-dropping before resampling, degenerate point-CI for `n=1` (never a
  fabricated spread), `n_groups==1` correctly reduces to instance-only resampling (no phantom
  between-seed variability invented when there is only one seed), and the general two-level loop
  (resample seeds with replacement, then resample instances within each *resampled* seed with
  replacement, pool, recompute) is the textbook cluster bootstrap for exactly this two-level
  structure. The design rationale for hierarchical-over-flat (a flat pool understates uncertainty
  whenever seeds differ systematically) is scientifically correct and matches the smoke-scale
  demonstration in `docs/m2_multiseed_and_pearl_carla.md` §1.4 (CARLA/CftsCOMTE at validity=1.0
  every seed → degenerate CI; CftsWachter's one bad seed, 0.80 vs 1.00/1.00 → visibly wider CI).
  No objection.
- `PearlCARLARecourse` (`causaltemp_xai/methods/counterfactual/carla.py`): read in full alongside
  the untouched `CARLARecourse`. The noise-abduction (`eps[t] = x_orig[t] -
  mechanism.forward_numpy(window)`) and Pearl rollout are mechanically correct and match
  `CFfaith`'s own `pearl_delta` construction exactly, as claimed. Keeping it as a fully independent
  class (not a subclass/refactor of `CARLARecourse`) rather than risk `CARLARecourse`'s pinned
  behavior is the right risk trade-off given this codebase's own existing convention (the six
  `Cfts*CF` classes already duplicate `generate_batch` rather than share a base).
- **`lam_prox=0.1` documentation — adequate for a future reader. Confidence: High.** The class
  docstring alone (not just the design doc) carries the full causal chain: the homogeneous
  recursion the Pearl delta obeys, why it decays under the stability requirement, why
  `lam_prox=0.5` quadratically over-penalizes the larger delta a long horizon needs, and the
  concrete empirical numbers (0.07 vs 0.40 validity at the longest tested horizon) that motivated
  `0.1`. A reader with zero conversational context gets the same explanation I have. Combined with
  commit `145c3f3`'s message recording the PI accept-now/revisit-later decision, this item is
  closed — no documentation fix needed.
- **One real gap found by independent verification, not by trusting the design doc's own framing.**
  `docs/m2_multiseed_and_pearl_carla.md` §1.4 frames its smoke validation as "scope deliberately
  reduced from a full 6-method run" (3 of the pre-M2 6 methods) — true, but incomplete: I checked
  `experiments/03_run_cf_methods.py::build_methods()` (the actual registry every phase-03/04/07 run
  reads) directly, and `PearlCARLARecourse` — the M2 milestone's own second deliverable — is **not
  in it at all**. It is only ever instantiated directly inside `tests/test_methods.py::
  TestPearlCARLA`. This means the horizon-sweep numbers in the design doc's §2.3 (the `lam_prox`
  tuning finding) were produced by some standalone/ad hoc invocation outside the committed
  experiment scripts, and `PearlCARLARecourse` has **never once flowed through the actual
  reproducible pipeline** — not in the 3-method smoke validation, not anywhere. This is a materially
  bigger gap than "3 of 6 methods" — it is "the new method has zero pipeline integration."
- **Second, independently-found gap, not mentioned in the design doc at all:** I checked
  `results/smoke_nl/lstm/` directly — it contains only `train_report.json` (Phase 02's output).
  There is no `cf/` subdirectory, and `results/tables/table_axis_c_cf_faith.csv` has **zero**
  `smoke_nl,lstm,<method>` rows (only the two `smoke_nl,oracle,OracleCF-{Pearl,Rollout}` rows from
  Phase 05). Phases 03–04 of the *current* phased pipeline have never been executed against the
  nonlinear smoke config at all, with any method roster — even though Phase 03's own docstring
  explicitly documents that it supports nonlinear configs and expects to be run there
  ("exploratory, not a validated benchmark claim," its own words). Whether the stale claim in this
  document's own §3 ("Results produced: smoke + smoke_nl runs... 6 methods + 2 oracles") refers to
  the superseded legacy harness (`run_all.py`/`run_cfts.py`) rather than the current phased
  pipeline was not resolved — irrelevant either way, since it is the *current* pipeline's smoke_nl
  coverage that "entire pipeline cleared at smoke scale" requires, and that coverage is currently
  zero.

**Verdict: "entire experiment pipeline cleared at smoke scale" is currently false, on two
independently-verified counts, not one.** Both are closeable at smoke scale with no full-scale
dependency. Filed as new P0 item **0** (top of file 5's blocking list — see there for the finding
detail) and dispatched now (Methodology & Coding, task brief below) as the single highest-priority
assignment of this review — nothing else (M3, M4, a preliminary H1/H3/H4 read) should be treated as
more urgent than closing this, since it is what "the pipeline is clean" actually depends on.

**Task-list numbering — a discrepancy I could not resolve, flagged rather than silently papered
over.** The coordinating brief referred to "Task #8/#10/#12/#13/#16" using an informal numbering
that does not match this document's own §5 P0/P1/P2 item numbers (e.g. the M2 design doc's own
title cites "Task #8, #10" for multi-seed-CI + Pearl-CARLA, but §5's item 10 is actually "run
full/full_nl," item 11 is Pearl-CARLA). I searched for an actual task-tracking tool
(`TaskList`/`TaskGet`/`TaskUpdate`, as referenced in my own brief) via `ToolSearch` and found none
available in this session — only `TaskStop` (kill a running background task) exists. I have
therefore tracked everything through this document's own §5 TODO list and this log, as in every
prior review, and mapped the brief's "#13 full-scale run" → item 10, "#12 H1/H3/H4 reassessment" →
the not-yet-separately-itemized O2 reassessment work, and "#16 M6 follow-ups" → the three open
items from §9.3 (now also itemized as new item 30). **If a separate, authoritative task tracker
exists outside this document that a different session/tool context can reach, it should be
reconciled against these item numbers by whoever can access it — I could not verify one exists.**

**Task #13 (full-scale run): confirmed still ON HOLD, not touched.** No work assigned or
recommended in this review depends on it or brings it closer to "imminent."

**Task #12 (H1/H3/H4 full re-assessment): stays on hold, by my own judgment, with one narrow
exception I am taking on myself, not delegating.** O2's real, falsifiable threshold (≥6 methods ×
≥5 seeds × n_cf≥100 with CIs) cannot be met at smoke scale and I am not pretending otherwise. But
once the pipeline-gap-closure task below returns real data (all 7 methods, 3 seeds, n_cf=10, smoke
scale), I will personally do a preliminary, explicitly-labeled-as-underpowered H1/H3/H4 read as
part of my own review of that data — not a separate dispatched task, not a claim that O2 is
powered, just an honest "does the qualitative shape still hold with the fuller method roster"
sanity check, the same spirit as the existing smoke validation's own framing.

**Dispatched this review (all smoke-scale-only; see each agent's own task brief for full detail
and hard constraints — summarized here):**

1. **Methodology & Coding — pipeline-gap closure (P0 item 0, top priority).** Wire
   `PearlCARLARecourse` into `build_methods()`; run Phases 01→04 on `smoke_nl` for the first time
   through the current pipeline; re-run Phase 07 on `smoke` with the full 7-method roster
   (3 seeds, n_cf=10, matching the existing validation's scale); add a regression test so this
   specific method-missing-from-registry class of gap cannot silently recur; regenerate the 3
   figures if the added methods change them. Explicitly required to make an *documented*, not
   silent, decision about `n_steps` consistency (the existing `CARLA` entry in `build_methods()`
   overrides `n_steps=300`, but the validated `lam_prox=0.1` finding used the class default of 500
   — wiring `PearlCARLA` in at 300 would run it at a step count never validated for that
   hyperparameter choice, exactly the kind of silent parameter drift a reviewer would catch).
2. **Methodology & Coding — Dynamask/TimeSHAP rename (P0 item 5, half).** Rename the two proxy
   attribution classes to honest names (e.g. `FDSaliency`, `MCMaskSHAP`), seed `TimeSHAP`'s
   currently-unseeded RNG, add real unit tests (currently zero), add a small provenance note.
   CBM-T explicitly excluded from this task (lower severity, separately scoped later). File-disjoint
   from task 1 (`causaltemp_xai/methods/attribution/*` vs. `experiments/03_run_cf_methods.py`) —
   dispatched in parallel, not sequentially, deliberately.
3. **Methodology & Coding — M4 ablation-preset code + smoke-only validation (new item 29).**
   Gaussian-noise negative control (H5 — confirmed by reading `benchmarks/generator.py` directly
   that only `("laplace", "uniform")` are currently supported; Gaussian is genuinely new code, not
   a flag flip), non-monotonic mechanism variant (H6), minimal regime-switching variant (H7). Code
   + tests + a smoke-scale-only preliminary finding per ablation, explicitly forbidden from
   recommending or approaching a full-scale run. File-disjoint from tasks 1–2
   (`causaltemp_xai/config.py`, `benchmarks/generator.py`) — dispatched in parallel.
4. **Literature Intelligence — the three open M6 follow-ups (new item 30).** The lagged-graph
   question for CAUKER/Causal-DAG-Prior (highest priority of the three — it changes how much of
   this project's novelty claim survives), the four omitted related-work benchmarks
   (CausalDynamics, TimeGraph, XTSC-Bench, AMEE), and a full (not abstract-only) read of
   CausalProfiler. No pipeline dependency; compatible with the smoke-only constraint trivially.

**Not dispatched, held by explicit choice (not oversight):** TSEvo/Glacier official-code
integration (M3) — held pending a literature feasibility check on official-code availability that
was not part of this review's scope, to avoid a Methodology & Coding task starting on an
unconfirmed premise; CBM-T/iVAE wiring into an Axis-A phase (M3) — held one cycle because it
plausibly touches `experiments/03_run_cf_methods.py`/`_common.py`, the same files task 1 above is
actively modifying, and sequencing avoids a conflict; task #12's full re-assessment — held per
above, pending task 1's data, with only the narrow preliminary read reserved for the PI directly.

**Note from the coordinating session (relayed, not independently re-verified by me — out of this
review's scope and already reported as done):** reference-list corrections from
`docs/references_verified.md` (title/URL/metadata fixes for 15 "verified-with-corrections" entries,
inline flags on 2 unverifiable entries, duplicate-pair annotations for #1/#33 and #17/#18 without
renumbering, given ~90+ inline citation groups in the body text that a manual renumber would risk
corrupting) have been applied directly to `docs/BenchmarkingTSCFEs.md` by the coordinating session
at the user's direct request, 2026-07-08. This is a separate, already-closed thread from this
review's M6 dispatch (item 4 above) — the three *open* M6 follow-ups this review dispatches are
unaffected by and independent of that reference-list edit.
