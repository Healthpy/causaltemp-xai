---
date: 2026-08-26T16:20:00+02:00
researcher: Oleksii Furman
topic: "NeurIPS and TAE (Trust-AI-Eval) benchmark papers: high-level contributions, structure, and story patterns for a future NeurIPS benchmark paper"
tags: [neurips, datasets-and-benchmarks, evaluations-and-datasets, tai-eval, construct-validity, counterfactual-explanations, causal-audit, paper-structure]
sources: [web, peer-reviewed, workshop-cfp, neurips-official]
status: complete
last_updated: 2026-08-26
last_updated_note: "Guidelines are distilled from existing papers only; no manuscript draft; results to be finished at claimed scale."
---

# Research: NeurIPS / TAE Benchmark Papers — Structure and Story

**Date**: 2026-08-26T16:20:00+02:00
**Researcher**: Oleksii Furman

## Research Question

Research the works and benchmarks published on NIPS and on this workshop: prefered with high citations or relevance. https://tai-eval.github.io/ Research the high level main contribution and structure. Distill the good structure and story to guidelines md doc for future NIPS benchmark paper

## Summary

NeurIPS 2026 treats evaluation as a scientific object in two places at once: the archival **Evaluations & Datasets (E&D)** track (renamed from Datasets & Benchmarks) and the non-archival **TAE (Trust-AI-Eval): Can We Trust AI Evaluation?** workshop ([TAE home](https://tai-eval.github.io/); [E&D CFP](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets)). TAE has no accepted papers yet; the CFP deadline is 29 August 2026 AoE, papers are 8 pages, double-blind, and non-archival ([TAE CFP](https://tai-eval.github.io/cfp/)). E&D’s 2026 paper deadline has already passed (6 May 2026).

The papers that changed how the field evaluates do not mainly ship a larger dataset. They **name a construct**, **state the assumptions that connect a protocol to a claim**, **put ranking tables in service of a remaining gap or a validity failure**, and **bound what a score may be used to say** ([Raji et al., NeurIPS D&B 2021](https://arxiv.org/abs/2111.15366); [Liao et al., NeurIPS D&B 2021](https://openreview.net/forum?id=mPducS1MsEK); [WILDS, ICML 2021](https://proceedings.mlr.press/v139/koh21a.html); [HELM, TMLR 2023](https://arxiv.org/abs/2211.09110)). SuperGLUE-style single-number ranking is the machine later evaluation-theory papers indict.

No NeurIPS/ICML/ICLR paper jointly scores independently produced **temporal** counterfactual explanations against a **ground-truth graph + Pearl structural CF + classification label**. Closest fragments are tabular SCM-consistency checks ([Höllig et al., xAI 2023](https://doi.org/10.1007/978-3-031-44067-0_32)), SCM-as-generation-penalty ([Bahri et al., IEEE BigData 2025](https://doi.org/10.1109/BigData66926.2025.11402392)), and TSCM generators that stop at intervention (CausalTimePrior) or forecasting (DoFlow). That joint object is the actual gap a CausalTemp-XAI paper can claim, if it leads with the **audit instrument and the finding**, not with “we also released synthetic series.”

The distilled writing guide lives at [`2026-08-26-neurips-benchmark-paper-guidelines.md`](2026-08-26-neurips-benchmark-paper-guidelines.md). It is a structure-and-story spec from published papers and CFPs, **not a draft**. Venue for this cycle: TAE 2026 (8 pages, non-archival, 29 August 2026 AoE).

## Detailed Findings

### TAE 2026: workshop object, not a paper corpus

TAE is NeurIPS 2026 Sydney, 11 or 12 December 2026. Official names disagree: the CFP binding string is **TAE (Trust-AI-Eval): Can We Trust AI Evaluation?**; the NeurIPS workshop list says **Trustworthy AI Evaluation (TAI-Eval)** ([workshop announcement](https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/)). There is no program, no talk titles, and no accepted-paper list as of 26 August 2026.

The homepage problem statement treats evaluation itself as the object of study: an evaluation can be precise but measure the wrong construct, stable on a familiar benchmark but brittle on new data, or impressive on a leaderboard while poorly aligned with real-world decisions ([TAE home](https://tai-eval.github.io/)). The CFP’s core gap: “Many evaluation claims do not state what is measured, how it becomes a protocol, or what inference the protocol supports” ([TAE CFP](https://tai-eval.github.io/cfp/)).

Eight CFP clusters: uncertainty/robustness; benchmark/leaderboard auditing; black-box auditing; **measurement and causal validity**; stress tests/judge reliability; domain coverage; application-domain evaluation; deployment risk/governance.

Submission: 8 pages excl. refs/appendix, NeurIPS 2026 workshop template, double-blind, OpenReview, in-person posters, **non-archival**. Previously-published / dual-submission policy is unspecified (JUDGe 2026 forbids prior major-venue papers; TAE is silent).

Organizers mix Melbourne/Fudan adversarial robustness (Huang, Ma, Erfani; speaker Bailey) with Pavia statistical-risk/finance (Giudici, Tarantino) and NLP-evaluation pedigree (Hovy). Confirmed speakers include Bin Yu (PCS / veridical data science) and Soheil Feizi (robustness). Organizer-adjacent I-SAFE ([arXiv:2605.21731](https://arxiv.org/abs/2605.21731)) is a black-box structural audit: accuracy-matched models can fail a structural prior. Relevant, not yet a field standard.

**Fit for CausalTemp-XAI:** primary cluster is **measurement and causal validity**; secondary are black-box auditing and uncertainty/robustness. Do not stretch “underserved domains” or LLM-judge reliability. Do not sell a method bake-off. TAE is not a causal-inference workshop and not an XAI methods workshop.

**Venue tension:** NeurIPS 2026 E&D already claimed “evaluation becomes an object of scientific study in its own right” ([E&D CFP](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets)). TAE is the late, non-archival, in-person conversation after E&D reviewing. For this cycle, TAE is the remaining NeurIPS-adjacent venue (deadline 29 Aug 2026). An archival E&D paper is a 2027 target unless a later workshop-to-track path is used.

Do not conflate TAE with Atlanta’s sibling **JUDGe: Can We Trust the Judge?** ([judge2026.github.io](https://judge2026.github.io/)), which owns LLM-as-judge systems.

### NeurIPS Datasets & Benchmarks → Evaluations & Datasets 2026

NeurIPS launched D&B in 2021 because algorithm-centric reviewing undervalued datasets, audits, and evaluation methodology ([2021 announcement](https://blog.neurips.cc/2021/04/07/announcing-the-neurips-2021-datasets-and-benchmarks-track/)). For 2026 the track is renamed **Evaluations & Datasets**: datasets are not endpoints; beating a baseline is **not** required; review is as stringent as the main track ([2026 rename blog](https://blog.neurips.cc/2026/03/23/introducing-the-evaluations-datasets-track-at-neurips-2026/)).

**In-scope:** new datasets *if* they state what claims they support; benchmarks; audits; new protocols/metrics; negative results; stress tests. **Out-of-scope:** dataset dump without an evaluative claim; novel architecture whose primary claim is better performance (that is main track).

Practical 2026 bars: 9 content pages (+1 camera-ready); mandatory checklist; double-blind default; Croissant + Responsible AI fields if releasing data; code required for executable artifacts at submission; data accessible to reviewers without a PI request ([2026 CFP](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets); [reviewer guidelines](https://neurips.cc/Conferences/2026/EvaluationsDatasetsReviewerGuidelines); [paper checklist](https://neurips.cc/public/guides/PaperChecklist)).

**What reviewers are told to reward:** a clear evaluative claim; significance = “does it change how we measure progress?”; originality does not require a new method; negative results when deep and controlled; a protocol another team could re-run ([2026 reviewer guidelines](https://neurips.cc/Conferences/2026/EvaluationsDatasetsReviewerGuidelines)).

**What they reject:** dataset-as-endpoint; inaccessible code/data; irreproducible protocol; thin “we tried and it failed”; method paper framed as SOTA in the eval track.

**Disagreement:** chairs admit they have not described “impact and scientific relevance” well enough to reviewers; some reviewers still grade D&B like a methods paper; 11% of 2025 D&B reviewers never inspected the data ([Art to Science](https://blog.neurips.cc/2025/12/05/neurips-datasets-benchmarks-track-from-art-to-science-in-ai-evaluations/)). Frame the paper so it still works after deleting any new architecture.

Track choice rule: if the paper is still true after deleting the new model, it is E&D. If you are arguing *that* evaluation is broken without a runnable protocol, that is a **Position** paper ([Position CFP](https://neurips.cc/Conferences/2026/CallForPositionPapers)). TAE is for an 8-page non-archival slice, a rejected-E&D recycle, or an incomplete validity argument that is not yet Croissant-ready.

### High-impact evaluation papers: contribution and narrative

Venue honesty: many of the templates the NeurIPS community cites were **not** NeurIPS D&B papers (WILDS is ICML 2021; HELM and BIG-bench are TMLR 2023; SWE-bench is ICLR 2024; CausalTime is ICLR 2024; CLIP is ICML 2021; ImageNet is CVPR 2009). Citation counts disagree across indexes; treat them as order-of-magnitude only.

| Paper | Venue | Scientific object (not the dataset) | Story beat to steal | Story beat to refuse |
|---|---|---|---|---|
| [SuperGLUE](https://proceedings.neurips.cc/paper/2019/file/4496bf24afe7fab6f046bf4923da8de6-Paper.pdf) | NeurIPS 2019 | Harder GLUE after saturation | Name the predecessor’s failure mode (headroom); diagnostics beside the score | Single-number “general language understanding” |
| [WILDS](https://proceedings.mlr.press/v139/koh21a.html) | ICML 2021 | ID–OOD **gap remaining after** robust-training methods | Remaining gap is the result; standardized package | Ranking a winner |
| [HELM](https://arxiv.org/abs/2211.09110) | TMLR 2023 | Scenarios × metrics matrix | Taxonomy first; 25 findings; raw artifacts; state missing cells | Later productized one-number leaderboard |
| [Raji et al.](https://arxiv.org/abs/2111.15366) | NeurIPS D&B 2021 | Construct validity of “general” benches | Finite data ≠ general ability; limitations as claim law | Advertising the paper as a general XAI bench |
| [Liao et al.](https://openreview.net/forum?id=mPducS1MsEK) | NeurIPS D&B 2021 | Internal vs external validity | Claim-chain figure; taxonomy of repeated failures | Printing dataset diagnostics as method scores |
| [Bowman & Dahl](https://aclanthology.org/2021.naacl-main.385/) | NAACL 2021 | Four criteria for a healthy NLU bench | High score that does not imply the ability is a failed gate | Adversarial OOD as a fix for a broken IID construct |
| [DecodingTrust](https://arxiv.org/abs/2306.11698) | NeurIPS D&B 2023 outstanding | Trustworthiness ≠ capability | Ranking **inversion** (GPT-4 worse under jailbreak) | Averaging axes into one trust score |
| [OpenXAI](https://arxiv.org/abs/2206.11104) | NeurIPS D&B 2022 | Tabular attribution bake-off + synthetic GT | Synthetic known-explanation controls; metrics allowed to conflict | CFfaith as a leaderboard axis |
| [CARLA](https://arxiv.org/abs/2108.00783) | NeurIPS D&B 2021 | Tabular CF/recourse comparability | Library + methods + metrics + seeds | Validity/proximity/sparsity as the scientific object |
| [CauseMe / C4C](https://proceedings.mlr.press/v123/runge20a.html) | NeurIPS 2019 competition | Discovery vs known (or high-confidence) graphs | Challenge types as factors; known mechanism | F1-of-edges as a CF-explanation headline |
| [SWE-bench](https://arxiv.org/abs/2310.06770) | ICLR 2024 | End-to-end issue resolution | Executable world; SOTA barely works; later Verified revises the construct | Treating classifier flip as pass@k |
| [Adebayo et al.](https://arxiv.org/abs/1810.03292) | NeurIPS 2018 | Saliency can be independent of model and DGP | Negative controls in the main figure | Visual/plausible explanations as validity |

**Three competing stories of why eval papers succeed** (do not pick a silent winner):

1. **Infrastructure / Common Task Framework** — shared test + single number + leaderboard (ImageNet, GLUE/SuperGLUE). Raji et al. call the over-extension of this into “general ability” a construct-validity failure.
2. **Hardness / remaining headroom** — SuperGLUE, SWE-bench, LiveCodeBench. Bowman & Dahl argue making the test adversarially harder does **not** restore construct validity.
3. **Scientific object, not score** — WILDS, HELM, DecodingTrust, Liao, Raji: change *what a table is allowed to mean*.

A quieter mechanism: someone else’s **method** paper made the benchmark famous (ALE via DQN; Gym via PPO). Machado et al. later argued ALE’s *evaluation protocol* was broken while the platform was canonical ([JAIR 2018](https://www.jair.org/index.php/jair/article/view/11122)).

### Causal / XAI neighborhood a CausalTemp paper must engage

**Evaluation culture that became the problem.** Wachter et al. ([2018](https://arxiv.org/abs/1711.00399)) canonized closest-world optimization against the **model**. DiCE ([FAT* 2020](https://arxiv.org/abs/1905.07697)) added diversity/feasibility. Guidotti ([DMKD 2022](https://doi.org/10.1007/s10618-022-00831-6)) and Verma et al. ([2020](https://arxiv.org/abs/2010.10596)) institutionalized a dashboard (validity, proximity, sparsity, plausibility) in which **causality is a checkbox**, not an oracle. Temporal CF methods (CoMTE, TSEvo, CELS, CONFETTI, CFTS) inherit that dashboard. None reports Pearl-trajectory disagreement, do-complexity, or intervention-to-label-site distance.

**Internal dissent on the estimand.** Karimi et al. ([FAccT 2021](https://arxiv.org/abs/2002.06278)) : counterfactual explanations say **where to get**, not **how to intervene**. Recourse should be minimal `do()` in an SCM. CausalTemp sits on Karimi’s side of the estimand but **audits other people’s CFs** rather than proposing a new generator.

**Adjacent artifacts, none with joint G+CF+Y** (ground-truth graph, ground-truth structural CF, classification label as explanation target):

- CARLA, OpenXAI: tabular, model-relative (OpenXAI has synthetic *attribution* GT).
- CFTS / Li et al. TSC-CF / XTSC-Bench / AMEE: temporal, model metrics or saliency GT, no Pearl CF of a written SCM.
- CausalTime (ICLR 2024), CausalDynamics (NeurIPS 2025 D&B), CausalRivers (ICLR 2025), CauseMe: graph recovery; CausalTime’s graph is **derived from a fitted NN** and is not claimed to be the true DGP.
- LEWIS (SIGMOD 2021), Watson et al. (UAI 2021): necessity/sufficiency of **features of f**, not of independently produced CF trajectories.

**Closest work, ranked:**

1. Höllig et al., xAI 2023 — nine tabular CF methods vs known SCMs; non-causal methods fail semantic consistency. Tabular, relation-satisfaction not Pearl-trajectory, no time, no *D*.
2. Beckers, arXiv:2301.02499 — DiCE/Wachter vs Pearl CFs on toy graphs; ~30% conflict; not a top venue.
3. Melistas et al., NeurIPS 2024 D&B — SCM **image generators**, not independently produced explanations of a classifier.
4. Bahri et al., IEEE BigData 2025 — SCM in the **generator’s loss** (“first effort to leverage SCMs to generate feasible CFs for time series”). Fair foil: penalise vs derive.

**Contested neighbors (cite and distinguish, do not dismiss):**

- **CauKer**, ICLR 2026 oral — SCM-coherent series for TSFM pretraining; CausalTimePrior reports it lacks temporal lags and intervention support. No explainer audit.
- **DoFlow**, ICLR 2026 — abduction–action–prediction **forecasting** on a known DAG. Competitor *mechanism*, not competing *benchmark*.
- **CausalTimePrior**, arXiv:2603.11090, ICLR 2026 workshop — paired observational/interventional TSCMs. Closest generator. Still L2 (`do`), not L3 instance-level abduction CFs of proposed explanations; no classifier; no Axis C. The authors themselves call generating counterfactual pairs a “natural extension” and “a strictly harder task than interventional prediction.”

**Safe novelty sentence:** SCMs have been used to generate CFs (Karimi, Mahajan, Bahri) and, in tabular xAI 2023, to check causal consistency of third-party CFs (Höllig). They have not been used to derive a **temporal oracle CF** against which independently produced TSC explanations are scored, with do-complexity reported beside trajectory disagreement.

Intervention-to-label-site distance as a **reporting condition** appears to be CausalTemp-specific; frame it as such, not as filling a named literature hole.

### Trust-in-evaluation literature TAE is built on

A TAE-style paper is trustworthy when it is a **validity argument**: construct → operationalization → assumption chain → uncertainty → controls → claim-scope. Documentation checklists (datasheets, model cards, NeurIPS checklist) are necessary and not sufficient ([Jacobs & Wallach, FAccT 2021](https://arxiv.org/abs/1912.05511); [Bean et al., NeurIPS 2025 D&B](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1967e0fc3aa6cbbace562f5cb8e3954e-Abstract-Datasets_and_Benchmarks_track.html) on 445 LLM benchmarks: almost all are weak on defining the phenomenon and justifying construct validity).

Bean et al.’s skeleton is the reusable diagram: **phenomenon → task → metric → claim**.

**Proxy ≠ construct** (examples TAE reviewers will know): BLEU ≠ meaning ([Callison-Burch et al., EACL 2006](https://aclanthology.org/E06-1032/)); plausibility ≠ faithfulness ([Jacovi & Goldberg, ACL 2020](https://aclanthology.org/2020.acl-main.386/)); association ≠ intervention (Pearl); safety-bench score can correlate with capability (“safetywashing”). Mutually exclusive CF-faith semantics must not be averaged.

**Controls that became standard:** Adebayo randomization (negative control); CheckList MFT/INV/DIR ([Ribeiro et al., ACL 2020](https://aclanthology.org/2020.acl-main.442/)); hypothesis-only baselines ([Gururangan et al., NAACL 2018](https://aclanthology.org/N18-2017/)); F-Fidelity (ICLR 2025) uses **degraded explainers with known ranking** as a construct-validity test of the *metric*. Positive control analog: oracle / known-good must pass; vacuous rewrite must fail.

**Uncertainty:** name the unit of variation (seed, split, prompt, judge, hyperparameter search). They are not interchangeable ([Reimers & Gurevych, EMNLP 2017](https://arxiv.org/abs/1707.09861); [Bouthillier et al., MLSys 2021](https://arxiv.org/abs/2103.03098); NeurIPS checklist Q7). Yu’s PCS treats stability under human judgment calls as a scientific requirement ([PNAS 2020](https://www.pnas.org/doi/10.1073/pnas.1901326117)) — complementary to, not a substitute for, Pearl identification.

**Disagreements to keep visible in the design:**

- More tasks vs deeper validity (HELM/BIG-bench vs Raji/Bean/Jacobs).
- Static IID vs adversarial/dynamic tests (Bowman & Dahl vs Dynabench/AFLite).
- Leaderboards as coordination vs contamination (GLUE/HELM vs LiveCodeBench/Sainz).
- Recht et al. (ImageNet drop ≈ distribution gap, rankings preserved) vs LLM contamination (training-set inclusion). Do not cite Recht to dismiss leakage.
- HELM argued against one number, then shipped aggregate leaderboards.
- Documentation vs claims: perfect datasheet for a benchmark that still does not measure “reasoning.”
- Jacovi graded faithfulness vs a hard CF-faith admission gate.
- PCS stability vs Pearl identification (TAE speakers and CFP bullets can be read as either).

### Recommended narrative spine for this project

Hybrid, not a clone of any one paper:

1. **Raji/Liao framing** — what the Wachter/DiCE score is *not* (model-relative dashboard; finite benches ≠ general CF quality).
2. **Karimi gap sentence** — temporal CF methods say whether *f* flipped, not whether the world would have. Sharpen to CausalTemp’s foil: the field uses SCMs to **penalise** CFs; nobody uses them to **derive** the oracle a proposal is scored against.
3. **WILDS remaining-gap result** — existing methods still disagree with the world, even when admitted by CF-faith; do-complexity shows they buy validity by intervening almost everywhere.
4. **CauseMe generation** — known additive-noise TSCM, challenge factors (linear/nonlinear/spring; label site; horizon), exact abduction.
5. **HELM tables** — Axes A/B/C as different scientific units; never average CF-faith into a super-score; report `(D, Δ)` as a pair.
6. **DecodingTrust inversion** — low trajectory disagreement at high *D* is not a win.
7. **Bowman limitations** — power, reading reliability (single-`do` vs schedule), real-data transfer is not GT; synthetic-only construct validation is a feature for TAE, not a hedge to bury.
8. **Adebayo/F-Fidelity controls** — positive control (PearlSCMRecourse, *D* = 1), negative/vacuous (full-trajectory rewrite), adversarial metric tests in the main figure.

Do **not** write SuperGLUE with extra causal keywords.

## Sources Consulted

- [TAE homepage](https://tai-eval.github.io/) — workshop object, organizers, speakers
- [TAE CFP](https://tai-eval.github.io/cfp/) — 8-page non-archival rules, topic clusters, dates
- [NeurIPS 2026 workshops](https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/) — official TAI-Eval listing
- [NeurIPS 2026 E&D CFP](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets) — archival eval-track contract
- [2026 E&D reviewer guidelines](https://neurips.cc/Conferences/2026/EvaluationsDatasetsReviewerGuidelines) — reward/reject criteria
- [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist) — claims vs scope, error bars, limitations
- [Raji et al. 2021](https://arxiv.org/abs/2111.15366) — construct validity of “general” benchmarks
- [Liao et al. 2021](https://openreview.net/forum?id=mPducS1MsEK) — internal vs external validity
- [Bowman & Dahl 2021](https://aclanthology.org/2021.naacl-main.385/) — four criteria; adversarial OOD is not a fix
- [Jacobs & Wallach 2021](https://arxiv.org/abs/1912.05511) — construct vs operationalization
- [WILDS](https://proceedings.mlr.press/v139/koh21a.html) — remaining gap after SOTA
- [HELM](https://arxiv.org/abs/2211.09110) — taxonomy × multi-metric
- [DecodingTrust](https://arxiv.org/abs/2306.11698) — ranking inversion
- [OpenXAI](https://arxiv.org/abs/2206.11104) / [CARLA](https://arxiv.org/abs/2108.00783) — XAI D&B templates
- [Wachter et al.](https://arxiv.org/abs/1711.00399) / [DiCE](https://arxiv.org/abs/1905.07697) / [Karimi et al.](https://arxiv.org/abs/2002.06278) — CF estimand split
- [CausalTime](https://arxiv.org/abs/2310.01753) / [CausalTimePrior](https://arxiv.org/abs/2603.11090) / [CauKer](https://arxiv.org/abs/2508.02879) / [DoFlow](https://arxiv.org/abs/2511.02137) / [Bahri et al. 2025](https://doi.org/10.1109/BigData66926.2025.11402392) — contested neighbors
- [Adebayo et al. 2018](https://arxiv.org/abs/1810.03292) / [Jacovi & Goldberg 2020](https://aclanthology.org/2020.acl-main.386/) — faithfulness controls
- [Yu & Kumbier, PCS](https://www.pnas.org/doi/10.1073/pnas.1901326117) — stability under judgment calls
- Project scientific contract: [`docs/general_plan.md`](../../general_plan.md)

## Key Insights

1. **The scientific object is the protocol, not the leaderboard.** TAE and 2026 E&D use almost the same sentence. A method-ranking table is evidence for a validity claim, not the contribution.
2. **Write the phenomenon → task → metric → claim chain in the main text.** Silent hops (validity → causal efficacy; low Δ_trajectory → the world agrees) are exactly the failure TAE named.
3. **Admission gates and ranking axes must not share a column.** For the TAE manuscript (clarified 2026-08-26), **graded** mechanism-faithfulness is the described quantity; the hard 0/1 check is a **reporting companion** (fraction passing), not the scientific object. `(D, Δ_total)` remains the ranking pair. Axis A remains a dataset diagnostic. OpenXAI/SuperGLUE are still the anti-templates: do not average graded CF-faith into a super-score.
4. **Positive, negative, and oracle controls belong in Figure 1.** Adebayo and F-Fidelity made “the metric must fail on a vacuous input” a standard. PearlSCMRecourse at *D* = 1 and full-rewrite at *D* = *T* are that figure for this project.
5. **Synthetic GT validates the metric; real data tests transfer.** Raji’s museum, WILDS’s ID vs OOD, and CausalTemp’s tier contract are the same rule. Presenting tiers as a realism ladder that strengthens construct validity is the error reviewers who know CausalTime will also catch (“GT graph” is not one thing).
6. **Fight Bahri/CauKer/DoFlow/CausalTimePrior on estimand, not on “we also use SCMs.”** Oracle vs penalty; independent methods vs generator; classification label vs forecasting/pretraining; L3 abduction vs L2 `do`.
7. **Do not make retractions the story, and do not ship a WIP snapshot.** Report finished evidence at the scale you claim (WILDS/DecodingTrust: the finding is complete enough to change the question). Dead numbers stay out of tables (Raji claim law; SWE-bench Verified is a construct revision, not an excuse to reprint bad scores).
8. **This cycle’s venue is TAE (8 pages, 29 Aug 2026 AoE, non-archival).** Cut to the workshop contract. These notes are not a manuscript.

## Conflicting Information

- **TAE vs TAI-Eval naming.** CFP `\workshoptitle` vs NeurIPS workshop list. Use the CFP string in the PDF; TAI-Eval for logistics.
- **TAE vs E&D uniqueness.** Both claim evaluation-as-science. Difference is venue contract (8-page non-archival late deadline vs 9-page archival hosting/Croissant), not conceptual novelty.
- **One number.** HELM argued against it, then shipped aggregates. SuperGLUE’s success *is* the ranking machine Raji indicted.
- **Faithfulness.** Jacovi: graded, not binary. The TAE manuscript (clarified 2026-08-26) **reports graded faithfulness** and keeps the hard 0/1 check only as a reporting rule. That is a **framing** choice against [`docs/general_plan.md`](../../general_plan.md) §7, which still treats hard CF-faith as the admission gate because it has no dynamic range across real methods. The paper must name the graded quantity (mechanism residual / soft score) and keep adversarial tests so a vacuous CF cannot buy the grade. PCS stability vs Pearl identification is a separate disagreement (TAE speakers vs CFP bullets).
- **CausalTime “ground truth.”** Derived from a fitted NN; authors deny it is a discovery of the underlying relationship. Additive-noise TSCMs with known *F* are a stronger object; reviewers who know CausalTime will attack “GT graph” unless this is explicit.
- **Causality in the loss vs in the score.** Bahri/Mahajan vs Höllig/Beckers vs Karimi (generate) vs CausalTemp (audit).
- **PCS vs Pearl.** Yu (TAE speaker) = stability of conclusions under judgment calls. TAE’s causal-validity bullet = identification. A vacuous high-*D* CF can be stable and still fail the construct.
- **ImageNet drops (Recht) vs LLM contamination.** Different mechanisms; do not cite one to settle the other.
- **Static shared tasks vs living benches.** Coordination vs contamination. A frozen seeded SCM generator is usually the right anti-memorization story here; living streams fight a different threat (GitHub scrape).

## Confidence Notes

- Citation counts were not retrieved from a single consistent index; OpenAlex often undercounts Google Scholar by an order of magnitude. Do not copy numeric ranks into a paper.
- TAE has no accepted papers; speaker list is single-source (workshop site); OpenReview group URL was not confirmed.
- TAE dual-submission / previously-published policy is unspecified.
- RewardBench / LiveCodeBench exact proceedings lines should be re-checked on OpenReview before camera-ready citation.
- I-SAFE and SAFE-AI are organizer-adjacent 2023–2026 work, not field-standard citations.
- “No paper reports intervention-to-label-site distance as a CF reporting condition” is a negative search result, not a proof of uniqueness.
- Höllig et al. 2023 and Beckers 2023 are the closest model-vs-world audits found; a missed NeurIPS 2025/2026 paper is possible.

## Open Questions

None remaining from the initial research pass.

## Clarifications Log

- **2026-08-26 16:45 +02:00 — Venue for this cycle.** Target is an 8-page TAE 2026 workshop paper (non-archival, deadline 29 August 2026 AoE).
- **2026-08-26 16:55 +02:00 — CF-faith framing.** Report **graded** faithfulness; hard 0/1 check is a reporting companion. Manuscript framing vs `docs/general_plan.md` §7; Jacovi & Goldberg (ACL 2020).
- **2026-08-26 16:44 +02:00 — Results and drafting.** The paper and results will be **finished** at the claimed scale. Do **not** draft the paper in these docs; distill guidelines from existing papers only. Do not frame the submission as a WIP snapshot.
