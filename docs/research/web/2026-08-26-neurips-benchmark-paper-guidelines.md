# Guidelines: NeurIPS / TAE Benchmark Paper — Structure and Story

**What this is:** structure and story distilled from published evaluation papers, NeurIPS Datasets & Benchmarks / Evaluations & Datasets CFPs, and the TAE 2026 workshop CFP. Not a manuscript, not a results plan, not a second general plan.
**What this is not:** a draft of the CausalTemp-XAI paper. Do not paste §9-style prose into a submission.
**Companion research (citations, disagreements):** [`2026-08-26-neurips-tae-benchmark-papers.md`](2026-08-26-neurips-tae-benchmark-papers.md)
**Date:** 2026-08-26

Venue for *this* cycle (clarified): **TAE 2026**, 8 pages, non-archival, deadline 29 August 2026 AoE. The patterns below still come from archival D&B/E&D and adjacent eval papers; cut to 8 pages, do not write a 9-page E&D paper in workshop clothing.

---

## 1. What winning eval papers actually contribute

Papers that changed how the field evaluates **name a scientific object** (a remaining gap, a validity failure, a trade-off, a leakage mechanism). Dataset-dump papers collapse that object into a leaderboard score.

| Paper | Venue | Object (not the files) | Story beat to copy | Story beat to refuse |
|---|---|---|---|---|
| [Raji et al.](https://arxiv.org/abs/2111.15366) | NeurIPS D&B 2021 | Construct validity of “general” benches | Finite data ≠ general ability; limitations as **claim law** | Advertising a new bench as general XAI / general CF quality |
| [Liao et al.](https://openreview.net/forum?id=mPducS1MsEK) | NeurIPS D&B 2021 | Internal vs external validity | Claim-chain figure; same failures recur | Printing dataset diagnostics as method scores |
| [Bowman & Dahl](https://aclanthology.org/2021.naacl-main.385/) | NAACL 2021 | Four criteria for a healthy bench | A high score that does not imply the ability is a failed gate | Adversarial OOD as a fix for a broken construct |
| [WILDS](https://proceedings.mlr.press/v139/koh21a.html) | ICML 2021 | ID–OOD **gap remaining after** existing methods | The remaining gap *is* the result | Ranking a winner |
| [HELM](https://arxiv.org/abs/2211.09110) | TMLR 2023 | Scenarios × metrics matrix | Taxonomy first; numbered findings; state missing cells | Later one-number leaderboard |
| [DecodingTrust](https://arxiv.org/abs/2306.11698) | NeurIPS D&B 2023 | Trustworthiness ≠ capability | Ranking **inversion** | Averaging axes into one score |
| [SuperGLUE](https://arxiv.org/abs/1905.00537) | NeurIPS 2019 | Harder GLUE after saturation | Name the predecessor’s failure mode; diagnostics **beside** the score | Single-number “general understanding” |
| [OpenXAI](https://arxiv.org/abs/2206.11104) | NeurIPS D&B 2022 | Attribution bake-off + synthetic GT | Synthetic known-explanation controls; metrics allowed to conflict | Faithfulness leaderboard as the product |
| [CARLA](https://arxiv.org/abs/2108.00783) | NeurIPS D&B 2021 | Tabular CF comparability | Library + methods + metrics + seeds | Validity/proximity/sparsity as the scientific object |
| [CausalTime](https://arxiv.org/abs/2310.01753) | ICLR 2024 | Discovery data with a (derived) graph | 3-column comparison of **prior designs** | Calling a fitted-NN graph the written DGP |
| [CauseMe / C4C](https://proceedings.mlr.press/v123/runge20a.html) | NeurIPS 2019 competition | Discovery vs known / high-confidence graphs | Challenge *types* as factors | F1-of-edges as an explanation headline |
| [Adebayo et al.](https://arxiv.org/abs/1810.03292) | NeurIPS 2018 | Saliency can ignore model and DGP | Negative controls in the **main** figure | Visual/plausible output as validity |
| [Jacovi & Goldberg](https://aclanthology.org/2020.acl-main.386/) | ACL 2020 | Faithfulness ≠ plausibility | Graded faithfulness; do not use human “looks good” as faithfulness | Binary faithfulness as the ranking axis |
| [Karimi et al.](https://arxiv.org/abs/2002.06278) | FAccT 2021 | Recourse = `do()`, not nearest point | One quote-sized gap sentence | Adding “causality” as a checkbox on Wachter metrics |
| [Höllig et al.](https://doi.org/10.1007/978-3-031-44067-0_32) | xAI 2023 | Tabular CFs vs known SCMs | Cite and outgrow: third-party CFs scored against a world | Treating relation-satisfaction as Pearl-trajectory `Δ` |
| [Bean et al.](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1967e0fc3aa6cbbace562f5cb8e3954e-Abstract-Datasets_and_Benchmarks_track.html) | NeurIPS 2025 D&B | 445 LLM benches | Diagram: phenomenon → task → metric → claim | Silent hops from score to slogan |

**Three competing success stories** (do not pick a silent winner): (1) Common Task Framework + one number (ImageNet, GLUE); (2) hardness / leftover headroom (SuperGLUE, SWE-bench); (3) scientific object, not score (WILDS, HELM, Raji, Liao). Evaluation-theory papers exist to attack (1) and (2).

**2026 E&D chairs** ([blog](https://blog.neurips.cc/2026/03/23/introducing-the-evaluations-datasets-track-at-neurips-2026/)): significance = does it **change how we measure progress**? Beating a baseline is **not** required. Originality does **not** require a new method. If the paper is still true after deleting a new architecture, it is an eval paper, not a methods paper.

**TAE CFP** ([cfp](https://tai-eval.github.io/cfp/)): “Many evaluation claims do not state what is measured, how it becomes a protocol, or what inference the protocol supports.” Primary cluster for a model-vs-world audit: **measurement and causal validity**.

---

## 2. Venue contract (from the CFPs, not from taste)

| | **E&D** (ex D&B) | **TAE 2026** | **Main track** |
|---|---|---|---|
| Object | Evaluation *is* the claim | Same object, workshop-scale | New method; eval is evidence |
| Pages | 9 + 1 camera-ready | **8** excl. refs/appendix | 9 + 1 |
| Archival | Yes | **Non-archival** | Yes |
| Beat baselines? | No | Optional | Usually yes |
| Artifact bar | Croissant+RAI if new data; code at submission | Reviewer-dependent | Encouraged |
| This cycle | Deadline passed (6 May 2026) | **29 Aug 2026 AoE** | Passed |

TAE PDF string: `\workshoptitle{TAE (Trust-AI-Eval): Can We Trust AI Evaluation?}`. NeurIPS list name: TAI-Eval. Not Atlanta **JUDGe** (LLM-as-judge). Dual-submission / later archival reuse: TAE is silent; confirm with `aiteval2026@gmail.com`.

---

## 3. Narrative spine (stolen pieces, not a clone)

Copy **beats**, not plots.

1. **Use-case constraint that makes the protocol inevitable** — Wachter led with “don’t open the box.” Invert when the claim *requires* opening the DGP.
2. **One gap sentence at estimand level** — Karimi: CEs say where to get, not how to intervene. DiCE named two properties. Guidotti dumped ten metrics and concluded nobody wins — that is the intro to avoid.
3. **Instrument, then finding that changes the question** — WILDS: methods don’t close the gap. DecodingTrust: capability ranking inverts under a different construct. SuperGLUE: ranking *is* the object — do not copy that.
4. **Taxonomy before the leaderboard** — HELM scenarios × metrics; Liao internal vs external; DecodingTrust axes; CauseMe challenge types.
5. **Controls in the main paper** — Adebayo randomization; F-Fidelity degraded explainers; OpenXAI synthetic GT; ALE published baseline agents.
6. **Limitations as claim law** — Raji: what the score may **not** be used to say. SWE-bench Verified: revise the construct in public if it was noisy; do not quietly reprint dead numbers.

---

## 4. Page skeleton distilled from D&B winners

Typical accepted D&B/E&D (9 pages). TAE: same spine, **8 pages**, appendix unread.

1. **Abstract** — problem, instrument, **finding**, non-claim. Not a leaderboard teaser. (DecodingTrust, WILDS)
2. **Introduction** — evaluation failure; why existing benches cannot answer it; artifact; 3–5 numbered claims; explicit non-claims. (Raji + WILDS)
3. **Related work as evaluation designs** — not a method zoo. CausalTime’s highest-leverage device is a **comparison table of prior benches**. Karimi vs Wachter is an estimand split, not “they also use causality.”
4. **Evaluative claims / protocol** — Bean et al.: phenomenon → task → metric → claim. Liao: the arrow from score to real-world progress, with named breaks. Assumptions in the **main text** (NeurIPS checklist Q1–Q2; TAE’s “what inference the protocol supports”).
5. **Data or generator as challenge factors** — CauseMe types, not a file dump. WILDS: when a shift “counts.”
6. **Baselines, oracles, controls** — missing them looks like a leaderboard dump (2026 E&D reviewers). Beating them is optional.
7. **Results, then analysis** — tables that change the question. Numbered findings (HELM: 25; prefer fewer). Uncertainty with a **named unit** of variation (Bouthillier; Reimers & Gurevych; checklist Q7).
8. **Limitations** — own section. Honesty is not punished ([checklist](https://neurips.cc/public/guides/PaperChecklist)).
9. **Availability** — protocol another team could re-run. E&D: Croissant. TAE: anonymous code URL is enough if the protocol is in the paper.

Reviewers are not required to read the appendix. If a sentence is needed for the validity argument or the headline finding, it is not an appendix sentence.

---

## 5. Writing rules (each from a source)

**Title the construct, then the dataset.** Jacobs & Wallach; Raji; Bean et al. “We measure *X* via protocol *P*.” Not “we propose the X benchmark.”

**Draw phenomenon → task → metric → claim.** Bean et al. 2025; *Measurement to Meaning* ([arXiv:2505.10573](https://arxiv.org/abs/2505.10573)). Refuse silent hops (BLEU ↛ meaning, Callison-Burch et al. EACL 2006; validity ↛ causal efficacy).

**Write the assumption chain.** Liao Figure 1; Yu PCS ([PNAS 2020](https://www.pnas.org/doi/10.1073/pnas.1901326117)) for stability under judgment calls — complementary to Pearl identification, not a substitute.

**Separate internal from external validity.** Liao et al. Synthetic known-mechanism tests can validate a metric; transfer tests cannot. Raji: the museum cannot contain the world. WILDS: ID vs OOD is the split, not a realism ladder that strengthens construct validity.

**Name the sources of variation.** Seeds, splits, prompts, judges, hyperparameter search are not interchangeable (Bouthillier et al. MLSys 2021; NeurIPS checklist Q7). Kocmi et al.: the decision is pairwise, not a correlation with humans.

**Do not let a single scalar be the object.** HELM’s matrix; Dehghani et al. *Benchmark Lottery*. If you rank, declare aggregation as a value choice.

**Keep the proxy from becoming the construct.** Jacovi & Goldberg: graded faithfulness, not human plausibility. Do not average mutually exclusive semantics. OpenXAI/SuperGLUE are the anti-templates for turning a gate or a proxy into a league table.

**Causal language only where `do()` is licensed.** Pearl: association ≠ intervention. Bahri-style residual penalty is a local check in a generator’s loss, not an oracle for independent proposals (Höllig / Beckers are the closer audit ancestors).

**Controls in the main results.** Oracle / known-good must pass; vacuous / randomized / null-submission must fail (Adebayo; CheckList; BenchJack). F-Fidelity: test the *metric* with degraded explainers of known rank.

**Negative results as findings when controlled.** WILDS, DecodingTrust. Superficial “we tried and it failed” is rejectable (2026 E&D reviewer guidelines).

**Document assets; bind them to claims.** Datasheets / model cards / NeurIPS checklist do not create construct validity (Jacobs & Wallach; Bean et al.). Add what the documentation **does not** support.

**Label oracles, methods, foils, and dataset diagnostics as different units.** HELM core vs targeted; Liao learning-problem vs task.

**Scope every headline number.** Population, protocol version, unit of resampling. Living benches name a frozen v1 (HELM, LiveCodeBench).

**Release the raw protocol, not only the mean.** HELM released prompts/completions; without instance-level outputs the paper is not evaluation research.

**Venue-honest related work.** WILDS is ICML; HELM is TMLR; CausalTime is ICLR; many temporal CF benches are ICMLA/BigData. Do not write “SOTA NeurIPS benchmarks” for them.

---

## 6. Related-work table (device from CausalTime)

Compare **evaluation designs**, not methods. Columns that make the gap visible (adapt to the construct):

| Prior resource | What it scores | What ground truth it has | What it cannot support |

CausalTime used fidelity / GT / coverage. For a CF-audit the informative columns are: model-relative dashboard; GT graph; GT structural (Pearl) CF; classification label as explanation target; scores **independently produced** explainers.

**Estimand split to keep visible** (do not paper over): nearest point in input space (Wachter / DiCE / CARLA / CFTS) vs minimal `do()` (Karimi) vs Pearl CF of the DGP (Höllig SMO; Beckers; this project’s audit).

**“Ground truth” is not one thing:** CausalTime’s graph is derived from a fitted NN (authors deny it is the underlying DGP); CauseMe real graphs are high-confidence; CausalRivers is physical topology; a written additive-noise `F` is a different object.

---

## 7. Anti-templates (from the same corpus)

| Template | Why it worked there | Why eval-theory papers reject copying it |
|---|---|---|
| SuperGLUE | Shared score after saturation | Makes the ranking axis the scientific object (Raji) |
| OpenXAI leaderboard | Systematic XAI bake-off | Ranks proxy faithfulness |
| “We also ran 30 datasets” | CLIP / HELM coverage | Coverage without a construct multiplies invalid claims (Raji / Bean) |
| Generator paper that “uses SCMs” | Bahri: first SCM+TSCF generation at BigData | Fights on “we also use SCMs,” not on oracle vs penalty |
| Position paper, no runnable protocol | Agenda | TAE/E&D want a protocol others can re-run; use the Position track if there is no artifact |

---

## 8. TAE 8-page cut (contract only)

Keep in the main text: construct, assumption chain, comparison-of-designs table, controls, headline finding with named variance, limitations as claim law.
Move if needed: extra methods, extra datasets, generator cards, extra figures.
Do not spend main-text pages on: E&D Croissant bureaucracy, LLM-as-judge (JUDGe’s object), dataset-reuse scientometrics, or a methods bake-off whose object is SOTA.

**Finish the results the protocol calls for.** These guidelines do not license a WIP snapshot, a “complete-but-unfinished” label, or a retractions-as-narrative feature. Report the finished evidence at the scale you claim. Do not reprint retracted numbers. Historical construct revisions belong in a short limitations or appendix sentence if a reader would otherwise cite a dead figure — not as the paper’s story.

---

## 9. Thin mapping to this repository (pointers only)

Apply the **patterns** above to objects already defined in [`docs/general_plan.md`](../../general_plan.md). This section is not an outline of sections to write.

| Pattern (source) | Object already in the project |
|---|---|
| Construct, not dataset (Raji / Bean) | Model-vs-world audit, not “the CausalTemp suite” |
| Remaining gap after methods (WILDS) | `(D, Δ)` after standard temporal CF methods |
| Graded faithfulness (Jacovi); companion hard check | Named graded mechanism-faithfulness; 0/1 as `frac_pass`, not the ranking axis |
| Axes not averaged (HELM) | A = dataset; B/C = methods; `(D, Δ)` ranks |
| Controls in main figure (Adebayo) | Oracle vs vacuous rewrite vs SCM-in-the-loss foil |
| Comparison table (CausalTime) | G / Pearl CF / label / independent explainers |
| GT honesty (CausalTime §1) | Written `F` ≠ fitted-NN “GT graph” |
| Internal vs external (Liao / Raji) | Synthetic validates the metric; real tests transfer |
| Claim law (Raji) | Additive noise, intervention reading, label site, `t_label − t0` |

When implementation and manuscript disagree on **metric semantics**, follow `docs/general_plan.md`, not this file. This file constrains **story and structure** stolen from published eval papers.
