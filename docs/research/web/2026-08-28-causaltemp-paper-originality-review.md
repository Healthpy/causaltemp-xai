---
date: 2026-08-28T17:00:48+02:00
researcher: Oleksii Furman
topic: "Originality, closest prior work, and venue expectations for the CausalTemp-XAI manuscript"
tags: [temporal-counterfactuals, causal-recourse, benchmarks, peer-review, TAE-2026]
sources: [peer-reviewed, proceedings, arxiv, workshop-guidance]
status: complete
last_updated: 2026-08-28
---

# Research: Originality and Venue Review of the CausalTemp-XAI Manuscript

**Date**: 2026-08-28T17:00:48+02:00
**Researcher**: Oleksii Furman

## Research Question

Review `tae_manuscript/main.tex` for technical quality, clarity, significance, and originality, including verification of its closest-work positioning.

## Summary

The paper is an unusually strong thematic fit for TAE because it studies measurement and causal validity directly. Its defensible originality is the joint audit of arbitrary post-hoc time-series classification trajectories: reconstructing an implied intervention schedule, replaying it for the same unit, and jointly reporting mechanism consistency, schedule complexity, and trajectory-versus-outcome disagreement. The current novelty framing is nevertheless too broad because temporal additive-noise causal recourse and known-equation downstream evaluation already exist. The empirical case is promising but remains vulnerable to single-seed variation, tolerance sensitivity, an untested interior-label claim, and insufficient baseline implementation detail.

## Detailed Findings

### Closest temporal causal-recourse work

- Han et al.'s *Algorithmic Recourse in Abnormal Multivariate Time Series* uses additive-noise temporal structural equations, abducts factual exogenous variables, applies a recourse action, and derives ground-truth downstream time series from known generation equations in semi-synthetic experiments ([TMLR/arXiv paper](https://arxiv.org/html/2309.16896v2)). It evaluates independently trained action-producing baselines. This invalidates the unqualified manuscript statement that the closest known-SCM evaluation of third-party counterfactuals is tabular.
- The distinction that remains is meaningful: Han et al. study anomaly-mitigation actions, whereas this manuscript audits arbitrary full-trajectory classification proposals that do not expose an action. The present paper further adds two explicit CF-faith readings, reconstructed timestep schedules, do-complexity, and the A/B/C localization.
- De Toni et al. formalize temporal causal recourse under time-indexed additive-noise models and show how time can invalidate recourse ([FAccT 2025 paper](https://facctconference.org/static/docs/facct2025-206archivalpdfs/facct2025-final48-acmpaginated.pdf)). Their object is an evolving user state rather than a full classifier-input trajectory, but the manuscript should cite and distinguish this causal/timing precedent.
- Tsirtsis et al. use an SCM to replay alternative action sequences in a finite-horizon decision process ([arXiv](https://arxiv.org/abs/2107.02776)). This is not time-series classification, but it is relevant sequential counterfactual precedent.

### Adjacent temporal benchmarks

- Kan et al. benchmark six temporal counterfactual methods over 20 UCR and 10 UEA datasets with three classifiers and revised sparsity, plausibility, and consistency metrics ([arXiv](https://arxiv.org/abs/2408.12666)).
- Li et al.'s 2026 *Counterfactual Explanation Bake-Off* evaluates nine methods and 16 variants on 20 UCR datasets ([Machine Learning article](https://link.springer.com/article/10.1007/s10994-026-07056-4)). These two papers are more direct temporal-CF comparison points than saliency-oriented XTSC-Bench or AMEE.
- DoFlow performs temporal abduction-action-prediction with a learned causal forecaster and retained latent state ([ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f2affea1fddbdf2e7b5542dac3614640-Abstract-Conference.html)). The safe distinction is learned-model self-evaluation versus a true-DGP oracle applied to independently generated proposals.
- DoTime provides temporal-SCM data with paired interventions and shared-noise counterfactuals but benchmarks causal estimators rather than post-hoc explanations ([arXiv](https://arxiv.org/abs/2607.27263)). EpiCF-Bench similarly targets counterfactual epidemic trajectories rather than classifier explanations ([arXiv](https://arxiv.org/abs/2606.05692)).
- The manuscript's description of TCS as a lagged-graph recovery benchmark is outdated. Its current title is *Adversarial Causal Tuning*, and it fits/selects graph, functions, and noise while supporting observational and interventional generation ([ACT/TCS](https://arxiv.org/abs/2506.02084)).

### Baseline semantics and reproducibility

- Original CoMTE substitutes complete channels across the full time axis ([primary paper](https://www.bu.edu/peaclab/files/2021/05/CoMTE___ICAPAI.pdf)). Consequently, timestep-counted `D=100` is largely induced by the benchmark's observed-variable action language; it should not be phrased as though CoMTE explicitly claimed 100 separate actions.
- M-CELS optimizes a channel-by-time saliency mask toward a nearest-unlike neighbor and has material optimizer hyperparameters ([paper](https://arxiv.org/abs/2411.02649)). Those settings directly control do-complexity and should be reported.
- CONFETTI uses CAM-guided subsequences, NSGA-III, a confidence threshold, and a selection weight, and its primary experiments use CAM-compatible FCN/ResNet classifiers ([AAAI 2026 paper](https://ojs.aaai.org/index.php/AAAI/article/download/38792/42754)). The manuscript must explain the LSTM adaptation and pin the implementation version.
- Wachter, COMTE, M-CELS, CONFETTI, and TSCausalCF need compact implementation cards because optimizer, stopping, reference-set, and guidance choices determine the proposal geometry that the new metrics judge.

### Venue fit and expected evidence

- TAE explicitly solicits work on measurement and causal validity, ground truth, and assumptions connecting protocols to claims ([TAE 2026 call](https://tai-eval.github.io/cfp/)). The topic is therefore an excellent fit.
- The same call emphasizes robustness to random seeds, metrics, and evaluator choices. The paper's one SCM draw and one classifier fit per family are therefore a particularly visible weakness, even though workshop submissions may present partially supported or developing work ([NeurIPS 2026 reviewer guidance](https://neurips.cc/Conferences/2026/ReviewerGuidelines)).
- The highest-value additions are multi-seed SCM/classifier replication with uncertainty, adversarial tolerance analysis, at least one interior or distributed label experiment, multiple defensible action readings, and an anonymous reproducibility artifact.

## Sources Consulted

- [Algorithmic Recourse in Abnormal Multivariate Time Series](https://arxiv.org/html/2309.16896v2) - closest temporal known-equation recourse evaluation.
- [Time Can Invalidate Algorithmic Recourse](https://facctconference.org/static/docs/facct2025-206archivalpdfs/facct2025-final48-acmpaginated.pdf) - temporal causal recourse and timing precedent.
- [Benchmarking Counterfactual Interpretability in Deep Learning Models for Time Series Classification](https://arxiv.org/abs/2408.12666) - direct temporal-CF benchmark.
- [Counterfactual Explanation Bake-Off](https://link.springer.com/article/10.1007/s10994-026-07056-4) - broad current temporal-CF comparison.
- [DoFlow](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f2affea1fddbdf2e7b5542dac3614640-Abstract-Conference.html) - learned temporal abduction-action-prediction.
- [DoTime](https://arxiv.org/abs/2607.27263) - shared-noise temporal counterfactual benchmark data.
- [Adversarial Causal Tuning](https://arxiv.org/abs/2506.02084) - current form of the cited TCS work.
- [CoMTE](https://www.bu.edu/peaclab/files/2021/05/CoMTE___ICAPAI.pdf), [M-CELS](https://arxiv.org/abs/2411.02649), and [CONFETTI](https://ojs.aaai.org/index.php/AAAI/article/download/38792/42754) - baseline semantics and implementation requirements.
- [TAE 2026 call for papers](https://tai-eval.github.io/cfp/) and [NeurIPS 2026 reviewer guidance](https://neurips.cc/Conferences/2026/ReviewerGuidelines) - venue fit and evidence expectations.

## Key Insights

The paper should claim novelty in the conjunction of method-agnostic full-trajectory auditing, same-unit replay, intervention-schedule reconstruction, and diagnostic decomposition. It should not claim that temporal SCM replay or known-equation causal recourse evaluation is unprecedented. Because the benchmark maps edited values to observed-variable interventions, method-native output semantics are part of the estimand rather than incidental implementation detail.

## Confidence Notes

No exact predecessor was found for the manuscript's full joint diagnostic contract, but absence-of-prior-work claims are inherently search-limited. RecAD is clearly close enough to require discussion; whether its every reported replay retains precisely the factual innovation path is less explicit than in this manuscript, so the distinction should be stated cautiously.

## Open Questions

None.

## Clarifications Log

None.
