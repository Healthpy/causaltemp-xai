# Plan: Regime-switching HMM (B3) + CITRIS/iCITRIS scope (B4)

**Date**: 2026-07-14
**Predecessor**: [`docs/PROJECT_PLAN.md`](../../PROJECT_PLAN.md)
**Scope**: This plan was trimmed on 2026-07-14 (user directive) to **only** items B3 and B4 of the original completion audit. Stage A (LOF, step-function ablation, decorrelated-label control, TV-Confounding Robustness) and Stage B items B1 (TSEvo/Glacier) and B2 (TCN/Transformer) are **out of scope** — deliberately dropped, not deferred.

The full 13-gap inventory from the audit still lives in this repo's git history if it is ever needed again; it is not reproduced here.

---

## B3 — Regime-switching ablation as a real HMM (#4)

**Gap**: `RegimeSwitchNlinearSCMT` does a single deterministic 2-regime switch at T/2 — not the HMM with $R\in\{2,3\}$ regimes and random change-points the plan (`updated_general_plan.md` §Benchmarks) specifies.

**Approach (chosen)**: Add a *new* generator `HMMRegimeSwitchNlinearSCMT` and a new preset `smoke_regime_hmm` (`mechanism_type="mlp_regime_hmm"`) rather than mutating the existing deterministic class. Rationale:
- The existing `RegimeSwitchNlinearSCMT` is locked by 7 tests and by `results/smoke_regime/`; mutating it in place would silently reinterpret those artifacts (the plan explicitly warns against this).
- Keeping both gives a clean A/B for the H7 writeup: deterministic single-break vs. stochastic HMM multi-break.

**Design**:
- $R\in\{2,3\}$ regimes, each an independent `MLPMechanism` over the shared causal graph (parameter switch, not graph switch — same contract as the deterministic class).
- A hidden Markov process with an $R\times R$ transition matrix (strong diagonal `p_stay`, default 0.9 → regimes persist, with occasional random change-points). Per-sequence, per-timestep regime path sampled from the chain, run across burn-in + observed window.
- Regime 0's mechanism is drawn as the first `MLPMechanism.random` call after the graph — bit-identical to a plain `NlinearSCMT` at the same seed — so the ablation isolates exactly the HMM structure.
- **Downstream single-mechanism contract** (unchanged from the deterministic class): `generate()` returns regime 0 as the nominal `"mechanism"` for CF-faith / oracle-CF, plus `"mechanisms"` (all R), `"transition_matrix"`, and `"regime_path"` (N, T ground-truth diagnostic) as metadata. CF-faith on this preset scores against the nominal regime-0 mechanism — the deliberate H7 stress condition, not a bug.

**Status**: IMPLEMENTED 2026-07-14 — generator, config preset, `data_io` dispatch + persistence, and tests. See the summary in the closing PR/commit.

---

## B4 — CITRIS / iCITRIS (#7, gates Axis B graph-error decomposition #10)

**Gap**: CITRIS/iCITRIS are named in the paradigm table (`updated_general_plan.md` §Methods) and gate the Axis B graph-error decomposition (already-written `graph_error_decomposition` in `axis_b.py` has no self-graphing method to feed it) and half of the H3/H7 comparison set. Neither is implemented — only a docstring mention.

**Why this is a decision, not just engineering**: CITRIS (Lippe et al., ICML 2022) and iCITRIS are self-supervised causal-representation learners with their own VAE + intervention-conditioned transition-prior training loops — **not** a thin wrapper like the vendored cfts methods. Critically, their identifiability result relies on **intervention-target supervision** (which causal factors were intervened on at each step, $I_t$). The current benchmark data model (`X, Y, graph, mechanism`) does **not** emit intervention-labeled temporal triplets, so a faithful CITRIS needs either new data-generation support or a documented adaptation. This is the fork that must be settled before code — see the open decision below.

**Decision**: Option 1 (faithful implementation) chosen by user 2026-07-14.

**Status (revised 2026-07-14 after PI review)**: **GENUINE CITRIS vendored from
upstream.** An earlier iteration shipped a hand-rolled identity-encoder linear-VAR
model under the name "CITRIS (faithful)". The PI end-to-end review correctly
flagged that as the same proxy-naming integrity violation the lab polices
elsewhere (Standing Decision #3): the identity-VAR collapse meant *none* of
CITRIS's identifiability machinery was exercised — it merely matched a ridge
baseline. **That proxy has been replaced** with the genuine upstream
implementation.

Delivered 2026-07-14 (revised):
- **Upstream vendored**: `third_party/citris_repo` = git submodule of
  https://github.com/phlippe/CITRIS (alongside `cfts_repo`, `dynamask_repo`).
- **Data support (complete)**: `causaltemp_xai/benchmarks/interventional.py` —
  `generate_interventional_sequences()` rolls the intervention-target-labeled
  sequences CITRIS requires. Added `mode="single"` (≤1 intervention per step →
  one-hot targets, the standard CITRIS regime). Tested.
- **CITRIS adapter (genuine)**: `causaltemp_xai/methods/causal/citris.py` uses
  the **unmodified upstream** `models.shared.transition_prior.TransitionPrior`
  (Gumbel-Softmax latent→causal assignment `psi` + `psi(0)` noise slot +
  intervention-conditioned prior) and `TargetClassifier` (the auxiliary loss
  that specialises `psi`), loaded via importlib with light stubs for the repo's
  image-pipeline deps (torchvision/seaborn) so no Lightning is needed. The
  adapter supplies only the observation-modality MLP encoder/decoder (the
  benchmark's channels are low-dim vectors, not images — the modality encoder
  is legitimately swappable) + the CITRIS-VAE ELBO training loop with
  Gumbel-temperature annealing. A test asserts the prior/classifier come from
  the `models.shared.*` upstream modules.
- **iCITRIS**: not built separately — the benchmark's SCM is purely time-lagged
  (no lag-0 edges), so iCITRIS reduces to CITRIS here (documented).

**Honest identification status**: genuine CITRIS is a heavy VAE whose
disentanglement (`psi` specialisation) needs proper-scale training; at
smoke-scale CPU training `psi` is near-uniform and graph recovery is ~chance —
an honest property of the real method, *not* a regression. (The removed proxy's
~0.84 AUC was an artefact of it being a linear model matched to a linear signal,
i.e. the very thing that made it not-CITRIS.) Full-scale training / GPU is the
path to a reportable Axis-B number — this aligns with the PI's action #3
(full-scale runs) and must be reported *with* its training config.

Remaining / follow-ups:
- **#10 (Axis B graph-error decomposition) — DONE**: `experiments/07_auxiliary_methods.py`
  fits CITRIS on intervention-labeled data, reads off the inferred lag-1 graph,
  and writes `results/<config>/citris/graph_error.json` with Axis-B recovery
  (SHD/LagAcc/AUC via density-matched binarisation) **and** the H3 graph-error
  decomposition — CF-faith (soft) of the oracle structural CF derived from the
  true vs. CITRIS-inferred graph, split into `graph_error` and
  `propagation_error` (the latter 0 for the oracle by construction). Verified on
  `smoke_nl`: AUC ~0.78, graph_error ~0.008 (small, because the SCM's cross-edge
  effects are small — an honest, consistent finding). Added to the README
  pipeline. The `soft` CF-faith is used deliberately: the `hard` score is a 0/1
  per-instance indicator that any edge error zeroes, so it can't grade the
  decomposition.
- **Per-method propagation error — DONE**: Phase 07 also decomposes every
  existing CF method's *actual* CFs (from Phase 03) — scoring each against the
  true SCM (`cf_faith_vs_gt` → `propagation_error = 1 − that`) and the
  CITRIS-inferred-graph SCM (→ the `graph_error` term), written under `methods`
  in `graph_error.json`. On `smoke_nl` this reproduces the H3 story cleanly:
  **CARLA (causal recourse) propagation_error ≈ 0.000** vs **CftsCOMTE
  (instance substitution) +0.202** (Wachter +0.109) — standard CF methods fail
  to propagate through the SCM while the causal method respects it. The oracle
  row is retained as the perfect-propagator reference (propagation_error ≡ 0).
- **General nonlinear-mixing path** (`identity_encoder=False`) does not yet
  identify at smoke scale — a genuine research item (representation
  regularisation, CITRIS-NF variant, hyperparameter search). Not needed for
  the current benchmark (which is identity-mixing by construction), so it does
  not block H3/H7.
- Report CITRIS Axis-B numbers **with the training config** (N, epochs) that
  produced them; recovery improves with more data/epochs.

## PI re-review response (2026-07-14): DYNOTEARS added as the load-bearing graph-aware method

The PI follow-up review confirmed both P0 fixes and flagged the exposed coverage
gap: with genuine CITRIS non-identifying at smoke scale, the "causal/graph-aware"
paradigm cell was empty and Axis B / H3 lacked a genuinely-recovering method
(PI action #2). **Resolved:**

- **DYNOTEARS vendored + wired**: `third_party/causalnex_repo` (git submodule of
  McKinsey CausalNex); `causaltemp_xai/methods/causal/dynotears.py` uses the
  genuine upstream `_learn_dynamic_structure` solver (numpy/scipy augmented
  Lagrangian) unmodified, with an adapter that reshapes `(N,T,k)` panels into
  DYNOTEARS `(X, Xlags)` format and maps the inter-slice matrix `A` to the
  `(k,k,L)` convention. **Observational** (no intervention targets).
- **It recovers the graph**: mean edge-ranking AUC ~0.91 across seeds; in Phase 07
  on `smoke_nl`, SHD=2, LagAcc=0.86, AUC=0.917. Tests assert above-chance
  recovery (`tests/test_dynotears.py`).
- **Phase 07 now defaults to `--method dynotears`** (the load-bearing graph-aware
  method); `--method citris` runs the honest secondary. Output goes to
  `results/<config>/<method>/graph_error.json`. This gives H3 a genuine,
  non-circular positive method and Axis B a well-recovered graph.

## Full-scale (`full_nl`) run — partial (2026-07-15)

- **Dataset**: `full_nl` (k=10, T=100, N=10000) generated; LSTM classifier
  trained to **98.95% test accuracy** (8.5 min, one seed).
- **Graph recovery, multi-seed (5 seeds) — DONE**: DYNOTEARS AUC **0.937 ± 0.020**
  (range 0.917–0.969), SHD 3.6 ± 0.8
  (`results/full_nl/dynotears_multiseed_recovery.json`). Reliable at scale.
- **Graph-error decomposition + sweep — DONE** (seed 42, n_cf=100): DYNOTEARS
  AUC 0.952; graph_error dynamic range is *smaller* at full scale (≤0.004 even for
  a random graph) — strengthens H3a (see `docs/hypotheses_assessment.md`).
- **Per-method propagation table (full scale)**: single-seed Phase 03/04/08 run
  in progress at time of writing.
- **Multi-seed CF-method bootstrap CIs — NOT feasible in-session**: the LSTM
  train (~8.5 min/seed) + 7 CF methods (CARLA 500-step × n_cf=100) × 5 seeds is a
  multi-hour job. Run detached/overnight (verified `07_aggregate_seeds` handles
  `mlp` configs): `uv run python experiments/06_aggregate_and_report.py seeds --config full_nl --seeds 0 1 2 3 4 --n-cf 100`.

Still open (PI P1): multi-seed CF-method CIs at full scale (overnight command
above); a fair CITRIS training budget (GPU); spec↔code reconciliation (R10 —
**DONE**, see `docs/spec_code_reconciliation.md`); the identity-mixing scope
decision (add nonlinear-mixing `x=g(z)` variant, or scope representation methods
out).
