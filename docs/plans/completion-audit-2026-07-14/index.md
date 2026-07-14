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

**Status**: WORKING — data support + faithful architecture landed, and graph identification now recovers the lag-1 causal graph **above chance** (edge-ranking AUC ~0.84–0.85 at N=1000/80 epochs, vs the ridge reference of ~0.79–0.98 that bounds the recoverable signal).

**Breakthrough (2026-07-14)**: the fix was recognising the benchmark *guarantees identity mixing* — the k channels are the causal variables — so the correct CITRIS encoder is the **identity** (`identity_encoder=True`, now the default). The nonlinear per-channel VAE latent was scrambling the small, near-linear cross-edge signal; with the identity encoder the model reduces to a directly-trainable sparse, intervention-gated structural transition (predictive MSE + L1 adjacency + linear per-source messages), and the signal survives. The general nonlinear-mixing path (`identity_encoder=False`, full VAE ELBO) is retained but does not yet identify at smoke scale — a documented research item.

Delivered 2026-07-14:
- **Data support (complete)**: `causaltemp_xai/benchmarks/interventional.py` —
  `generate_interventional_sequences()` rolls intervention-target-labeled
  temporal sequences (the `I_t` supervision CITRIS requires) from a dataset's
  ground-truth mechanism. Fully tested (`tests/test_citris.py`).
- **CITRIS model (architecture complete + faithful)**:
  `causaltemp_xai/methods/causal/citris.py` — per-channel factorised VAE
  (leverages the benchmark's guaranteed identity mixing to pin latent block
  `i` to channel `i`), intervention-conditioned severing transition prior,
  explicit L1-sparse learned lag-1 adjacency, additive non-compensable
  message-passing, CITRIS ELBO, and the `inferred_graph()` interface Axis B
  consumes. Trains (recon+KL decrease); wired into `methods/__init__`.
- **iCITRIS**: not built as a separate model — the benchmark is purely
  time-lagged, so iCITRIS's instantaneous-effect discovery reduces to CITRIS
  here (documented in `methods/causal/__init__.py` and the class docstring).

Remaining / follow-ups:
- **#10 (Axis B graph-error decomposition) — DONE**: `experiments/08_citris_graph.py`
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
- **Per-method propagation error — DONE**: Phase 08 also decomposes every
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
