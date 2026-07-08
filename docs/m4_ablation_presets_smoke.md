# M4 Ablation Presets — Smoke-Scale Preliminary Findings (H5, H6, H7)

**Date:** 2026-07-08
**Scope:** smoke-scale only (`k=5, T=30, N=500`, `n_cf=10`, single seed), per the standing
constraint that no full-scale run happens until the pipeline is cleared at smoke scale. These are
**exploratory** findings to (a) prove the three ablation presets run end-to-end through the phased
pipeline and (b) give a first read on the H5/H6/H7 directions — **not** powered verdicts. Powered
re-assessment (≥5 seeds, n_cf≥100, bootstrap CIs) is deferred to the full-scale milestone.
**Pipeline:** phases 01→04 (generate → train LSTM → run 7 CF methods → evaluate) for each preset.
**Presets:** `causaltemp_xai/config.py` — `SMOKE_GAUSSIAN` (H5), `SMOKE_NONMONOTONIC` (H6),
`SMOKE_REGIME` (H7). Code + structural-property tests committed separately (`feat(m4)`); this doc
records the runs.

## Baselines (from the M2 pipeline-clearance runs, same n_cf=10-scale story)

The core phenomenon this benchmark exists to expose reproduces on **every** config: the causal
recourse method (`CARLA`) is rollout-faithful **and** valid (`roll_hard_valid = 1.0`), while every
standard CF method reaches `validity = 1.0` while being causally arbitrary (`rollout_hard = 0`,
`pearl_hard = 0`). That contrast is config-invariant and is the constant backdrop for all three
ablations below. The one method whose behaviour *varies* across configs — and therefore the one
that carries the ablation signal — is `PearlCARLA` (pearl-faithful by construction; its *validity*
is the free variable):

| Config | mechanism | noise | PearlCARLA validity | PearlCARLA pearl_hard_valid |
|---|---|---|---:|---:|
| `smoke` | linear VAR | Laplace | 0.95 | 0.95 |
| `smoke_nl` | nonlinear MLP | Laplace | 0.00 | 0.00 |

PearlCARLA is faithful-and-valid on the **linear** mechanism but its validity **collapses** on the
**nonlinear** one (the documented long-horizon collapse: the Pearl delta obeys a homogeneous
recursion that decays under the SCM's stability requirement). The ablations test whether this split,
and the core phenomenon, are artifacts of specific modelling choices.

## H5 — Gaussian-noise negative control (`SMOKE_GAUSSIAN`)

**Pre-registered expectation:** the validity-vs-causal-faithfulness divergence is a property of the
*causal structure*, not of the non-Gaussian (Laplace) innovation choice. So switching the noise to
Gaussian — everything else identical to linear `smoke` — should leave the phenomenon **unchanged**
(this is a negative control: we expect *no* effect).

**Result (smoke-scale): expectation holds.** `smoke_gaussian` reproduces linear `smoke` on every
method:

| method | validity | roll_hard_valid | pearl_hard | pearl_hard_valid |
|---|---:|---:|---:|---:|
| CARLA | 1.00 | **1.00** | 0.00 | 0.00 |
| PearlCARLA | 1.00 | 0.00 | 1.00 | **1.00** |
| CftsCOMTE / CftsWachter / CftsCels | 1.00 | 0.00 | 0.00 | 0.00 |
| CftsConfeti | 0.90 | 0.00 | 0.10 | 0.00 |
| CftsCounts | 0.50 | 0.00 | 0.00 | 0.00 |

CARLA stays rollout-faithful+valid; standard methods stay valid-but-arbitrary; PearlCARLA stays
faithful **and** valid (validity 1.00, matching linear `smoke`'s 0.95). **Verdict (preliminary):**
the phenomenon is *not* an artifact of the Laplace noise choice — it survives the Gaussian negative
control. (The Axis-B `TV_Confounding` for this config, 0.607, also tracks linear `smoke`'s 0.628,
confirming the Gaussian variant is structurally the linear regime.)

## H6 — Non-monotonic mechanism (`SMOKE_NONMONOTONIC`)

**Pre-registered expectation:** the phenomenon should be robust to the *shape* of the nonlinear
hidden activation, not tied to the monotonic `tanh`. Swapping the per-node MLP's hidden activation
to a bounded, non-monotonic one (odd `sin`) — output branch still `gain·tanh(·)`, so boundedness is
preserved — should behave like the standard nonlinear config `smoke_nl`.

**Result (smoke-scale): expectation holds.** `smoke_nonmonotonic` reproduces `smoke_nl` essentially
row-for-row:

| method | validity | roll_hard_valid | pearl_hard | pearl_hard_valid |
|---|---:|---:|---:|---:|
| CARLA | 1.00 | **1.00** | 0.00 | 0.00 |
| PearlCARLA | **0.00** | 0.00 | 1.00 | 0.00 |
| CftsConfeti | 0.60 | 0.00 | 0.40 | 0.00 |
| CftsCOMTE / CftsWachter / CftsCels | 1.00 | 0.00 | 0.00 | 0.00 |
| CftsCounts | 0.80 | 0.00 | 0.00 | 0.00 |

Identical qualitative structure to `smoke_nl`, including PearlCARLA's validity collapse to 0.00 and
CftsConfeti's partial-faithfulness-via-tiny-edits (0.60/0.40, matching `smoke_nl`'s 0.60/0.40).
**Verdict (preliminary):** the phenomenon — and PearlCARLA's nonlinear-regime validity collapse — is
robust to a non-monotonic mechanism; it is a property of nonlinearity, not of `tanh` specifically.

## H7 — Regime-switching mechanism (`SMOKE_REGIME`)

**Pre-registered expectation:** a structural break partway through the observed horizon (two
distinct nonlinear regimes, deterministic switch at `T/2`) stresses temporal stability. The
relevant axis is **Shift-VR** (validity retention under distribution shift) and whether causal
faithfulness survives a mid-trajectory regime change. Being a nonlinear mechanism, it should share
`smoke_nl`'s CF-faith profile (PearlCARLA validity collapse), with the regime break as an added
robustness stressor.

**Result (smoke-scale): expectation broadly holds, with an informative shift-VR contrast.**

CF-faith profile (`smoke_regime`):

| method | validity | roll_hard_valid | pearl_hard | pearl_hard_valid |
|---|---:|---:|---:|---:|
| CARLA | 1.00 | **1.00** | 0.00 | 0.00 |
| PearlCARLA | **0.20** | 0.00 | 1.00 | 0.20 |
| CftsCOMTE / CftsWachter / CftsCels / CftsConfeti | 1.00 | 0.00 | 0.00 | 0.00 |
| CftsCounts | 0.90 | 0.00 | 0.00 | 0.00 |

Same core structure as the other nonlinear configs — CARLA rollout-faithful+valid, standard methods
valid-but-arbitrary. PearlCARLA's validity is 0.20: collapsed (nonlinear regime) but not fully to
zero (the milder post-switch regime 2 leaves a few instances flippable), consistent with the
long-horizon-collapse mechanism rather than a new failure mode.

Shift-VR (validity retention, base vs. noise-shifted test set — H7's primary axis):

| method | validity_base | validity_shift | shift_vr |
|---|---:|---:|---:|
| CARLA | 1.00 | 1.00 | 1.00 |
| CftsWachter / CftsCOMTE / CftsCels / CftsCounts | 1.00 | 1.00 | 1.00 |
| CftsConfeti | 1.00 | 0.90 | 0.90 |
| PearlCARLA | 0.20 | 0.00 | *(suppressed: base < 0.3)* |

**Verdict (preliminary):** the phenomenon survives a mid-trajectory regime break. The shift-VR
result is itself informative: the standard CF methods retain full validity across the regime shift
(`shift_vr ≈ 1.0`) — but this "robustness" is exactly the benchmark's point, since they flip the
classifier *regardless of the regime structure* (valid but causally arbitrary). PearlCARLA, the
method that respects causal structure, has low base validity that collapses further under shift; per
the M1 guard its ratio is correctly suppressed (`validity_base = 0.20 < 0.3`) and only the
`(base, shift)` pair is reported. Robustness-of-validity under regime change is not evidence of
causal faithfulness — the H7 direction the ablation was built to probe.

## Summary

| Hypothesis | Ablation | Preliminary verdict (smoke-scale) |
|---|---|---|
| H5 | Gaussian negative control | **Phenomenon survives** — not a Laplace-noise artifact |
| H6 | Non-monotonic mechanism | **Phenomenon robust** — behaves like standard nonlinear |
| H7 | Regime-switching | **Phenomenon survives** the regime break; standard methods' validity-robustness under shift is *not* causal faithfulness |

**Caveats.** All at `n_cf=10`, single seed, smoke scale — directional reads, not powered verdicts.
CftsConfeti's low/variable validity makes its per-config `shift_vr` ratios small-sample-noisy (the
M1 guard suppresses the ratio below `validity_base = 0.3`). These presets and this protocol carry
forward to the full-scale milestone for the actual pre-registered H5/H6/H7 verdicts with CIs.
