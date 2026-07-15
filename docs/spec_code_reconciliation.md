# Spec ↔ Code Reconciliation (R10)

**Date:** 2026-07-14 · **Trigger:** PI review action (R10) — three definitions in
`docs/updated_general_plan.md` contradicted the implemented code. Per the PI
directive, each divergence is resolved to **one source of truth, chosen
deliberately**, and `updated_general_plan.md` is edited to describe the
implemented reality so the eventual methods section is reproducible.

**Governing principle.** For a *benchmark* paper the source of truth must be the
code that actually produces the reported numbers — i.e. the **wired, tested**
pipeline (`config.py` presets → `data_io` → `experiments/01–08` → `results/`).
Where a spec-matching implementation exists but is **dormant** (not wired into
any config/experiment, no tests), it cannot be canonical without wiring +
validation work that is out of scope here; it is documented as a dormant
alternative, not silently presented as the method.

---

## 1. Mechanism family (Eq. 1)

| | |
|---|---|
| **Spec (drafted)** | `X_t^{(j)} = Σ w_ij · φ_ij(X_{t−τ}^{(i)}) + U`, with `φ_ij` sampled from the invertible operator dictionary `Φ = {identity, sin, cos, tanh, |·|, (·)², exp(−|·|)}`. |
| **Code — LIVE** | Two wired families: **LinearSCM-T** (`LinearSCMT`, VAR, i.e. `φ = identity` — a *special case* of Eq. 1, fully consistent); **NlinearSCM-T** (`NlinearSCMT` / `MLPMechanism`): additive-noise per-node MLP `X_t^i = decay_i·X_{t−1}^i + gain·tanh(MLP_i(masked lagged parents)) + U_t^i`, spectral-norm-capped. Used by every config, experiment, result, and test. |
| **Code — DORMANT** | `causaltemp_xai/scm/operators.py` implements the exact operator dictionary `Φ` (+ `sample_mechanism`, per-edge `Mechanism`). **Unwired** (no config/data_io/experiment uses it) and **untested** (no `tests/test_scm*`). |

**Decision: canonical = the LIVE additive-noise ANM (linear + MLP instantiations).**
Rationale: Eq. 1 is really an umbrella additive-noise structural equation
`X_t^j = f_j(pa(X_t^j)) + U`; the linear case is `φ=identity`, and NlinearSCM-T's
per-node MLP over masked parents *subsumes* a sum of per-edge operators while
preserving **additive noise** (which is what makes Pearl abduction the exact
subtraction the whole CF-faith/oracle machinery relies on). The per-edge
operator-dictionary form is a legitimate alternative parameterisation but is
dormant.

**Spec edit:** rewrite Eq. 1 as the additive-noise ANM with its two implemented
instantiations; move the operator dictionary `Φ` to a footnote flagged as the
non-monotonic ablation's operator source (the step function `𝟙[·>0]` is drawn
from this pool) and as a dormant alternative parameterisation, not the primary
mechanism.

---

## 2. ICC (Axis A)

| | |
|---|---|
| **Spec (drafted)** | Latent-traversal flip-rate: `ICC_i = (1/N) Σ_n 𝟙[ f(D_ψ(z^n + δ_i e_i)) ≠ f(x^n) ]` — perturb one latent axis, decode, count label flips. Requires an encoder/decoder `D_ψ`. |
| **Code — LIVE** | `axis_a.icc(attribution, int_channel)` — a **decoder-free** proxy: fraction of a saliency map's attribution mass on the causally-relevant channel. Wired via `compute_axis_a` (used in `experiments/_common.py`). |
| **Code — matches spec but DORMANT** | `axis_a.icc_latent(encoder, decoder, ...)` implements the spec formula *exactly* (perturb latent, decode, flip-rate). Exported, but **not referenced by any experiment or test**. |

**Decision: keep BOTH as two operationalisations of one concept, with the
latent-traversal `icc_latent` as the *definitional* form.** Rationale: the spec's
`D_ψ`-based ICC only applies to methods that *have* an encoder/decoder (concept /
representation learners — iVAE, CITRIS, CBM-T); attribution methods (TimeSHAP,
Dynamask, IG) have no decoder, so a decoder-free proxy (`icc`) is needed to score
them at all. These are complementary, not contradictory.

**Spec edit:** keep the latent-traversal formula as the ICC definition (it is
implemented as `icc_latent`), and add a sentence that attribution methods —
which lack `D_ψ` — are scored with a decoder-free attribution-mass proxy
(`icc`). **Open wiring TODO (not a definition change):** `icc_latent` is not yet
run in an experiment; wiring it for the decoder-based methods is a P1 follow-up
(tracked with the identity-mixing / Axis-A-signal item).

---

## 3. CF-faith (Axis C)

| | |
|---|---|
| **Spec (drafted)** | Normalised DTW to the analytical oracle CF: `CF-faith = 1 − DTW(X'_exp, X'_CF) / DTW(X, X'_CF)`; hard = `𝟙[DTW(X'_exp, X'_CF) < ε]`. |
| **Code — LIVE** | `metrics/cf_faith.py::CFfaith` — **residual-based SCM-consistency**: `soft = exp(−residual/scale)`, `hard = 𝟙[no retroactive pre-intervention change AND forward mechanism-residual < tol]`, under **two semantics** (`noiseless_rollout`, `pearl_delta`). No DTW. This is the tested, load-bearing metric the entire pipeline (oracle CFs, Phase 04/05/08, the rollout-vs-pearl contrast) is built on. |

**Decision: canonical = the residual SCM-consistency CF-faith (two semantics).**
Rationale: it is strictly more general than DTW-to-a-single-oracle — it tests
whether a CF is consistent with the *mechanism* via abduction–rollout, which (i)
does not require committing to one oracle trajectory, and (ii) yields the
`noiseless_rollout` vs `pearl_delta` distinction that is central to the
benchmark's headline finding. Replacing it with DTW would discard that and break
every downstream phase.

**Spec edit:** replace the DTW definition with the residual/two-semantics
definition as implemented. The DTW-to-oracle distance is noted as an optional
**complementary** metric (cheaply computable — the analytical oracle CF already
exists via `structural_counterfactual`) that MAY be added as a secondary
reported number, but is not the canonical CF-faith.

---

## Summary of resolutions

| Divergence | Source of truth | Action on code | Action on spec |
|---|---|---|---|
| Eq. 1 mechanism | LIVE additive-noise ANM (linear + MLP) | none (dormant `scm/operators.py` kept, flagged) | rewrite Eq. 1 to the ANM + instantiations; `Φ` → footnote |
| ICC | latent-traversal (definitional) + attribution proxy | none (`icc_latent` exists; wiring is a P1 TODO) | keep formula; add attribution-proxy sentence |
| CF-faith | residual SCM-consistency, two semantics | none | replace DTW definition with residual definition |

None of the three requires a code change for the reconciliation itself — the
live code stands; the spec is corrected to match it. Two P1 follow-ups are noted
(wire `icc_latent`; optionally add DTW-to-oracle as a complementary CF-faith
number). After the `updated_general_plan.md` edits below, spec and code tell one
consistent story, clearing the methods-section gate.
