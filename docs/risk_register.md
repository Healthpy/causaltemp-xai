# CausalTemp-XAI — Risk Register

**Extracted 2026-07-31** from `docs/PROJECT_PLAN.md` §6 (that file is now
deleted — `DECISIONS.md` 2026-07-31). Original PI audit dated 2026-07-07.

> **IDs renamed `RISK-nn` deliberately.** The register previously used `R1–R11`,
> colliding with the normative standing rules `R1–R10` in `CLAUDE.md` — a
> collision that required a standing warning paragraph in `CLAUDE.md` to
> navigate. Risk IDs and rule IDs are now lexically distinct and the warning is
> retired. **Nothing in this file is normative.** These are descriptive risks;
> the rules live in `CLAUDE.md`.

Mapping from the old IDs: `R1 → RISK-01`, …, `R11 → RISK-11`.

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| RISK-01 | **Loss/corruption of uncommitted work** — pipeline untracked in a OneDrive-synced tree; `references_verified.md` has already vanished this way | ~~High~~ **Mitigated** (M0 closed) | Severe | Commit early, push often; OneDrive + git is a known hazard |
| RISK-02 | **Metric-validity attacks** — reviewers find the CELS artifact, the MCC ceiling, or a gameability loophole before we fix them | High if unfixed | Fatal to the paper | M1 gated *before* at-scale runs; adversarial metric tests in CI (R6) |
| RISK-03 | **Circularity criticism** — CARLA scores `rollout_hard=1` *by construction* against the metric it optimises; a reviewer will call the headline contrast rigged | **Realised and conceded** (2026-07-30) | Major | Positive controls labelled as such; the claim rests on the *standard* methods' failure, never CARLA's success. Addressed structurally by the 2026-07-31 reordering: PNS's `C` term is empirical, and the former H3b is relabelled a construction check |
| RISK-04 | **Misrepresented baselines** — proxy code named Dynamask/TimeSHAP/CBM-T; unseeded RNG in TimeSHAP | ~~Certain~~ **Moot** — those methods descoped 2026-07-29 | Fatal if published | R3: official implementations or renamed proxies with disclosure; `docs/method_provenance.md` |
| RISK-05 | **Underpowered claims** — n_cf=10, single seed, 3-method ρ; H4 was the headline and is descriptive only | Certain today | Major | M2. **Structurally addressed 2026-07-31**: H4 demoted, so no headline claim now depends on a rank statistic over a degenerate column |
| RISK-06 | **Novelty overrun** — the M6 literature pass (2026-07-08) found CAUKER (ICLR 2026 Oral) and "A Causal DAG Prior for Synthetic TS Classification Datasets" (arXiv 2606.21776) already jointly provide ground-truth-DAG + classification labels, and DoFlow already provides analytical structural CFs (for forecasting) | **Realised** | Major, not fatal | Narrow the claim: the structural-CF ground truth must be used **as an evaluation oracle for independently-produced, post-hoc CF explainers**. This delta survives the M6 pass. Reinforced by the 2026-07-31 reordering, which leads with the audit instrument and the horizon result rather than the artifact |
| RISK-07 | **Scope creep** — 4 axes × 3 benchmarks × 10+ methods × 7 hypotheses is a 2-paper program; the plan already collapsed to an MVP once for this reason | ~~High~~ **Reduced** by the 2026-07-29 and 2026-07-31 narrowings | Schedule | Milestone gates; SepsisSim conditional; M4b repurposed rather than expanded |
| RISK-08 | **Synthetic-only realism objection** | High if M5 no-go | Moderate | SepsisSim if feasible; otherwise foreground the identifiability rationale and position as a *diagnostic* benchmark (as synthetic causal-discovery suites are). M4b's horizon external-validity test now partially answers this without needing a graph |
| RISK-09 | **Reproducibility drift** — bit-identical golden tests already fail cross-platform; label design once silently capped classifier accuracy (avg-pool vs endpoint label) | Medium | Moderate | Tolerance-based goldens with documented bounds; seed audit; clean-checkout reproduction by a non-author (R10) |
| RISK-10 | **Spec–code divergence** — plan math (operator dictionary, DTW CF-faith, latent ICC) ≠ shipped code | ~~Certain~~ **Reconciled 2026-07-14** | Major | `docs/spec_code_reconciliation.md`; live code is the source of truth (R5) |
| RISK-11 | **Single-classifier evidence** — with TCN and Transformer cut, H1/H3 are demonstrated on LSTM only; a reviewer may argue the divergence is architecture-specific | Medium–High (near-certain to be asked) | Major to the generalisation claim, not fatal to the core phenomenon | **PI decision 2026-07-08 (deferred, not open):** LSTM-only through M2; classifier-diversity go/no-go re-decided at the M4 gate against real M2 results, backstopped at M7 reviewer simulation. Candidates if revisited: small 1D-CNN or flattened-window MLP — deliberately *not* Transformer. Until then the two-mechanism-family axis (linear VAR vs nonlinear MLP) is the stated generalisation evidence, stated honestly |

---

## Risks added 2026-07-31

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| RISK-12 | **Degenerate gate metric** — CF-faith (hard) assigns exactly 0.00 to every non-control method, so a reviewer may argue the benchmark cannot discriminate and therefore cannot rank | Certain (it is true) | Major if unaddressed, neutral if owned | Owned explicitly in `general_plan.md` §7 and §10: CF-faith is published as an *admission gate*, PNS is the ranking axis. The paper states this before a reviewer can |
| RISK-13 | **Horizon result reframes our own published validity numbers** — if validity is uninterpretable without `T − t0`, then some of this project's own earlier tables are too | Certain | Moderate | Every validity/PNS figure ships with its `T − t0` (M2 DoD). Earlier figures are retracted, not re-explained |
| RISK-14 | **PNS antecedents** — LEWIS (Galhotra et al. 2021) and Watson et al. (2021) precede necessity/sufficiency-for-explanation; a novelty claim on PNS itself would not survive review | High if claimed loosely | Major | The claim is narrowed to the *temporal* setting with *exact* (not Tian–Pearl-bounded) abduction plus the model-vs-world decomposition. To be verified against `docs/references_verified.md` before any manuscript claim |
