# Archive — superseded documents

**Nothing in this directory may be cited.** These files are retained for
provenance: to show what was believed, when, and on what evidence. Every
number in them predates at least one correction that invalidates it.

Archived 2026-07-31 (`DECISIONS.md`).

| File | Why archived |
|---|---|
| `hypotheses_assessment.md` | Verdicts computed 2026-06-13 on a **TCN classifier** (descoped 2026-07-08) and **DiCE** (deleted 2026-07-17), using metric code predating the `99a2e9f` construct-validity fixes. Commit `d3c377d` then retracted its CF-faith column outright (CftsConfeti 0.46→0.00, CftsCels 0.60→0.00, PearlCARLA 1.00→NaN). Its ρ = −0.87 rank inversion was retired as the paper's headline on 2026-07-31. Live verdicts now live in `docs/05_evaluation_plan.md` §5. |
| `m2_multiseed_and_pearl_carla.md` | Task memo for the multi-seed/bootstrap-CI machinery and the Pearl-CARLA variant. Its §2.3 recommendation (lower `lam_prox` at full scale) was **explicitly ruled out** 2026-07-30: the intervention decays below float32 precision, so no penalty tuning can recover it. |
| `m4_ablation_presets_smoke.md` | Smoke-scale H5/H6/H7 exploration dated 2026-07-08 — pre-`99a2e9f`, pre-CF-faith-fix. Exploratory even when written. |
| `plans/` | Three stale plan trees (`mvp-v0.1-completion`, `nlinearscm-t`, `completion-audit-2026-07-14`). Superseded by `ROADMAP.md` and `PROGRESS_DASHBOARD.md`. |

Cross-references *inside* these files are deliberately left pointing at their
original targets, several of which no longer exist. That is what makes them a
record rather than a live document.

## Two exceptions — design content that is still cited from source

Two of these files mix retracted *numbers* with design *rationale* that live
code still points at. The rationale remains valid; the numbers do not.

- **`m4_ablation_presets_smoke.md`** — the construction of `SMOKE_GAUSSIAN`,
  `SMOKE_NONMONOTONIC` and `SMOKE_REGIME` (and the reasoning behind the
  mechanism shapes) is cited from `benchmarks/generator.py`,
  `benchmarks/mechanisms.py`, `config.py` and `tests/test_config.py`. Read
  those sections for *why the presets are built the way they are*. Its H5/H6/H7
  findings are exploratory smoke-scale results from 2026-07-08 and are
  superseded — M4 verdicts belong in `docs/05_evaluation_plan.md` §5.
- **`m2_multiseed_and_pearl_carla.md` §3** — the PearlCARLA registry-wiring
  contract is enforced by `tests/test_experiment_registry.py` and referenced
  from `experiments/03_run_cf_methods.py`. That contract is live. **§2.3 is
  not** — its `lam_prox` recommendation was ruled out 2026-07-30.

If either file's design content is edited again, promote that section back into
a live document rather than un-archiving the whole file.

**Where the live content went:**

- Scientific claim, contributions, protocol → `docs/general_plan.md`
- Pre-registration, RQs, hypotheses, verdicts → `docs/05_evaluation_plan.md`
- Risks → `docs/risk_register.md`
- Milestones and status → `ROADMAP.md`, `PROGRESS_DASHBOARD.md`
