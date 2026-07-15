# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All Python tooling goes through `uv` — never `pip`.

```bash
uv sync --extra dev                        # env setup (creates .venv from uv.lock)
uv run pytest tests/ -q                    # all tests
uv run pytest tests/test_axis_c.py -q      # single file
uv run pytest tests/test_axis_c.py::test_name   # single test
uv run pytest --cov=causaltemp_xai         # with coverage
```

Lint/format — CI runs these over `causaltemp_xai tests experiments` and fails on any diff:

```bash
uv run black causaltemp_xai tests experiments
uv run isort causaltemp_xai tests experiments
uv run ruff check causaltemp_xai tests experiments
```

Pipeline (each phase is independently re-runnable; `--config smoke` is the fast path CI uses):

```bash
uv run python experiments/01_generate_benchmarks.py --config smoke
uv run python experiments/02_train_classifiers.py --config smoke   # linear configs only
uv run python experiments/03_run_cf_methods.py --config smoke --n-cf 20
uv run python experiments/04_evaluate_axes.py --config smoke
uv run python experiments/05_run_oracle_control.py --config smoke  # oracle control; any config, no classifier
uv run python experiments/06_make_figures.py
uv run python experiments/07_aggregate_seeds.py                    # multi-seed pooling + bootstrap CIs
uv run python experiments/08_citris_graph.py --config smoke_nl     # DYNOTEARS self-graphing (nonlinear configs)
uv run python experiments/09_axis_a_icc.py --config smoke          # decoder-based Axis-A ICC
```

`uv.lock` pins a CUDA torch build. On CPU-only machines reinstall CPU torch
(`uv pip install torch --index-url https://download.pytorch.org/whl/cpu --reinstall-package torch`)
and then use `uv run --no-sync`, or the sync reverts it. See the CI comments in `.github/workflows/ci.yml`.

Submodules are required (`third_party/{cfts_repo,dynamask_repo,citris_repo,causalnex_repo}`) — `git submodule update --init --recursive`.

## Standing rules (governance)

This is a research benchmark aimed at publication, not an app. `PROJECT_GOVERNANCE.md` and
`AGENT_CHARTERS.md` treat this file as the normative source of the rules R1–R10 and bind AI
assistants to them, but no such file existed until now — the rules below were **reconstructed** from
how each is cited in `PROJECT_GOVERNANCE.md`, `AGENT_CHARTERS.md`, and `.github/ISSUE_TEMPLATE/`.
Treat them as pending PI confirmation, not as settled text.

- **R1** — No new feature work while any test is failing. Exception: a fix-branch targeting the failing test. (Gate: `pytest -q` — 0 failures, 0 errors.)
- **R2** — No experiment runs on uncommitted code. Confirm `git status` is clean before running any `experiments/` script.
- **R3** — No proxy carries its original method's name. An implementation that approximates a published method must not be exported under that method's name (see `docs/method_provenance.md`).
- **R5** — On code/doc disagreement the code is correct; the doc is updated in the same commit.
- **R6** — No metric appears in a result table before its adversarial test is green (construct a CF that *should* fail the metric; verify rejection).
- **R7** — No result file without `seed` and `git_commit` fields.
- **R8** — Work happens on a feature branch, never a direct commit to `main`; branch CI must be green.
- **R9** — No new public export that is not wired to a test or an experiment.
- **R10** — The pipeline reproduces from a clean checkout (smoke run attached at milestone review).
- **R11** — Claude should never list itself as a co-author or credit itself in commit metadata.

**Rule R4 is defined nowhere** — it appears only inside the range "R1–R10". Ask the PI rather than inventing it.

Two unrelated R-series share the same names, which is a live trap when reading docs:

- **Rules R1–R10** (this file, the governance docs, the issue templates) — normative.
- **Risks R1–R11** (`docs/PROJECT_PLAN.md` §6 risk register) — descriptive. Risk R4 is "misrepresented baselines", risk R10 is "spec–code divergence", and neither is the rule of the same number. `docs/spec_code_reconciliation.md` is titled "(R10)" but tracks the **risk**, not the rule.

Scope is not an implementation detail. Adding or removing any method, metric, experiment phase, or
publication target requires a recorded PI decision in `ROADMAP.md` **before** code is written. If an
assigned task would violate a standing rule, say so before starting rather than reporting it afterward.

Results contradicting a hypothesis get recorded honestly — the hypothesis is never adjusted to match.

## Architecture

Synthetic SCMs with a **known** ground-truth causal graph and mechanism, so a proposed counterfactual
can be scored against the true data-generating process rather than against a proxy. That known-truth
property is what the whole design protects.

**Generation → explanation → scoring:**

- `benchmarks/generator.py` — `LinearSCMT` (VAR(L)) and `NlinearSCMT` (per-node MLP transitions). Both are **additive-noise**, which is load-bearing: it makes Pearl abduction an exact subtraction (`eps = x − f(parents)`), so CF-faith and the oracle structural-CF carry over unchanged from linear to nonlinear.
- `benchmarks/mechanisms.py` — `Mechanism` / `LinearMechanism` / `MLPMechanism`. `generate()` returns the `mechanism` object alongside `X`/`Y`/`graph`; downstream scoring needs it.
- `benchmarks/structural_cf.py` — oracle abduction-action-prediction CF (the positive control).
- `methods/counterfactual/` — CF methods behind one interface. `carla.py` is causal noiseless-rollout recourse; `cfts_methods.py` wraps the vendored `cfts` repo (Wachter/COMTE/CONFETI/CounTS/CELS).
- `metrics/cf_faith.py` — `CFfaith` with two **mutually exclusive** semantics: `noiseless_rollout` (default, what CARLA satisfies) and `pearl_delta`. A single CF cannot be `hard=1` under both — that contrast is itself a benchmark result, not a bug to fix.

**Axis routing** — the four metric axes are deliberately scored on different kinds of method, and the
reasoning is documented at length in the `experiments/_common.py` module docstring. Read it before
touching axis code:

- **Axis C** (`axis_c.py`) + CF-faith → every CF-*generating* method.
- **Axis D** (`axis_d.py`) → Shift-VR for CF methods; input-sensitivity for attribution methods.
- **Axis A** (`axis_a.py`) → attribution methods, scored against ground-truth oracle interventions.
- **Axis B** (`axis_b.py`) → not a per-method score; computed once per **dataset** in Phase 01 as a structural diagnostic, because no graph-discovery method is wired into the main pipeline. Phase 08 is the exception (DYNOTEARS self-graphing on nonlinear configs).

**Configs are Python, not YAML.** Presets (`SMOKE`, `FULL`, `SMOKE_NL`, `FULL_NL`, `SMOKE_GAUSSIAN`,
`SMOKE_NONMONOTONIC`, `SMOKE_REGIME`, …) are `BenchmarkConfig` instances in `causaltemp_xai/config.py`,
resolved by name via `get_config()`. A new preset is documented in that module's docstring.
`shifted_config` / `seeded_variant` derive variants for Shift-VR and multi-seed runs.

**Phases communicate only through path convention** under `results/<config>/<classifier>/…` — there is
no shared runtime object between phases. `results/` is fully regenerated output; nothing there is
source of truth. Datasets and checkpoints under `data/scm_t/` are gitignored and regenerated
deterministically from seeds. See `results/README.md` for the exact layout.

`causaltemp_xai/scm/operators.py` is **dormant** — it implements the spec's operator dictionary but is
unwired and untested. Do not treat it as canonical; `docs/spec_code_reconciliation.md` records why.

## Docs

`DEVELOPMENT.md` is substantially stale: it describes `configs/*.yaml`, a TCN classifier, `axis_p.py`/
`axis_f.py`, and modules (`benchmark/`, `scm/generator.py`, `exemplar_cf.py`) that do not exist. Trust
`README.md`, `results/README.md`, and the code. Per R5, fix the doc in the same commit as any code you
touch that it misdescribes.

`ROADMAP.md` (milestones M0–M6 + current blockers), `PROJECT_GOVERNANCE.md` (who decides what),
`docs/PROJECT_PLAN.md`, and `docs/hypotheses_assessment.md` (H1/H3/H4 verdicts) are current.

## Commits

`[component] brief description` — e.g. `[metrics] add temporal sparsity decomposition`.
