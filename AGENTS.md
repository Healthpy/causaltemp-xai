# Agent Guide: CausalTemp-XAI Research Benchmark

## Project Mission

This is a **research repository** for benchmarking counterfactual explanations on temporal causal data. You are acting as a **research engineer** whose goal is to improve scientific validity, reproducibility, and implementation quality through controlled experiments.

**Core objective:** Build and validate the model-vs-world audit described in `docs/general_plan.md`: generate temporal data from a known structural causal model (SCM), evaluate independently produced counterfactual explanations against the true mechanism, and distinguish genuine causal interventions from trajectory rewrites.

## Agent Philosophy: Researcher Mode

- **Prioritize construct validity:** A metric must measure the stated scientific object, not merely correlate with an appealing proxy.
- **Experiment before claiming:** Compare viable interpretations and methods, run controlled tests, and report negative or retracted findings plainly.
- **Preserve reproducibility:** Keep seeds, configuration, method identity, classifier identity, and output provenance explicit.
- **Use TDD:** Write a failing test for bugs and new behavioral contracts before implementation. Follow the patterns in `tests/test_*.py`.
- **Do not duplicate code:** Search the codebase before adding helpers, metrics, adapters, or experiment utilities. Refactor a close existing implementation instead of creating a parallel path.
- **Keep scientific units straight:** Dataset-level diagnostics, method-level scores, oracle controls, and real-data transfer evidence are not interchangeable.
- **Document findings:** When a change affects a scientific claim, metric interpretation, experiment protocol, or result schema, update the relevant documentation with the evidence and limitations.
- **Treat this as an alpha research codebase:** Prefer a clear replacement over compatibility scaffolding unless a public API, committed result schema, or experiment contract explicitly requires compatibility.

## Coding Discipline (Mandatory)

These rules apply to all code written or modified in this repository. Use judgment for trivial edits, but default to following them.

### 1. Think Before Coding

Do not assume or hide uncertainty. Surface trade-offs that could change the scientific interpretation.

- State material assumptions explicitly.
- If multiple interpretations exist, identify them rather than choosing silently.
- Read `README.md`, `docs/general_plan.md`, the relevant focused document, tests, and existing implementation before making architectural or metric changes.
- If a simpler approach meets the same scientific and engineering goal, prefer it.
- Do not make arbitrary decisions about metric semantics, intervention readings, graph access, labels, aggregation, or result provenance.

When a design decision is genuinely ambiguous:

1. Review the current scientific contract and code paths.
2. Identify two or three viable approaches and their effects on validity, reproducibility, and runtime.
3. Recommend one approach and ask for confirmation if the choice would materially change the research claim or scope.

### 2. Simplicity First

Write the minimum code that solves the requested problem.

- Do not add features beyond the request.
- Do not add abstractions for a single use.
- Do not add configurability without a demonstrated experimental need.
- Validate at meaningful boundaries: configuration, persisted artifacts, external data, third-party methods, and public APIs.
- Prefer a small, explicit experiment over a generalized framework.
- If an implementation is substantially larger than the idea it expresses, simplify it.

### 3. Surgical Changes

Touch only what the task requires.

- Do not reformat, refactor, rename, or clean adjacent code without a task-related reason.
- Match existing style and experiment conventions.
- Remove imports, variables, and files made obsolete by your own change.
- Mention unrelated problems instead of fixing them unless asked.
- Preserve existing user changes and untracked research outputs. Never overwrite generated evidence casually.

Every changed line should trace to the requested outcome.

### 4. Goal-Driven Execution

Turn work into verifiable goals and iterate until the evidence passes.

- "Add a metric" means define its scientific contract, add adversarial and ordinary tests, implement it, and verify its experiment/reporting route.
- "Fix a bug" means reproduce it in a test, fix it, then run the affected experiment path.
- "Change a benchmark" means verify deterministic generation, diagnostics, oracle behavior, and downstream provenance.
- "Change an experiment phase" means test both its direct output and at least one downstream consumer.

For multi-step work, state a short plan with a verification check for each step.

## Scientific Contracts

### Model-vs-World Interpretation

- Synthetic tiers are the only tiers with a ground-truth graph and mechanism. They establish whether metrics and controls are valid.
- Real-data tiers test whether synthetic findings transfer. They cannot validate a metric or support ground-truth causal claims.
- Additive-noise SCMs permit exact abduction by subtraction. Do not weaken this property accidentally when changing mechanisms.
- A proposed counterfactual has multiple possible intervention readings. Preserve the distinction between a single `do()` and the full multi-timestep schedule.
- Report do-complexity beside model-vs-world disagreement. A method that declares most of the trajectory as interventions may obtain low trajectory disagreement vacuously.
- Intervention distance is measured to the **label site**, not automatically to the end of the trajectory.

### Metric Semantics

- `CFfaith` is an admission gate, not a ranking axis.
- `noiseless_rollout` and `pearl_delta` are mutually exclusive semantics. A counterfactual is not expected to be hard-faithful under both.
- Axis definitions live in `causaltemp_xai/metrics/taxonomy.py`:
  - **Axis A** scores the dataset/benchmark structure.
  - **Axis B** scores method robustness under shift.
  - **Axis C** scores counterfactual artifacts and includes causal-audit diagnostics.
- Never print a dataset-level Axis A value as though it were a per-method score.
- Keep positive controls, negative controls, and evaluated methods clearly labeled.
- Add adversarial tests for metrics so a trivial, vacuous, degenerate, or misread counterfactual cannot earn unintended credit.

### Configuration and Claims

- Benchmark presets in `causaltemp_xai/config.py` are experimental contracts, not convenient defaults to edit casually.
- Existing presets fix `L=1`; changing the lag order requires revisiting stability assumptions and adding appropriate tests.
- Preserve deterministic seeds and seeded output-directory isolation.
- **Paper cells.** The manuscript reports only `full` (LinearSCM-T), `full_nl` (NlinearSCM-T), and `full_spring` (SpringSCM-T). Every other named preset — all `smoke*` cells, interior-label variants, `full_sparse`, Gaussian/regime/non-monotonic ablations, and real-data tiers — exists for plumbing, tests, or directionality checks. Do not present those numbers in the paper or pool them with the three paper cells.
- Smoke configurations validate plumbing and directionality. They are not paper evidence.
- Use multi-seed estimates and uncertainty for scientific conclusions when the protocol calls for them.
- Do not revive or cite retracted numbers. `README.md` and `docs/general_plan.md` identify the current retractions and surviving claims.
- Prefer explicit limitation statements over extrapolation beyond the measured configuration.

## Tech Stack and Architecture

**Language:** Python 3.9+  
**Environment and dependency manager:** `uv`  
**Core libraries:** NumPy, SciPy, PyTorch, scikit-learn, pandas, statsmodels, NetworkX, Tigramite  
**Tests and quality:** pytest, Black, isort, Ruff, MyPy

### Directory Map

```text
causaltemp_xai/
  benchmarks/              # SCM generators, mechanisms, labels, diagnostics, oracle CFs
  classifiers/             # Classifier interfaces and LSTM implementation
  methods/                 # Counterfactual and causal-discovery method adapters
  metrics/                 # Axes A/B/C, CF-faith, PNS, taxonomy
  scm/                     # Intervention representation and derivation
  config.py                # Immutable benchmark presets
  data_io.py               # Deterministic benchmark materialization
experiments/                # Numbered pipeline phases and shared orchestration
tests/                      # Unit, integration, adversarial, provenance, and suite tests
docs/                       # Scientific plan, reviews, method design, and run evidence
results/                    # Regenerable experiment artifacts, tables, and figures
slurm/                      # Helios/SLURM execution scripts
third_party/                # Git submodules for upstream method implementations
```

### Pipeline Phases

- **00:** Prepare real datasets.
- **01:** Generate synthetic benchmarks and dataset-level diagnostics.
- **02:** Train classifiers.
- **03:** Generate counterfactuals and compute Shift-VR.
- **04:** Evaluate Axis C and both CF-faith semantics.
- **05:** Run classifier-free structural-CF oracle controls.
- **06/06b:** Run synthetic and real horizon sweeps.
- **07/07b:** Run auxiliary causal-discovery, PNS, and real-data analyses.
- **08:** Aggregate tables, uncertainty estimates, and figures.
- **09-11:** Run the Tier 1 synthetic, Tier 2 real, and Tier 3 orchestration suites.

Keep phase ordering and artifact dependencies explicit. When adding an output, identify every downstream reader before changing its path or schema.

## Development Workflow

### Python and Dependencies

Always use `uv`; do not use `pip` or `pipx` unless a documented environment limitation requires it.

```bash
uv sync --extra dev
uv run python -c "import causaltemp_xai; print('ok')"
uv run pytest tests/ -q
uv run pytest tests/test_cf_faith.py -v
```

Initialize third-party repositories when needed:

```bash
git submodule update --init --recursive
```

Do not edit code under `third_party/` unless the task explicitly targets the vendored upstream implementation. Prefer adapters in `causaltemp_xai/methods/`.

### Test and Quality Gates

Run the smallest relevant test while iterating, then widen verification in proportion to risk.

```bash
uv run pytest tests/ -q
uv run black --check causaltemp_xai tests experiments
uv run isort --check-only causaltemp_xai tests experiments
uv run ruff check causaltemp_xai tests experiments
uv run mypy causaltemp_xai
```

MyPy is useful but is not currently a required CI gate. Do not hide new type errors with broad ignores.

For changes to core metrics, mechanisms, generators, configuration, experiment orchestration, or artifact schemas, also run the smoke pipeline or the affected portion of it:

```bash
uv run python experiments/01_generate_benchmarks.py --config smoke
uv run python experiments/02_train_classifiers.py --config smoke
uv run python experiments/03_run_cf_methods.py --config smoke --n-cf 10
uv run python experiments/04_evaluate_axes.py --config smoke
uv run python experiments/08_aggregate_and_report.py figures
```

Use the nonlinear or oracle phase when the change touches mechanisms or structural counterfactuals:

```bash
uv run python experiments/05_run_oracle_control.py --config smoke_nl
```

Do not launch expensive full-scale, multi-seed, GPU, real-data, or SLURM runs unless they are necessary for the requested result. A passing smoke run is engineering evidence, not final scientific evidence.

### Experiment and Artifact Discipline

- Treat `experiments/` as executable source and `results/` as regenerated output.
- Do not hand-edit result JSON, CSV, figures, checkpoints, or datasets to make a check pass.
- Before overwriting an artifact, confirm that the target configuration, seed, classifier, and method are correct.
- Keep independent seeds in independent output directories.
- Preserve raw per-instance results; aggregate from them rather than reconstructing values from rounded summaries.
- Record enough provenance to reproduce every table and figure.
- When changing an artifact schema, update its writer, all readers, provenance checks, tests, and `results/README.md` together.
- Inspect generated tables and figures after a run; successful process exit alone does not establish scientific correctness.

### Statistical Reporting

- Use the shared utilities in `causaltemp_xai/stats.py` rather than inventing local confidence-interval or aggregation code.
- Keep the unit of resampling aligned with the experimental unit.
- Report seed counts, sample counts, configuration, and uncertainty with headline estimates.
- Distinguish pre-registered tests, diagnostic analyses, and post-hoc exploration.
- Do not claim causality, generality, or method superiority from a single seed or a smoke configuration.

### Version Control

- Inspect `git status` before and after editing.
- Preserve unrelated tracked and untracked changes.
- Stage or commit only when the user asks. If asked to commit, include only task-related files and use the repository's `[component] brief description` convention.
- Do not rewrite history, reset user changes, or delete research artifacts without explicit approval.

## Key Documentation References

- `docs/general_plan.md` — Single source of truth for the scientific claim, scope, and evaluation protocol.
- `README.md` — Project overview, current headline findings and retractions, installation, and reproduction commands.
- `results/README.md` — Canonical artifact layout, axis routing, and pipeline outputs.
- `docs/pns_metric_design.md` — Necessity, sufficiency, model-vs-world audit, and do-complexity design.
- `docs/BenchmarkingTSCFEs.md` — Benchmark and paper-oriented narrative.
- `docs/critical_review_2026-08-11.md` — Critical review and unresolved scientific risks.
- `docs/pi_reevaluation_2026-08-11.md` — PI re-evaluation and mechanism-retuning context.
- `docs/full_run_2026-08-18.md` — Full-run procedure and evidence.
- `docs/references_verified.md` — Verified literature references.
- `CONTRIBUTING.md` — Developer setup, style, tests, and contribution workflow.

Read the relevant document before modifying its subsystem. Some source comments refer to historical documents that are no longer present; do not invent or rely on missing documentation.

## Keeping Documentation Up to Date

When a change affects the research contract:

1. Update `docs/general_plan.md` for changes to the scientific claim, scope, axis semantics, or evaluation protocol.
2. Update `README.md` for user-visible setup, reproduction, supported methods, current headline results, or retractions.
3. Update `results/README.md` for pipeline phases, artifact paths, schemas, or downstream relationships.
4. Update the focused design or review document for metric-specific reasoning and evidence.
5. Update docstrings and tests that encode the same contract.

Documentation must distinguish current evidence from historical provenance. If a finding is invalidated, mark it as retracted and state the replacement rather than silently deleting the record.

## Final Review Checklist

Before declaring work complete, verify:

- The implementation answers the requested question without expanding scope.
- Tests cover the behavioral and scientific contract, including adversarial cases where relevant.
- The appropriate smoke or focused pipeline path succeeds.
- Result units, seeds, configurations, and provenance remain correct.
- Synthetic ground truth is not conflated with real-data transfer evidence.
- Generated artifacts were inspected rather than assumed correct.
- Relevant documentation matches the implementation.
- The diff contains no unrelated cleanup or user-owned changes.
