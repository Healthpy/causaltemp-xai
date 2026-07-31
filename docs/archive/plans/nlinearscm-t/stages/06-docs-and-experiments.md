# Stage 6: Docs, experiments wiring, future-work notes

**Goal**: Make NlinearSCM-T runnable from the experiment harness at smoke scale, document the benchmark and its scope boundaries (nonlinear-mixing as future work), and record findings to memory.
**Dependencies**: Stages 1–5.

---

## Steps

1. **Wire `experiments/run_all.py` to accept nonlinear configs.**
   - *Prerequisite already done*: the `data["mechanism"]` rename and the duplicate CARLA introspection (`run_all.py:74`) were fixed in **Stage 1** (the data-contract sweep) — do **not** redo that here. This stage only adds *nonlinear-config* support.
   - The harness loads a dataset by config name, so passing `--config smoke_nl` should already flow through `load_dataset` → `mechanism`. Confirm the metric path (`evaluate_method`, both `CFfaith` semantics) runs on a nonlinear dataset.
   - **CARLA/Wachter/DiCE on nonlinear are out of scope** (Backlog #2). For the smoke nonlinear run, exercise CF-faith using the **oracle structural-CF** (Stage 4) as the in-house "method" so the harness produces a faithful-CF datapoint without a real CF method. Clearly comment that real CF methods on NlinearSCM-T are the collaborator's track.
   - If retraining a classifier on nonlinear data is needed for validity, either reuse the LSTM training entry point against `smoke_nl` or note it as a follow-up — do not block the stage on classifier accuracy (CF-faith is classifier-agnostic).

2. **Update project docs.**
   - `docs/general_plan.md` §Benchmarks #2: note NlinearSCM-T is implemented as additive-noise MLP transitions; nonlinear *mixing* `g(z)` is a documented future extension (link Backlog #1). Reconcile the §Scope-Boundaries line ("NlinearSCM-T exercises nonlinearity but without tight identifiability guarantees").
   - `README.md`: add a short "NlinearSCM-T" subsection — what it is, how to generate (`--config smoke_nl` / `full_nl`), and the CF-faith/oracle story.
   - Optionally add `docs/plans/nlinearscm-t/resources/` references to a short design note summarizing the additive-noise + abduction rationale and citations (Rhino, ANM, bijective SCM, Stable Recurrent Models).

3. **Document the scope boundary explicitly.**
   - In code (module docstring of `mechanisms.py` / `generator.py`) and `general_plan.md`: this iteration = nonlinear transitions, additive noise; mixing + non-additive noise stress tests are future work. State the abduction-validity boundary (additive ⇒ exact; non-invertible mixing / non-monotone noise ⇒ partial identification).

4. **Update the plan index + memory.**
   - Mark all stages DONE in `index.md`; ensure Fixed Issues / Backlog reflect reality.
   - Write/refresh a NeoCortex memory: NlinearSCM-T design decisions (additive-noise MLP, polymorphic `Mechanism`, abduction-based Pearl generalization that is linear-backward-compatible, oracle structural-CF as positive control), and the open Backlog items (nonlinear mixing; CF methods on nonlinear). Link `[[cf-faith-intervention-crux]]`.

---

## Verification

> Run with `uv run --offline …` (see `resources/commands.md`).

- [ ] `uv run --offline python experiments/run_all.py --config smoke_nl` runs end-to-end and writes `results.json` with CF-faith populated (via oracle CF).
- [ ] `README.md` + `docs/general_plan.md` mention NlinearSCM-T and the nonlinear-mixing future extension.
- [ ] `index.md` progress tracker fully DONE; Backlog accurate.
- [ ] Full suite (`uv run --offline pytest -q`) green.

---

## Commit

`docs(scm): document NlinearSCM-T, wire smoke experiment, record future-work scope`
