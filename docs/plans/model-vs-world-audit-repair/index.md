# Plan: model-vs-world-audit-repair

**Date**: 2026-08-26
**Branch**: [branch-name]
**Predecessors**: [Links to predecessor plans, or "None"]
**Goal**: [One sentence. What is true when this is done that is not true now.]

Executed per [PROTOCOL.md](PROTOCOL.md). Status of record: [state.json](state.json).
Runtime record: [journal.md](journal.md) · [decisions.md](decisions.md) · [backlog.md](backlog.md)

---

## Context

[The problem, and only the background needed to route work. Target <=60 lines.
Long derivations, measurement tables, and code archaeology go in resources/ and are
linked from here or from the stage that needs them.]

---

## Strategy

[High-level approach. If stages group into phases, describe them here -- 1-2 lines each.]

---

## Success Criteria

Every row declares a **Kind**. GATE blocks the stage; REPORT is measured and published and
never blocks. **Default to REPORT** -- a row earns GATE only if you can name the deciding
command *and* its tolerance today. `If unmeasurable` must differ from `If missed`:
a measurement that returns no signal is information about the instrument, not the change.

A GATE value must be **derived from a measurement of this run's own inputs** -- never a literal, a
default, a band midpoint, or a row generated to satisfy a count. Keep the provenance row below if
any gate here is a measurement; delete it only if none are.

| Metric | Baseline | Target | Kind | If missed | If unmeasurable |
|--------|----------|--------|------|-----------|-----------------|
| [test suite] | green | green | GATE | block stage | n/a |
| Every GATE value is derived from a measurement of this run's own inputs | n/a | no status is a literal, a default, a band midpoint, or a row generated to satisfy a count | GATE | block stage | REPORT `NOT MEASURED` and block |
| [measurement] | [current] | [target +/- tolerance] | REPORT | publish + continue | publish `NOT MEASURED` |

---

## Files That May Be Changed

- `path/to/file` -- [What changes and why]

---

## Stages

Routing table only. **Status, notes and commits live in `state.json` and nowhere else** --
never mirror them here. To read the current status:

```bash
jq -r '.stages[] | "\(.id)  \(.status)  \(.title)"' state.json
```

| # | Stage |
|---|-------|
| 1 | [Stage 1](stages/01-.md) |
| 2 | [Stage 2](stages/02-.md) |
| 3 | [Stage 3](stages/03-.md) |
| 4 | [Stage 4](stages/04-.md) |
| 5 | [Stage 5](stages/05-.md) |
| 6 | [Stage 6](stages/06-.md) |
| 7 | [Stage 7](stages/07-.md) |
