# Necessity/Sufficiency Gap — metric design

**Status: implemented 2026-07-30** (`6d9f277`) — `causaltemp_xai/metrics/pns.py`,
10 adversarial tests in `tests/test_pns.py`, wired to Phase 07 as
`--method pns`. PS direction only; PN still outstanding (see build order).

## Result of the falsifiable self-check

The design committed to a prediction: CARLA on `full` should show
`Δ_total ≈ 1.0` attributed almost entirely to `Δ_trajectory`, else "the design
is wrong, not the finding". **Confirmed exactly** — `Δ_total = 1.00`,
`Δ_trajectory = 1.00`, `Δ_outcome = 0.00`.

The broader result is stronger than expected. On `full`, *every* method posts
`Δ_total` 0.87–1.00, essentially all of it `Δ_trajectory`, with `B = 0.00`
across the board: when the world realises the method's own intervention, the
classifier never flips. Measured directly — the oracle CF differs from the
factual by **~1e-6** at the label site while the methods' proposed
trajectories differ by **~0.2–0.3**. They are not making causal
interventions; they edit the outcome's neighbourhood directly and the causal
structure does no work. `PearlCARLA`/`full` is the sole exception at
`Δ_total = 0.00` — it claims nothing and achieves nothing, the only
well-calibrated method in the set.

## Motivation

CF-faith asks *"is this trajectory mechanism-consistent?"* — a **trajectory**
question. It cannot ask whether the intervention the CF proposes actually
**causes** the outcome change it claims. The two come apart, and today's
results show both failure directions:

- **CARLA on `full`:** CF-faith(rollout) = 1.00, validity = 1.00 — yet its
  intervention's effect on the final timestep is 2.6e-04 (≈ zero). The label
  flip comes from noiseless rollout discarding the abducted noise, not from
  the intervention (2026-07-30 diagnosis, `DECISIONS.md`). A metric that
  compared the *model's* causal claim against the *world's* would have caught
  this automatically.
- **CARLA on `full_nl`:** CF-faith = 1.00, validity = 0.00 — perfectly
  faithful, causally inert.

So this is an orthogonal axis, not a re-packaging of CF-faith.

**Why this benchmark can do it exactly.** Necessity/sufficiency probabilities
are normally only *bounded* (Tian & Pearl 2000) because both potential
outcomes are never jointly observed. Here additive noise makes abduction an
exact subtraction, so for a given instance both potential outcomes are
**deterministic and computable** — no bounding. That is a genuine
methodological advantage of the design and the main reason this is worth
doing here rather than elsewhere.

## Definition

For a selected instance `n` with factual trajectory `x`, the method's CF
`x_cf`, and the benchmark's uniform intervention heuristic
`t0 = derive_intervention_t(x, x_cf)` (reused, so this is scored on the same
intervention CF-faith is):

| symbol | outcome fn | trajectory | meaning |
|---|---|---|---|
| `A` | classifier `f` | method's `x_cf` | what the explanation claims |
| `B` | classifier `f` | oracle CF of the *same* intervention | claim, on the world's trajectory |
| `C` | SCM label rule | oracle CF of the same intervention | what the world actually does |

Oracle CF = `structural_counterfactual(x, mech, t0, …, noiseless=False)` —
**Pearl semantics (reuse the abducted factual noise)**, because the question
is what would have happened *to this instance*, not what the noiseless
skeleton does.

SCM label rule (verified in `benchmarks/generator.py`, all four generators):
`Y = 1[x[-1, 0] > θ]` — threshold on the **final timestep of variable 0**.

The headline number is the gap, with an **additive decomposition**:

```
Δ_total      = A − C
Δ_trajectory = A − B     (proposed CF differs from what the world would produce)
Δ_outcome    = B − C     (classifier disagrees with the true label mechanism)
Δ_total      = Δ_trajectory + Δ_outcome
```

Signed, and both directions are informative: **> 0** the explanation claims a
causal efficacy the world denies (spurious / model artifact); **< 0** the
model is blind to a real causal dependence. The decomposition mirrors the
existing `graph_error_decomposition` pattern.

**Predicted behaviour on CARLA/`full`** (a design self-check, to be verified,
not assumed): A = 1.00 (validity), C ≈ 0 (inert intervention), so
Δ_total ≈ 1.0 attributed almost entirely to `Δ_trajectory` — because the
oracle Pearl CF of that same intervention has no noise-stripping displacement
to ride on. If the implementation does *not* reproduce this, the design is
wrong, not the finding.

## Three problems that must be resolved first

### 1. The label threshold θ is not persisted — hard blocker

`generate()` returns `{X, Y, graph, mechanism}`; θ appears nowhere, and
`data_io` does not save it. Without θ the world outcome `C` cannot be
computed at all. Options:

**Both recovery routes verified on `full` (2026-07-30), no regeneration needed:**

- **(a) Recover from labels.** `Y = 1[v > θ]` with `v = X[:, -1, 0]` pins θ to
  `[max{v : Y=0}, min{v : Y=1})`. Measured bracket width **4.6e-05**;
  midpoint reproduces every stored label exactly. ✅
- **(b) Recompute the median** over concatenated train+val+test. The splits
  *do* partition the full set (N=10000 = config N, no drops), so this
  recovers θ **exactly**: `-0.007906164455137353`, reproduces every stored
  label. ✅
- **(c) Persist θ in the generator** — durable fix, needs regeneration or a
  migration.

**(b) recommended** — it recovers the true θ rather than an
observationally-equivalent value, and is verified exact. **A wrong θ silently
corrupts every world-side number**, so this needs a test asserting the
recovered θ reproduces the stored `Y` exactly, on every config (verified on
`full` only so far; `full_nl` and the ablation presets still to check).

### 2. `structural_counterfactual` is single-node only

Its signature is `(x_orig, mechanism, t0, node: int, value: float)`. Real CF
methods edit **many channels** at `t0` (e.g. CftsCOMTE edits whole channels).
Restricting to one node would score a different intervention than the one the
method actually proposed. Needs a vector-valued generalisation — accepting the
full intervened row `x_cf[t0]`, or a `(nodes, values)` pair — kept
backward-compatible with the existing callers and tests.

### 3. Naming: on flip-candidates this is **PS**, not PNS (R3)

`select_flip_candidates` keeps only instances the classifier currently puts in
the non-target class, so `Y_{x'} = 0` holds **by construction** on the model
side. Then

```
PNS_model = P(Y_x = 1 ∧ Y_{x'} = 0) = P(Y_x = 1) = validity
```

— the model side collapses into validity exactly, and the quantity being
estimated is `P(Y_x=1 | X=x', Y=0)`, which is the **probability of
sufficiency (PS)**, not PNS. Calling it PNS would misname it (R3). Two ways
out:

- **Minimal — `PS-gap`.** Flip-candidates only, named honestly. Cheap (reuses
  existing CFs). The model side is validity, so the *contribution* is the
  world side and the decomposition, which should be stated plainly rather than
  dressed up.
- **Full — genuine PNS.** Also generate CFs in the reverse direction (target
  → non-target) to estimate PN, then combine via
  `PNS = P(x,y)·PN + P(x',y')·PS`. Genuinely PNS, symmetric, ~2× CF
  generation cost.

> **PI decision 2026-07-30: full PNS (both directions).** PN is estimated on a
> complementary candidate set — instances the classifier puts in the *target*
> class, with CFs sought toward non-target — and combined with PS via the
> identity above. The name "PNS" is then earned rather than borrowed, and both
> `Δ` decompositions are reported per direction.
>
> Consequences to plan for:
> - **~2× CF generation.** On this CPU-only box the `full`+`full_nl` CF phase
>   already runs multi-hour; the reverse direction doubles it. Prove the metric
>   out at `smoke` scale first.
> - **A complementary selector is needed.** `select_flip_candidates` only
>   returns non-target instances; PN needs its mirror. Either generalise it
>   with a `from_class` argument or add a sibling — the former is preferable
>   (one code path, R9).
> - **PN may be weakly estimated.** The reverse direction inherits the same
>   horizon limit diagnosed on 2026-07-30: if an intervention at
>   `t0 ∈ {0.25T, 0.5T}` cannot move the outcome in either direction, PN
>   collapses toward 0 for structural reasons and the PNS combination is
>   dominated by the PS term. This is a *predicted* outcome to check, not a
>   reason to avoid PN — but it means `P(x,y)` and `P(x',y')` weights and the
>   PN/PS terms must all be reported separately, never just the combined PNS.

## Prior work (R3 — needs verification before any novelty claim)

Necessity/sufficiency for explanation is **not new**: LEWIS (Galhotra,
Pradhan & Salimi, 2021) and Watson et al. (2021) on necessity/sufficiency
explanations are the obvious antecedents. The defensible contribution here is
the *temporal* setting with *exact* (not bounded) abduction and the
model-vs-world decomposition — **not** PNS-for-explanation itself. To be
checked against `docs/references_verified.md` conventions before the claim
appears in writing.

## Decisions taken (2026-07-30)

1. **Full PNS**, both directions — see the boxed note above.
2. **Placement: Phase 07 auxiliary first**, alongside DYNOTEARS/iVAE, on one
   config. Phase 04's published numbers stay stable while the metric earns a
   track record; promotion to the main pipeline is a later, separate decision.
3. θ recovery: **(b)** recompute the median over concatenated splits (verified
   exact); **(c)** persisting it in the generator remains the durable fix.

## Build order

1. ✅ θ recovery verified on all 7 configs (`full`, `full_nl`, `smoke`,
   `smoke_nl`, `smoke_gaussian`, `smoke_nonmonotonic`, `smoke_regime`) —
   exact on every one.
2. ✅ `structural_counterfactual` takes vector-valued interventions,
   backward-compatible (scalars still work).
3. ✅ `select_flip_candidates(from_class=…)`.
4. ✅ Metric module + 10 adversarial tests (R6).
5. ✅ CARLA/`full` prediction confirmed.
6. ✅ Wired to Phase 07 (`--method pns`), run on `full` + `full_nl`.

7. ✅ **PN direction** — `--with-pn` generates the necessity-direction CFs
   (target → non-target, via `build_methods(..., target_class=0)` reusing
   Phase 03's registry through `importlib`), caches them under `cf_pn/`, and
   combines into genuine PNS. Run on `smoke`.

## PNS results (`smoke`, n=20, 2026-07-30)

| method | PNS_world | Δ_total (PS) | Δ_total (PN) |
|---|---|---|---|
| **PearlCARLA** | **0.90** | +0.10 | +0.05 |
| CARLA | 0.15 | +0.95 | +0.75 |
| CftsCOMTE / CftsCels / CftsWachter | 0.10 | +0.95 | +0.85 |
| CftsConfeti | 0.10 | +1.00 | +0.80 |

This **reverses** the PS-only reading. At this horizon PearlCARLA is by a
wide margin the most causally honest method: its interventions genuinely
produce the outcome change in the world (PN=0.95, PS=0.85) and its claims
almost match (gap ≤ 0.10). Every other method claims near-total efficacy
(A≈1.00) while the world delivers ~0.05–0.25.

**Read `Δ_trajectory` for PearlCARLA with care.** It is 0.00 in *both*
directions **by construction** — PearlCARLA emits a Pearl-semantics rollout
and the oracle is a Pearl rollout of that same intervention, so they coincide
tautologically (the same caveat as H3b's "faithful by construction"). What is
*not* tautological, and is the real result, is `C` — whether the world
actually delivers the outcome change. That is empirical, and it is high.

Contrast with `full`/`full_nl`, where PearlCARLA's interventions decay to
nothing before the outcome (2026-07-30 horizon finding) and it claims nothing.
Same method, opposite verdict, entirely explained by horizon.

### Outstanding

- **Full-scale runs (both directions) have been executed**; the numbers,
  verdicts and their consequences are recorded in this project's decision log
  and planning documents, which are kept out of version control. This file
  documents the metric's *design*, not its results.
- **Literature verification** of the LEWIS / Watson et al. positioning before
  any novelty claim reaches a manuscript.
- **Promotion to the main pipeline** (Phase 04) is a separate later decision;
  it stays auxiliary until it has a track record.
