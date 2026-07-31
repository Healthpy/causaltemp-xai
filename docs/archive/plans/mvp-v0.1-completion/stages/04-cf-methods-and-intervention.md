# Stage 4: CF methods & intervention-time rule

**Goal**: Implement the three CF generators (Wachter, DiCE, CARLA-causal) against the `TCNClassifier`, and add the shared `derive_intervention_t` rule that makes CF-faith comparable across methods.
**Dependencies**: Stage 3

> **Propagation convention (do NOT "fix" this).** The generator uses `X @ A.T`
> (batch), `CFfaith` uses `A @ x` (per-sample), and for a vector these are
> **identical** (`x @ A.T == A @ x`). All three — generator, `CFfaith`, and the
> CARLA propagation below — are already consistent. Implement CARLA's rollout as
> `x_t = Σ_l A_l @ x_{t-l}` (per-sample) to match `cf_faith.py:114`. Introducing a
> transpose here to "align" with the generator would create a real bug.

> **Stage size note.** This is the heaviest stage (intervention util + 3 methods).
> If implementation grows large, it is acceptable to split CARLA into its own
> follow-up commit (`feat(methods): CARLA-causal recourse`) while keeping the same
> stage file — but keep Wachter + DiCE + intervention in the first commit.

---

## Steps

1. Add the intervention-time utility (the experiment's hinge — see index "crux").
   - File: `causaltemp_xai/methods/intervention.py` (new)
   - `derive_intervention_t(x, x_cf, tol=1e-6) -> int`: smallest `t` with `max_j |x_cf[t,j]−x[t,j]| > tol`; return `T-1` if none differ. Exact spec in [resources/configs.md](../resources/configs.md#intervention-time-convention).
   - This is applied uniformly to every method's CF before CF-faith scoring (Stage 5). Add focused unit tests in `tests/test_intervention.py`.

2. Implement Wachter (from scratch, torch).
   - File: `causaltemp_xai/methods/wachter.py`
   - Replace the stub body. Minimise `L(cf) = lam · CE(model.torch_logits(cf), target) + ||cf − x||_2²` via Adam on a `cf` leaf tensor initialised at `x`. Early-stop when predicted class == `target_class`. Increase `lam` (or re-init) if no flip within `n_steps`.
   - Add `generate_batch(X) -> (N,T,k)` looping `generate`. Keep the documented `__init__` signature but allow passing the classifier (decide: pass `model` to `generate(x, model)` per the stub's signature; `model` is the `TCNClassifier`, methods call `model.torch_logits`). Keep the stub's `(x, model)` interface — do **not** adopt run_all.py's `WachterCF(predict_fn)` form.

3. Implement DiCE via the official library (user-selected), with fallback.
   - File: `causaltemp_xai/methods/dice.py`
   - Add `dice-ml` via `uv add dice-ml`. Wrap it via the **PyTorch gradient path**: build a small `nn.Module` adapter that accepts a 2D `(N, T·k)` tensor, reshapes to `(N, T, k)`, and calls `TCNClassifier.torch_logits`; register it with `dice_ml.Model(model=..., backend="PYT", func=...)` and a `dice_ml.Data` built from a pandas DataFrame whose columns are the `T·k` flattened features (all `continuous_features`, outcome = label). Use `method="gradient"`. Reshape returned CFs back to `(n_cfs, T, k)`.
   - **Decision gate (not open-ended):** if this dice-ml path does not return correctly-shaped, label-flipping CFs on the **smoke** config within one focused attempt, **switch to the from-scratch fallback** and record the deviation in the index Decisions section. Do not iterate repeatedly on dice-ml internals.
   - **Fallback:** the from-scratch DPP-diversity optimiser already specified in the stub docstring (`-logdet K` diversity + proximity + prediction loss, joint Adam over `n_cfs` candidates). The MVP risk table explicitly permits this deviation.
   - Expose the same `generate(x, model)` / `generate_batch(X)` interface; for the harness, select 1 CF per instance.

4. Implement CARLA-causal recourse (the high-CF-faith comparison point). **Specify precisely — this is the most novel piece; do not improvise.**
   - File: `causaltemp_xai/methods/carla.py`
   - Signature: `generate(x, model, graph, mechanisms)` + `generate_batch`. `x` is `(T,k)`; `graph`/`mechanisms` come from the dataset (Stage 2).
   - **(a) Intervention point.** Do not search over all timesteps. Use a small fixed candidate set of *early* intervention timesteps, e.g. `t0 ∈ {T//4, T//2}` (configurable). For each candidate, run the optimisation below; keep the CF with the best (flip achieved, lowest proximity) outcome.
   - **(b) Free variables.** Only the values at `x[t0]` on actionable/parent variables are free parameters (a leaf tensor initialised at `x[t0]`). Everything before `t0` stays exactly equal to `x` (guarantees zero retroactive change → passes the `CFfaith` retroactive check).
   - **(c) Differentiable deterministic rollout.** Construct the post-`t0` trajectory as the **noiseless** VAR rollout in torch so gradients flow from the classifier back to the free `x[t0]` values:
     `x_cf[t] = Σ_l A_l @ x_cf[t-l]` for `t > t0`, with `A_l = mechanisms[l]` (per-sample `A @ x`, matching `cf_faith.py:114`). This is the exact operation `CFfaith` re-simulates, so a correctly-built CARLA CF scores **hard=1 by construction** (only `x[t0]` is set; all later timesteps are determined by the rollout — never re-noise them).
   - **(d) Objective.** Minimise `lam_pred · CE(model.torch_logits(x_cf), target) + lam_prox · ||x_cf[t0] − x[t0]||²`, optimising only the free `x[t0]` values; the classifier sees the fully propagated `x_cf`. Because the label lives at `x[-1,0]` (generator.py:118), the gradient must reach the final timestep through the rollout — that is why the rollout must be differentiable, not a post-hoc numpy step.
   - This method, by construction, has no retroactive change and SCM-consistent propagation → high CF-faith, contrasting Wachter/DiCE.

5. Update `methods/__init__.py` to export the implemented classes + `derive_intervention_t`. Decide on a consistent constructor: keep the stubs' parameter-first `__init__` (target_class, lams, lr, n_steps) and `generate(x, model, …)`.

---

## Verification

- [ ] `uv run pytest tests/test_intervention.py -q` passes (first-changed-timestep logic, all-equal → T-1, single-timestep change).
- [ ] `tests/test_methods.py` (new): on the smoke config + trained smoke TCN, each method returns CFs of shape `(T,k)` (or `(n_cfs,T,k)` for DiCE) with finite values; Wachter flips the label on ≥1 example.
- [ ] CARLA-causal output has zero retroactive change before its intervention timestep (assert in test).
- [ ] If the DiCE fallback was used, the index Decisions section records it.

---

## Commit

`feat(methods): implement Wachter, DiCE, CARLA-causal recourse + intervention-time rule`
