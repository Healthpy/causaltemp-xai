# Stage 1: Mechanism abstraction + linear refactor (no behavior change)

**Goal**: Introduce a polymorphic `Mechanism` abstraction (`forward_numpy`/`forward_torch`) and refactor every `A @ x` call site onto it, with **zero observable change** to the frozen v0.1 LinearSCM-T benchmark.
**Dependencies**: None.

> ⚠️ This is the riskiest stage. Its success criterion is *behavior preservation*. Before refactoring, pin the current outputs with a **golden test**; the refactor is only correct if those values do not move.

---

## Steps

1. **Pin v0.1 behavior with a golden test (do this FIRST, before touching any code).**
   - File: `tests/test_golden_linear.py` (new)
   - **Capture the constants from current behavior**: run the *current* code once (interactively) to obtain the numbers, hardcode them, then write the test against the **new** `Mechanism` API. The frozen artifact is the **numeric constants**, not the call syntax — the test file legitimately uses the post-refactor API and must pass after the refactor. (The current code takes a `mechanisms` *list*; the refactored code takes a `mechanism` *object* — the same test can't literally run against both APIs, so don't claim "passes before and after"; claim "constants captured from current truth, asserted after refactor.")
   - Capture: the **full `X` array** (not just a few slices — a slice-only check can miss an RNG-order regression from the Stage 3 `_sample_graph` extraction), the labels, and — critically — `CFfaith(semantics="noiseless_rollout").score(...)` and `CFfaith(semantics="pearl_delta").score(...)` on a small fixed set of (x, x_cf) pairs (build one faithful CF via the existing `CARLARecourse` noiseless rollout, one retroactive CF, one identical CF). Hardcode the resulting hard/soft numbers as expected constants.
   - **Include at least one `L=2` case** in the CF-faith golden set so the partial-window / negative-lag boundary (see step 2) is actually exercised, not assumed. Do not edit the expected constants later.

2. **Create the `Mechanism` abstraction.**
   - File: `causaltemp_xai/benchmark/mechanisms.py` (new)
   - Define an abstract base / Protocol `Mechanism` with:
     - attributes `k: int`, `L: int`.
     - `forward_numpy(self, history: np.ndarray) -> np.ndarray`: given the lag window — shape `(L, k)` for a single step, or batched `(N, L, k)` — return the **deterministic next-step mean** (pre-noise), shape `(k,)` or `(N, k)`. History is ordered oldest→newest: `history[..., -1, :]` is lag 1, `history[..., -L, :]` is lag L.
     - `forward_torch(self, history: torch.Tensor) -> torch.Tensor`: same contract, differentiable, torch dtypes.
     - **Partial-window / boundary contract (preserve current behavior):** the current `cf_faith.py` forward loop guards `if lag_t >= 0` — i.e. a lag reaching before `t=0` contributes **zero**. The abstraction must replicate this. Define the contract as: **callers always pass exactly `L` rows**, zero-padding the oldest rows when fewer than `L` are available (so a missing lag contributes 0). This is a no-op for `L=1` (every config + every current cf_faith test), but it is a correctness landmine for `L>1` with an early `intervention_t` — which is why the Stage 1 golden adds an `L=2` case. Document this on the `Mechanism` base.
     - `state_dict(self) -> dict[str, np.ndarray | str | float]` and classmethod `from_state_dict(cls, d) -> Mechanism`: round-trippable serialization. Include a `"__type__"` discriminator (`"linear"` / `"mlp"`).
     - **npz-coercion contract (correctness landmine):** `state_dict` is persisted via `np.savez(**d)` and reloaded via `dict(np.load(...))` (Stage 1 step 7, Stage 5). `np.load` returns **every** value as an `np.ndarray` — including scalars and strings, which come back as 0-d arrays (e.g. `d["__type__"]` is `array("linear")`, `d["k"]` is `array(3)`). Therefore `from_state_dict` **must coerce** non-array fields (`str(d["__type__"])`, `int(d["k"])`, `float(d["gain"])`, …), and `mechanism_from_state_dict` must coerce before comparing (`str(d["__type__"]) == "linear"`). The existing linear loader sidestepped this by keying on key-*names*, not values — the `__type__`/scalar fields are new surface, so this must be handled from the start or it detonates at the Stage 5 data_io round-trip.
   - Add a module-level `mechanism_from_state_dict(d)` dispatcher keyed on `str(d["__type__"])`.

3. **Implement `LinearMechanism(Mechanism)`.**
   - File: `causaltemp_xai/benchmark/mechanisms.py`
   - Holds `A_list: list[np.ndarray]` (length L, each `(k, k)`).
   - `forward_numpy(history)` = `sum_l A_list[l] @ history[..., -(l+1), :]` matching the existing convention (`x_t = sum_l A_l @ x_{t-l-1}`). Support batched matmul via `np.einsum`/broadcasting so the generator's batched path stays vectorized.
   - `forward_torch` = the torch equivalent (`A @ x` per lag, summed).
   - `state_dict` stores `{"__type__": "linear", "A_0": ..., ..., "k": k, "L": L}`; `from_state_dict` restores lag order by sorting `A_<l>` keys (mirror current `data_io.load_dataset`).
   - **Backward-compat helper**: expose `A_list` as a public attribute so any leftover linear-only code can still reach the matrices.

4. **Refactor `LinearSCMT` to build and use a `LinearMechanism`.**
   - File: `causaltemp_xai/benchmark/generator.py`
   - `_build_graph_and_mechanisms` returns `(graph, LinearMechanism(A_list))` instead of `(graph, list)`.
   - The forward loop calls `self.mechanism.forward_numpy(window)` instead of the inline `@ A.T` sum. Keep batched-over-N vectorization (pass `(N, L, k)` window). Verify the `_stabilise` spectral-radius logic stays inside `LinearMechanism` construction or the generator (unchanged math).
   - `generate()` returns `{"X", "Y", "graph", "mechanism": <Mechanism>}`. **Contract change**: key `mechanisms` (list) → `mechanism` (object). Keep the rest of the dict identical.

5. **Refactor the library consumers onto the abstraction.**
   - `causaltemp_xai/metrics/cf_faith.py`: `score(..., mechanism)` (was `mechanisms`). Replace the inner `x_t_pred += A @ simulated[lag_t]` loop with `simulated[t] = mechanism.forward_numpy(window_ending_at(t-1))`. Keep the retroactive check, the rollout/pearl branch selection, and tol/scale exactly as-is for this stage (pearl stays linear-delta here — it is reformulated in Stage 4). Routing the pearl rollout (`target = x_cf - x_orig`) through `LinearMechanism.forward_numpy` still computes `sum_l A_l @ delta[lag]` — i.e. identical linear-delta behavior. Update `L = mechanism.L`.
   - `causaltemp_xai/methods/carla.py`: `_rollout` builds rows by `mechanism.forward_torch(stack of recent rows)` instead of `acc += A @ rows[lag]`. Rename the `generate(..., mechanisms)` / `generate_batch(..., mechanisms)` param to `mechanism` (note: `carla.py:113` and `:189`).
   - `causaltemp_xai/eval.py`: thread `mechanism` (was `mechanisms`) through `evaluate_method`, `_generate_batch`, `shift_vr`. Update the signature-introspection in `_generate_batch` (`eval.py:128`) from `if "graph" in params or "mechanisms" in params` to check `"mechanism"`.

6. **Sweep the experiment-harness consumers of the data contract (P1 — these break the moment step 4/5 lands).**
   The key rename (`data["mechanisms"]` → `data["mechanism"]`) and the carla param rename ripple beyond the library into `experiments/`. These are part of the *data contract*, not Stage-6 docs — fix them here or the tree won't run.
   - `experiments/run_all.py`:
     - `run_all.py:217` `graph, mech = data["graph"], data["mechanisms"]` → `data["mechanism"]`.
     - `run_all.py:74` has its **own duplicate** of the introspection: `if "graph" in params or "mechanisms" in params` → check `"mechanism"`.
     - Verify how `mech` reaches `evaluate_method` / `shift_vr` (positional vs keyword). If passed as `mechanisms=`, rename the keyword; positional needs no change.
   - `experiments/phenomenon_check.py`:
     - `phenomenon_check.py:67` `graph, mech = data["graph"], data["mechanisms"]` → `data["mechanism"]`. (Note: it builds `LinearSCMT` directly and calls `gen.generate()`, so it also depends on the `generate()` key rename.)
     - `phenomenon_check.py:88` `carla.generate_batch(X_sel, clf, graph, mech)` is positional → no change needed beyond `mech` now being a `Mechanism` object.

7. **Refactor `data_io.py` persistence to the `state_dict` contract.**
   - File: `causaltemp_xai/data_io.py`
   - Save: `np.savez(dest / "mechanism.npz", **mechanism.state_dict())` (replaces `mechanisms.npz` with the `A_l` convention; for linear the contents are equivalent + a `__type__` marker). Keep `graph.npy`, splits, `meta.json` as-is; record `mechanism_type` in `meta.json`.
   - Load: `out["mechanism"] = mechanism_from_state_dict(dict(np.load(...)))`. Keep returning `graph`.
   - **Note**: this changes the on-disk filename/shape. Datasets are gitignored & regenerated in CI (per the v0.1 decision), so no migration needed — but the rename must be reflected in all loaders.

8. **Update all existing tests to the new contract.**
   - `tests/test_generator.py`: `data["mechanism"]` is a `LinearMechanism`; assertions on `len(mechanisms)`/`A.shape` become assertions on `mechanism.L` and `mechanism.A_list[l].shape`. The persistence round-trip compares `mechanism.state_dict()`.
   - `tests/test_cf_faith.py`: **the fix is concentrated in one helper** — `_make_simple_scm` (`test_cf_faith.py:21-40`) currently returns a raw `list[np.ndarray]`; change it to return a `LinearMechanism` wrapping the sampled `A_l` list. Every `scorer.score(x_orig, x_cf, t, graph, mechanisms)` call then passes that object unchanged. Leave the manual CF-construction loops (e.g. `test_cf_faith.py:62-66`, which build raw arrays with their own `if t-lag >= 0` guard) as-is — they construct `x_cf`, not mechanisms.
   - `tests/test_eval.py`, `tests/test_methods.py`, `tests/test_intervention.py`: pass `mechanism` instead of `mechanisms` (most build it via `LinearSCMT(...).generate()["mechanism"]`, so the rename flows automatically). **Do not weaken assertions** — only adapt the plumbing. The locked `TestCFFaithPearl` / `TestCFFaithCompliant` expectations must still hold byte-for-byte.

---

## Verification

> **Run all `pytest`/`python` commands with `uv run --offline …`** (or `.venv/bin/python -m …`). Bare `uv run` re-resolves against a private index that 401s on `setuptools` and fails before running anything — see `resources/commands.md`.

- [ ] `uv run --offline pytest tests/test_golden_linear.py -v` passes (pinned v0.1 CF-faith numbers reproduced after refactor).
- [ ] `uv run --offline pytest -q` — full suite green, same count as before (82 tests) plus the golden test.
- [ ] Grep shows no remaining `@ A.T`, `A @ simulated`, `A @ rows` in `generator.py`/`cf_faith.py`/`carla.py` (all routed through `mechanism.forward_*`).
- [ ] `LinearMechanism.forward_numpy` and `forward_torch` agree to < 1e-6 on a random window.
- [ ] `state_dict` → `from_state_dict` round-trips a `LinearMechanism` to bit-identical `A_list`. **Also round-trip through `np.savez`→`np.load`** (not just in-memory): `np.load` returns every scalar/string field (`__type__`, `k`, `L`) as a 0-d array, so `from_state_dict` / `mechanism_from_state_dict` must coerce (`str(...)`/`int(...)`/`float(...)`) — see step 2.
- [ ] **End-to-end harness gate (no checkpoint required):** `uv run --offline python -m experiments.phenomenon_check --config smoke` runs to completion. Use this rather than `run_all.py` — `run_all.py:220` hard-requires a trained `lstm.pt` checkpoint that an autonomous run won't have, so it fails for reasons unrelated to the refactor. `phenomenon_check.py` self-trains a smoke classifier (`_train_smoke_classifier`) and exercises the full `generate → mechanism → carla/cf_faith` path. (Optionally also confirm `run_all.py --config smoke` if a checkpoint exists.)

---

## Commit

`refactor(scm): introduce polymorphic Mechanism abstraction (linear path preserved)`
