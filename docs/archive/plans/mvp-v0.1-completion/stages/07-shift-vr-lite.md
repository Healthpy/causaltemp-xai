# Stage 7: Shift-VR-lite (Axis D)

**Goal**: Generate a shifted environment (same SCM, uniform noise), re-evaluate validity and CF-faith on it, and report Shift-VR = fraction of originally-valid CFs that remain valid under shift.
**Dependencies**: Stages 4–5 (methods + eval pipeline)

---

## Steps

1. Define the shifted environment.
   - Use the convention in [resources/configs.md](../resources/configs.md#shift-vr-lite-environments): identical `graph`, `mechanisms`, and `seed` as the base FULL config, but `noise_type="uniform"`. This isolates a distributional shift in the innovations while holding causal structure fixed.
   - Generate `E_shift` test split via `data_io` (reuse the same SCM object so graph/mechanisms are bit-identical — extend `data_io` to support a `--shift-noise` flag or a derived config that overrides only `noise_type`).

2. Implement Shift-VR-lite. **Definition is pinned (see [resources/configs.md](../resources/configs.md#shift-vr-lite-environments)) — do not reinterpret.**
   - File: `causaltemp_xai/eval.py` (extend)
   - Protocol: keep the **frozen base classifier** (no retraining). For each method, **generate fresh CFs for `E_shift`'s test inputs** and evaluate validity on them, exactly as was done for `E_base`.
   - `shift_vr(model, methods, X_base_test, X_shift_test, graph, mechanisms, target_class) -> {method: ratio}` where `ratio = validity(E_shift) / validity(E_base)` per method (clip denominator-zero to a documented sentinel, e.g. `nan`).
   - **Do NOT** use the "fraction of the *same* CFs that remain valid" framing: under a single frozen classifier a fixed CF array's validity cannot change between environments, so that framing is incoherent. The validity difference comes from generating recourse for shifted inputs.

3. Tests.
   - File: `tests/test_shift.py` (new)
   - `E_shift` has the same `graph`/`mechanisms` as base (assert array-equal) but a different noise marginal (KS test distinguishes). `shift_vr` returns a value in [0,1].

---

## Verification

- [ ] `uv run pytest tests/test_shift.py -q` passes.
- [ ] Shifted dataset generated with identical SCM structure (graph/mechanisms equal) and different noise distribution.
- [ ] `shift_vr` produces a retention ratio per method on the smoke config (CFs generated for shifted test inputs; same frozen classifier).

---

## Commit

`feat(robustness): add Shift-VR-lite shifted environment and validity-retention metric`
