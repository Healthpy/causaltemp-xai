# Stage 6: Integrated-Gradients attribution foil (WP3)

**Goal**: Add an Integrated-Gradients attribution baseline against the TCN plus deletion/insertion curves, to demonstrate that high attribution-saliency quality coexists with low CF-faith — the central empirical gap.
**Dependencies**: Stage 3 (trained classifier); can run in parallel with Stages 4–5 conceptually but commit after Stage 5.

---

## Steps

1. Implement Integrated Gradients.
   - File: `causaltemp_xai/attribution/integrated_gradients.py` (new) + `causaltemp_xai/attribution/__init__.py`
   - `integrated_gradients(model, x, target_class, baseline=None, steps=50) -> (T,k)` attribution map. Use the `TCNClassifier.torch_logits` hook from Stage 3. Default baseline = zeros (document choice). Optionally use `captum` (`uv add captum`) instead of hand-rolling — either is acceptable; document which.

2. Deletion / insertion curves (lightweight saliency-quality proxy, MVP §WP3).
   - File: `causaltemp_xai/attribution/perturbation_curves.py` (new) or same module
   - `deletion_curve` / `insertion_curve`: rank timesteps (or (t,feature) cells) by |attribution|, progressively mask (deletion) or reveal from baseline (insertion), record model confidence on `target_class`, return curve + AUC.
   - Keep it intentionally lightweight — a foil, not a full attribution benchmark.

3. Tests.
   - File: `tests/test_attribution.py` (new)
   - IG completeness sanity (hard assertion): `sum(IG) ≈ logit_target(x) − logit_target(baseline)` within tolerance (the IG axiom), attributing the **target-class logit** (a scalar). Attribution shape == `(T,k)`.
   - Deletion/insertion: assert only that both curves are **finite** and have the correct length (n_steps+1) and that endpoints are sensible (deletion starts at full confidence, insertion ends at full confidence). Treat the `deletion AUC ≤ insertion AUC` ordering as a **logged diagnostic, not a hard assertion** — it need not hold on a tiny smoke model and would make CI flaky.

4. (Optional, time-permitting) TimeSHAP — explicitly out of MVP scope unless trivial; leave a documented TODO. Do not block the stage on it.

---

## Verification

- [ ] `uv run pytest tests/test_attribution.py -q` passes including the IG completeness check.
- [ ] Running IG on a few smoke-config test instances yields finite `(T,k)` maps and deletion/insertion AUCs.

---

## Commit

`feat(attribution): integrated gradients + deletion/insertion curves (WP3 foil)`
