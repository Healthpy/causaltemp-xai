# Stage 3: TCNClassifier wrapper & training

**Goal**: Add a sklearn-style `TCNClassifier` (fit with val early-stopping, predict / predict_proba, a differentiable forward for gradient-based CFs, checkpoint save/load), train it on the locked config to >90% test accuracy, and freeze the checkpoint.
**Dependencies**: Stage 2

---

## Steps

1. Add the classifier wrapper.
   - File: `causaltemp_xai/classifiers/tcn.py` (extend; keep `TCN` module + `train_tcn`)
   - Implement `class TCNClassifier`:
     - `__init__(config-ish hyperparams: n_inputs/k, n_classes=2, n_levels, n_channels, kernel_size, dropout, lr, batch_size, max_epochs, patience=10, target_acc=0.90, device=None, seed)`.
     - `fit(X_train, y_train, X_val=None, y_val=None)`: trains the underlying `TCN`; **early stopping on val loss (patience=10)** per MVP §WP1; restores best-val weights. Accepts `(N,T,k)` and handles the `(N,T,k)→(N,k,T)` permute internally (matching `TCN.forward`).
     - `predict(X) -> (N,)` int labels; `predict_proba(X) -> (N, n_classes)` softmax.
     - `torch_logits(x_tensor)`: a differentiable forward accepting a `(N,k,T)` (or `(N,T,k)` + internal permute) **torch tensor with grad**, returning logits — this is the hook CF methods optimise through. Document the exact expected input layout in the docstring (CF methods in Stage 4 depend on it).
     - `save(path)` / `load(path)` (torch state_dict + hyperparams).
   - Update `classifiers/__init__.py` to export `TCN, train_tcn, TCNClassifier`.
   - Add a `__main__` CLI: `--config {smoke,full} --train` → loads dataset (Stage 2), fits, prints train/val/test accuracy, saves checkpoint to `data/linearscm_t/<config>/tcn.pt`.

2. Train on the locked config.
   - Run smoke first to validate the pipeline, then `full`.
   - Target ≥0.90 test accuracy. If unreached after a reasonable hyperparam attempt, relax to ≥0.85 and **document the gap** in the index Issues section (MVP risk table sanctions this — CF-faith is classifier-agnostic).
   - Freeze: the saved `tcn.pt` is the single checkpoint used by all later stages. Do not retrain after Stage 4 begins.

3. Add classifier tests.
   - File: `tests/test_classifier.py` (new)
   - `TCNClassifier` overfits a tiny synthetic set to ≥0.9 train acc in few epochs; `predict`/`predict_proba` shapes and proba-sums-to-1; `save`/`load` round-trip yields identical predictions; `torch_logits` returns a grad-enabled tensor and `.backward()` populates `x.grad`.
   - **Shape-contract test (prevents layout bugs across modules):** standardise on `(T,k)` at every public boundary except inside `TCN` (which wants `(N,k,T)`). Add a round-trip test that takes one generator instance `(T,k)`, runs it through `TCNClassifier.predict` / `predict_proba` / `torch_logits` and through `CFfaith.score` **without any manual transpose at the call site** — the permute lives only inside `TCNClassifier`. Assert all accept `(T,k)`/`(N,T,k)` directly.

---

## Verification

- [ ] `uv run python -m causaltemp_xai.classifiers.tcn --config smoke --train` runs end-to-end and prints test accuracy.
- [ ] On `full`: test accuracy ≥0.90 (or ≥0.85 documented) logged to `meta`/stdout; `data/linearscm_t/full/tcn.pt` exists.
- [ ] `uv run pytest tests/test_classifier.py -q` passes, including the gradient-flow test.

---

## Commit

`feat(classifier): add TCNClassifier wrapper, train to target accuracy, freeze checkpoint`
