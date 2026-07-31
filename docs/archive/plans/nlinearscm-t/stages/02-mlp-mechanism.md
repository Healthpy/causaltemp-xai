# Stage 2: MLPMechanism implementation

**Goal**: Implement `MLPMechanism` — a per-node, additive-noise MLP transition with bounded, contractive dynamics — implementing the `Mechanism` interface in both numpy and torch.
**Dependencies**: Stage 1 (the `Mechanism` base + `state_dict` contract).

---

## Background (read before implementing)

The deterministic next-step mean for node `i` is:

```
mean_i(history)  =  decay_i · history[-1, i]  +  gain · tanh( MLP_i( masked_lagged_parents_i ) )
```

- `masked_lagged_parents_i` = the lagged parent values of node `i`, selected by the graph mask `graph[i, :, :]` (shape `(k, L)`), flattened over (parent, lag). Non-parents are zeroed so the MLP only sees true causal inputs.
- `MLP_i`: a small 2-layer MLP (input = `k·L` masked lags, hidden width `H`, output 1), per node. Stack the `k` node MLPs into batched weight tensors `W1 (k, H, k*L)`, `b1 (k, H)`, `W2 (k, 1, H)`, `b2 (k, 1)` so forward is a single batched op (no Python loop over nodes).
- Noise `eps` is added by the **generator**, not the mechanism — `forward_*` returns the pre-noise mean (mirrors `LinearMechanism`).

**Stability (from research — Rhino / CausalDynamics / Stable Recurrent Models):**
- `tanh` output bound + `decay_i ∈ [0.3, 0.8]` leaky term + `gain ∈ [0.5, 1.0]`.
- **Spectral-norm cap** on `W1`, `W2` so each layer's spectral norm ≤ `s` (default `s ≈ 0.9`), giving Lipschitz < 1 for the nonlinear branch. Apply at construction (one-shot rescale of each per-node weight matrix by `min(1, s / spectral_norm)`), analogous to `generator._stabilise`.
- **Init**: `std = 0.7 · √(1/fan_in)`, bias 0. Scale the masked input (divide by `√(n_active_parents)` or normalize) so pre-activations land in tanh's curved region — guarantees genuine nonlinearity, not linear-in-disguise.

---

## Steps

1. **Add `MLPMechanism(Mechanism)` to `causaltemp_xai/benchmark/mechanisms.py`.**
   - Constructor params: `graph (k, k, L)`, `hidden (int)`, `decay (np.ndarray (k,))`, `gain (float)`, `activation="tanh"`, plus the weight tensors (`W1,b1,W2,b2`). Provide a classmethod `random(graph, hidden, rng, decay_range, gain, spectral_cap, init_gain)` that samples + stabilizes weights from an `np.random.default_rng`.
   - Store `k`, `L` from `graph.shape`. Precompute the boolean parent mask per node `(k, k*L)` from `graph`.

2. **Implement `forward_numpy(history)`.**
   - Accept `(L, k)` or `(N, L, k)`. Build the masked lag vector per node: gather `history` into `(N, k, k*L)` (for each node, its `k*L` lagged inputs), apply the parent mask, scale.
   - **Input scaling — guard zero-parent nodes:** scale each node's masked input by `1/√(max(n_active_parents_i, 1))`. Under Bernoulli sparsity a node can receive **0 incoming edges**, so `n_active_parents_i = 0` is reachable and an unguarded `1/√(n)` divides by zero. A 0-parent node then correctly reduces to pure decay (`mean_i = decay_i · x_{t-1}^i`, since its masked input — and hence `MLP_i` — is all-zero → `tanh(0)=0`). Precompute the per-node active-parent count once in the constructor.
   - Batched MLP: `h = tanh_or_act(einsum('khd,nkd->nkh', W1, masked) + b1)`, then **add the output bias before squeezing the singleton output dim** — `out = (einsum('koh,nkh->nko', W2, h) + b2)[..., 0]`. (Writing `einsum(...)[...,0] + b2` is wrong: after `[...,0]` the result is `(N,k)` and `b2` is `(k,1)`, which broadcasts as `(N,k)+(k,1)` → mis-shaped unless `N==k`.)
   - Return `decay * history[..., -1, :] + gain * tanh(out)`. Shapes `(k,)` / `(N, k)`.

3. **Implement `forward_torch(history)`.**
   - Mirror `forward_numpy` with torch ops; keep it differentiable w.r.t. `history` (needed later by CARLA, Backlog #2) and free of in-place ops on leaf tensors. Cache weights as torch tensors (lazily, dtype/device from `history`).

4. **Implement `state_dict` / `from_state_dict`.**
   - Serialize `__type__="mlp"`, `W1,b1,W2,b2`, `decay`, scalars (`gain`, `hidden`, `k`, `L`, `activation`), and the parent mask (or the `graph` to rederive it). Flatten into the flat `npz`-friendly dict the Stage 1 dispatcher expects.

5. **Write `tests/test_mechanisms.py`.**
   - `forward_numpy` vs `forward_torch` agree < 1e-5 on random windows (single + batched).
   - **Contractivity / boundedness**: iterate `x_{t+1} = mechanism.forward_numpy(window)` (no noise) from random init for 300 steps — assert finite and bounded (e.g. `max|x| < C`).
   - **Nonlinearity**: fit the best linear map from input window → output via least squares; assert the residual is non-trivial (mechanism is provably not linear) — but also assert it's not saturated/constant (output variance > threshold). Build the test graph so ≥1 node has ≥2 active parents (don't rely on a sparse random draw giving every node parents — a 0-parent node is pure decay and would dilute the nonlinearity signal).
   - **Masking**: perturbing a non-parent lagged input does not change the output (mask respected). Pick a node with ≥1 parent *and* ≥1 non-parent for this assertion.
   - **Serialization (two round-trips)**: (a) in-memory — `from_state_dict(m.state_dict())` reproduces `forward_numpy` outputs bit-for-bit; (b) **through npz** — `np.savez(tmp, **m.state_dict())`, reload via `mechanism_from_state_dict(dict(np.load(tmp)))`, and assert forward outputs still bit-identical. The npz hop is what exercises the 0-d-array coercion contract (Stage 1 step 2); the in-memory hop alone does **not** catch it and would let a coercion bug slip to Stage 5.
   - **LinearMechanism still passes its Stage-1 tests** (no cross-contamination).

---

## Verification

> Run with `uv run --offline …` (see `resources/commands.md`).

- [ ] `uv run --offline pytest tests/test_mechanisms.py -v` green.
- [ ] numpy/torch parity < 1e-5; serialization round-trip exact **both in-memory and through npz**.
- [ ] 300-step noiseless rollout stays finite & bounded across ≥10 random seeds.
- [ ] Linear-fit residual confirms nonlinearity; output variance confirms non-saturation.
- [ ] Full suite (`uv run --offline pytest -q`) still green.

---

## Commit

`feat(scm): add MLPMechanism (additive-noise, contractive nonlinear transition)`
