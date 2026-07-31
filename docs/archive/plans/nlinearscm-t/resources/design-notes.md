# Design Notes — NlinearSCM-T

Condensed, citation-grounded rationale behind the implementation choices. Source: web
research synthesis (2026-06-21), Rhino / CausalDynamics / ANM literature.

## Functional form (additive-noise per-node MLP transition)

```
x_t^i  =  decay_i · x_{t-1}^i  +  gain · tanh( MLP_i( masked lagged parents_{<t} ) )  +  eps_t^i
```

- Per-node 2-layer MLP over **masked** lagged parents (graph-respecting), à la **Rhino**
  (embed-then-aggregate; arXiv 2210.14706, Eq. 6–7) and **CausalDynamics** (random nonlinear
  couplings; arXiv 2505.16620).
- **Additive non-Gaussian noise** (Laplace/uniform, same as LinearSCM-T). This is the
  defining property — it makes the model an **Additive Noise Model** (Hoyer et al., NeurIPS
  2008), which is identifiable and, crucially, **counterfactually identifiable**
  (Nasr-Esfahany et al., *Bijective Causal Models*, ICML 2023).

## Why additive noise (the linchpin)

With additive noise, Pearl's **abduction is exact subtraction**:
`eps[t] = x[t] − f(parents[t])`. This makes:
1. the **noiseless-rollout** CF-faith check provably correct;
2. the **oracle structural-CF** closed-form (abduct → intervene → re-roll reusing eps);
3. the **Pearl-delta** metric generalizable to nonlinear (abduction-action-prediction),
   reducing *exactly* to the locked linear `delta[t]=A@delta[t-l]` when `f` is linear.

Boundary (document as scope limit): non-additive/heteroscedastic noise needs an invertible
(monotone-in-noise) `g_i` for exact abduction; non-invertible mixing breaks it.

## Stability over T = 50–200 (no spectral-radius analogue for nonlinear)

Combine (Miller & Hardt, *Stable Recurrent Models*, ICLR 2019; Lipschitz-RNN):
- **tanh** output → hard per-variable bound;
- **leaky decay** `decay_i · x_{t-1}` (∈[0.3,0.8]) → contraction analogue of VAR spectral radius;
- **spectral-norm cap** (≤0.9) on MLP weights → Lipschitz < 1 for the nonlinear branch;
- **burn-in** + **divergence rejection/resample** as empirical guardrails.

## Init for stable-but-genuinely-nonlinear dynamics

- `std = 0.7 · √(1/fan_in)`, bias 0 (contraction, below Xavier);
- scale inputs so pre-activations hit tanh's **curved** region (~N(0,1.5)) — otherwise tanh
  ≈ identity and the model is linear-in-disguise. Validate via a linear-fit residual test.

## Two axes the plan deliberately separates

| | nonlinear **transition** (built) | nonlinear **mixing** (deferred, Backlog #1) |
|---|---|---|
| form | `x_t = f(parents)+eps` | `x = g(z)` over latents |
| buys | counterfactual tractability + ANM direction-identifiability | latent-factor identifiability (iVAE/CITRIS) |
| needs | additive noise | invertible `g` + auxiliary labels |

The plan's phrase "MLP mechanisms **and** nonlinear mixing" conflates these; we ship the
transition axis and document mixing as a separately-toggled, invertible future layer.

## Key citations

- Rhino — arXiv 2210.14706
- CausalDynamics — arXiv 2505.16620
- Hoyer et al., Additive Noise Models — NeurIPS 2008
- Nasr-Esfahany et al., Counterfactual Identifiability of Bijective Causal Models — ICML 2023
- Miller & Hardt, Stable Recurrent Models — ICLR 2019
- Khemakhem et al. (iVAE) / Lippe et al. (CITRIS) — nonlinear-mixing identifiability
