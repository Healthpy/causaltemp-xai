"""Tests for the Mechanism abstraction (Stage 1: LinearMechanism)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from causaltemp_xai.benchmarks.mechanisms import (
    LinearMechanism,
    MLPMechanism,
    mechanism_from_state_dict,
)


def _random_linear(k=3, L=2, seed=0):
    rng = np.random.default_rng(seed)
    return LinearMechanism([rng.uniform(-0.3, 0.3, (k, k)) for _ in range(L)])


def _dense_graph(k=4, L=2):
    """Fully connected lagged graph (every node ≥2 active parents, ≥1 non-parent
    only if a self/missing edge exists — see callers that craft sparser graphs)."""
    graph = np.ones((k, k, L), dtype=float)
    # Drop lag-1 self loops (mirrors generator convention) so each node still has
    # many parents but at least the diagonal differs.
    for i in range(k):
        graph[i, i, 0] = 0.0
    return graph


def _random_mlp(graph=None, hidden=8, seed=0, **kw):
    if graph is None:
        graph = _dense_graph()
    rng = np.random.default_rng(seed)
    return MLPMechanism.random(graph, hidden=hidden, rng=rng, **kw)


class TestLinearForward:
    def test_single_window_matches_manual_sum(self):
        mech = _random_linear(k=4, L=2, seed=1)
        rng = np.random.default_rng(2)
        history = rng.normal(size=(mech.L, mech.k))  # (L, k) oldest->newest
        expected = np.zeros(mech.k)
        for l, A in enumerate(mech.A_list):
            expected += A @ history[-(l + 1)]
        np.testing.assert_array_equal(mech.forward_numpy(history), expected)

    def test_batched_matches_per_sample(self):
        mech = _random_linear(k=3, L=2, seed=3)
        rng = np.random.default_rng(4)
        batch = rng.normal(size=(7, mech.L, mech.k))
        out = mech.forward_numpy(batch)
        assert out.shape == (7, mech.k)
        for n in range(7):
            # Batched (mat @ A.T) vs single (A @ vec) agree up to float rounding.
            np.testing.assert_allclose(out[n], mech.forward_numpy(batch[n]), rtol=1e-12)

    def test_numpy_torch_agree(self):
        mech = _random_linear(k=5, L=3, seed=5)
        rng = np.random.default_rng(6)
        history = rng.normal(size=(mech.L, mech.k))
        np_out = mech.forward_numpy(history)
        torch_out = mech.forward_torch(torch.as_tensor(history)).numpy()
        assert np.max(np.abs(np_out - torch_out)) < 1e-6

    def test_zero_padding_is_noop_contribution(self):
        """A zero-padded (missing) lag row contributes nothing."""
        mech = _random_linear(k=3, L=2, seed=7)
        rng = np.random.default_rng(8)
        # Only lag-1 present (oldest row zero) → only A_0 contributes.
        history = np.zeros((mech.L, mech.k))
        history[-1] = rng.normal(size=mech.k)
        expected = mech.A_list[0] @ history[-1]
        np.testing.assert_array_equal(mech.forward_numpy(history), expected)


class TestLinearSerialization:
    def test_in_memory_round_trip(self):
        mech = _random_linear(k=4, L=2, seed=9)
        restored = mechanism_from_state_dict(mech.state_dict())
        assert restored.L == mech.L and restored.k == mech.k
        for A_a, A_b in zip(mech.A_list, restored.A_list):
            np.testing.assert_array_equal(A_a, A_b)

    def test_npz_round_trip_coerces_scalars(self, tmp_path):
        """Through np.savez/np.load, __type__/k/L come back as 0-d arrays;
        the loader must coerce them."""
        mech = _random_linear(k=3, L=2, seed=10)
        path = tmp_path / "mechanism.npz"
        np.savez(path, **mech.state_dict())
        with np.load(path) as loaded:
            restored = mechanism_from_state_dict(dict(loaded))
        assert isinstance(restored, LinearMechanism)
        assert restored.L == mech.L and restored.k == mech.k
        for A_a, A_b in zip(mech.A_list, restored.A_list):
            np.testing.assert_array_equal(A_a, A_b)

    def test_type_discriminator(self):
        mech = _random_linear()
        assert mech.state_dict()["__type__"] == "linear"


class TestMLPForward:
    def test_numpy_torch_agree_single(self):
        mech = _random_mlp(hidden=8, seed=11)
        rng = np.random.default_rng(12)
        history = rng.normal(size=(mech.L, mech.k))
        np_out = mech.forward_numpy(history)
        torch_out = mech.forward_torch(torch.as_tensor(history)).numpy()
        assert np_out.shape == (mech.k,)
        assert np.max(np.abs(np_out - torch_out)) < 1e-5

    def test_numpy_torch_agree_batched(self):
        mech = _random_mlp(hidden=16, seed=13)
        rng = np.random.default_rng(14)
        batch = rng.normal(size=(9, mech.L, mech.k))
        np_out = mech.forward_numpy(batch)
        torch_out = mech.forward_torch(torch.as_tensor(batch)).numpy()
        assert np_out.shape == (9, mech.k)
        assert np.max(np.abs(np_out - torch_out)) < 1e-5

    def test_batched_matches_per_sample(self):
        mech = _random_mlp(hidden=8, seed=15)
        rng = np.random.default_rng(16)
        batch = rng.normal(size=(5, mech.L, mech.k))
        out = mech.forward_numpy(batch)
        for n in range(5):
            np.testing.assert_allclose(out[n], mech.forward_numpy(batch[n]), rtol=1e-12)

    def test_forward_torch_is_differentiable(self):
        """forward_torch must be autograd-safe w.r.t. history (needed by CARLA)."""
        mech = _random_mlp(hidden=8, seed=17)
        rng = np.random.default_rng(18)
        history = torch.as_tensor(rng.normal(size=(mech.L, mech.k)))
        history.requires_grad_(True)
        out = mech.forward_torch(history)
        out.sum().backward()
        assert history.grad is not None
        assert torch.isfinite(history.grad).all()
        assert history.grad.shape == history.shape

    def test_contractive_boundedness(self):
        """300-step noiseless rollout stays finite and bounded across seeds."""
        graph = _dense_graph(k=5, L=2)
        for seed in range(10):
            mech = _random_mlp(graph=graph, hidden=8, seed=seed)
            window = np.random.default_rng(1000 + seed).normal(size=(mech.L, mech.k))
            for _ in range(300):
                nxt = mech.forward_numpy(window)
                window = np.vstack([window[1:], nxt[None, :]])
                assert np.all(np.isfinite(window))
            assert np.max(np.abs(window)) < 100.0

    def test_nonlinearity_and_non_saturation(self):
        """Best linear fit leaves non-trivial residual; output is not constant."""
        graph = _dense_graph(k=4, L=2)
        # Confirm the test graph gives ≥1 node ≥2 active parents.
        assert (graph.reshape(graph.shape[0], -1).sum(axis=1) >= 2).any()
        mech = _random_mlp(graph=graph, hidden=16, seed=21)
        rng = np.random.default_rng(22)
        # Scale inputs so pre-activations reach tanh's curved region.
        windows = rng.normal(scale=1.5, size=(400, mech.L, mech.k))
        Y = mech.forward_numpy(windows)
        Xflat = windows.reshape(400, -1)
        Xd = np.hstack([Xflat, np.ones((400, 1))])  # design + bias
        coef, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
        resid = Y - Xd @ coef
        rel_resid = np.linalg.norm(resid) / (np.linalg.norm(Y - Y.mean(0)) + 1e-12)
        assert rel_resid > 1e-3, f"mechanism is linear-in-disguise (rel_resid={rel_resid})"
        assert Y.var(axis=0).max() > 1e-4, "output saturated/constant"

    def test_masking_respected(self):
        """Perturbing a non-parent lagged input does not change the output."""
        k, L = 4, 2
        # Craft a graph where node 0 has parent (var 1, lag 1) but NOT (var 2, lag 1).
        graph = np.zeros((k, k, L), dtype=float)
        graph[0, 1, 0] = 1.0
        graph[0, 3, 1] = 1.0
        # give other nodes some parents so the mechanism is well-formed
        graph[1, 2, 0] = 1.0
        graph[2, 0, 1] = 1.0
        mech = _random_mlp(graph=graph, hidden=8, seed=23)
        rng = np.random.default_rng(24)
        history = rng.normal(size=(L, k))
        base = mech.forward_numpy(history)
        # var 2 at lag 1 is a non-parent of node 0 → perturb it.
        perturbed = history.copy()
        perturbed[-1, 2] += 5.0
        out = mech.forward_numpy(perturbed)
        assert abs(out[0] - base[0]) < 1e-12


class TestMLPSerialization:
    def test_in_memory_round_trip(self):
        mech = _random_mlp(hidden=8, seed=31)
        restored = mechanism_from_state_dict(mech.state_dict())
        assert isinstance(restored, MLPMechanism)
        rng = np.random.default_rng(32)
        history = rng.normal(size=(mech.L, mech.k))
        np.testing.assert_array_equal(mech.forward_numpy(history), restored.forward_numpy(history))

    def test_npz_round_trip_coerces_scalars(self, tmp_path):
        """Through np.savez/np.load, scalars/strings come back as 0-d arrays;
        the loader must coerce them and forward outputs stay bit-identical."""
        mech = _random_mlp(hidden=8, seed=33)
        path = tmp_path / "mlp.npz"
        np.savez(path, **mech.state_dict())
        with np.load(path) as loaded:
            restored = mechanism_from_state_dict(dict(loaded))
        assert isinstance(restored, MLPMechanism)
        rng = np.random.default_rng(34)
        for shape in [(mech.L, mech.k), (6, mech.L, mech.k)]:
            history = rng.normal(size=shape)
            np.testing.assert_array_equal(
                mech.forward_numpy(history), restored.forward_numpy(history)
            )

    def test_type_discriminator(self):
        mech = _random_mlp()
        assert mech.state_dict()["__type__"] == "mlp"

    def test_zero_parent_node_is_pure_decay(self):
        """A 0-parent node reduces to decay·x_{t-1} (no divide-by-zero)."""
        k, L = 3, 1
        graph = np.zeros((k, k, L), dtype=float)
        graph[1, 0, 0] = 1.0  # node 0 has no parents
        mech = _random_mlp(graph=graph, hidden=4, seed=35)
        rng = np.random.default_rng(36)
        history = rng.normal(size=(L, k))
        out = mech.forward_numpy(history)
        assert np.isclose(out[0], mech.decay[0] * history[-1, 0])


# ---------------------------------------------------------------------------
# M4/H6 non-monotonic mechanism ablation
# ---------------------------------------------------------------------------


class TestMLPNonmonotonicActivation:
    def test_random_accepts_nonmonotonic(self):
        mech = _random_mlp(hidden=8, seed=40, activation="nonmonotonic")
        assert mech.activation == "nonmonotonic"

    def test_numpy_torch_agree(self):
        mech = _random_mlp(hidden=8, seed=41, activation="nonmonotonic")
        rng = np.random.default_rng(42)
        history = rng.normal(size=(mech.L, mech.k))
        np_out = mech.forward_numpy(history)
        torch_out = mech.forward_torch(torch.as_tensor(history)).numpy()
        assert np.max(np.abs(np_out - torch_out)) < 1e-5

    def test_numpy_torch_agree_batched(self):
        mech = _random_mlp(hidden=16, seed=46, activation="nonmonotonic")
        rng = np.random.default_rng(47)
        batch = rng.normal(size=(9, mech.L, mech.k))
        np_out = mech.forward_numpy(batch)
        torch_out = mech.forward_torch(torch.as_tensor(batch)).numpy()
        assert np.max(np.abs(np_out - torch_out)) < 1e-5

    def test_forward_torch_is_differentiable(self):
        """forward_torch must stay autograd-safe w.r.t. history (needed by
        CARLA) with the nonmonotonic hidden activation too."""
        mech = _random_mlp(hidden=8, seed=43, activation="nonmonotonic")
        rng = np.random.default_rng(44)
        history = torch.as_tensor(rng.normal(size=(mech.L, mech.k)))
        history.requires_grad_(True)
        out = mech.forward_torch(history)
        out.sum().backward()
        assert history.grad is not None
        assert torch.isfinite(history.grad).all()
        assert history.grad.shape == history.shape

    def test_contractive_boundedness(self):
        """Same 300-step rollout stability check as the tanh test above --
        the output branch is always tanh regardless of hidden activation,
        so boundedness must be unaffected by this ablation."""
        graph = _dense_graph(k=5, L=2)
        for seed in range(10):
            mech = _random_mlp(graph=graph, hidden=8, seed=seed, activation="nonmonotonic")
            window = np.random.default_rng(2000 + seed).normal(size=(mech.L, mech.k))
            for _ in range(300):
                nxt = mech.forward_numpy(window)
                window = np.vstack([window[1:], nxt[None, :]])
                assert np.all(np.isfinite(window))
            assert np.max(np.abs(window)) < 100.0

    def test_activation_is_actually_non_monotonic_in_domain(self):
        """Structural check on the function itself: sin has a local max in
        [0, pi] (unlike tanh, which is strictly increasing everywhere)."""
        from causaltemp_xai.benchmarks.mechanisms import _ACTIVATIONS_NP

        f = _ACTIVATIONS_NP["nonmonotonic"]
        y0, y1, y2 = f(0.0), f(np.pi / 2), f(np.pi)
        assert y0 < y1 and y1 > y2

    def test_activation_is_bounded_like_tanh(self):
        from causaltemp_xai.benchmarks.mechanisms import _ACTIVATIONS_NP

        f = _ACTIVATIONS_NP["nonmonotonic"]
        x = np.linspace(-50, 50, 5000)
        assert np.all(np.abs(f(x)) <= 1.0 + 1e-9)

    def test_invalid_activation_still_rejected(self):
        with pytest.raises(ValueError):
            _random_mlp(hidden=4, seed=45, activation="relu")

    def test_state_dict_round_trip_preserves_nonmonotonic_activation(self):
        mech = _random_mlp(hidden=8, seed=48, activation="nonmonotonic")
        restored = mechanism_from_state_dict(mech.state_dict())
        assert restored.activation == "nonmonotonic"
        rng = np.random.default_rng(49)
        history = rng.normal(size=(mech.L, mech.k))
        np.testing.assert_array_equal(mech.forward_numpy(history), restored.forward_numpy(history))


# ---------------------------------------------------------------------------
# P0-1 acceptance gate for the M4/P0-2 mechanism redesign
# ---------------------------------------------------------------------------


def _shipped_nl_dataset(config_name: str = "smoke_nl"):
    """Generate the *shipped* nonlinear config, not a hand-built toy.

    The claim under test is about the benchmark tier researchers actually run,
    so the hyperparameters must come from ``config.py`` rather than this
    module's ``_random_mlp`` defaults -- a toy mechanism could pass thresholds
    the shipped preset fails.
    """
    from causaltemp_xai.config import get_config
    from causaltemp_xai.data_io import build_generator

    result = build_generator(get_config(config_name)).generate()
    return result["X"], result["mechanism"]


def _histories(X: np.ndarray, L: int) -> np.ndarray:
    """Every length-``L`` window in ``X``, flattened to ``(N*(T-L), L, k)``."""
    windows = [X[:, t - L : t, :] for t in range(L, X.shape[1])]
    return np.stack(windows, axis=1).reshape(-1, L, X.shape[2])


class TestMLPMechanismIsMeasurablyNonlinear:
    """P0-1 gate for P0-2 (`docs/pi_reevaluation_2026-08-11.md`).

    Instrumentation on 2026-08-11 showed ``NlinearSCM-T`` is a linear VAR(1):
    hidden pre-activations sit at ``|z| ~ 0.015`` where ``tanh`` is the
    identity, an affine fit reproduces the mechanism at ``R^2 = 1.000000``, and
    the graph-carrying MLP branch holds ~1% of output variance against ~99% for
    the ``decay_i * x_{t-1}^i`` self-term -- which is *not* an edge in ``graph``.

    Every assertion below is a P0-2 acceptance target. They are expected to fail
    until the mechanism is reparameterised; remove the ``xfail`` markers then.
    A passing suite must not be read as "the nonlinear tier is nonlinear".
    """

    @pytest.mark.xfail(
        strict=True,
        reason="NOT fixable by reparameterisation -- needs a PI scope decision "
        "on the mechanism's functional form. P0-2's retune (2026-08-11) lifted "
        "this from 3.0e-07 to ~0.06 mean, but the MINIMUM across seeds stays "
        "~0.002: a randomly-initialised 2-layer net sits in the lazy regime and "
        "tracks its own linearisation regardless of |z|, because per-unit "
        "curvature cancels across the hidden layer. Swept gain, init_gain, "
        "spectral_cap, decay_range, hidden (1..16) and non-zero b1 -- the "
        "configurations that approached 5% lost contraction. Remove this marker "
        "only after the functional form changes.",
    )
    def test_nonlinear_share_of_variance_is_at_least_5_percent(self):
        """An affine map must NOT be able to reproduce the mechanism.

        ``1 - R^2`` of the best least-squares affine fit to the mechanism's own
        conditional mean is the fraction of its behaviour that is genuinely
        nonlinear. If that is ~0, the family is a VAR wearing an MLP costume and
        no "carries over from linear to nonlinear" claim is supported.
        """
        X, mech = _shipped_nl_dataset()
        hist = _histories(X, mech.L)
        mu = mech.forward_numpy(hist)

        design = np.concatenate([hist.reshape(len(hist), -1), np.ones((len(hist), 1))], axis=1)
        coef, *_ = np.linalg.lstsq(design, mu, rcond=None)
        residual = mu - design @ coef
        ss_res = (residual**2).sum(axis=0)
        ss_tot = ((mu - mu.mean(axis=0)) ** 2).sum(axis=0)
        r_squared = float(np.mean(1.0 - ss_res / ss_tot))
        nonlinear_share = 1.0 - r_squared  # the part no affine map can reach

        assert nonlinear_share >= 0.05, (
            f"affine fit reproduces the mechanism at R^2 = {r_squared:.8f} "
            f"(nonlinear share {nonlinear_share:.2e}); the 'nonlinear' tier is a "
            f"linear VAR"
        )

    def test_graph_carrying_branch_holds_at_least_40_percent_of_variance(self):
        """The branch the causal graph flows through must not be a rounding error.

        ``mean_i = decay_i * x_{t-1}^i + gain * tanh(MLP_i(masked parents))``.
        Only the second term touches ``graph`` at all, so its share of output
        variance bounds how much of the data the ground-truth causal structure
        can explain. At ~1% the benchmark scores counterfactuals against a
        process the graph barely participates in.
        """
        X, mech = _shipped_nl_dataset()
        hist = _histories(X, mech.L)
        mu = mech.forward_numpy(hist)

        decay_branch = mech.decay * hist[:, -1, :]
        mlp_branch = mu - decay_branch  # the gain * tanh(MLP(...)) term
        mlp_share = float(mlp_branch.var() / (decay_branch.var() + mlp_branch.var()))

        assert mlp_share >= 0.40, (
            f"graph-carrying MLP branch holds only {mlp_share:.4f} of output "
            f"variance; the remaining {1 - mlp_share:.4f} flows through decay_i, "
            f"which is not an edge in graph (trace(graph[:, :, 0]) == 0)"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Improved ~30,000x by the P0-2 retune (3.9e-06 -> ~0.12) but "
        "still not robust: the swap clears the innovation-noise floor (0.141) "
        "on only 1 of 3 seeds. Shares a root cause with the nonlinear-share "
        "gate above -- if tanh is near-affine over the realised |z|, so is sin, "
        "and the two agree. Blocked on the same functional-form decision.",
    )
    def test_nonmonotonic_swap_exceeds_innovation_noise(self):
        """The H5(ii) ablation must actually ablate something.

        ``SMOKE_NONMONOTONIC`` swaps the hidden activation ``tanh`` -> ``sin``.
        If that swap moves the conditional mean by less than one innovation-noise
        standard deviation, the ablation is undetectable in the generated data
        and cannot support or refute any hypothesis about non-monotonicity.
        """
        X, mech_tanh = _shipped_nl_dataset("smoke_nl")
        hist = _histories(X, mech_tanh.L)

        mu_tanh = mech_tanh.forward_numpy(hist)
        # Same weights, only the hidden activation differs -- isolates the swap
        # from any change in the sampled parameters.
        mech_sin = MLPMechanism(
            graph=mech_tanh.graph,
            hidden=mech_tanh.hidden,
            decay=mech_tanh.decay,
            gain=mech_tanh.gain,
            W1=mech_tanh.W1,
            b1=mech_tanh.b1,
            W2=mech_tanh.W2,
            b2=mech_tanh.b2,
            activation="nonmonotonic",
        )
        mu_sin = mech_sin.forward_numpy(hist)

        swap_effect = float(np.abs(mu_tanh - mu_sin).mean())
        # Innovation noise: what the additive-noise SCM adds on top of the mean.
        innovation_std = float(
            (X[:, mech_tanh.L :, :] - mu_tanh.reshape(X.shape[0], -1, X.shape[2])).std()
        )

        assert swap_effect > innovation_std, (
            f"tanh->sin moves the conditional mean by {swap_effect:.3e}, below "
            f"the innovation-noise std {innovation_std:.3e} -- the ablation is "
            f"invisible in the generated data"
        )


def _decoupled(mech):
    """Return ``mech`` with all graph-mediated coupling removed, weights intact.

    Isolates the graph's contribution from every other design choice: the
    decoupled mechanism differs from the original *only* in that no influence
    can flow along a ``graph`` edge. Each family has a natural zero-coupling
    form: an all-zero parent mask (MLP), a zero spring matrix, a zero adjacency
    (Kuramoto), or zero coefficient matrices (linear, where ``A`` *is* the
    graph, so the decoupled mechanism is the null dynamic).
    """
    import copy

    from causaltemp_xai.benchmarks.mechanisms import KuramotoMechanism, SpringMechanism

    if isinstance(mech, MLPMechanism):
        # Zero parent mask => masked input 0 => tanh(0) = 0 => mean = decay * x.
        return MLPMechanism(
            graph=np.zeros_like(mech.graph),
            hidden=mech.hidden,
            decay=mech.decay,
            gain=mech.gain,
            W1=mech.W1,
            b1=mech.b1,
            W2=mech.W2,
            b2=mech.b2,
            activation=mech.activation,
        )
    if isinstance(mech, SpringMechanism):
        bare = copy.deepcopy(mech)
        bare.K = np.zeros_like(mech.K)  # no cross-particle acceleration
        return bare
    if isinstance(mech, KuramotoMechanism):
        bare = copy.deepcopy(mech)
        bare.A = np.zeros_like(mech.A)  # no phase coupling
        return bare
    if isinstance(mech, LinearMechanism):
        return LinearMechanism([np.zeros_like(A) for A in mech.A_list])
    raise TypeError(f"no decoupled form defined for {type(mech).__name__}")


def _graph_share_of_increment(config_name: str) -> float:
    """Fraction of what the mechanism *does* each step that flows through ``graph``.

    Measured on the **increment** ``mean(x_t) - x_{t-1}``, not the state level.
    Level is the wrong denominator for the non-dissipative families: Kuramoto's
    phase is unwrapped and unbounded, so its own carry-forward term dominates
    any level-based share asymptotically no matter how strong the coupling is
    (measured: 0.0002% on level vs 2.14% on increment). The increment asks the
    question the benchmark actually needs -- of the change the mechanism
    introduces at each step, how much is causal-graph-mediated -- which is also
    what an intervention perturbs and then accumulates.
    """
    from causaltemp_xai.config import get_config
    from causaltemp_xai.data_io import build_generator

    result = build_generator(get_config(config_name)).generate()
    X, mech = result["X"], result["mechanism"]
    hist = _histories(X, mech.L)
    last = hist[:, -1, :]

    bare = _decoupled(mech)
    bare_mu = bare.forward_numpy(hist)
    graph_part = mech.forward_numpy(hist) - bare_mu
    ungraphed_increment = bare_mu - last
    return float(graph_part.var() / (graph_part.var() + ungraphed_increment.var()))


class TestGraphDrivesEverySyntheticFamily:
    """P0-1 gate, extended to all four families (2026-08-11 measurement).

    P0-2 as written targets ``MLPMechanism`` only, but the pathology is
    benchmark-wide: every family except ``linear`` carves a dominant self-term
    out of ``graph`` and calls it structural -- ``decay_i * x_{t-1}^i`` (MLP),
    position<-own-velocity (spring), ``theta + dt*omega_i`` (Kuramoto). Fixing
    one family leaves the benchmark's other tiers scoring counterfactuals
    against processes their graphs barely drive.

    The 20% bar is set against the empirical ceiling rather than picked: the
    ``linear`` family, where ``A`` *is* the graph and no term sits outside it,
    reaches 47.9%. 20% therefore asks each family to clear roughly 40% of what
    a fully graph-driven mechanism achieves -- demanding, but demonstrably
    attainable, and spring already passes at 30.6%.

    Passing this gate does **not** clear spring's separate defect: its strongest
    coupling (every position on its own velocity) is deliberately excluded from
    ``graph``, so its discovery AUC is scored against an incomplete reference
    graph. That is tracked separately under P0-4.
    """

    MIN_GRAPH_SHARE = 0.20

    @pytest.mark.parametrize(
        "config_name",
        [
            "smoke",  # linear: A *is* the graph, nothing sits outside it
            "smoke_spring",
            "smoke_nl",  # 2.33% -> ~0.50 after the P0-2 retune
            "smoke_kuramoto",  # 2.14% -> ~0.46 after k_coupling 0.5 -> 5.0
        ],
    )
    def test_graph_carries_at_least_20_percent_of_the_increment(self, config_name):
        share = _graph_share_of_increment(config_name)
        assert share >= self.MIN_GRAPH_SHARE, (
            f"{config_name}: only {share:.2%} of the per-step increment flows "
            f"through graph edges; the rest is a self-term the ground-truth "
            f"graph does not contain, so counterfactuals are scored against a "
            f"process the causal structure barely drives"
        )
