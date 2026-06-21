"""Tests for the Mechanism abstraction (Stage 1: LinearMechanism)."""

from __future__ import annotations

import numpy as np
import torch

from causaltemp_xai.benchmark.mechanisms import (
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
        np.testing.assert_array_equal(
            mech.forward_numpy(history), restored.forward_numpy(history)
        )

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
