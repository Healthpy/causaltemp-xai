"""Tests for the Mechanism abstraction (Stage 1: LinearMechanism)."""

from __future__ import annotations

import numpy as np
import torch

from causaltemp_xai.benchmark.mechanisms import (
    LinearMechanism,
    mechanism_from_state_dict,
)


def _random_linear(k=3, L=2, seed=0):
    rng = np.random.default_rng(seed)
    return LinearMechanism([rng.uniform(-0.3, 0.3, (k, k)) for _ in range(L)])


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
