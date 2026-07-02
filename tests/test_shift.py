"""Tests for the Shift-VR-lite shifted environment + validity-retention metric."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import ks_2samp

from causaltemp_xai.benchmark.generator import LinearSCMT
from causaltemp_xai.classifiers import LSTMClassifier
from causaltemp_xai.config import SMOKE, shifted_config
from causaltemp_xai.eval import shift_vr
from causaltemp_xai.methods import CARLARecourse, WachterCF


def _gen(config):
    return LinearSCMT(
        k=config.k,
        L=config.L,
        sparsity=config.sparsity,
        noise_type=config.noise_type,
        T=config.T,
        N=config.N,
        seed=config.seed,
    )


class TestShiftedEnvironment:
    def test_shifted_config_changes_only_noise(self):
        shift = shifted_config(SMOKE, noise_type="uniform")
        assert shift.noise_type == "uniform"
        assert shift.name == "smoke_shift"
        for field in ("k", "L", "sparsity", "T", "N", "seed"):
            assert getattr(shift, field) == getattr(SMOKE, field)

    def test_identical_scm_structure(self):
        base = _gen(SMOKE)
        shift = _gen(shifted_config(SMOKE, noise_type="uniform"))
        # Graph and mechanisms are built from the seed before noise sampling,
        # so they must be bit-identical across the two environments.
        np.testing.assert_array_equal(base.graph, shift.graph)
        assert base.mechanism.L == shift.mechanism.L
        for a, b in zip(base.mechanism.A_list, shift.mechanism.A_list):
            np.testing.assert_array_equal(a, b)

    def test_noise_marginal_differs(self):
        base = _gen(SMOKE)
        shift = _gen(shifted_config(SMOKE, noise_type="uniform"))
        base_noise = np.asarray(base._sample_noise(20000))
        shift_noise = np.asarray(shift._sample_noise(20000))
        _, p = ks_2samp(base_noise, shift_noise)
        # Laplace (heavy-tailed) vs uniform (bounded) are clearly distinguishable.
        assert p < 0.01

    def test_rejects_bad_noise_type(self):
        with pytest.raises(ValueError):
            shifted_config(SMOKE, noise_type="gaussian")


class TestShiftVR:
    def test_ratio_in_unit_interval(self):
        base = _gen(SMOKE)
        shift = _gen(shifted_config(SMOKE, noise_type="uniform"))
        base_data = base.generate()
        shift_data = shift.generate()

        X, Y = base_data["X"], base_data["Y"]
        clf = LSTMClassifier(
            n_inputs=SMOKE.k,
            hidden_size=16,
            num_layers=2,
            dropout=0.0,
            lr=5e-3,
            batch_size=64,
            max_epochs=15,
            patience=15,
            seed=0,
        )
        clf.fit(X[:400], Y[:400], X[400:], Y[400:])

        X_base_test = X[400:404]
        X_shift_test = shift_data["X"][:4]
        graph, mech = base_data["graph"], base_data["mechanism"]

        methods = {
            "wachter": WachterCF(target_class=1, n_steps=50, lr=0.1),
            "carla": CARLARecourse(target_class=1, n_steps=50, t0_fractions=(0.5,)),
        }
        result = shift_vr(clf, methods, X_base_test, X_shift_test, graph, mech, 1)

        assert set(result) == {"wachter", "carla"}
        for name, m in result.items():
            assert 0.0 <= m["validity_base"] <= 1.0
            assert 0.0 <= m["validity_shift"] <= 1.0
            ratio = m["shift_vr"]
            assert math.isnan(ratio) or 0.0 <= ratio  # ratio ≥ 0 (nan if base=0)
