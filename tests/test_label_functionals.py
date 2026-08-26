"""Tests for label functionals and the label-site confound (R6, RISK-19).

Contract:

1. the default functional reproduces the pre-M2b hardcoded rule **exactly** —
   every committed preset, dataset and ``results/`` row depends on it;
2. ``interior_threshold`` genuinely moves the label site off ``T - 1`` while
   leaving the trajectory length alone, which is what makes H8c a controlled
   comparison rather than a shorter benchmark;
3. threshold recovery round-trips under every functional, and **raises** rather
   than returning a plausible-looking number when the wrong functional is
   passed — a silently wrong ``theta`` corrupts every world-side PNS term;
4. a label functional that ignores the trajectory is *visible* as degenerate
   rather than producing a readable ``delta_outcome``.
"""

from __future__ import annotations

import numpy as np
import pytest

from causaltemp_xai.benchmarks.labels import (
    INTERIOR_THRESHOLD,
    TERMINAL_THRESHOLD,
    LabelFunctional,
    get_label_functional,
)
from causaltemp_xai.benchmarks.mechanisms import LinearMechanism
from causaltemp_xai.config import get_config
from causaltemp_xai.data_io import build_generator
from causaltemp_xai.metrics.pns import recover_label_threshold, scm_label


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.normal(size=(200, 30, 4))


class TestDefaultUnchanged:
    def test_terminal_threshold_is_the_original_rule(self, X):
        assert np.array_equal(TERMINAL_THRESHOLD.latent_batch(X), X[:, -1, 0])

    def test_none_resolves_to_terminal(self):
        assert get_label_functional(None) is TERMINAL_THRESHOLD

    def test_scm_label_default_matches_the_hardcoded_form(self, X):
        theta = 0.1
        for x in X[:20]:
            assert scm_label(x, theta) == int(x[-1, 0] > theta)

    def test_locked_presets_still_use_the_terminal_rule(self):
        for name in ("smoke", "full", "smoke_nl", "full_nl"):
            cfg = get_config(name)
            assert cfg.label_fn == "terminal_threshold"
            assert cfg.label_params is None

    def test_generated_labels_unchanged_for_smoke(self):
        """The end-to-end guard: routing labels through the functional must not
        move a single label on a locked preset."""
        cfg = get_config("smoke")
        data = build_generator(cfg).generate()
        X_gen, Y = data["X"], data["Y"]
        latent = X_gen[:, -1, 0]
        expected = (latent > float(np.median(latent))).astype(int)
        assert np.array_equal(Y, expected)


class TestInteriorThreshold:
    def test_label_site_moves_but_trajectory_does_not(self, X):
        T = X.shape[1]
        assert INTERIOR_THRESHOLD.label_site(T) == 18  # int(0.6 * 30)
        assert TERMINAL_THRESHOLD.label_site(T) == T - 1
        assert np.array_equal(INTERIOR_THRESHOLD.latent_batch(X), X[:, 18, 0])

    def test_horizon_is_strictly_shorter_at_the_same_t0(self, X):
        """The point of the preset: t_label - t0 < T - t0 by a wide margin."""
        T, t0 = X.shape[1], 5
        assert INTERIOR_THRESHOLD.label_site(T) - t0 < TERMINAL_THRESHOLD.label_site(T) - t0

    def test_preset_is_registered_and_differs_only_in_label(self):
        base, interior = get_config("full"), get_config("full_interior_label")
        assert interior.label_fn == "interior_threshold"
        assert interior.label_params == {"frac": 0.6}
        for field in ("k", "L", "sparsity", "noise_type", "T", "N", "seed", "mechanism_type"):
            assert getattr(base, field) == getattr(interior, field), field

    def test_label_site_is_clamped_within_the_trajectory(self):
        assert get_label_functional("interior_threshold", {"frac": 1.0}).label_site(30) == 29

    def test_rejects_out_of_range_frac(self):
        with pytest.raises(ValueError, match="frac must be in"):
            get_label_functional("interior_threshold", {"frac": 0.0})


class TestOtherFunctionals:
    def test_window_mean_averages_the_tail(self, X):
        fn = get_label_functional("window_mean", {"window": 5})
        assert np.allclose(fn.latent_batch(X), X[:, -5:, 0].mean(axis=1))

    def test_window_mean_is_not_flippable_by_one_cell(self, X):
        """A single-cell rewrite moves the window mean by 1/window of the edit —
        the failure mode a sparsity=0.02 method exploits on a terminal label."""
        fn = get_label_functional("window_mean", {"window": 10})
        x = X[0]
        before = fn.latent_one(x)
        x_edit = x.copy()
        x_edit[-1, 0] += 1.0
        assert fn.latent_one(x_edit) - before == pytest.approx(0.1)

    def test_multichannel_linear_uses_every_channel(self, X):
        fn = get_label_functional("multichannel_linear")
        assert np.allclose(fn.latent_batch(X), X[:, -1, :].mean(axis=1))

    def test_multichannel_linear_rejects_wrong_width(self, X):
        fn = get_label_functional("multichannel_linear", {"weights": [1.0, 0.0]})
        with pytest.raises(ValueError, match="expected"):
            fn.latent_batch(X)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown label functional"):
            get_label_functional("no_such_rule")

    def test_batch_and_single_paths_agree(self, X):
        """latent_one is derived from latent_batch precisely so these cannot
        disagree; assert it for every registered functional."""
        for name in ("terminal_threshold", "interior_threshold", "window_mean"):
            fn = get_label_functional(name)
            assert fn.latent_one(X[3]) == pytest.approx(fn.latent_batch(X)[3])


class TestThresholdRecovery:
    @pytest.mark.parametrize(
        "name", ["terminal_threshold", "interior_threshold", "window_mean", "multichannel_linear"]
    )
    def test_round_trips_under_every_functional(self, X, name):
        fn = get_label_functional(name)
        latent = fn.latent_batch(X)
        theta_true = float(np.median(latent))
        Y = (latent > theta_true).astype(int)
        assert recover_label_threshold(X, Y, label_fn=fn) == pytest.approx(theta_true)

    def test_wrong_functional_raises_rather_than_guessing(self, X):
        """The guard that matters: a silently wrong theta corrupts C, and
        therefore delta_total, delta_outcome, PN and PS, with no visible sign."""
        latent = INTERIOR_THRESHOLD.latent_batch(X)
        Y = (latent > float(np.median(latent))).astype(int)
        with pytest.raises(ValueError, match="cannot recover label threshold"):
            recover_label_threshold(X, Y, label_fn=TERMINAL_THRESHOLD)

    def test_recovered_theta_reproduces_generated_labels(self):
        """End-to-end on the real preset, since this is how the audit uses it."""
        cfg = get_config("smoke_interior_label")
        data = build_generator(cfg).generate()
        fn = cfg.label_functional()
        theta = recover_label_threshold(data["X"], data["Y"], label_fn=fn)
        got = np.array([scm_label(x, theta, label_fn=fn) for x in data["X"][:100]])
        assert np.array_equal(got, data["Y"][:100])


class TestWorldSideRespectsLabelSite:
    """A `do()` placed after the label site cannot move the label.

    This is the invariant that caught the 2026-08-03 Phase-08 bug: `C` was
    computed with the default terminal rule while `theta` came from the config's
    interior rule, so an intervention at `t=95` appeared to move a label at
    `t=90` — it was really reading `x[99]`. Any call path that scores the world
    side must be handed the same functional the threshold was recovered under.
    """

    @staticmethod
    def _scm(T=30, k=3, seed=0):
        rng = np.random.default_rng(seed)
        A = rng.uniform(-0.3, 0.3, (k, k))
        x = np.zeros((T, k))
        for t in range(1, T):
            x[t] = A @ x[t - 1] + rng.laplace(0, 0.05, k)
        return x, LinearMechanism([A])

    def test_post_label_intervention_leaves_the_world_label_unchanged(self):
        from causaltemp_xai.benchmarks.structural_cf import structural_counterfactual

        x, mech = self._scm()
        fn = get_label_functional("interior_threshold", {"frac": 0.6})  # t_label = 18
        theta = float(fn.latent_one(x)) - 0.5  # factual label is 1
        before = scm_label(x, theta, label_fn=fn)
        # Intervene well past the label site, with a large value.
        x_cf = structural_counterfactual(x, mech, t0=25, node=0, value=50.0)
        assert scm_label(x_cf, theta, label_fn=fn) == before

    def test_the_same_cf_does_move_a_terminal_label(self):
        """The control: the intervention is not inert, it is just out of reach
        of the interior label. Without this, the test above would pass on a CF
        that did nothing."""
        from causaltemp_xai.benchmarks.structural_cf import structural_counterfactual

        x, mech = self._scm()
        term = get_label_functional("terminal_threshold")
        x_cf = structural_counterfactual(x, mech, t0=25, node=0, value=50.0)
        assert term.latent_one(x_cf) != pytest.approx(term.latent_one(x))

    def test_pns_direction_world_side_follows_the_passed_functional(self):
        """`pns_direction` must not fall back to the terminal rule."""
        from causaltemp_xai.benchmarks.structural_cf import structural_counterfactual
        from causaltemp_xai.metrics.pns import pns_direction

        x, mech = self._scm()
        fn = get_label_functional("interior_threshold", {"frac": 0.6})
        theta = float(fn.latent_one(x)) - 0.5
        x_cf = structural_counterfactual(x, mech, t0=25, node=0, value=50.0)

        target = scm_label(x, theta, label_fn=fn)
        out = pns_direction(
            x[None],
            x_cf[None],
            _ThresholdModel(theta),
            mech,
            theta,
            target_class=target,
            label_fn=fn,
        )
        # The world cannot have changed its mind about a label the intervention
        # never touched.
        assert out["C_world_oracle"] == pytest.approx(1.0)


class _ThresholdModel:
    def __init__(self, theta):
        self.theta = theta

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X[:, -1, 0] > self.theta).astype(int)


class TestDegenerateLabelIsVisible:
    def test_trajectory_independent_label_cannot_be_recovered_silently(self, X):
        """A label functional that ignores the trajectory must fail loudly.
        Otherwise C is a constant and delta_outcome reads as a finding."""

        class _Constant(LabelFunctional):
            name = "constant"

            def latent_batch(self, arr):
                return np.zeros(len(arr))

            def label_site(self, T):
                return T - 1

        fn = _Constant()
        Y = np.random.default_rng(0).integers(0, 2, len(X))
        with pytest.raises(ValueError, match="cannot recover label threshold"):
            recover_label_threshold(X, Y, label_fn=fn)

    def test_constant_latent_gives_a_constant_world_label(self, X):
        """And where it *is* recoverable (all-one-class), C is constant — which
        `n_scorable` / a degenerate C column must make visible to the reader."""

        class _Constant(LabelFunctional):
            name = "constant"

            def latent_batch(self, arr):
                return np.zeros(len(arr))

            def label_site(self, T):
                return T - 1

        fn = _Constant()
        labels = {scm_label(x, 0.5, label_fn=fn) for x in X[:50]}
        assert labels == {0}
