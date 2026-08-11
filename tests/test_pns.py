"""Adversarial tests for the necessity/sufficiency gap (R6).

Contract, per ``docs/pns_metric_design.md``'s build order — the metric must:

1. score the **oracle** structural CF at ``delta_total ≈ 0`` (it *is* the
   world, so model and world cannot disagree about it);
2. **flag** a CF that flips the classifier while having no world effect —
   the exact CARLA/``full`` artifact this metric exists to catch;
3. **abstain** on a no-op CF rather than scoring it 0;
4. satisfy the additive identity ``delta_total == delta_trajectory +
   delta_outcome`` exactly;
5. recover the label threshold exactly, since a wrong ``theta`` silently
   corrupts every world-side number.
"""

from __future__ import annotations

import numpy as np
import pytest

from causaltemp_xai.benchmarks.generator import SpringSCMT
from causaltemp_xai.benchmarks.mechanisms import LinearMechanism
from causaltemp_xai.benchmarks.structural_cf import structural_counterfactual
from causaltemp_xai.metrics.pns import (
    extract_intervention,
    pns_direction,
    pns_from_directions,
    recover_label_threshold,
)


def _make_scm(k=4, L=1, T=30, seed=0, noise_scale=0.05):
    rng = np.random.default_rng(seed)
    A_list = [rng.uniform(-0.3, 0.3, (k, k)) for _ in range(L)]
    x = np.zeros((T, k))
    noise = rng.laplace(0, noise_scale, (T, k))
    for t in range(L, T):
        for lag, A in enumerate(A_list, start=1):
            x[t] += A @ x[t - lag]
        x[t] += noise[t]
    return x, LinearMechanism(A_list)


def _make_spring(n_particles=3, T=30, seed=0):
    """(x, mechanism) from a small SpringSCMT (M4c)."""
    gen = SpringSCMT(n_particles=n_particles, T=T, N=1, seed=seed)
    data = gen.generate(burn_in=20)
    return data["X"][0], data["mechanism"]


class _AlwaysTarget:
    """Classifier that says 'target' for everything — maximally over-claiming."""

    def predict(self, X):
        return np.ones(len(np.atleast_3d(X)), dtype=int)


class _ThresholdModel:
    """A classifier that *agrees* with the SCM label rule."""

    def __init__(self, theta):
        self.theta = theta

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X[:, -1, 0] > self.theta).astype(int)


class TestThresholdRecovery:
    def test_recovers_median_threshold_exactly(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 10, 3))
        theta_true = float(np.median(X[:, -1, 0]))
        Y = (X[:, -1, 0] > theta_true).astype(int)
        assert recover_label_threshold(X, Y) == pytest.approx(theta_true)

    def test_raises_when_labels_are_not_a_threshold_rule(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, 10, 3))
        Y = rng.integers(0, 2, size=50)  # random labels, not a threshold
        with pytest.raises(ValueError):
            recover_label_threshold(X, Y)


class TestInterventionExtraction:
    def test_extracts_all_changed_channels_not_just_one(self):
        """Real CF methods edit several channels at t0; scoring one would
        evaluate a different intervention than the method proposed."""
        x, _ = _make_scm(seed=2)
        x_cf = x.copy()
        x_cf[10, [0, 2]] += 1.5
        t0, nodes, values = extract_intervention(x, x_cf)
        assert t0 == 10
        assert sorted(nodes.tolist()) == [0, 2]
        assert values == pytest.approx(x_cf[10, [0, 2]])

    def test_no_op_yields_no_intervention(self):
        x, _ = _make_scm(seed=3)
        _, nodes, _ = extract_intervention(x, x.copy())
        assert nodes.size == 0


class TestPNSAdversarial:
    def test_oracle_cf_has_zero_gap(self):
        """The oracle structural CF *is* the world's trajectory, so a model
        that agrees with the SCM label rule must show no gap on it."""
        x, mech = _make_scm(seed=4)
        theta = 0.0
        t0 = 10
        oracle = structural_counterfactual(x, mech, t0, [0], [x[t0, 0] + 2.0], noiseless=False)
        model = _ThresholdModel(theta)
        out = pns_direction(np.stack([x]), np.stack([oracle]), model, mech, theta, target_class=1)
        assert out["n_scorable"] == 1
        assert out["delta_trajectory"] == pytest.approx(0.0)
        assert out["delta_outcome"] == pytest.approx(0.0)
        assert out["delta_total"] == pytest.approx(0.0)

    def test_flags_model_artifact_cf(self):
        """The metric's reason to exist: a CF the model calls a successful
        flip, whose intervention the world says does nothing.

        Built to mimic the CARLA/`full` artifact -- the proposed trajectory is
        displaced (so the model flips) while the *intervention* at t0 is
        negligible, so the world's realisation of that same intervention does
        not cross the label threshold.
        """
        x, mech = _make_scm(seed=5)
        t0 = 10
        x_cf = x.copy()
        x_cf[t0, 0] += 1e-3  # a real but tiny intervention
        x_cf[t0 + 1 :] += 50.0  # trajectory displaced by something else entirely
        # theta far above anything the true mechanism would reach from a 1e-3 nudge
        theta = 10.0
        out = pns_direction(
            np.stack([x]), np.stack([x_cf]), _AlwaysTarget(), mech, theta, target_class=1
        )
        assert out["A_model_proposed"] == 1.0, "model claims a successful flip"
        assert out["C_world_oracle"] == 0.0, "world says the intervention does nothing"
        assert out["delta_total"] == pytest.approx(1.0), "the over-claim must be flagged"

    def test_isolates_trajectory_error_from_outcome_error(self):
        """The two halves of the decomposition must be separately reachable.

        ``test_flags_model_artifact_cf`` uses a model that ignores its input,
        which pins ``B == A`` and so can only ever exercise ``delta_outcome``.
        Here the model *does* read the trajectory, and the proposed CF's final
        value crosses the threshold while the world's realisation of the same
        intervention does not -- so the gap must land on ``delta_trajectory``
        instead. Without this, half the decomposition is untested.
        """
        x, mech = _make_scm(seed=8)
        theta = 5.0
        t0 = 10
        x_cf = x.copy()
        x_cf[t0, 0] += 1e-3  # negligible intervention...
        x_cf[-1, 0] = theta + 1.0  # ...but the proposed endpoint clears theta
        out = pns_direction(
            np.stack([x]), np.stack([x_cf]), _ThresholdModel(theta), mech, theta, target_class=1
        )
        assert out["A_model_proposed"] == 1.0, "model flips on the proposed CF"
        assert out["B_model_oracle"] == 0.0, "but not on the world's version of it"
        assert out["delta_trajectory"] == pytest.approx(1.0)
        assert out["delta_outcome"] == pytest.approx(0.0)

    def test_no_op_abstains_rather_than_scoring_zero(self):
        """A no-op CF carries no evidence about causal efficacy -- it must be
        excluded, not counted as a (correct) zero-effect intervention, which
        would let a method that does nothing look well-calibrated."""
        x, mech = _make_scm(seed=6)
        out = pns_direction(
            np.stack([x]), np.stack([x.copy()]), _AlwaysTarget(), mech, 0.0, target_class=1
        )
        assert out["n_scorable"] == 0
        assert out["frac_no_intervention"] == 1.0
        assert np.isnan(out["delta_total"])

    def test_additive_identity_holds(self):
        x, mech = _make_scm(seed=7)
        rng = np.random.default_rng(0)
        X = np.stack([x, x, x])
        CFs = []
        for i in range(3):
            c = x.copy()
            c[12, 0] += float(rng.uniform(0.5, 2.0))
            c[13:] += rng.normal(scale=0.2, size=c[13:].shape)
            CFs.append(c)
        out = pns_direction(X, np.stack(CFs), _AlwaysTarget(), mech, 0.0, target_class=1)
        assert out["delta_total"] == pytest.approx(out["delta_trajectory"] + out["delta_outcome"])


# ---------------------------------------------------------------------------
# M4c: PNS adversarial tests on the two new non-dissipative families (R6).
# ---------------------------------------------------------------------------


class TestPNSAdversarialNewFamilies:
    @pytest.mark.parametrize("make_scm", [_make_spring], ids=["spring"])
    def test_oracle_cf_has_zero_gap(self, make_scm):
        x, mech = make_scm(seed=4)
        theta = 0.0
        t0 = 10
        oracle = structural_counterfactual(x, mech, t0, [0], [x[t0, 0] + 2.0], noiseless=False)
        model = _ThresholdModel(theta)
        out = pns_direction(np.stack([x]), np.stack([oracle]), model, mech, theta, target_class=1)
        assert out["n_scorable"] == 1
        assert out["delta_trajectory"] == pytest.approx(0.0)
        assert out["delta_outcome"] == pytest.approx(0.0)
        assert out["delta_total"] == pytest.approx(0.0)

    @pytest.mark.parametrize("make_scm", [_make_spring], ids=["spring"])
    def test_flags_model_artifact_cf(self, make_scm):
        x, mech = make_scm(seed=5)
        t0 = 10
        x_cf = x.copy()
        x_cf[t0, 0] += 1e-3  # a real but tiny intervention
        x_cf[t0 + 1 :] += 50.0  # trajectory displaced by something else entirely
        theta = 10.0
        out = pns_direction(
            np.stack([x]), np.stack([x_cf]), _AlwaysTarget(), mech, theta, target_class=1
        )
        assert out["A_model_proposed"] == 1.0, "model claims a successful flip"
        assert out["C_world_oracle"] == 0.0, "world says the intervention does nothing"
        assert out["delta_total"] == pytest.approx(1.0), "the over-claim must be flagged"


class TestPNSCombination:
    def test_reports_terms_separately_so_a_collapsed_pn_is_visible(self):
        """PN can collapse to 0 for structural reasons (long-horizon decay).
        The combined PNS must not be able to hide that."""
        ps_dir = {"C_world_oracle": 0.8, "delta_total": 0.1, "n_scorable": 10}
        pn_dir = {"C_world_oracle": 0.0, "delta_total": 0.9, "n_scorable": 10}
        out = pns_from_directions(ps_dir, pn_dir, p_xy=0.5, p_xpyp=0.5)
        assert out["PNS_world"] == pytest.approx(0.4)
        assert out["PN_world"] == 0.0  # visible, not folded away
        assert out["PS_world"] == 0.8
