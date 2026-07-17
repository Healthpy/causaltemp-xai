"""End-to-end validation of ``evaluate_method`` against manual SCM simulation.

This covers the MVP DoD item "CF-faith validated vs manual SCM simulation on
≥10 examples": we hand-build ≥10 SCM-faithful counterfactuals (intervene at
``t0``, propagate noiselessly via the mechanisms) and ≥10 retroactively-edited
ones, and confirm the pipeline's aggregate CF-faith matches the manual
expectation (rollout-hard == 1 for the faithful set, 0 for the edited set).
"""

from __future__ import annotations

import numpy as np

from causaltemp_xai.benchmarks.generator import LinearSCMT
from causaltemp_xai.eval import evaluate_method
from causaltemp_xai.metrics.cf_faith import CFfaith
from causaltemp_xai.scm.intervention import derive_intervention_t

N_EXAMPLES = 12


class _AllTargetModel:
    """Mock classifier that predicts the target class for every instance."""

    def predict(self, X):
        return np.ones(np.asarray(X).shape[0], dtype=int)


def _noiseless_cf(x: np.ndarray, t0: int, delta: np.ndarray, mechanism) -> np.ndarray:
    """Build an SCM-faithful CF: x held before ``t0``, intervened at ``t0``,
    noiseless VAR rollout after (mirrors CARLA's construction in numpy)."""
    T, k = x.shape
    cf = x.copy().astype(float)
    cf[t0] = x[t0] + delta
    for t in range(t0 + 1, T):
        acc = np.zeros(k)
        for l, A in enumerate(mechanism.A_list):
            lag = t - l - 1
            if lag >= 0:
                acc += A @ cf[lag]
        cf[t] = acc
    return cf


def _dataset():
    gen = LinearSCMT(k=5, L=1, T=20, N=N_EXAMPLES + 30, seed=0)
    return gen.generate()


class TestCFFaithValidation:
    def test_faithful_cfs_score_hard_one(self):
        data = _dataset()
        X, mech, graph = data["X"], data["mechanism"], data["graph"]
        rng = np.random.default_rng(0)
        t0 = 5

        cfs = np.stack(
            [
                _noiseless_cf(X[i], t0, rng.normal(scale=1.0, size=X.shape[2]), mech)
                for i in range(N_EXAMPLES)
            ]
        )
        result = evaluate_method(
            _AllTargetModel(), X[:N_EXAMPLES], cfs, X, graph, mech, target_class=1
        )
        assert result["n"] == N_EXAMPLES
        # Built as a noiseless rollout → every example is rollout-faithful.
        assert result["cf_faith_rollout_hard"] == 1.0
        # Pearl semantics rewards a different CF; the noiseless rollout is not it.
        assert result["cf_faith_pearl_hard"] == 0.0

    def test_retroactive_cfs_score_hard_zero(self):
        data = _dataset()
        X, mech, graph = data["X"], data["mechanism"], data["graph"]
        rng = np.random.default_rng(1)

        # Perturb every timestep → not a noiseless rollout of itself → hard 0.
        cfs = X[:N_EXAMPLES] + rng.normal(scale=0.5, size=X[:N_EXAMPLES].shape)
        result = evaluate_method(
            _AllTargetModel(), X[:N_EXAMPLES], cfs, X, graph, mech, target_class=1
        )
        assert result["cf_faith_rollout_hard"] == 0.0

    def test_aggregate_matches_per_pair_manual(self):
        """The pipeline mean must equal a from-scratch per-pair computation."""
        data = _dataset()
        X, mech, graph = data["X"], data["mechanism"], data["graph"]
        rng = np.random.default_rng(2)
        t0 = 4

        # Mix faithful and edited so the mean is strictly between 0 and 1.
        cfs = []
        for i in range(N_EXAMPLES):
            if i % 2 == 0:
                cfs.append(_noiseless_cf(X[i], t0, rng.normal(size=X.shape[2]), mech))
            else:
                cfs.append(X[i] + rng.normal(scale=0.5, size=X[i].shape))
        cfs = np.stack(cfs)

        result = evaluate_method(
            _AllTargetModel(), X[:N_EXAMPLES], cfs, X, graph, mech, target_class=1
        )

        scorer = CFfaith(semantics="noiseless_rollout")
        manual = []
        for i in range(N_EXAMPLES):
            t = derive_intervention_t(X[i], cfs[i])
            manual.append(scorer.score(X[i], cfs[i], t, graph, mech)["hard"])
        manual_mean = float(np.mean(manual))

        assert result["cf_faith_rollout_hard"] == manual_mean
        assert 0.0 < manual_mean < 1.0  # genuine mix, not a degenerate all-0/all-1

    def test_returns_all_expected_keys(self):
        data = _dataset()
        X, mech, graph = data["X"], data["mechanism"], data["graph"]
        cfs = X[:N_EXAMPLES] + 0.1
        result = evaluate_method(
            _AllTargetModel(), X[:N_EXAMPLES], cfs, X, graph, mech, target_class=1
        )
        expected = {
            "validity",
            "proximity_l1",
            "proximity_l2",
            "sparsity",
            "frac_altered",
            "ood",
            "cf_faith_rollout_hard",
            "cf_faith_rollout_soft",
            "cf_faith_pearl_hard",
            "cf_faith_pearl_soft",
        }
        assert expected.issubset(result.keys())
        assert 0.0 <= result["validity"] <= 1.0
