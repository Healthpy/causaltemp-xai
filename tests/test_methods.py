"""Tests for the native CF generators on a small trained LSTM.

Verifies output shapes/finiteness, that Wachter flips at least one label, that
CARLA-causal has zero retroactive change and is CF-faith (rollout) hard=1 by
construction, and that TSCausalCF's FISTA proximal step and
causal-residual masking match the paper's equations cell-for-cell.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from causaltemp_xai.benchmarks.generator import LinearSCMT, exogenous_channels
from causaltemp_xai.classifiers import LSTMClassifier
from causaltemp_xai.methods import (
    CARLARecourse,
    CftsCelsCF,
    CftsCOMTECF,
    CftsConfetiCF,
    CftsCountsCF,
    CftsWachterCF,
    PearlCARLARecourse,
    TSCausalCF,
    derive_intervention_t,
)
from causaltemp_xai.methods.counterfactual.carla import _resolve_t0_candidates
from causaltemp_xai.methods.counterfactual.causal_feasibility import (
    _batched_lag_windows,
    _build_loss_masks,
    _shift_forward,
    _soft_threshold,
)
from causaltemp_xai.metrics.cf_faith import CFfaith


@pytest.fixture(scope="module")
def trained():
    """Small VAR dataset + lightly trained LSTMClassifier (shared across tests)."""
    gen = LinearSCMT(k=5, L=1, T=30, N=300, seed=0)
    data = gen.generate()
    X, Y = data["X"], data["Y"]
    clf = LSTMClassifier(
        n_inputs=5,
        hidden_size=16,
        num_layers=2,
        dropout=0.0,
        lr=5e-3,
        batch_size=64,
        max_epochs=25,
        patience=25,
        seed=0,
    )
    clf.fit(X[:240], Y[:240], X[240:], Y[240:])
    return clf, data


# ---------------------------------------------------------------------------
# CARLA-causal
# ---------------------------------------------------------------------------


class TestCARLA:
    def test_shape_and_finite(self, trained):
        clf, data = trained
        x = data["X"][0]
        cf = CARLARecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        assert cf.shape == x.shape
        assert np.all(np.isfinite(cf))

    def test_zero_retroactive_change(self, trained):
        clf, data = trained
        x = data["X"][3]
        cf = CARLARecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        t0 = derive_intervention_t(x, cf)
        # Nothing before the derived intervention point may change.
        assert np.allclose(cf[:t0], x[:t0], atol=1e-6)

    def test_cf_faith_rollout_hard_is_one(self, trained):
        """CARLA emits a noiseless rollout → rollout-faith hard=1 by construction."""
        clf, data = trained
        x = data["X"][3]
        cf = CARLARecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        t0 = derive_intervention_t(x, cf)
        scorer = CFfaith(tol=1e-3, semantics="noiseless_rollout")
        result = scorer.score(x, cf, t0, data["graph"], data["mechanism"])
        assert result["hard"] == 1.0, f"expected rollout hard=1, got {result}"


# ---------------------------------------------------------------------------
# Pearl-CARLA (M2, 2026-07-08) — noise-reinjecting recourse variant
# ---------------------------------------------------------------------------


class TestPearlCARLA:
    """Mirrors TestCARLA, but the invariant that holds "by construction" is
    ``cf_faith_pearl_hard == 1`` (not ``rollout_hard``) — see
    ``causaltemp_xai/methods/counterfactual/carla.py``'s module docstring for
    why the two variants differ, and the docstring of ``PearlCARLARecourse``
    for the empirical long-horizon-validity finding (not a naive "always
    recovers validity" result — smoke-scale confirmed instead: zero
    retroactive change still holds bit-identically pre-t0 because both
    variants copy the factual prefix verbatim regardless of rollout
    semantics; only the *forward* (post-t0) reference trajectory differs
    between rollout and pearl_delta semantics).
    """

    def test_shape_and_finite(self, trained):
        clf, data = trained
        x = data["X"][0]
        cf = PearlCARLARecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        assert cf.shape == x.shape
        assert np.all(np.isfinite(cf))

    def test_zero_retroactive_change(self, trained):
        """Still holds: both CARLARecourse and PearlCARLARecourse copy
        ``x[:t0]`` verbatim into the CF regardless of forward-rollout
        semantics (only the post-t0 region differs between the two)."""
        clf, data = trained
        x = data["X"][3]
        cf = PearlCARLARecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        t0 = derive_intervention_t(x, cf)
        assert np.allclose(cf[:t0], x[:t0], atol=1e-6)

    def test_cf_faith_pearl_hard_is_one(self, trained):
        """PearlCARLARecourse reinjects the abducted factual noise -> Pearl-faith
        hard=1 by construction (the rollout-semantics scorer should generally
        NOT also score hard=1 -- the two are structurally different CFs)."""
        clf, data = trained
        x = data["X"][3]
        cf = PearlCARLARecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        t0 = derive_intervention_t(x, cf)
        pearl_scorer = CFfaith(tol=1e-3, semantics="pearl_delta")
        result = pearl_scorer.score(x, cf, t0, data["graph"], data["mechanism"])
        assert result["hard"] == 1.0, f"expected pearl hard=1, got {result}"

    def test_generate_batch_shape(self, trained):
        clf, data = trained
        X = data["X"][:4]
        cfs = PearlCARLARecourse(target_class=1, n_steps=40, t0_fractions=(0.5,)).generate_batch(
            X, clf, data["graph"], data["mechanism"]
        )
        assert cfs.shape == X.shape
        assert np.all(np.isfinite(cfs))


# ---------------------------------------------------------------------------
# TSCausalCF (M3, 2026-08-04) — SCM-regularised recourse
# (Bahri et al., IEEE BigData 2025)
# ---------------------------------------------------------------------------


class TestTSCausalCFHelpers:
    """The FISTA building blocks, tested in isolation: soft-thresholding is a
    real proximal operator (not folded into the loss like Adam+L1 would be),
    and the lag-window construction matches
    ``causaltemp_xai.benchmarks.mechanisms.lag_window``'s per-timestep,
    zero-pad-before-t=0 contract, just vectorised over every t at once.
    """

    def test_soft_threshold_shrinks_toward_zero(self):
        z = torch.tensor([-5.0, -0.5, 0.0, 0.5, 5.0])
        out = _soft_threshold(z, torch.tensor(1.0))
        assert torch.allclose(out, torch.tensor([-4.0, 0.0, 0.0, 0.0, 4.0]))

    def test_soft_threshold_zero_threshold_is_identity(self):
        z = torch.tensor([-3.0, 0.0, 2.5])
        out = _soft_threshold(z, torch.tensor(0.0))
        assert torch.allclose(out, z)

    def test_shift_forward_zero_pads_the_front(self):
        x = torch.arange(5.0).reshape(5, 1)  # [[0],[1],[2],[3],[4]]
        shifted = _shift_forward(x, shift=2)
        assert torch.allclose(shifted, torch.tensor([[0.0], [0.0], [0.0], [1.0], [2.0]]))

    def test_shift_forward_shift_zero_is_identity(self):
        x = torch.arange(4.0).reshape(4, 1)
        assert torch.allclose(_shift_forward(x, shift=0), x)

    def test_shift_forward_shift_exceeding_length_is_all_zero(self):
        x = torch.ones(3, 2)
        assert torch.allclose(_shift_forward(x, shift=10), torch.zeros(3, 2))

    def test_batched_lag_windows_matches_per_timestep_construction(self):
        """window[t, -1] must equal x[t-1] (lag 1, zero before t=0) -- the same
        contract as ``lag_window(arr, t, L, k)`` called once per t, but built
        here for every t in one vectorised pass."""
        from causaltemp_xai.benchmarks.mechanisms import lag_window

        rng = np.random.default_rng(0)
        x_np = rng.normal(size=(6, 3)).astype(np.float32)
        x = torch.as_tensor(x_np)
        L = 2
        windows = _batched_lag_windows(x, L)
        assert windows.shape == (6, L, 3)
        for t in range(6):
            expected = lag_window(x_np, t, L, 3)
            assert np.allclose(windows[t].numpy(), expected)

    def test_masks_no_typing_is_all_causal(self):
        """U_s and U_d both empty (this benchmark's actual full/full_nl case,
        2026-08-04): every cell is causal, the proximal step is
        a structural no-op everywhere."""
        prox_mask, causal_mask, thresh = _build_loss_masks(
            T=4, k=3, u_s=[], u_d=[], lam_s=1.0, lam_d=13.0
        )
        assert not prox_mask.any()
        assert causal_mask.all()
        assert not thresh.any()

    def test_masks_static_exogenous_is_prox_at_every_t(self):
        prox_mask, _causal_mask, thresh = _build_loss_masks(
            T=4, k=3, u_s=[1], u_d=[], lam_s=2.0, lam_d=13.0
        )
        assert prox_mask[:, 1].all()
        assert not prox_mask[:, [0, 2]].any()
        assert torch.allclose(thresh[:, 1], torch.full((4,), 2.0))

    def test_masks_dynamic_exogenous_is_prox_only_at_t0(self):
        prox_mask, causal_mask, thresh = _build_loss_masks(
            T=4, k=3, u_s=[], u_d=[2], lam_s=1.0, lam_d=13.0
        )
        assert bool(prox_mask[0, 2])
        assert not prox_mask[1:, 2].any(), "U_d is only prox-only at t=0 (eq. 3/4)"
        assert causal_mask[1:, 2].all(), "U_d must be causally checked for t >= 1"
        assert float(thresh[0, 2]) == 13.0
        assert not thresh[1:, 2].any()

    def test_masks_prox_and_causal_partition_every_cell(self):
        """Every (t, channel) cell is exactly one of prox or causal -- never
        both, never neither -- for an arbitrary typing."""
        prox_mask, causal_mask, _ = _build_loss_masks(
            T=5, k=4, u_s=[0], u_d=[3], lam_s=1.0, lam_d=13.0
        )
        assert torch.equal(causal_mask, ~prox_mask)
        assert (prox_mask.to(torch.int) + causal_mask.to(torch.int)).eq(1).all()


class TestTSCausalCF:
    def test_shape_and_finite(self, trained):
        clf, data = trained
        x = data["X"][0]
        cf = TSCausalCF(target_class=1, n_steps=60).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        assert cf.shape == x.shape
        assert np.all(np.isfinite(cf))

    def test_flips_at_least_one(self, trained):
        """Unlike CARLARecourse, this method has no do()-timestep search --
        Delta is free over the whole trajectory (see the class module
        docstring on why: the paper has no intervention time at all)."""
        clf, data = trained
        X = data["X"]
        preds = clf.predict(X[:20])
        src = [i for i in range(20) if preds[i] == 0][:3] or list(range(3))
        method = TSCausalCF(target_class=1, n_steps=150)
        flips = 0
        for i in src:
            cf = method.generate(X[i], clf, data["graph"], data["mechanism"])
            if clf.predict(cf) == 1:
                flips += 1
        assert flips >= 1, "TSCausalCF failed to flip any selected instance"

    def test_generate_batch_shape(self, trained):
        clf, data = trained
        X = data["X"][:4]
        cfs = TSCausalCF(target_class=1, n_steps=40).generate_batch(
            X, clf, data["graph"], data["mechanism"]
        )
        assert cfs.shape == X.shape
        assert np.all(np.isfinite(cfs))

    def test_lam_zero_disables_the_causal_penalty(self, trained):
        """With lam=0 the objective is plain Wachter (L_pred only) -- the
        causal-residual and proximity terms both vanish (lam_d = lam_v = lam
        per the paper's own D1 trade-off study), so this is a sanity check
        that the causal term is genuinely additive, not silently always-on."""
        clf, data = trained
        x = data["X"][3]
        cf_reg = TSCausalCF(target_class=1, n_steps=80, lam=13.0).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        cf_unreg = TSCausalCF(target_class=1, n_steps=80, lam=0.0).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        assert not np.allclose(cf_reg, cf_unreg, atol=1e-4), (
            "lam=13 and lam=0 produced the same trajectory -- the causal "
            "penalty is not affecting the optimisation"
        )

    def test_typing_uses_the_benchmark_exogenous_channels(self, trained):
        """Whatever :func:`exogenous_channels` reports for this fixture's
        graph is exactly what ends up proximity-only at t=0 -- the one place
        this benchmark's U_s/U_d/V decision (2026-08-04) is
        exercised end-to-end, not just in the mask-construction unit tests
        below."""
        _clf, data = trained
        exo = exogenous_channels(data["graph"])
        T, k = data["X"][0].shape
        prox_mask, causal_mask, thresh = _build_loss_masks(T, k, [], exo, lam_s=1.0, lam_d=13.0)
        for j in range(k):
            assert bool(prox_mask[0, j]) == (j in exo)
            assert bool(causal_mask[0, j]) == (j not in exo)
            assert float(thresh[0, j]) == (13.0 if j in exo else 0.0)
        # Every cell at t >= 1 is causal regardless of typing (eq. 4: U_d only
        # gets the proximity exemption at the literal first timestep).
        if T > 1:
            assert bool(causal_mask[1:].all())


# ---------------------------------------------------------------------------
# cfts-backed methods (M3 DoD: every wired method needs an output-contract
# test — these five were registered in build_methods() with zero direct
# behavioural coverage; test_experiment_registry.py only checked isinstance,
# never called .generate()/.generate_batch() on any of them).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfts_dataset(trained):
    """The same (X, y) split ``trained`` fit the classifier on, wrapped for
    the cfts-backed methods' reference-dataset argument."""
    from causaltemp_xai.methods.counterfactual.cfts_methods import _DatasetAdapter

    _clf, data = trained
    return _DatasetAdapter(data["X"][:240], data["Y"][:240])


class TestCftsWachter:
    def test_shape_and_finite(self, trained, cfts_dataset):
        clf, data = trained
        x = data["X"][3]
        cf = CftsWachterCF(target_class=1, dataset=cfts_dataset, max_cfs=50).generate(x, clf)
        assert cf.shape == x.shape
        assert np.all(np.isfinite(cf))

    def test_generate_batch_shape(self, trained, cfts_dataset):
        clf, data = trained
        X = data["X"][:3]
        cfs = CftsWachterCF(target_class=1, dataset=cfts_dataset, max_cfs=50).generate_batch(X, clf)
        assert cfs.shape == X.shape
        assert np.all(np.isfinite(cfs))

    def test_flips_at_least_one(self, trained, cfts_dataset):
        """Carried over from the removed native WachterCF's test (2026-08-05,
        -- CftsWachterCF is what the pipeline actually uses, so
        this is the one that should carry the invariant now."""
        clf, data = trained
        X = data["X"]
        preds = clf.predict(X[:20])
        src = [i for i in range(20) if preds[i] == 0][:3] or list(range(3))
        method = CftsWachterCF(target_class=1, dataset=cfts_dataset, max_cfs=100)
        flips = 0
        for i in src:
            cf = method.generate(X[i], clf)
            if clf.predict(cf) == 1:
                flips += 1
        assert flips >= 1, "CftsWachterCF failed to flip any of the selected instances"


class TestCftsCOMTE:
    def test_shape_and_finite(self, trained, cfts_dataset):
        clf, data = trained
        x = data["X"][3]
        cf = CftsCOMTECF(target_class=1, dataset=cfts_dataset).generate(x, clf)
        assert cf.shape == x.shape
        assert np.all(np.isfinite(cf))

    def test_generate_batch_shape(self, trained, cfts_dataset):
        clf, data = trained
        X = data["X"][:3]
        cfs = CftsCOMTECF(target_class=1, dataset=cfts_dataset).generate_batch(X, clf)
        assert cfs.shape == X.shape
        assert np.all(np.isfinite(cfs))


class TestCftsConfeti:
    def test_shape_and_finite(self, trained, cfts_dataset):
        clf, data = trained
        x = data["X"][3]
        cf = CftsConfetiCF(
            target_class=1, dataset=cfts_dataset, max_iterations=10, population_size=10
        ).generate(x, clf)
        assert cf.shape == x.shape
        assert np.all(np.isfinite(cf))

    def test_generate_batch_shape(self, trained, cfts_dataset):
        clf, data = trained
        X = data["X"][:3]
        cfs = CftsConfetiCF(
            target_class=1, dataset=cfts_dataset, max_iterations=10, population_size=10
        ).generate_batch(X, clf)
        assert cfs.shape == X.shape
        assert np.all(np.isfinite(cfs))


class TestCftsCounts:
    """CounTS trains its own internal VAE per call (not our LSTM) -- kept to
    a handful of epochs/steps here purely for test speed; the output-contract
    check does not depend on the VAE having converged."""

    def test_shape_and_finite(self, trained, cfts_dataset):
        clf, data = trained
        x = data["X"][3]
        cf = CftsCountsCF(
            target_class=1, dataset=cfts_dataset, train_epochs=5, max_iter=20
        ).generate(x, clf)
        assert cf.shape == x.shape
        assert np.all(np.isfinite(cf))

    def test_generate_batch_shape(self, trained, cfts_dataset):
        clf, data = trained
        X = data["X"][:2]  # CounTS retrains its VAE per instance -- keep this one small
        cfs = CftsCountsCF(
            target_class=1, dataset=cfts_dataset, train_epochs=5, max_iter=20
        ).generate_batch(X, clf)
        assert cfs.shape == X.shape
        assert np.all(np.isfinite(cfs))


class TestCftsCels:
    def test_shape_and_finite(self, trained, cfts_dataset):
        clf, data = trained
        x = data["X"][3]
        cf = CftsCelsCF(target_class=1, dataset=cfts_dataset, max_iter=20).generate(x, clf)
        assert cf.shape == x.shape
        assert np.all(np.isfinite(cf))

    def test_generate_batch_shape(self, trained, cfts_dataset):
        clf, data = trained
        X = data["X"][:3]
        cfs = CftsCelsCF(target_class=1, dataset=cfts_dataset, max_iter=20).generate_batch(X, clf)
        assert cfs.shape == X.shape
        assert np.all(np.isfinite(cfs))


class TestT0CandidateResolution:
    """`t0` selection is the M4d horizon-sweep axis. It must be pinnable in
    absolute steps, must never silently substitute a horizon the caller did not
    ask for, and must leave every pre-M4d run byte-identical when unused.
    """

    def test_absolute_steps_take_precedence_over_fractions(self):
        got = _resolve_t0_candidates(100, (0.25, 0.5), t0_steps=(90,))
        assert got == [90], "t0_steps must win; a sweep pinned to 90 cannot run at 25/50"

    def test_fractions_are_unchanged_when_steps_is_none(self):
        """Regression guard on every committed result: the default path must
        resolve exactly as it did before t0_steps existed."""
        assert _resolve_t0_candidates(100, (0.25, 0.5)) == [25, 50]
        assert _resolve_t0_candidates(30, (0.25, 0.5)) == [8, 15]

    def test_absolute_is_scale_free_where_fractional_is_not(self):
        """The reason the sweep is specified in absolute steps: the same
        fraction leaves a different horizon at different T, which is what makes
        smoke-validated values hopeless at full scale."""
        assert _resolve_t0_candidates(30, (0.25,)) == [8]  # horizon 22
        assert _resolve_t0_candidates(100, (0.25,)) == [25]  # horizon 75
        for T in (30, 100, 250):
            assert _resolve_t0_candidates(T, (0.25,), t0_steps=(T - 10,)) == [T - 10]

    def test_out_of_range_steps_raise_rather_than_fall_back(self):
        """Silently falling back to T//2 would label a sweep point with a
        horizon it never evaluated -- a fabricated row in the decay curve."""
        with pytest.raises(ValueError, match="no t0 in t0_steps"):
            _resolve_t0_candidates(100, (0.25,), t0_steps=(99,))
        with pytest.raises(ValueError, match="no t0 in t0_steps"):
            _resolve_t0_candidates(100, (0.25,), t0_steps=(0,))

    def test_partially_valid_steps_keep_the_valid_ones(self):
        assert _resolve_t0_candidates(100, (0.25,), t0_steps=(0, 50, 99)) == [50]

    def test_degeneracy_bound_is_respected(self):
        """t0 == T-1 is NaN'd by the CF-faith degeneracy gate, so it must never
        be proposed -- it would manufacture unscorable CFs at the far end of the
        sweep, exactly where the horizon claim is decided."""
        with pytest.raises(ValueError):
            _resolve_t0_candidates(50, (0.25,), t0_steps=(49,))
        assert _resolve_t0_candidates(50, (0.25,), t0_steps=(48,)) == [48]

    def test_both_variants_accept_the_pin(self):
        """R9 + a real hazard: if only one variant honoured t0_steps the sweep
        would compare CARLA at a pinned horizon against PearlCARLA at 25/50."""
        for cls in (CARLARecourse, PearlCARLARecourse):
            m = cls(target_class=1, t0_steps=(42,))
            assert m.t0_steps == (42,)
            assert _resolve_t0_candidates(100, m.t0_fractions, m.t0_steps) == [42]
