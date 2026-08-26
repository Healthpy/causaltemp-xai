"""Tests for the native CF generators on a small trained LSTM.

Verifies output shapes/finiteness, that Wachter flips at least one label, that
Noiseless SCM recourse has zero retroactive change and is CF-faith (rollout) hard=1 by
construction, and that TSCausalCF's FISTA proximal step and
causal-residual masking match the paper's equations cell-for-cell.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from causaltemp_xai.benchmarks.generator import LinearSCMT, NlinearSCMT, exogenous_channels
from causaltemp_xai.classifiers import LSTMClassifier
from causaltemp_xai.config import SMOKE_NL
from causaltemp_xai.methods import (
    CftsCelsCF,
    CftsCOMTECF,
    CftsConfetiCF,
    CftsCountsCF,
    CftsWachterCF,
    NoiselessSCMRecourse,
    PearlSCMRecourse,
    TSCausalCF,
    derive_intervention_t,
)
from causaltemp_xai.methods.counterfactual.causal_feasibility import (
    _batched_lag_windows,
    _build_loss_masks,
    _shift_forward,
    _soft_threshold,
)
from causaltemp_xai.methods.counterfactual.scm_recourse import _resolve_t0_candidates
from causaltemp_xai.metrics.cf_faith import CFfaith
from causaltemp_xai.scm.intervention import INTERVENTION_TOL

_phase03 = importlib.import_module("experiments.03_run_cf_methods")
_phase04 = importlib.import_module("experiments.04_evaluate_axes")


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


@pytest.fixture(scope="module")
def nonlinear_current():
    """Small fixture with the exact MLP hyperparameters registered for smoke_nl."""
    cfg = SMOKE_NL
    gen = NlinearSCMT(
        k=cfg.k,
        L=cfg.L,
        sparsity=cfg.sparsity,
        noise_type=cfg.noise_type,
        T=cfg.T,
        N=8,
        seed=cfg.seed,
        **cfg.nonlinear,
    )
    return gen.generate()


class _ReferenceShiftModel:
    """Differentiable target reached by moving away from a factual reference."""

    device = torch.device("cpu")

    def __init__(self, references, threshold=0.05, reachable=True):
        self.references = torch.as_tensor(references, dtype=torch.float32)
        self.reference_sums = self.references.sum(dim=(1, 2))
        self.threshold = float(threshold)
        self.reachable = reachable

    def torch_logits(self, Z):
        Z = torch.as_tensor(Z, dtype=torch.float32)
        if Z.ndim == 2:
            Z = Z.unsqueeze(0)
        distances = ((Z[:, None] - self.references[None]) ** 2).mean(dim=(2, 3))
        nearest = distances.argmin(dim=1)
        shift = Z.sum(dim=(1, 2)) - self.reference_sums[nearest]
        if self.reachable:
            score = shift - self.threshold
        else:
            # Always below the class-1 logit, but retains a gradient so failed
            # candidates have meaningfully different prediction losses.
            score = torch.tanh(shift / 10.0) - 2.0
        return torch.stack((-score, score), dim=1)

    def predict(self, Z):
        with torch.no_grad():
            return self.torch_logits(Z).argmax(dim=1).cpu().numpy()


class TestSCMRecourseRegression:
    @pytest.mark.parametrize("method_cls", [NoiselessSCMRecourse, PearlSCMRecourse])
    def test_current_nonlinear_fixture_produces_real_interventions(
        self, method_cls, nonlinear_current
    ):
        X = nonlinear_current["X"][:3].astype(np.float32)
        model = _ReferenceShiftModel(X)
        method = method_cls(n_steps=30, lr=0.1, t0_fractions=(0.25, 0.5))

        cfs, no_cf_found = method.generate_batch_with_status(
            X, model, nonlinear_current["graph"], nonlinear_current["mechanism"]
        )

        delta_max = np.asarray(
            [diagnostic["selected_delta_max"] for diagnostic in method.last_batch_diagnostics]
        )
        assert cfs.shape == X.shape
        assert no_cf_found.dtype == np.bool_
        assert no_cf_found.shape == (len(X),)
        assert np.mean(delta_max > INTERVENTION_TOL) >= 0.9
        assert not no_cf_found.any()

    @pytest.mark.parametrize("method_cls", [NoiselessSCMRecourse, PearlSCMRecourse])
    def test_forced_zero_delta_failure_has_status_without_breaking_legacy_api(
        self, method_cls, nonlinear_current
    ):
        X = nonlinear_current["X"][:1].astype(np.float32)
        model = _ReferenceShiftModel(X)
        method = method_cls(
            actionable_mask=np.zeros(X.shape[-1], dtype=bool),
            n_steps=2,
            t0_fractions=(0.5,),
        )

        cfs, no_cf_found = method.generate_batch_with_status(
            X, model, nonlinear_current["graph"], nonlinear_current["mechanism"]
        )
        assert isinstance(cfs, np.ndarray)
        assert no_cf_found.dtype == np.bool_
        assert no_cf_found.tolist() == [True]
        assert method.last_batch_diagnostics[0]["selected_delta_max"] == 0.0

        legacy_one = method.generate(
            X[0], model, nonlinear_current["graph"], nonlinear_current["mechanism"]
        )
        legacy_batch = method.generate_batch(
            X, model, nonlinear_current["graph"], nonlinear_current["mechanism"]
        )
        assert isinstance(legacy_one, np.ndarray) and legacy_one.shape == X[0].shape
        assert isinstance(legacy_batch, np.ndarray) and legacy_batch.shape == X.shape

    @pytest.mark.parametrize("method_cls", [NoiselessSCMRecourse, PearlSCMRecourse])
    def test_failed_candidates_are_ranked_by_prediction_loss(self, method_cls, nonlinear_current):
        x = nonlinear_current["X"][0].astype(np.float32)
        model = _ReferenceShiftModel(x[None], reachable=False)
        per_t0 = {}
        for t0 in (8, 15):
            method = method_cls(n_steps=3, lr=0.1, t0_steps=(t0,))
            _cf, found = method._generate_one(
                x, model, nonlinear_current["graph"], nonlinear_current["mechanism"]
            )
            assert found is False
            per_t0[t0] = (
                method.last_generation_diagnostics["selected_prediction_loss"],
                method.last_generation_diagnostics["selected_proximity"],
            )

        combined = method_cls(n_steps=3, lr=0.1, t0_steps=(8, 15))
        _cf, found = combined._generate_one(
            x, model, nonlinear_current["graph"], nonlinear_current["mechanism"]
        )
        assert found is False
        expected_t0 = min(per_t0, key=lambda t0: per_t0[t0])
        assert combined.last_generation_diagnostics["selected_t0"] == expected_t0
        assert combined.last_generation_diagnostics["selected_prediction_loss"] == pytest.approx(
            per_t0[expected_t0][0]
        )


class TestRecourseStatusPipeline:
    def test_phase03_writes_matching_cf_and_status_sidecars(self, tmp_path, monkeypatch):
        X = np.zeros((3, 6, 2), dtype=np.float32)
        status = np.asarray([False, True, False], dtype=bool)

        class _Method:
            def generate_batch_with_status(self, X, model, graph, mechanism):
                return X + 1.0, status.copy()

        class _Classifier:
            def score(self, X, y):
                return 1.0

            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        cfg = SimpleNamespace(
            name="test_nl", seed=0, mechanism_type="mlp", k=2, T=6, noise_type="laplace"
        )
        data = {
            "X_train": X,
            "Y_train": np.zeros(len(X), dtype=int),
            "X_test": X,
            "Y_test": np.zeros(len(X), dtype=int),
            "graph": np.zeros((2, 2, 1)),
            "mechanism": object(),
        }
        out_dir = tmp_path / "data"
        ckpt = out_dir / cfg.name / "lstm.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.touch()
        results_dir = tmp_path / "results"

        monkeypatch.setattr(_phase03, "get_config", lambda name: cfg)
        monkeypatch.setattr(_phase03, "load_dataset", lambda name, out_dir: data)
        monkeypatch.setattr(
            _phase03, "LSTMClassifier", SimpleNamespace(load=lambda path: _Classifier())
        )
        monkeypatch.setattr(_phase03, "select_flip_candidates", lambda clf, X, n: np.arange(n))
        monkeypatch.setattr(
            _phase03,
            "build_methods",
            lambda X, y: {
                "NoiselessSCMRecourse": _Method(),
                "PearlSCMRecourse": _Method(),
            },
        )
        monkeypatch.setattr(
            _phase03,
            "config_dir",
            lambda name, classifier="lstm": results_dir / name / classifier,
        )

        _phase03.run(cfg.name, len(X), out_dir, skip_aux=True)

        cf_dir = results_dir / cfg.name / "lstm" / "cf"
        for method_name in ("NoiselessSCMRecourse", "PearlSCMRecourse"):
            assert np.load(cf_dir / f"X_cf_{method_name}.npy").shape == X.shape
            saved_status = np.load(cf_dir / f"no_cf_found_{method_name}.npy")
            assert saved_status.dtype == np.bool_
            assert np.array_equal(saved_status, status)

    @pytest.mark.parametrize("method_name", ["NoiselessSCMRecourse", "PearlSCMRecourse"])
    @pytest.mark.parametrize("failure", ["missing", "dtype", "length"])
    def test_phase04_rejects_invalid_required_sidecars(self, tmp_path, method_name, failure):
        path = tmp_path / f"no_cf_found_{method_name}.npy"
        if failure == "dtype":
            np.save(path, np.zeros(3, dtype=np.int8))
        elif failure == "length":
            np.save(path, np.zeros(2, dtype=bool))

        with pytest.raises(SystemExit):
            _phase04.load_no_cf_found(tmp_path, method_name, n=3)

    def test_phase04_accepts_matching_boolean_sidecar(self, tmp_path):
        expected = np.asarray([False, True, False], dtype=bool)
        np.save(tmp_path / "no_cf_found_NoiselessSCMRecourse.npy", expected)
        got = _phase04.load_no_cf_found(tmp_path, "NoiselessSCMRecourse", n=3)
        assert np.array_equal(got, expected)


# ---------------------------------------------------------------------------
# Noiseless SCM recourse
# ---------------------------------------------------------------------------


class TestNoiselessSCMRecourse:
    def test_shape_and_finite(self, trained):
        clf, data = trained
        x = data["X"][0]
        cf = NoiselessSCMRecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        assert cf.shape == x.shape
        assert np.all(np.isfinite(cf))

    def test_zero_retroactive_change(self, trained):
        clf, data = trained
        x = data["X"][3]
        cf = NoiselessSCMRecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        t0 = derive_intervention_t(x, cf)
        # Nothing before the derived intervention point may change.
        assert np.allclose(cf[:t0], x[:t0], atol=1e-6)

    def test_cf_faith_rollout_hard_is_one(self, trained):
        """NoiselessSCMRecourse emits a noiseless rollout → rollout-faith hard=1 by construction."""
        clf, data = trained
        x = data["X"][3]
        cf = NoiselessSCMRecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        t0 = derive_intervention_t(x, cf)
        scorer = CFfaith(tol=1e-3, semantics="noiseless_rollout")
        result = scorer.score(x, cf, t0, data["graph"], data["mechanism"])
        assert result["hard"] == 1.0, f"expected rollout hard=1, got {result}"


# ---------------------------------------------------------------------------
# Pearl-NoiselessSCMRecourse (M2, 2026-07-08) — noise-reinjecting recourse variant
# ---------------------------------------------------------------------------


class TestPearlSCMRecourse:
    """Mirrors TestNoiselessSCMRecourse, but the invariant that holds "by construction" is
    ``cf_faith_pearl_hard == 1`` (not ``rollout_hard``) — see
    ``causaltemp_xai/methods/counterfactual/scm_recourse.py``'s module docstring for
    why the two variants differ, and the docstring of ``PearlSCMRecourse``
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
        cf = PearlSCMRecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        assert cf.shape == x.shape
        assert np.all(np.isfinite(cf))

    def test_zero_retroactive_change(self, trained):
        """Still holds: both NoiselessSCMRecourse and PearlSCMRecourse copy
        ``x[:t0]`` verbatim into the CF regardless of forward-rollout
        semantics (only the post-t0 region differs between the two)."""
        clf, data = trained
        x = data["X"][3]
        cf = PearlSCMRecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        t0 = derive_intervention_t(x, cf)
        assert np.allclose(cf[:t0], x[:t0], atol=1e-6)

    def test_cf_faith_pearl_hard_is_one(self, trained):
        """PearlSCMRecourse reinjects the abducted factual noise -> Pearl-faith
        hard=1 by construction (the rollout-semantics scorer should generally
        NOT also score hard=1 -- the two are structurally different CFs)."""
        clf, data = trained
        x = data["X"][3]
        cf = PearlSCMRecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        t0 = derive_intervention_t(x, cf)
        pearl_scorer = CFfaith(tol=1e-3, semantics="pearl_delta")
        result = pearl_scorer.score(x, cf, t0, data["graph"], data["mechanism"])
        assert result["hard"] == 1.0, f"expected pearl hard=1, got {result}"

    def test_generate_batch_shape(self, trained):
        clf, data = trained
        X = data["X"][:4]
        cfs = PearlSCMRecourse(target_class=1, n_steps=40, t0_fractions=(0.5,)).generate_batch(
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
        """Unlike NoiselessSCMRecourse, this method has no do()-timestep search --
        Delta is free over the whole trajectory (see the class module
        docstring on why: the paper has no intervention time at all).

        The budget here is deliberately not tiny: with the backtracking line
        search the iterates move by an admissible step rather than the old
        fixed ``lr``-sized one, and the first flip lands a few hundred steps
        in rather than immediately. A flip inside 150 steps used to "pass"
        only because the unstable iteration was wandering far off-manifold.
        """
        clf, data = trained
        X = data["X"]
        preds = clf.predict(X[:20])
        src = [i for i in range(20) if preds[i] == 0][:3] or list(range(3))
        method = TSCausalCF(target_class=1, n_steps=600)
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
        would compare NoiselessSCMRecourse at a pinned horizon against PearlSCMRecourse at 25/50."""
        for cls in (NoiselessSCMRecourse, PearlSCMRecourse):
            m = cls(target_class=1, t0_steps=(42,))
            assert m.t0_steps == (42,)
            assert _resolve_t0_candidates(100, m.t0_fractions, m.t0_steps) == [42]
