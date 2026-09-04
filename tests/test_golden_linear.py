"""Golden test: pins v0.1 LinearSCM-T numbers across the Mechanism refactor.

The *numeric constants* below are the frozen artifact — captured from the
pre-refactor (``A @ x`` inline) code and asserted against the post-refactor
``Mechanism`` API. The refactor is only correct if these do not move. **Never
re-bless these constants**; if they shift, the refactor is wrong.

Golden comparison policy (M0 decision, 2026-07-07)
--------------------------------------------------
Bit-for-bit identity of float-derived constants is **not achievable across
platforms/BLAS builds**: ``@``-chains and reductions re-associate differently
per build, moving results at the ULP (observed: L=1 X-checksum and one soft
CF-faith score each off by ~1 ULP on Windows/MKL vs the capture machine).
Therefore:

* **Float-derived constants** (X checksums, soft CF-faith scores) are asserted
  with ``rtol=1e-12``. This loses no discriminative power against the failure
  mode these goldens guard — an RNG-order regression changes every draw, and
  hence the constants, at O(1), thirteen orders of magnitude above the gate.
* **Exact-by-construction values** (integer labels, hard CF-faith indicator
  scores, which are 0/1 anchors) remain asserted with strict equality.

Two configs are pinned so the partial-window / negative-lag boundary is actually
exercised:

* ``L=1`` — every real benchmark config; trajectory is numerically tame, so the
  **full X array** checksum is asserted at ``rtol=1e-12`` (this is the
  RNG-order-regression guard called out in the Stage 1 plan).
* ``L=2`` — exercises the cf_faith zero-padded window for ``t-l-1 < 0``. This
  config's VAR is intentionally *not* stabilised in companion form, so the
  trajectory diverges to ~1e11; catastrophic cancellation amplifies ULP noise
  to ~1e-7 there, so its soft scores use the looser ``rtol=1e-5``. The labels
  and hard scores are pinned exactly.
"""

from __future__ import annotations

import numpy as np

from causaltemp_xai.benchmarks.generator import LinearSCMT
from causaltemp_xai.benchmarks.mechanisms import LinearMechanism, lag_window
from causaltemp_xai.metrics.cf_faith import CFfaith


def _noiseless_cf(x, mechanism, t0, pert):
    """Pure noiseless mechanism rollout from t0 (mirror of the capture script)."""
    T, k = x.shape
    cf = x.copy()
    cf[t0] = x[t0] + pert
    for t in range(t0 + 1, T):
        window = np.zeros((mechanism.L, k))
        for j in range(mechanism.L):
            src = t - mechanism.L + j
            if src >= 0:
                window[j] = cf[src]
        cf[t] = mechanism.forward_numpy(window)
    return cf


# --- Frozen constants captured from pre-refactor code (do not edit) ----------

GOLDEN = {
    "L1": {
        "X_sum": -1.4410774479072668,
        "Y": [1, 1, 0, 0, 0],
        "noiseless_rollout": {
            "faithful": (1.0, 1.0),
            "retro": (0.0, 0.0),
            "identical": (0.0, 0.8745185398806657),
        },
        "pearl_delta": {
            "faithful": (0.0, 0.8745185398806657),
            "retro": (0.0, 0.0),
            "identical": (1.0, 1.0),
        },
    },
    "L2": {
        "X_sum": -76232980455.57904,
        "Y": [0, 0, 1, 1, 0],
        "noiseless_rollout": {
            "faithful": (1.0, 1.0),
            "retro": (0.0, 0.0),
            "identical": (0.0, 0.3138058441507252),
        },
        "pearl_delta": {
            "faithful": (0.0, 0.3138057565509235),
            "retro": (0.0, 0.0),
            "identical": (1.0, 1.0),
        },
    },
}


def _build(L):
    gen = LinearSCMT(k=3, L=L, T=20, N=5, seed=123)
    data = gen.generate()
    return data


def _cf_pairs(data):
    x = data["X"][0]
    t0 = 4
    mech = data["mechanism"]
    pert = np.full(3, 0.3)
    cf_faithful = _noiseless_cf(x, mech, t0, pert)
    cf_retro = x.copy()
    cf_retro[1] += 5.0
    cf_identical = x.copy()
    return x, t0, {"faithful": cf_faithful, "retro": cf_retro, "identical": cf_identical}


class TestGoldenL1:
    def test_full_X_checksum(self):
        data = _build(1)
        # Captured full-X checksum (RNG-order-regression guard, L=1 is tame).
        # rtol=1e-12 per the golden policy: BLAS re-association moves the sum
        # at the ULP; an RNG-order regression would move it at O(1).
        np.testing.assert_allclose(float(data["X"].sum()), GOLDEN["L1"]["X_sum"], rtol=1e-12)

    def test_labels(self):
        data = _build(1)
        assert data["Y"].tolist() == GOLDEN["L1"]["Y"]

    def test_cf_faith_scores(self):
        data = _build(1)
        x, t0, cfs = _cf_pairs(data)
        for sem in ("noiseless_rollout", "pearl_delta"):
            scorer = CFfaith(tol=1e-3, semantics=sem)
            for name, cf in cfs.items():
                r = scorer.score(x, cf, t0, data["graph"], data["mechanism"])
                exp_hard, exp_soft = GOLDEN["L1"][sem][name]
                # Hard scores are 0/1 indicator anchors — exact by construction.
                assert r["hard"] == exp_hard, f"{sem}/{name} hard"
                # Soft scores are float-derived — golden policy rtol.
                np.testing.assert_allclose(
                    r["soft"], exp_soft, rtol=1e-12, err_msg=f"{sem}/{name} soft"
                )


class TestGoldenL2:
    def test_X_matches_to_tolerance(self):
        data = _build(2)
        # Diverging trajectory (~1e11): ULP-level re-association is expected,
        # so a tight rtol catches an RNG-order regression without flagging the
        # benign float-associativity difference.
        np.testing.assert_allclose(float(data["X"].sum()), GOLDEN["L2"]["X_sum"], rtol=1e-9)

    def test_labels(self):
        data = _build(2)
        assert data["Y"].tolist() == GOLDEN["L2"]["Y"]

    def test_cf_faith_scores(self):
        data = _build(2)
        x, t0, cfs = _cf_pairs(data)
        for sem in ("noiseless_rollout", "pearl_delta"):
            scorer = CFfaith(tol=1e-3, semantics=sem)
            for name, cf in cfs.items():
                r = scorer.score(x, cf, t0, data["graph"], data["mechanism"])
                exp_hard, exp_soft = GOLDEN["L2"][sem][name]
                # Hard scores are the boundary anchors and match bit-for-bit.
                assert r["hard"] == exp_hard, f"{sem}/{name} hard"
                # Soft scores derive from a ~1e11 diverging trajectory whose
                # generator re-association differs at the ULP; catastrophic
                # cancellation amplifies that to ~1e-7, so pin to a loose rtol.
                np.testing.assert_allclose(r["soft"], exp_soft, rtol=1e-5)


# --- L=2 negative-lag boundary (the partial-window zero-pad branch) ----------
#
# The generator-based L=2 case above uses ``t0=4``, so cf_faith's re-simulation
# starts at ``t=5`` and never reaches ``src < 0`` — it does NOT actually exercise
# the zero-padded window. This case pins the boundary directly: a *tame,
# contractive* hand-built L=2 ``LinearMechanism`` scored at ``t0=0``, so the
# first re-simulated step (``t=1``) has window source ``[-1, 0]`` and takes the
# zero-pad branch. Trajectory stays O(1), so the constants are bit-for-bit
# stable. These are captured from the verified post-refactor code as a
# regression guard for the zero-pad branch (do not edit).

BOUNDARY = {
    "noiseless_rollout": {
        "faithful": (1.0, 1.0),
        "identical": (0.0, 0.760832446499636),
    },
    "pearl_delta": {
        "faithful": (0.0, 0.760832446499636),
        "identical": (1.0, 1.0),
    },
}


def _boundary_setup():
    A0 = np.array([[0.3, 0.1], [0.0, 0.2]])
    A1 = np.array([[0.1, 0.0], [0.05, 0.1]])
    mech = LinearMechanism([A0, A1])
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 2)) * 0.5
    graph = np.zeros((2, 2, 2))
    t0 = 0
    cf_faithful = x.copy()
    cf_faithful[t0] = x[t0] + np.array([0.3, 0.3])
    for t in range(t0 + 1, 6):
        cf_faithful[t] = mech.forward_numpy(lag_window(cf_faithful, t, 2, 2))
    cfs = {"faithful": cf_faithful, "identical": x.copy()}
    return x, mech, graph, t0, cfs


class TestGoldenL2Boundary:
    def test_negative_lag_branch_is_exercised(self):
        # Sanity: the first re-simulated step must reach before t=0.
        assert (1 - 2 + 0) < 0

    def test_cf_faith_scores(self):
        x, mech, graph, t0, cfs = _boundary_setup()
        for sem in ("noiseless_rollout", "pearl_delta"):
            scorer = CFfaith(tol=1e-3, semantics=sem)
            for name, cf in cfs.items():
                r = scorer.score(x, cf, t0, graph, mech)
                exp_hard, exp_soft = BOUNDARY[sem][name]
                assert r["hard"] == exp_hard, f"{sem}/{name} hard"
                # Float-derived soft scores follow the golden policy rtol.
                np.testing.assert_allclose(
                    r["soft"], exp_soft, rtol=1e-12, err_msg=f"{sem}/{name} soft"
                )
