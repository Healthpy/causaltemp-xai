"""Regression test: every implemented, unit-tested CF method must actually be
wired into ``experiments/03_run_cf_methods.py::build_methods()`` -- the one
registry every phase-03/04/06 run reads from.

This guards against exactly the gap found during the M2 pipeline-wiring
review (see ``docs/m2_multiseed_and_pearl_carla.md``, S3): ``PearlCARLARecourse``
was fully implemented (``causaltemp_xai/methods/counterfactual/carla.py``) and
unit-tested (``tests/test_methods.py::TestPearlCARLA``) but was absent from
``build_methods()``, so it had never flowed through a real experiment run
despite looking "done" from the unit-test suite alone. A method can be
correct and tested and still never actually run -- this test exists so that
gap cannot recur silently for *any* of the seven currently-registered methods.

Phase 03 is a numbered-prefix module (not a valid ``import`` target), so it is
loaded the same way ``experiments/06_aggregate_and_report.py`` already does:
``importlib.import_module("experiments.03_run_cf_methods")``.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from causaltemp_xai.methods import (
    CARLARecourse,
    CftsCelsCF,
    CftsCOMTECF,
    CftsConfetiCF,
    CftsCountsCF,
    CftsWachterCF,
    PearlCARLARecourse,
)

_phase03 = importlib.import_module("experiments.03_run_cf_methods")

#: method key -> expected class, mirroring build_methods()'s current roster.
EXPECTED_REGISTRY = {
    "CARLA": CARLARecourse,
    "PearlCARLA": PearlCARLARecourse,
    "CftsWachter": CftsWachterCF,
    "CftsCOMTE": CftsCOMTECF,
    "CftsConfeti": CftsConfetiCF,
    "CftsCounts": CftsCountsCF,
    "CftsCels": CftsCelsCF,
}


@pytest.fixture(scope="module")
def methods():
    """``build_methods()`` needs a training set only to build the cfts-*
    dataset adapter; a tiny random array is enough to construct the registry
    (no fitting happens at construction time)."""
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(20, 10, 3)).astype(np.float32)
    y_train = rng.integers(0, 2, size=20)
    return _phase03.build_methods(X_train, y_train)


def test_pearl_carla_is_registered(methods):
    """The specific gap this test was written for: PearlCARLARecourse must be
    present in the real experiment registry, not just importable/unit-tested."""
    assert "PearlCARLA" in methods, (
        "'PearlCARLA' missing from experiments/03_run_cf_methods.py::build_methods() -- "
        "a fully-implemented, unit-tested method is not reachable by any real "
        "experiment run. See docs/m2_multiseed_and_pearl_carla.md S3."
    )
    assert isinstance(methods["PearlCARLA"], PearlCARLARecourse)


def test_pearl_carla_uses_validated_n_steps(methods):
    """Pinned design decision (S3 of the M2 doc): PearlCARLA must NOT silently
    inherit CARLA's speed-motivated n_steps=300 override -- its lam_prox=0.1
    default was only empirically validated at n_steps=500 (its own class
    default). If this test starts failing because someone added an explicit
    n_steps= override to the PearlCARLA entry, that is a deliberate parameter
    change that needs its own documented justification, not a silent drift.
    """
    assert methods["PearlCARLA"].n_steps == 500


def test_full_registry_has_no_silent_gaps(methods):
    """Every method this repo implements and unit-tests for CF generation is
    reachable from the real pipeline's method registry -- no method is
    fully-built but orphaned from every actual experiment run."""
    missing = sorted(set(EXPECTED_REGISTRY) - set(methods))
    assert not missing, f"missing from build_methods(): {missing}"
    for name, cls in EXPECTED_REGISTRY.items():
        assert isinstance(methods[name], cls), f"{name} is not a {cls.__name__}"
