"""Smoke tests for CF explainer and attribution method wrappers."""

import numpy as np
import pytest

from causal_tscf_bench.classifiers.tcn import TCNClassifier
from causal_tscf_bench.methods.counterfactual.comte import CoMTE
from causal_tscf_bench.methods.counterfactual.tsevo import TSEvo
from causal_tscf_bench.methods.counterfactual.dice import DiCE
from causal_tscf_bench.methods.attribution.timeshap import TimeSHAP
from causal_tscf_bench.methods.attribution.dynamask import Dynamask


def make_data_and_clf(N=60, T=15, M=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, T, M)).astype(np.float32)
    Y = (X[:, -1, 0] > 0).astype(int)
    clf = TCNClassifier()
    clf.fit(X[:40], Y[:40], epochs=3)
    return X, Y, clf


@pytest.mark.parametrize("CFClass", [CoMTE, TSEvo, DiCE])
def test_cf_explainer_output_shape(CFClass):
    X, Y, clf = make_data_and_clf()
    explainer = CFClass()
    explainer.fit(X[:40], clf)
    x_query = X[40]
    x_cf = explainer.explain(x_query, target_class=1, classifier=clf)
    assert x_cf.shape == x_query.shape


@pytest.mark.parametrize("CFClass", [CoMTE, TSEvo, DiCE])
def test_cf_explainer_no_nans(CFClass):
    X, Y, clf = make_data_and_clf()
    explainer = CFClass()
    explainer.fit(X[:40], clf)
    x_cf = explainer.explain(X[40], target_class=0, classifier=clf)
    assert not np.isnan(x_cf).any()


@pytest.mark.parametrize("AttrClass", [TimeSHAP, Dynamask])
def test_attribution_output_shape(AttrClass):
    X, Y, clf = make_data_and_clf()
    method = AttrClass()
    if hasattr(method, "fit"):
        method.fit(X[:40])
    phi = method.attribute(X[0], clf)
    assert phi.shape == X[0].shape


@pytest.mark.parametrize("AttrClass", [TimeSHAP, Dynamask])
def test_attribution_no_nans(AttrClass):
    X, Y, clf = make_data_and_clf()
    method = AttrClass()
    if hasattr(method, "fit"):
        method.fit(X[:40])
    phi = method.attribute(X[0], clf)
    assert not np.isnan(phi).any()
