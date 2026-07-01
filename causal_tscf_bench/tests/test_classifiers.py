"""Smoke tests for classifier training and inference."""

import numpy as np
import pytest

from causal_tscf_bench.classifiers.tcn import TCNClassifier
from causal_tscf_bench.classifiers.lstm import LSTMClassifier
from causal_tscf_bench.classifiers.transformer import TransformerClassifier


def make_data(N=80, T=20, M=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, T, M)).astype(np.float32)
    Y = (X[:, -1, 0] > 0).astype(int)
    return X[:60], Y[:60], X[60:], Y[60:]


@pytest.mark.parametrize("CLF", [TCNClassifier, LSTMClassifier, TransformerClassifier])
def test_classifier_fit_predict(CLF):
    X_tr, Y_tr, X_val, Y_val = make_data()
    clf = CLF()
    clf.fit(X_tr, Y_tr, X_val=X_val, Y_val=Y_val, epochs=2)
    preds = clf.predict(X_val)
    assert preds.shape == (len(X_val),)
    assert set(preds.tolist()).issubset({0, 1})


@pytest.mark.parametrize("CLF", [TCNClassifier, LSTMClassifier, TransformerClassifier])
def test_classifier_predict_proba_shape(CLF):
    X_tr, Y_tr, X_val, Y_val = make_data()
    clf = CLF()
    clf.fit(X_tr, Y_tr, epochs=2)
    proba = clf.predict_proba(X_val)
    assert proba.shape == (len(X_val), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("CLF", [TCNClassifier, LSTMClassifier, TransformerClassifier])
def test_classifier_single_instance(CLF):
    X_tr, Y_tr, _, _ = make_data()
    clf = CLF()
    clf.fit(X_tr, Y_tr, epochs=2)
    x_single = X_tr[[0]]
    pred = clf.predict(x_single)
    assert pred.shape == (1,)
