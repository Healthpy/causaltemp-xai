"""Wrappers around the cfts library (visual-xai-for-time-series/counterfactual-explanations-for-time-series).

The cfts library is vendored as a git submodule at third_party/cfts_repo/.
Three methods are exposed via the same ``generate(x, model)`` interface used by
the rest of this project:

- ``CftsWachterCF``   — gradient-based Wachter (wachter_gradient_cf)
- ``CftsNativeGuideCF`` — instance-based Native Guide (native_guide_uni_cf)
- ``CftsCOMTECF``     — shapelet-replacement COMTE (comte_cf)

Shape contract
--------------
Our project uses (T, k) time-first arrays.  The cfts library uses channel-first
(C, L) internally and passes (N, C, L) batches to the model.  Two adapters
bridge the gap:

``_ChannelFirstAdapter`` — wraps LSTMClassifier.model (an nn.Module) so that
  cfts can call ``model(x)`` with (N, C, L) tensors and get back logits.
  Internally transposes (N, C, L) → (N, L, C) = (N, T, k) before forwarding.

``_DatasetAdapter`` — wraps (X, y) numpy arrays into an indexed sequence of
  (x, y) tuples required by dataset-consuming cfts methods.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ------------------------------------------------------------------
# Register the submodule so `import cfts.*` works
# ------------------------------------------------------------------

# parents[3] = repo root (this file lives at
# causaltemp_xai/methods/counterfactual/cfts_methods.py). Was parents[2]
# before the restructure moved the file one level deeper; the stale index
# silently worked only while the dead duplicate methods/cfts_methods.py
# (correct depth) registered the path first — fixed 2026-07-07 (M1 rerun).
_CFTS_ROOT = Path(__file__).parents[3] / "third_party" / "cfts_repo"
if str(_CFTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CFTS_ROOT))


# ------------------------------------------------------------------
# Shared adapters
# ------------------------------------------------------------------


class _ChannelFirstAdapter(nn.Module):
    """Wrap LSTMClassifier to accept (N, C, L) input from cfts methods.

    cfts detects the device via ``next(model.parameters()).device`` and passes
    tensors of shape (N, C, L).  Our LSTM expects (N, T, k) = (N, L, C), so
    we transpose before forwarding and return raw logits.
    """

    def __init__(self, lstm_classifier) -> None:
        super().__init__()
        self._inner = lstm_classifier.model  # the underlying nn.Module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L)  →  (N, L, C) = (N, T, k) for LSTM
        return self._inner(x.transpose(1, 2))


class _TimeFirstAdapter(nn.Module):
    """Wrap LSTMClassifier to accept (N, T, k) input — no transpose.

    Used by cfts methods (like CounTS) that pass the sample directly in
    time-first format and handle their own internal transposing.
    """

    def __init__(self, lstm_classifier) -> None:
        super().__init__()
        self._inner = lstm_classifier.model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._inner(x)


class _DatasetAdapter:
    """Wrap (X, y) numpy arrays into indexed (sample, label) tuples.

    cfts methods that use a reference dataset call ``dataset[i]`` and expect
    ``(x_array, label)`` tuples.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = np.asarray(X, dtype=np.float32)  # (N, T, k)
        self.y = np.asarray(y, dtype=int)  # (N,)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], int(self.y[i])


# ------------------------------------------------------------------
# Wachter
# ------------------------------------------------------------------


class CftsWachterCF:
    """Gradient-based Wachter CF via the cfts library.

    Parameters
    ----------
    target_class:
        Desired output class.
    dataset:
        Reference dataset as ``(X, y)`` numpy arrays or a pre-built
        ``_DatasetAdapter``.  Used to seed the initial CF candidate.
    max_cfs:
        Maximum gradient-descent iterations.
    distance:
        Distance metric — ``'euclidean'`` or ``'manhattan'``.
    """

    def __init__(
        self,
        target_class: int = 1,
        dataset=None,
        max_cfs: int = 1000,
        distance: str = "euclidean",
    ) -> None:
        self.target_class = target_class
        self.dataset = dataset
        self.max_cfs = max_cfs
        self.distance = distance

    def generate(self, x: np.ndarray, model) -> np.ndarray:
        """Generate a single gradient-based Wachter CF for ``x``.

        Parameters
        ----------
        x:
            Original instance, shape ``(T, k)``.
        model:
            ``LSTMClassifier`` exposing ``torch_logits``.

        Returns
        -------
        cf : ndarray of shape ``(T, k)``, or ``x`` unchanged if optimisation fails.
        """
        from cfts.cf_wachter.wachter import wachter_gradient_cf

        adapter = _ChannelFirstAdapter(model)
        ds = (
            self.dataset
            if self.dataset is not None
            else _DatasetAdapter(x[np.newaxis], np.array([0]))
        )
        cf, _ = wachter_gradient_cf(
            x,
            ds,
            adapter,
            target=self.target_class,
            max_cfs=self.max_cfs,
            distance=self.distance,
        )
        return (
            np.asarray(cf, dtype=np.float32) if cf is not None else np.asarray(x, dtype=np.float32)
        )

    def generate_batch(self, X: np.ndarray, model) -> np.ndarray:
        """Generate one CF per instance in ``X`` of shape ``(N, T, k)``."""
        return np.stack([self.generate(x, model) for x in X], axis=0)


# ------------------------------------------------------------------
# Native Guide
# ------------------------------------------------------------------


class CftsNativeGuideCF:
    """Instance-based Native Guide CF via the cfts library.

    Finds the nearest neighbour of the target class, then uses
    GradientShap to identify influential time windows and iteratively
    replaces them with segments from the guide.

    Requires ``captum`` (``pip install captum``).

    Parameters
    ----------
    target_class:
        Desired output class.
    dataset:
        Reference dataset ``(X, y)`` or ``_DatasetAdapter``.  Required —
        Native Guide needs training examples of the target class.
    sub_len:
        Starting window size for the iterative replacement.
    """

    def __init__(
        self,
        target_class: int = 1,
        dataset=None,
        sub_len: int = 1,
    ) -> None:
        self.target_class = target_class
        self.dataset = dataset
        self.sub_len = sub_len

    def generate(self, x: np.ndarray, model) -> np.ndarray:
        """Generate a Native Guide CF for ``x``.

        Parameters
        ----------
        x:
            Original instance, shape ``(T, k)``.
        model:
            ``LSTMClassifier``.

        Returns
        -------
        cf : ndarray of shape ``(T, k)``.
        """
        from captum.attr import GradientShap
        from cfts.cf_native_guide.native_guide import native_guide_uni_cf

        if self.dataset is None:
            raise ValueError(
                "CftsNativeGuideCF requires a reference dataset. "
                "Pass dataset=(X_train, y_train) to the constructor."
            )
        ds = (
            self.dataset
            if isinstance(self.dataset, _DatasetAdapter)
            else _DatasetAdapter(*self.dataset)
        )
        adapter = _ChannelFirstAdapter(model)
        cf, _ = native_guide_uni_cf(
            x,
            ds,
            adapter,
            weight_function=GradientShap,
            sub_len=self.sub_len,
        )
        return np.asarray(cf, dtype=np.float32)

    def generate_batch(self, X: np.ndarray, model) -> np.ndarray:
        """Generate one CF per instance in ``X`` of shape ``(N, T, k)``."""
        return np.stack([self.generate(x, model) for x in X], axis=0)


# ------------------------------------------------------------------
# COMTE
# ------------------------------------------------------------------


class CftsCOMTECF:
    """Shapelet-replacement COMTE CF via the cfts library.

    COMTE iteratively replaces time-series segments with segments from
    a reference instance to steer the prediction toward the target class.

    Parameters
    ----------
    target_class:
        Desired output class.
    dataset:
        Reference dataset ``(X, y)`` or ``_DatasetAdapter``.
    """

    def __init__(
        self,
        target_class: int = 1,
        dataset=None,
    ) -> None:
        self.target_class = target_class
        self.dataset = dataset

    def generate(self, x: np.ndarray, model) -> np.ndarray:
        """Generate a COMTE CF for ``x``.

        Parameters
        ----------
        x:
            Original instance, shape ``(T, k)``.
        model:
            ``LSTMClassifier``.

        Returns
        -------
        cf : ndarray of shape ``(T, k)``.
        """
        from cfts.cf_comte.comte import comte_cf

        if self.dataset is None:
            raise ValueError(
                "CftsCOMTECF requires a reference dataset. "
                "Pass dataset=(X_train, y_train) to the constructor."
            )
        ds = (
            self.dataset
            if isinstance(self.dataset, _DatasetAdapter)
            else _DatasetAdapter(*self.dataset)
        )
        adapter = _ChannelFirstAdapter(model)
        cf, _ = comte_cf(x, ds, adapter, target_class=self.target_class)
        return (
            np.asarray(cf, dtype=np.float32) if cf is not None else np.asarray(x, dtype=np.float32)
        )

    def generate_batch(self, X: np.ndarray, model) -> np.ndarray:
        """Generate one CF per instance in ``X`` of shape ``(N, T, k)``."""
        return np.stack([self.generate(x, model) for x in X], axis=0)


# ------------------------------------------------------------------
# CONFETTI
# ------------------------------------------------------------------


class CftsConfetiCF:
    """Genetic CONFETTI CF via the cfts library.

    Selects subsequences from a reference pool and combines them with
    the original via a genetic algorithm to reach the target class.

    Parameters
    ----------
    target_class : int
    dataset : (X, y) tuple or _DatasetAdapter — reference pool.
    max_iterations : int
    population_size : int
    mutation_rate : float
    """

    def __init__(
        self,
        target_class: int = 1,
        dataset=None,
        max_iterations: int = 100,
        population_size: int = 50,
        mutation_rate: float = 0.1,
    ) -> None:
        self.target_class = target_class
        self.dataset = dataset
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def generate(self, x: np.ndarray, model) -> np.ndarray:
        from cfts.cf_confetti.confetti import confetti_genetic_cf

        if self.dataset is None:
            raise ValueError("CftsConfetiCF requires a reference dataset.")
        ds = (
            self.dataset
            if isinstance(self.dataset, _DatasetAdapter)
            else _DatasetAdapter(*self.dataset)
        )

        # confetti expects reference_data as (N, C, L); our data is (N, T, k) — transpose
        ref = np.stack([ds[i][0] for i in range(len(ds))], axis=0)  # (N, T, k)
        ref_cl = ref.transpose(0, 2, 1)  # (N, k, T) = (N, C, L)

        adapter = _ChannelFirstAdapter(model)
        cf, _ = confetti_genetic_cf(
            x,
            adapter,
            reference_data=ref_cl,
            target=self.target_class,
            max_iterations=self.max_iterations,
            population_size=self.population_size,
            mutation_rate=self.mutation_rate,
        )
        return (
            np.asarray(cf, dtype=np.float32) if cf is not None else np.asarray(x, dtype=np.float32)
        )

    def generate_batch(self, X: np.ndarray, model) -> np.ndarray:
        return np.stack([self.generate(x, model) for x in X], axis=0)


# ------------------------------------------------------------------
# CounTS
# ------------------------------------------------------------------


class CftsCountsCF:
    """VAE-based CounTS CF via the cfts library.

    Trains a conditional VAE jointly encoding the time series and class
    label, then optimises in the latent space toward the target class.

    Parameters
    ----------
    target_class : int
    dataset : (X, y) tuple or _DatasetAdapter.
    latent_dim : int
    hidden_dim : int
    train_epochs : int — VAE training epochs (per generate call if no cached model).
    max_iter : int — latent optimisation steps.
    """

    def __init__(
        self,
        target_class: int = 1,
        dataset=None,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        train_epochs: int = 30,
        max_iter: int = 500,
    ) -> None:
        self.target_class = target_class
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.train_epochs = train_epochs
        self.max_iter = max_iter

    def generate(self, x: np.ndarray, model) -> np.ndarray:
        from cfts.cf_counts.counts import counts_cf_with_pretrained_model

        if self.dataset is None:
            raise ValueError("CftsCountsCF requires a reference dataset.")
        ds = (
            self.dataset
            if isinstance(self.dataset, _DatasetAdapter)
            else _DatasetAdapter(*self.dataset)
        )
        # CounTS passes the sample directly in time-first format to the classifier and also
        # uses x_tensor in its actionability loss — so we must NOT transpose here.  We wrap
        # with _TimeFirstAdapter so the LSTM receives (N, T, k) without any transposing.
        # CounTS's shape-matching heuristic then returns (k, T) because shape[0]=T > shape[1]=k;
        # we transpose that back to (T, k).
        adapter = _TimeFirstAdapter(model)
        cf, _ = counts_cf_with_pretrained_model(
            x,
            ds,
            adapter,
            target=self.target_class,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            train_epochs=self.train_epochs,
            max_iter=self.max_iter,
        )
        if cf is None:
            return np.asarray(x, dtype=np.float32)
        cf_arr = np.asarray(cf, dtype=np.float32)
        # CounTS returns (k, T) when it detects T>k; transpose back to (T, k)
        if cf_arr.ndim == 2 and cf_arr.shape == (x.shape[1], x.shape[0]):
            cf_arr = cf_arr.T
        return cf_arr

    def generate_batch(self, X: np.ndarray, model) -> np.ndarray:
        return np.stack([self.generate(x, model) for x in X], axis=0)


# ------------------------------------------------------------------
# CELS / M-CELS
# ------------------------------------------------------------------


class CftsCelsCF:
    """CELS / M-CELS CF via the cfts library.

    ``cels_auto`` routes ``(T, k)`` multivariate input to M-CELS automatically.

    Parameters
    ----------
    target_class : int
    dataset : (X, y) tuple or _DatasetAdapter — training set for saliency.
    max_iter : int
    learning_rate : float
    """

    def __init__(
        self,
        target_class: int = 1,
        dataset=None,
        max_iter: int = 100,
        learning_rate: float = 0.01,
    ) -> None:
        self.target_class = target_class
        self.dataset = dataset
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def generate(self, x: np.ndarray, model) -> np.ndarray:
        from cfts.cf_cels.cels import cels_auto

        if self.dataset is None:
            raise ValueError("CftsCelsCF requires a reference dataset.")
        ds = (
            self.dataset
            if isinstance(self.dataset, _DatasetAdapter)
            else _DatasetAdapter(*self.dataset)
        )
        # M-CELS expects channel-first (k, T) samples and (N, k, T) X_train so the nearest
        # unlike neighbor (NUN) shape matches the sample for mask interpolation.
        X_train = np.stack([ds[i][0] for i in range(len(ds))], axis=0).transpose(
            0, 2, 1
        )  # (N, k, T)
        y_train = np.array([ds[i][1] for i in range(len(ds))])

        adapter = _ChannelFirstAdapter(model)
        cf, _ = cels_auto(
            x.T,  # (k, T) channel-first
            adapter,
            X_train,
            y_train,
            target=self.target_class,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
        )
        if cf is None:
            return np.asarray(x, dtype=np.float32)
        cf_arr = np.asarray(cf, dtype=np.float32)
        # M-CELS returns (1, k, T); squeeze batch dim and transpose back to (T, k)
        if cf_arr.ndim == 3:
            cf_arr = cf_arr.squeeze(0).T
        return cf_arr

    def generate_batch(self, X: np.ndarray, model) -> np.ndarray:
        return np.stack([self.generate(x, model) for x in X], axis=0)
