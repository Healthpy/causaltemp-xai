"""Abstract base class for black-box time-series classifiers.

All classifiers are trained once per benchmark config and then frozen.
Explainers treat them as black-boxes via predict() and predict_proba().

Ported from causal_tscf_bench/classifiers/base.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch


class TSClassifier(ABC):
    """Abstract black-box time-series classifier."""

    def __init__(self, n_classes: int = 2, device: str = "cpu"):
        self.n_classes = n_classes
        self.device = device
        self.model: torch.nn.Module | None = None

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        **kwargs,
    ):
        """Train the classifier."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions (N,)."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities (N, n_classes)."""
        ...

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        torch.save({
            "state_dict": self.model.state_dict(),
            "init_params": getattr(self, "_init_params", {}),
        }, Path(path) / "model.pt")

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(Path(path) / "model.pt", map_location=self.device)
        self._build_model(**ckpt.get("init_params", {}))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def _build_model(self, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement _build_model.")

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        """X: (N, T, k) — kept time-first (N, T, k); no permute for LSTM."""
        return torch.tensor(X, dtype=torch.float32, device=self.device)

    def _run_inference(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Batch inference returning softmax probabilities."""
        import torch.nn.functional as F
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                x_batch = self._to_tensor(X[i:i + batch_size])
                logits = self.model(x_batch)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)
