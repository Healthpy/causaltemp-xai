"""
Abstract base class for black-box time-series classifiers.

All classifiers are trained once per benchmark config and then frozen.
Explainers treat them as black-boxes via predict() and predict_proba().
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
        X_val: np.ndarray,
        Y_val: np.ndarray,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
    ) -> dict:
        """Train the classifier. Returns dict with final val_acc."""
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
        # Save state dict + init params so load() can reconstruct architecture
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
        # X: (N, T, M) -> (N, M, T) for Conv1d layers
        return torch.tensor(X, dtype=torch.float32, device=self.device).permute(0, 2, 1)

    def _run_inference(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Batch inference returning softmax probabilities."""
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                x_batch = self._to_tensor(X[i:i + batch_size])
                logits = self.model(x_batch)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)

    def _train_loop(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None,
        Y_val: np.ndarray | None,
        epochs: int,
        lr: float,
        batch_size: int,
    ) -> dict:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        N = len(X_train)
        best_val_acc = 0.0
        rng = np.random.default_rng(0)

        for epoch in range(epochs):
            self.model.train()
            idx = rng.permutation(N)
            epoch_loss = 0.0
            for i in range(0, N, batch_size):
                batch_idx = idx[i:i + batch_size]
                x_b = self._to_tensor(X_train[batch_idx])
                y_b = torch.tensor(Y_train[batch_idx], dtype=torch.long, device=self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(x_b), y_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if X_val is not None and Y_val is not None:
                val_preds = self.predict(X_val)
                val_acc = float((val_preds == Y_val).mean())
                best_val_acc = max(best_val_acc, val_acc)
                if (epoch + 1) % 10 == 0:
                    print(f"       epoch {epoch+1}/{epochs}  loss={epoch_loss/max(N//batch_size,1):.4f}  val_acc={val_acc:.4f}")

        return {"best_val_acc": best_val_acc, "epochs": epochs}
