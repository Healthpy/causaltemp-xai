"""Temporal Convolutional Network (TCN) for time-series classification.

Architecture follows Bai et al. (2018) "An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling", with:

* Dilated *causal* convolutions — no future leakage, exponentially growing
  receptive field.
* Residual connections with a 1×1 projection when channel counts differ.
* Weight normalisation on all convolution layers.
* Global average pooling over time before the linear classification head so
  the model accepts variable-length sequences at inference time.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _CausalConv1d(nn.Module):
    """Single dilated causal 1-D convolution with weight norm and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Left-padding to ensure the output has the same length as the input
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.pad,
                dilation=dilation,
            )
        )
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Trim right padding to maintain causal alignment
        out = self.conv(x)[:, :, : x.size(2)]
        return self.dropout(self.relu(out))


class _TCNBlock(nn.Module):
    """One TCN residual block: two dilated causal convolutions + skip."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(in_channels, out_channels, kernel_size, dilation, dropout)
        self.conv2 = _CausalConv1d(out_channels, out_channels, kernel_size, dilation, dropout)
        # 1×1 projection only when channel dimensions differ
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x if self.downsample is None else self.downsample(x)
        out = self.conv2(self.conv1(x))
        return self.relu(out + skip)


class TCN(nn.Module):
    """Temporal Convolutional Network.

    Parameters
    ----------
    n_inputs:
        Number of input channels (variables per time step).
    n_classes:
        Number of output classes.
    n_levels:
        Number of TCN blocks.  The dilation at block *i* is ``2**i``.
    n_channels:
        Number of convolutional filters in each block.  Constant across all
        levels (use a list of length *n_levels* to vary per block).
    kernel_size:
        Kernel size for all dilated convolutions.
    dropout:
        Dropout probability inside each TCN block.
    """

    def __init__(
        self,
        n_inputs: int,
        n_classes: int = 2,
        n_levels: int = 4,
        n_channels: int | list[int] = 64,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if isinstance(n_channels, int):
            channels = [n_channels] * n_levels
        else:
            if len(n_channels) != n_levels:
                raise ValueError(
                    f"len(n_channels)={len(n_channels)} must equal n_levels={n_levels}"
                )
            channels = list(n_channels)

        blocks = []
        for i in range(n_levels):
            in_ch = n_inputs if i == 0 else channels[i - 1]
            blocks.append(
                _TCNBlock(in_ch, channels[i], kernel_size, dilation=2 ** i, dropout=dropout)
            )
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(channels[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(N, k, T)`` — batch-first, channels second.

        Returns
        -------
        logits : Tensor of shape ``(N, n_classes)``.
        """
        features = self.network(x)          # (N, C, T)
        pooled = features.mean(dim=-1)       # global average pooling → (N, C)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Convenience training function
# ---------------------------------------------------------------------------


def train_tcn(
    dataset: tuple[np.ndarray, np.ndarray],
    n_classes: int = 2,
    n_levels: int = 4,
    n_channels: int = 64,
    kernel_size: int = 3,
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 100,
    target_acc: float = 0.90,
    device: Optional[str] = None,
    verbose: bool = True,
) -> TCN:
    """Train a TCN classifier until *target_acc* is reached or *max_epochs* elapse.

    Parameters
    ----------
    dataset:
        Tuple ``(X, y)`` where ``X`` has shape ``(N, T, k)`` and ``y`` has
        shape ``(N,)`` with integer class labels.
    n_classes:
        Number of output classes.
    n_levels:
        Number of TCN blocks.
    n_channels:
        Number of filters per block.
    kernel_size:
        Kernel size.
    dropout:
        Dropout rate.
    lr:
        Learning rate for Adam.
    batch_size:
        Mini-batch size.
    max_epochs:
        Hard ceiling on training epochs.
    target_acc:
        Training is stopped early once the training accuracy reaches this
        threshold.  Set to 1.0 to always run for *max_epochs*.
    device:
        Torch device string.  Auto-detected if ``None``.
    verbose:
        Print epoch summary when ``True``.

    Returns
    -------
    TCN
        Trained model in eval mode.
    """
    X_np, y_np = dataset
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    k = X_np.shape[2] if X_np.ndim == 3 else X_np.shape[1]
    model = TCN(
        n_inputs=k,
        n_classes=n_classes,
        n_levels=n_levels,
        n_channels=n_channels,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(dev)

    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb = xb.permute(0, 2, 1).to(dev)   # (N, T, k) → (N, k, T)
            yb = yb.to(dev)
            optimiser.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        acc = correct / total
        if verbose:
            print(f"Epoch {epoch:03d}/{max_epochs} | loss={total_loss/total:.4f} | acc={acc:.4f}")
        if acc >= target_acc:
            if verbose:
                print(f"  → target accuracy {target_acc:.0%} reached, stopping early.")
            break

    model.eval()
    return model


# ---------------------------------------------------------------------------
# sklearn-style wrapper
# ---------------------------------------------------------------------------


class TCNClassifier:
    """sklearn-style wrapper around :class:`TCN` for the benchmark.

    **Shape contract.** Every public method accepts data in ``(T, k)`` (single
    instance) or ``(N, T, k)`` (batch) layout — the same convention used by the
    generator and :class:`~causaltemp_xai.metrics.cf_faith.CFfaith`. The
    ``(N, T, k) → (N, k, T)`` permute required by :class:`TCN` lives **only**
    inside this wrapper, so callers (including the Stage 4 CF methods) never
    transpose at the call site.

    Parameters
    ----------
    n_inputs:
        Number of variables ``k`` (input channels).
    n_classes, n_levels, n_channels, kernel_size, dropout:
        Forwarded to :class:`TCN`.
    lr, batch_size, max_epochs:
        Adam optimiser / training-loop hyperparameters.
    patience:
        Early-stopping patience (epochs without monitor-metric improvement).
        The monitor is val loss when a validation set is supplied to
        :meth:`fit`, otherwise train loss.
    target_acc:
        If train accuracy reaches this, training stops early regardless of the
        early-stopping monitor.
    device:
        Torch device string; auto-detected when ``None``.
    seed:
        Seeds torch for reproducible initialisation/training.
    """

    def __init__(
        self,
        n_inputs: int,
        n_classes: int = 2,
        n_levels: int = 4,
        n_channels: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        target_acc: float = 0.90,
        device: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        self.hparams = dict(
            n_inputs=n_inputs,
            n_classes=n_classes,
            n_levels=n_levels,
            n_channels=n_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.target_acc = target_acc
        self.seed = seed
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch.manual_seed(seed)
        self.model = TCN(**self.hparams).to(self.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _as_batch_tensor(self, X) -> tuple[torch.Tensor, bool]:
        """Return ``((N, k, T)`` float tensor, was_single) from ``(T,k)``/``(N,T,k)``."""
        if isinstance(X, torch.Tensor):
            x = X.to(dtype=torch.float32, device=self.device)
        else:
            x = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        single = x.dim() == 2
        if single:
            x = x.unsqueeze(0)  # (1, T, k)
        return x.permute(0, 2, 1), single  # (N, k, T)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose: bool = False):
        """Train the underlying TCN with early stopping; restore best weights.

        ``X_train`` / ``X_val`` are ``(N, T, k)``; labels are ``(N,)``.
        """
        torch.manual_seed(self.seed)
        Xtr = torch.as_tensor(np.asarray(X_train), dtype=torch.float32)
        ytr = torch.as_tensor(np.asarray(y_train), dtype=torch.long)
        loader = DataLoader(
            TensorDataset(Xtr, ytr), batch_size=self.batch_size, shuffle=True
        )
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        has_val = X_val is not None and y_val is not None
        if has_val:
            Xva, _ = self._as_batch_tensor(X_val)
            yva = torch.as_tensor(np.asarray(y_val), dtype=torch.long, device=self.device)

        best_monitor = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        epochs_no_improve = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0
            for xb, yb in loader:
                xb = xb.permute(0, 2, 1).to(self.device)  # (N,T,k) → (N,k,T)
                yb = yb.to(self.device)
                optimiser.zero_grad()
                logits = self.model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                optimiser.step()
                total_loss += loss.item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)
            train_acc = correct / total
            train_loss = total_loss / total

            # Monitor metric for early stopping
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(Xva)
                    monitor = F.cross_entropy(val_logits, yva).item()
            else:
                monitor = train_loss

            if verbose:
                tag = "val_loss" if has_val else "train_loss"
                print(
                    f"Epoch {epoch:03d}/{self.max_epochs} | "
                    f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                    f"{tag}={monitor:.4f}"
                )

            if monitor < best_monitor - 1e-5:
                best_monitor = monitor
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if train_acc >= self.target_acc and not has_val:
                # No val set: target accuracy is the stopping signal.
                best_state = copy.deepcopy(self.model.state_dict())
                break
            if epochs_no_improve >= self.patience:
                if verbose:
                    print(f"  → early stopping at epoch {epoch} (patience {self.patience}).")
                break

        self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities, shape ``(N, n_classes)`` (softmax)."""
        x, _ = self._as_batch_tensor(X)
        self.model.eval()
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=-1)
        return probs.cpu().numpy()

    def predict(self, X) -> np.ndarray:
        """Return predicted integer labels.

        ``(N, T, k)`` → ``(N,)``; a single ``(T, k)`` instance → scalar ``int``.
        """
        single = np.asarray(X).ndim == 2 if not isinstance(X, torch.Tensor) else X.dim() == 2
        proba = self.predict_proba(X)
        labels = proba.argmax(axis=-1)
        return int(labels[0]) if single else labels

    def score(self, X, y) -> float:
        """Mean accuracy on ``(X, y)``."""
        preds = self.predict(X)
        return float((np.asarray(preds) == np.asarray(y)).mean())

    def torch_logits(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """Differentiable forward for gradient-based CF methods.

        Parameters
        ----------
        x_tensor:
            A **torch tensor** in ``(T, k)`` or ``(N, T, k)`` layout (the public
            convention). To optimise an input, create it with
            ``requires_grad_()`` and backprop through the returned logits — the
            ``(N, T, k) → (N, k, T)`` permute is applied internally so gradients
            flow back to the original tensor.

        Returns
        -------
        logits : Tensor of shape ``(N, n_classes)`` (always batched, even for a
            single ``(T, k)`` input).
        """
        x = x_tensor
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1).to(self.device)  # (N,k,T), preserves grad
        self.model.eval()
        return self.model(x)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Save state_dict + hyperparameters to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hparams": self.hparams,
                "train_cfg": {
                    "lr": self.lr,
                    "batch_size": self.batch_size,
                    "max_epochs": self.max_epochs,
                    "patience": self.patience,
                    "target_acc": self.target_acc,
                    "seed": self.seed,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str, device: Optional[str] = None) -> "TCNClassifier":
        """Reconstruct a classifier from a checkpoint written by :meth:`save`."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        train_cfg = ckpt.get("train_cfg", {})
        clf = cls(device=device, **ckpt["hparams"], **train_cfg)
        clf.model.load_state_dict(ckpt["state_dict"])
        clf.model.eval()
        return clf


# ---------------------------------------------------------------------------
# CLI: train on a locked config and freeze the checkpoint
# ---------------------------------------------------------------------------


def _train_cli(argv: list[str] | None = None) -> None:
    import argparse

    from causaltemp_xai.config import get_config
    from causaltemp_xai.data_io import DEFAULT_OUT_DIR, generate_and_save, load_dataset

    parser = argparse.ArgumentParser(description="Train a TCNClassifier on a locked config.")
    parser.add_argument("--config", choices=["smoke", "full", "full_sparse"], required=True)
    parser.add_argument("--train", action="store_true", help="Run training (required).")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-levels", type=int, default=4)
    parser.add_argument("--n-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    cfg = get_config(args.config)
    out_dir = Path(args.out_dir)
    if not (out_dir / cfg.name / "meta.json").exists():
        print(f"Dataset for '{cfg.name}' not found — generating (deterministic).")
        generate_and_save(cfg, out_dir=out_dir)
    data = load_dataset(cfg.name, out_dir=out_dir)

    clf = TCNClassifier(
        n_inputs=cfg.k,
        seed=cfg.seed,
        n_levels=args.n_levels,
        n_channels=args.n_channels,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    clf.fit(
        data["X_train"], data["Y_train"], data["X_val"], data["Y_val"], verbose=args.verbose
    )

    train_acc = clf.score(data["X_train"], data["Y_train"])
    val_acc = clf.score(data["X_val"], data["Y_val"])
    test_acc = clf.score(data["X_test"], data["Y_test"])
    print(
        f"[{cfg.name}] train_acc={train_acc:.4f} | "
        f"val_acc={val_acc:.4f} | test_acc={test_acc:.4f}"
    )

    ckpt_path = out_dir / cfg.name / "tcn.pt"
    clf.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    if test_acc < 0.90:
        print(
            f"WARNING: test accuracy {test_acc:.4f} < 0.90 target "
            f"(>=0.85 is the documented fallback)."
        )


if __name__ == "__main__":
    _train_cli()
