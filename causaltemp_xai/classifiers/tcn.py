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
