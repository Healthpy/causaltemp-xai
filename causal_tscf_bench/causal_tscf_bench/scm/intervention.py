"""
Do-operator (Action step of the causal ladder).

Applying do(X_{T_int}^(i) = x_int) severs all incoming causal edges to channel i
at time T_int and replaces the structural equation with the constant x_int.
Downstream effects (channels that have i in their parent set at later time steps)
propagate naturally during forward simulation.

Reference:
  Pearl (2009), Causality (2nd ed.) — do-operator and surgery on SCMs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Intervention:
    """Specification of a single do-operator intervention."""
    channel: int       # i — the intervened channel
    time: int          # T_int — the time step of intervention
    value: float       # x'_int — the constant value imposed

    def affects_time(self, t: int) -> bool:
        """True if time t is at or after the intervention time."""
        return t >= self.time


def apply_intervention(
    X_running: "np.ndarray",  # noqa: F821
    t: int,
    j: int,
    intervention: Intervention | None,
) -> "np.ndarray":
    """
    During forward simulation, override X[t, j] if an intervention targets (j, t).

    Called inside simulate_with_intervention at each (t, j) step.
    Returns the (possibly overridden) value for channel j at time t.

    Parameters
    ----------
    X_running : np.ndarray
        Partially filled simulation array, shape (N, T_padded, M).
    t : int
        Current absolute time index in X_running.
    j : int
        Current channel being computed.
    intervention : Intervention | None

    Returns
    -------
    np.ndarray
        Shape (N,) — the value for X[:, t, j] after potential override.
    """
    value = X_running[:, t, j].copy()
    if intervention is not None and intervention.channel == j and intervention.time == t:
        value[:] = intervention.value
    return value
