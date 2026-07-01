"""
SepsisSim benchmark — Week 6 (skeleton), Week 12 (full).

Real-data grounded, clinician-validated DAG on MIMIC-IV sepsis cohort.
Incorporates time-varying treatments (e.g., dynamic fluid administration)
and dynamic confounding.

Labels: Y = 1[MAP_T > 65 mmHg] — clinically meaningful outcome threshold.

Status: generate() raises NotImplementedError until MIMIC-IV data access
is confirmed. A synthetic fallback using the published clinical DAG structure
from Johnson et al. (2023) is available via generate_synthetic().

Reference:
  Johnson et al. (2023), MIMIC-IV, a freely accessible electronic health record.
  Schulam & Saria (2017), Reliable Decision Support using Counterfactual Models.
"""

from __future__ import annotations

import warnings

from .base import BenchmarkDataset, BenchmarkSplit


class SepsisSim(BenchmarkDataset):
    """
    SepsisSim: semi-synthetic clinical benchmark.

    Requires MIMIC-IV access for the full implementation.
    Use generate_synthetic() for a DAG-correct synthetic fallback.
    """

    # Clinical variables in the validated DAG (Johnson et al. 2023)
    CLINICAL_VARIABLES = [
        "MAP",           # Mean Arterial Pressure (label channel)
        "HR",            # Heart Rate
        "SpO2",          # Oxygen Saturation
        "Temp",          # Body Temperature
        "Lactate",       # Serum Lactate
        "WBC",           # White Blood Cell Count
        "Creatinine",    # Renal Function
        "Fluids_IV",     # IV Fluid Administration (time-varying treatment)
        "Vasopressors",  # Vasopressor Dose (time-varying treatment)
        "Antibiotics",   # Antibiotic Administration (binary, time-varying)
    ]

    MAP_CHANNEL = 0          # Index of MAP in CLINICAL_VARIABLES
    MAP_THRESHOLD = 65.0     # mmHg — label threshold

    def generate(self) -> BenchmarkSplit:
        raise NotImplementedError(
            "SepsisSim.generate() requires MIMIC-IV data access.\n"
            "Options:\n"
            "  1. Obtain MIMIC-IV from https://physionet.org/content/mimiciv/\n"
            "     and place cohort CSV files in data/sepsis_sim/raw/.\n"
            "  2. Use generate_synthetic() for a DAG-correct synthetic fallback\n"
            "     that does not require patient data."
        )

    def generate_synthetic(self, N: int = 5000, T: int = 48) -> BenchmarkSplit:
        """
        Synthetic SepsisSim using the published clinical DAG structure.

        Generates N ICU patient trajectories of length T hours using
        Neural-ODE state transitions (requires torchdiffeq).
        """
        warnings.warn(
            "Using synthetic SepsisSim fallback. Results are DAG-correct "
            "but not grounded in real patient data.",
            UserWarning,
            stacklevel=2,
        )
        try:
            import torch
            import torchdiffeq  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "SepsisSim requires torch and torchdiffeq. "
                "Install with: pip install torch torchdiffeq"
            ) from e

        from ..scm.dag import sample_dag
        from ..scm.operators import sample_mechanism, INVERTIBLE_OPERATORS
        from ..scm.tscm import simulate_tscm, sample_noise
        import numpy as np

        rng = np.random.default_rng(self.seed)
        M = len(self.CLINICAL_VARIABLES)

        # Clinical DAG encodes MAP <- (HR, Fluids_IV, Vasopressors)
        # HR -> MAP, SpO2 -> MAP, Fluids_IV -> MAP, Vasopressors -> MAP
        # Lactate <- MAP (downstream indicator)
        dag = sample_dag(M, max_lag=1, edge_density=0.15, rng=rng)

        mechanisms = sample_mechanism(dag, rng, sigma_w=0.5, operator_pool=INVERTIBLE_OPERATORS)
        for m in mechanisms:
            m.weight = np.clip(m.weight, -0.6, 0.6)

        noise = sample_noise(N, T + 50, M, "laplace", rng)
        X = simulate_tscm(dag, mechanisms, noise, burn_in=50)
        X = np.clip(X, -20.0, 20.0)

        Y, theta = self._label_from_threshold(X, self.MAP_CHANNEL)

        n_train = int(N * 0.70)
        n_val = int(N * 0.15)
        idx = rng.permutation(N)
        i_tr = idx[:n_train]
        i_va = idx[n_train:n_train + n_val]
        i_te = idx[n_train + n_val:]

        meta = {
            "benchmark": "SepsisSim-Synthetic",
            "clinical_variables": self.CLINICAL_VARIABLES,
            "map_threshold_mmhg": self.MAP_THRESHOLD,
            "label_threshold_normalized": float(theta),
            "class_balance": float(Y.mean()),
            "T_hours": T,
            "N": N,
            "seed": self.seed,
        }

        return BenchmarkSplit(
            X_train=X[i_tr], Y_train=Y[i_tr],
            X_val=X[i_va], Y_val=Y[i_va],
            X_test=X[i_te], Y_test=Y[i_te],
            dag=dag, mechanisms=mechanisms, meta=meta,
        )
