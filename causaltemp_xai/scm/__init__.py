"""SCM sub-package: intervention-time derivation and the shared "is this changed?" predicate.

Reduced to :mod:`~causaltemp_xai.scm.intervention` on 2026-08-03
. The package previously also carried a ported
``causal_tscf_bench`` cluster — ``operators``, ``abduction``, ``counterfactual``,
``dag``, ``tscm`` — which was a **second, untested implementation of the
benchmark's ground truth**, unimported from outside this package and
self-described as "the preferred path for new code". It was deleted.

The canonical implementations live in :mod:`causaltemp_xai.benchmarks`:
mechanisms in ``benchmarks/mechanisms.py``, oracle abduction–action–prediction
counterfactuals in ``benchmarks/structural_cf.py``, generation in
``benchmarks/generator.py``. There is exactly one of each.
"""

from .intervention import INTERVENTION_TOL, derive_intervention_t, is_vacuous_intervention

__all__ = [
    "INTERVENTION_TOL",
    "derive_intervention_t",
    "is_vacuous_intervention",
]
