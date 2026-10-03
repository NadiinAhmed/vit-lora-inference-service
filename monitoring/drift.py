"""Drift statistics: how different is production data from the reference?

PSI (Population Stability Index) - the decision metric
    Split values into buckets, compare each bucket's share in reference vs production:
        PSI = sum( (prod% - ref%) * ln(prod% / ref%) )
    0 means identical. Industry rule of thumb (Session 14):
        < 0.10 OK   |   0.10 - 0.25 WARNING   |   > 0.25 ALERT

KS distance (Kolmogorov-Smirnov statistic) - a second, intuitive view
    The largest gap between the two cumulative distributions, from 0 to 1.
    "At worst, 40% more production images are below this brightness than in the reference."
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

OK, WARNING, ALERT = "OK", "WARNING", "ALERT"
PSI_WARNING, PSI_ALERT = 0.10, 0.25

_EPSILON = 1e-4  # stands in for an empty bucket, so ln(0) and division by 0 never happen


def psi_status(psi: float) -> str:
    return ALERT if psi > PSI_ALERT else WARNING if psi >= PSI_WARNING else OK


def worst_status(statuses: Sequence[str]) -> str:
    for status in (ALERT, WARNING):
        if status in statuses:
            return status
    return OK


def _psi(reference_shares: np.ndarray, production_shares: np.ndarray) -> float:
    ref = np.clip(reference_shares, _EPSILON, None)
    prod = np.clip(production_shares, _EPSILON, None)
    return round(float(np.sum((prod - ref) * np.log(prod / ref))), 4)


def numeric_psi(reference: Sequence[float], production: Sequence[float], buckets: int = 10) -> float:
    """PSI for a numeric feature, with bucket edges taken from the reference's quantiles.

    Two extra buckets catch values below the reference minimum and above its maximum.
    Without them, a feature that is constant in the reference (every TrashNet image is
    512x384, so aspect_ratio is always 1.3333) could never show drift.
    """
    reference, production = np.asarray(reference, float), np.asarray(production, float)
    low, high = reference.min(), reference.max()
    inner_edges = np.unique(np.quantile(reference, np.linspace(0, 1, buckets + 1)[1:-1]))

    def shares(values: np.ndarray) -> np.ndarray:
        index = np.digitize(values, inner_edges, right=True) + 1
        index[values < low] = 0                        # below anything in the reference
        index[values > high] = len(inner_edges) + 2    # above anything in the reference
        counts = np.bincount(index, minlength=len(inner_edges) + 3)
        return counts / counts.sum()

    return _psi(shares(reference), shares(production))


def categorical_psi(reference: Sequence[str], production: Sequence[str]) -> float:
    """PSI over category shares, e.g. how often each class is predicted."""
    categories = sorted(set(reference) | set(production))
    ref_shares = np.array([list(reference).count(c) for c in categories]) / len(reference)
    prod_shares = np.array([list(production).count(c) for c in categories]) / len(production)
    return _psi(ref_shares, prod_shares)


def ks_distance(reference: Sequence[float], production: Sequence[float]) -> float:
    """Largest gap between the two cumulative distributions (0 = identical, 1 = no overlap)."""
    reference, production = np.sort(np.asarray(reference, float)), np.sort(np.asarray(production, float))
    points = np.concatenate([reference, production])
    ref_cdf = np.searchsorted(reference, points, side="right") / len(reference)
    prod_cdf = np.searchsorted(production, points, side="right") / len(production)
    return round(float(np.max(np.abs(ref_cdf - prod_cdf))), 4)
