from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Iterable


def pearson_r(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("x and y must have same length")
    if x.size == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fisher_z_ci(r: float, n: int, alpha: float = 0.05):
    """Return correlation and Fisher z-transformed confidence interval."""
    if n <= 3 or abs(r) >= 1:
        return r, float("nan"), float("nan")
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zcrit = 1.96  # approximate 95% two-sided
    lo = np.tanh(z - zcrit * se)
    hi = np.tanh(z + zcrit * se)
    return r, float(lo), float(hi)


def _theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Median slope of all pairs (Theil-Sen estimator)."""
    n = x.size
    idx_i, idx_j = np.triu_indices(n, k=1)
    slopes = (y[idx_j] - y[idx_i]) / (x[idx_j] - x[idx_i])
    return float(np.median(slopes))


def theil_sen_subsample_ci(x, y, B=100, pairs_per_boot=200, seed=None):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m_hat = _theil_sen_slope(x, y)
    boot = []
    n = x.size
    for _ in range(B):
        i = rng.integers(0, n, size=pairs_per_boot)
        j = rng.integers(0, n, size=pairs_per_boot)
        mask = i < j
        i = i[mask]; j = j[mask]
        if i.size == 0:
            continue
        slopes = (y[j] - y[i]) / (x[j] - x[i])
        boot.append(np.median(slopes))
    if boot:
        lo, hi = np.percentile(boot, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return m_hat, float(lo), float(hi)


@dataclass
class SliceParams:
    frac: float = 0.1
    kmin: int = 10


def analyze_port(df, params: SliceParams):
    """Split a DataFrame into bottom/middle/top slices and compute slopes."""
    x = np.asarray(df["SP"], dtype=float)
    y = np.asarray(df["VP"], dtype=float)
    n = x.size
    k = max(int(params.frac * n), params.kmin)
    slices = {
        "bottom": slice(0, k),
        "middle": slice(max((n - k) // 2, 0), max((n + k) // 2, 0)),
        "top": slice(n - k, n),
    }
    rows = []
    for name, sl in slices.items():
        xs = x[sl]; ys = y[sl]
        if xs.size >= params.kmin:
            m = _theil_sen_slope(xs, ys)
            rows.append(dict(Slice=name, theil_sen=m))
    import pandas as pd
    return pd.DataFrame(rows)
