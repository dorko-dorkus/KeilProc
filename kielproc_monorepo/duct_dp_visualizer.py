import numpy as np
import pandas as pd
from dataclasses import dataclass


def pearson_r(x, y):
    """Compute Pearson correlation coefficient between *x* and *y*.

    Parameters
    ----------
    x, y : array-like
        Input sequences.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return np.nan
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    denom = np.sqrt(np.nansum(x * x) * np.nansum(y * y))
    if denom == 0:
        return np.nan
    return float(np.nansum(x * y) / denom)


def fisher_z_ci(r, n, alpha=0.05):
    """Fisher :math:`z` transformation confidence interval for a correlation.

    Returns the original correlation ``r`` along with lower and upper bounds for
    a two-sided confidence interval of size ``1-alpha``.
    """
    r = float(r)
    if not np.isfinite(r) or n <= 3 or abs(r) >= 1.0:
        # Perfect correlations have undefined sampling variance; return NaNs
        return r, np.nan, np.nan
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    # 1.96 is the z critical value for 95% CI.  This is sufficient for tests
    # here without requiring scipy.
    zcrit = 1.96 if alpha == 0.05 else float(np.abs(np.sqrt(2) * np.erfcinv(alpha)))
    lo = np.tanh(z - zcrit * se)
    hi = np.tanh(z + zcrit * se)
    return r, lo, hi


def _theil_sen_slopes(x, y, rng, pairs):
    """Helper to compute median slope from ``pairs`` random index pairs."""
    n = len(x)
    idx_i = rng.integers(0, n, size=pairs)
    idx_j = rng.integers(0, n, size=pairs)
    mask = idx_i != idx_j
    dx = x[idx_j][mask] - x[idx_i][mask]
    dy = y[idx_j][mask] - y[idx_i][mask]
    valid = dx != 0
    slopes = dy[valid] / dx[valid]
    if slopes.size == 0:
        return np.nan
    return float(np.median(slopes))


def theil_sen_subsample_ci(x, y, B=1000, pairs_per_boot=200, seed=None, alpha=0.05):
    """Estimate Theil–Sen slope and a bootstrap CI using random subsampling.

    Parameters
    ----------
    x, y : array-like
        Data vectors.
    B : int
        Number of bootstrap replicates.
    pairs_per_boot : int
        Number of point pairs sampled per bootstrap replicate.  Defaults to 200
        which keeps runtime reasonable while giving stable estimates.
    seed : int or ``None``
        Optional RNG seed for reproducibility.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(int(B)):
        m = _theil_sen_slopes(x, y, rng, int(pairs_per_boot))
        slopes.append(m)
    slopes = np.asarray(slopes)
    m = float(np.nanmedian(slopes))
    lo = float(np.nanpercentile(slopes, 100 * alpha / 2))
    hi = float(np.nanpercentile(slopes, 100 * (1 - alpha / 2)))
    return m, lo, hi


@dataclass
class SliceParams:
    """Parameters controlling port slice analysis."""
    frac: float = 0.15
    kmin: int = 10


def analyze_port(df, params: SliceParams, sp_col="SP", vp_col="VP"):
    """Analyze port data by slicing into bottom, middle and top segments.

    The data frame ``df`` must contain static pressure ``sp_col`` and velocity
    pressure ``vp_col`` columns.  The lowest ``params.frac`` fraction (at least
    ``params.kmin`` samples) of the sorted ``sp_col`` values are considered the
    *bottom* slice and the highest fraction the *top* slice.  The remainder is
    the *middle* slice.  For each slice the Theil–Sen slope of ``vp_col`` versus
    ``sp_col`` is estimated.
    """
    df = df[[sp_col, vp_col]].dropna().sort_values(sp_col).reset_index(drop=True)
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["Slice", "theil_sen"])
    k = max(int(params.frac * n), int(params.kmin))
    if 2 * k >= n:
        k = n // 3
    slices = {
        "bottom": df.iloc[:k],
        "middle": df.iloc[k:n - k],
        "top": df.iloc[n - k:],
    }
    rows = []
    for name, sub in slices.items():
        if len(sub) >= 2:
            slope, _, _ = theil_sen_subsample_ci(
                sub[sp_col].to_numpy(),
                sub[vp_col].to_numpy(),
                B=200,
                pairs_per_boot=min(200, len(sub) * (len(sub) - 1) // 2),
                seed=0,
            )
        else:
            slope = np.nan
        rows.append({"Slice": name, "theil_sen": slope})
    return pd.DataFrame(rows)
