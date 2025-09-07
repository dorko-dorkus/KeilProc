from __future__ import annotations
"""
Span fitter for transmitter linearity over a calibration curve.

Algorithm
---------
1) Sort points by x and build prefix sums for x, y, x^2, y^2, x*y.
2) For each contiguous interval [i, j] wide enough in x, compute closed-form LS fit
   y ≈ a + b x via prefix sums.
3) Score each interval by worst-case inverse residual in x, normalized by span,
   with a light R² tie-breaker. Choose argmax.
4) Return span endpoints and even 4–20 mA setpoints mapped across that span.

The score directly targets x-domain setpoint error: maximize
    score = -max(|x - x_hat|)/span + weight_r2 * R²,
where x_hat = (y - a)/b.

Numerical guards:
- Denominator scaled guard for collinearity.
- R² = 0 when TSS ≤ 0 instead of 1.0 (degenerate y).
- Reject tiny |b| to avoid exploding inverse residuals.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, Any
import math
import numpy as np


@dataclass(frozen=True)
class OptimalSpan:
    x_low: float
    x_high: float
    slope: float
    intercept: float
    r2: float
    rmse: float            # RMSE of inverse residuals in x
    max_abs_dev: float     # worst-case inverse residual in x
    n_points: int
    coverage_frac: float
    setpoints: Dict[str, Any]


def _build_prefix(x: np.ndarray, y: np.ndarray):
    N = len(x)
    x_pref = np.zeros(3 * (N + 1))
    y_pref = np.zeros(2 * (N + 1))
    x_pref[1 : N + 1] = np.cumsum(x)
    y_pref[1 : N + 1] = np.cumsum(y)
    x_pref[N + 1 : 2 * (N + 1)] = np.cumsum(np.r_[0.0, x * x])
    x_pref[2 * (N + 1) :] = np.cumsum(np.r_[0.0, x * y])
    y_pref[N + 1 :] = np.cumsum(np.r_[0.0, y * y])
    return x_pref, y_pref


def _linfit_interval(i: int, j: int, xp: np.ndarray, yp: np.ndarray, N: int):
    n = j - i + 1
    Sx = float(xp[j + 1] - xp[i])
    Sy = float(yp[j + 1] - yp[i])
    Sxx = float(xp[j + 1 + (N + 1)] - xp[i + (N + 1)])
    Syy = float(yp[j + 1 + (N + 1)] - yp[i + (N + 1)])
    Sxy = float(xp[j + 1 + 2 * (N + 1)] - xp[i + 2 * (N + 1)])

    denom = n * Sxx - Sx * Sx
    # Scale-aware guard against degeneracy
    if denom <= 0 or denom <= 1e-12 * max(n * Sxx, 1.0):
        return math.nan, math.nan, math.inf, math.inf, 0.0

    b = (n * Sxy - Sx * Sy) / denom
    a = (Sy - b * Sx) / n
    SSE = Syy - 2 * a * Sy - 2 * b * Sxy + a * a * n + 2 * a * b * Sx + b * b * Sxx
    TSS = Syy - (Sy * Sy) / n
    if TSS <= 0:
        r2 = 0.0
    else:
        r2 = 1.0 - SSE / TSS
    r2 = float(max(0.0, min(1.0, r2)))
    return float(b), float(a), float(SSE), float(TSS), r2


def find_optimal_transmitter_span(
    x: Iterable[float],
    y: Iterable[float],
    *,
    min_fraction_of_range: float = 0.5,
    weight_max_span_dev: float = 1.0,
    weight_r2: float = 0.1,
    slope_sign: int = 0,  # 0 unconstrained; +1 require positive slope; -1 require negative
    x_allowed: Tuple[float, float] | None = None,
    setpoint_ticks: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    transmitter_ma: Tuple[float, float] = (4.0, 20.0),
) -> OptimalSpan:
    """Scan all contiguous spans and return the one with best linearity in x.

    Parameters mirror the original implementation with small robustness fixes.
    """

    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if len(x) < 3:
        raise ValueError("Need at least 3 points")

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    full_span = x_max - x_min
    if not np.isfinite(full_span) or full_span <= 0:
        raise ValueError("x has zero span")

    min_pts = 3  # rely on width constraint; avoids uneven grids biasing selection
    xp, yp = _build_prefix(x, y)
    N = len(x)

    best = None
    best_payload = None
    for i in range(N):
        j0 = max(i + min_pts - 1, i + 2)
        if j0 >= N:
            break
        for j in range(j0, N):
            x_lo = x[i]
            x_hi = x[j]
            width = x_hi - x_lo
            if width <= 0 or width < min_fraction_of_range * full_span:
                continue
            if x_allowed and not (x_allowed[0] <= x_lo <= x_hi <= x_allowed[1]):
                continue

            b, a, SSE, TSS, r2 = _linfit_interval(i, j, xp, yp, N)
            if not np.isfinite(b) or abs(b) < 1e-12:
                continue
            if slope_sign == +1 and b <= 0:
                continue
            if slope_sign == -1 and b >= 0:
                continue

            x_seg = x[i : j + 1]
            y_seg = y[i : j + 1]
            x_hat = (y_seg - a) / b
            inv_resid = np.abs(x_seg - x_hat)
            max_abs = float(np.max(inv_resid))
            rmse = float(np.sqrt(np.mean(inv_resid ** 2)))
            score = -weight_max_span_dev * (max_abs / width) + weight_r2 * r2
            if (best is None) or (score > best):
                best = score
                best_payload = dict(i=i, j=j, a=a, b=b, r2=float(r2), rmse=rmse, max_abs=max_abs)

    if best_payload is None:
        raise RuntimeError("No valid interval found")

    i = best_payload["i"]
    j = best_payload["j"]
    a = best_payload["a"]
    b = best_payload["b"]
    r2 = best_payload["r2"]
    rmse = best_payload["rmse"]
    max_abs = best_payload["max_abs"]
    x_lo = float(x[i])
    x_hi = float(x[j])
    coverage = (x_hi - x_lo) / full_span

    ticks = np.asarray(list(setpoint_ticks), dtype=float)
    mA_lo, mA_hi = transmitter_ma
    x_ticks = x_lo + ticks * (x_hi - x_lo)
    mA_ticks = mA_lo + ticks * (mA_hi - mA_lo)

    return OptimalSpan(
        x_low=x_lo,
        x_high=x_hi,
        slope=b,
        intercept=a,
        r2=r2,
        rmse=rmse,
        max_abs_dev=max_abs,
        n_points=(j - i + 1),
        coverage_frac=float(coverage),
        setpoints={
            "fractions": ticks.tolist(),
            "x_values": x_ticks.tolist(),
            "transmitter_mA": mA_ticks.tolist(),
            "mapping": [
                {"pct": float(fr * 100.0), "x": float(xv), "mA": float(ma)}
                for fr, xv, ma in zip(ticks, x_ticks, mA_ticks)
            ],
            "span": {"x_low": x_lo, "x_high": x_hi, "mA_low": mA_lo, "mA_high": mA_hi},
            "fit": {"slope": b, "intercept": a},
        },
    )


__all__ = ["OptimalSpan", "find_optimal_transmitter_span"]

if __name__ == "__main__":  # quick smoke test
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 10.0, 101)
    y = 2.0 + 3.0 * x + rng.normal(scale=0.01, size=x.size)
    opt = find_optimal_transmitter_span(x, y)
    print(opt)
