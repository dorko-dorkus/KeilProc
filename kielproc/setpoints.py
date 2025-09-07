from __future__ import annotations
"""Span fitter for transmitter linearity over a calibration curve.

This module scans all contiguous spans of a calibration curve and chooses the
interval with the best linearity in the *x* domain. Setpoints are returned as a
simple mapping suitable for insertion into ``summary.json``.
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
    rmse: float
    max_abs_dev: float
    n_points: int
    coverage_frac: float
    setpoints: Dict[str, Any]


def _build_prefix(x: np.ndarray, y: np.ndarray):
    N = len(x)
    xp = np.zeros(3 * (N + 1))
    yp = np.zeros(2 * (N + 1))
    xp[1:N+1] = np.cumsum(x)
    yp[1:N+1] = np.cumsum(y)
    xp[N+1:2*(N+1)] = np.cumsum(np.r_[0.0, x * x])
    yp[N+1:]        = np.cumsum(np.r_[0.0, y * y])
    xp[2*(N+1):]    = np.cumsum(np.r_[0.0, x * y])
    return xp, yp


def _linfit_interval(i: int, j: int, xp: np.ndarray, yp: np.ndarray, N: int):
    n   = j - i + 1
    Sx  = float(xp[j+1] - xp[i])
    Sy  = float(yp[j+1] - yp[i])
    Sxx = float(xp[j+1+(N+1)] - xp[i+(N+1)])
    Syy = float(yp[j+1+(N+1)] - yp[i+(N+1)])
    Sxy = float(xp[j+1+2*(N+1)] - xp[i+2*(N+1)])
    denom = n * Sxx - Sx * Sx
    if denom <= 0 or denom <= 1e-12 * max(n * Sxx, 1.0):
        return math.nan, math.nan, math.inf, math.inf, 0.0
    b = (n * Sxy - Sx * Sy) / denom
    a = (Sy - b * Sx) / n
    SSE = Syy - 2*a*Sy - 2*b*Sxy + a*a*n + 2*a*b*Sx + b*b*Sxx
    TSS = Syy - (Sy*Sy)/n
    r2 = 0.0 if TSS <= 0 else max(0.0, min(1.0, 1.0 - SSE/TSS))
    return float(b), float(a), float(SSE), float(TSS), float(r2)


def find_optimal_transmitter_span(
    x: Iterable[float],
    y: Iterable[float],
    *,
    min_fraction_of_range: float = 0.5,
    weight_max_span_dev: float = 1.0,
    weight_r2: float = 0.1,
    slope_sign: int = 0,
    x_allowed: Tuple[float, float] | None = None,
    setpoint_ticks: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    transmitter_ma: Tuple[float, float] = (4.0, 20.0),
) -> OptimalSpan:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        raise ValueError("Need at least 3 points")
    order = np.argsort(x)
    x, y = x[order], y[order]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    full_span = x_max - x_min
    if not np.isfinite(full_span) or full_span <= 0:
        raise ValueError("x has zero span")
    xp, yp = _build_prefix(x, y); N = len(x)
    best, best_payload = None, None
    min_width = min_fraction_of_range * full_span
    for i in range(N):
        for j in range(i+2, N):
            width = x[j] - x[i]
            if width < min_width:
                continue
            if x_allowed and (x[i] < x_allowed[0] or x[j] > x_allowed[1]):
                continue
            b,a,_,_,r2 = _linfit_interval(i, j, xp, yp, N)
            if not np.isfinite(b):
                continue
            if slope_sign>0 and b <= 0:
                continue
            if slope_sign<0 and b >= 0:
                continue
            x_seg, y_seg = x[i:j+1], y[i:j+1]
            x_hat = (y_seg - a) / b
            inv_resid = np.abs(x_seg - x_hat)
            max_abs = float(np.max(inv_resid))
            rmse    = float(np.sqrt(np.mean(inv_resid**2)))
            score = -weight_max_span_dev * (max_abs / width) + weight_r2 * r2
            if (best is None) or (score > best):
                best = score
                best_payload = dict(i=i, j=j, a=a, b=b, r2=r2, rmse=rmse, max_abs=max_abs, width=width)
    if not best_payload:
        raise RuntimeError("No valid interval found")
    i, j = best_payload["i"], best_payload["j"]
    a, b = best_payload["a"], best_payload["b"]
    x_lo, x_hi = float(x[i]), float(x[j])
    coverage = (x_hi - x_lo) / full_span
    ticks = np.asarray(list(setpoint_ticks), dtype=float)
    mA_lo, mA_hi = transmitter_ma
    x_ticks = x_lo + ticks * (x_hi - x_lo)
    mA_ticks = mA_lo + ticks * (mA_hi - mA_lo)
    return OptimalSpan(
        x_low=x_lo, x_high=x_hi, slope=b, intercept=a, r2=float(best_payload["r2"]),
        rmse=float(best_payload["rmse"]), max_abs_dev=float(best_payload["max_abs"]),
        n_points=j - i + 1, coverage_frac=float(coverage),
        setpoints={"mA": dict(zip([f"p{int(t*100)}" for t in ticks], mA_ticks)),
                   "x":  dict(zip([f"p{int(t*100)}" for t in ticks], x_ticks)),
                   "span": {"x_low": x_lo, "x_high": x_hi, "mA_low": mA_lo, "mA_high": mA_hi},
                   "fit": {"slope": b, "intercept": a}},
    )


__all__ = ["OptimalSpan", "find_optimal_transmitter_span"]

