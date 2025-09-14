from __future__ import annotations
import numpy as np, pandas as pd
from typing import Tuple, Optional


def current_to_dp_raw_mbar(I_mA: np.ndarray, lrv_mbar: float, urv_mbar: float) -> np.ndarray:
    span = float(urv_mbar - lrv_mbar)
    return lrv_mbar + np.clip((np.asarray(I_mA, float) - 4.0) / 16.0, -1e3, 1e3) * span


def fit_current_to_dp(I_mA: np.ndarray, dp_pred_mbar: np.ndarray, huber_delta: float = 0.5) -> Tuple[float, float]:
    I = np.asarray(I_mA, float); y = np.asarray(dp_pred_mbar, float)
    m = np.isfinite(I) & np.isfinite(y)
    I, y = I[m], y[m]
    if I.size < 10:
        a = (np.nanpercentile(y,95) - np.nanpercentile(y,5)) / 16.0
        b = float(np.nanmedian(y)) - a * float(np.nanmedian(I))
        return float(a), float(b)
    X = np.c_[I, np.ones_like(I)]; w = np.ones_like(y)
    for _ in range(3):
        beta, *_ = np.linalg.lstsq((w[:,None]*X), w*y, rcond=None)
        r = y - X@beta
        s = np.nanmedian(np.abs(r)) * 1.4826 or 1.0
        z = r / max(s, 1e-6)
        w = np.where(np.abs(z) <= huber_delta, 1.0, huber_delta/np.abs(z))
    a, b = beta[0], beta[1]
    return float(a), float(b)


def build_pred_dp_from_qs_mbar(qs_pa: np.ndarray, r: Optional[float], beta: Optional[float], Cf: float = 1.0) -> np.ndarray:
    if r is None or beta is None: 
        return np.full_like(qs_pa, np.nan, dtype=float)
    dp_pa = Cf * (1.0 - beta**4) * (float(r)**2) * np.asarray(qs_pa, float)
    return dp_pa / 100.0


def build_pred_dp_series_from_qs(qs_pa: np.ndarray, r: Optional[float], beta: Optional[float], Cf: float = 1.0) -> np.ndarray:
    """Kept for backward compatibility; identical to build_pred_dp_from_qs_mbar."""
    return build_pred_dp_from_qs_mbar(qs_pa, r, beta, Cf)
