from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple


def current_to_dp_raw_mbar(I_mA: np.ndarray,
                           lrv_mbar: float,
                           urv_mbar: float) -> np.ndarray:
    """Linear 4–20 mA map to DP (mbar)."""
    span = float(urv_mbar - lrv_mbar)
    return lrv_mbar + np.clip((np.asarray(I_mA) - 4.0) / 16.0, -1e3, 1e3) * span


def fit_current_to_dp(I_mA: np.ndarray,
                      dp_pred_mbar: np.ndarray,
                      huber_delta: float = 0.5) -> Tuple[float, float]:
    """
    Robust fit DP ≈ a*I + b (mbar, mA).
    Huber-type IRLS (2–3 iterations) to tame outliers; returns (a,b).
    """
    I = np.asarray(I_mA, float); y = np.asarray(dp_pred_mbar, float)
    m = np.isfinite(I) & np.isfinite(y)
    I, y = I[m], y[m]
    if I.size < 10:
        # fallback: slope from span, bias from median
        a = (np.nanpercentile(y, 95) - np.nanpercentile(y, 5)) / 16.0
        b = float(np.nanmedian(y)) - a * float(np.nanmedian(I))
        return float(a), float(b)
    X = np.c_[I, np.ones_like(I)]
    w = np.ones_like(y)
    for _ in range(3):
        # weighted least squares
        W = np.diag(w)
        beta = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
        y_hat = X @ beta
        r = y - y_hat
        # Huber weights
        s = np.nanmedian(np.abs(r)) * 1.4826 or 1.0
        z = r / max(s, 1e-6)
        w = np.where(np.abs(z) <= huber_delta, 1.0, huber_delta/np.abs(z))
    a, b = beta[0], beta[1]
    return float(a), float(b)


def build_pred_dp_series_from_qs(qs_pa: np.ndarray,
                                 r: Optional[float],
                                 beta: Optional[float],
                                 Cf: float = 1.0) -> np.ndarray:
    """Predicted Piccolo DP (mbar) from plane dynamic q_s sequence via geometry."""
    if r is None or beta is None or not np.isfinite(r) or not np.isfinite(beta):
        return np.full_like(qs_pa, np.nan, dtype=float)
    dp_pa = Cf * (1.0 - beta**4) * (r**2) * np.asarray(qs_pa, float)
    return dp_pa / 100.0  # Pa→mbar
