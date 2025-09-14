from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict, Optional, Tuple


def _best_col(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns: return c
    return None


def primary_leg_mask_from_z(z: np.ndarray, edge_frac: float = 0.05) -> np.ndarray:
    """Keep the longest monotone span (primary leg) in measured position."""
    z = np.asarray(z, float)
    n = len(z)
    ok = np.isfinite(z)
    if n < 20 or ok.sum() < 10: return np.ones(n, dtype=bool)
    # Find global max/min and choose the longer monotone run leading to that extremum
    i_max, i_min = int(np.nanargmax(z)), int(np.nanargmin(z))
    # Build monotone up run ending at max:
    up = np.zeros(n, dtype=bool)
    prev = -np.inf
    for i in range(0, i_max + 1):
        if np.isfinite(z[i]) and z[i] >= prev:
            up[i] = True; prev = z[i]
        else:
            up[i] = False
    # Build monotone down run ending at min:
    dn = np.zeros(n, dtype=bool)
    prev = +np.inf
    for i in range(0, i_min + 1):
        if np.isfinite(z[i]) and z[i] <= prev:
            dn[i] = True; prev = z[i]
        else:
            dn[i] = False
    # Choose longer
    keep = up if up.sum() >= dn.sum() else dn
    # Pad a little at both ends
    k = max(1, int(edge_frac * n))
    first = int(np.argmax(keep)); last = n - 1 - int(np.argmax(keep[::-1]))
    first = max(0, first - k); last = min(n - 1, last + k)
    m = np.zeros(n, dtype=bool); m[first:last + 1] = True
    return m


def aggregate_by_xi(per_sample: pd.DataFrame,
                    qcols_by_port: Dict[int, str],
                    xi_col_candidates = ("xi","Xi","XI","xi_norm","xi_index"),
                    aj_col_candidates = ("Aj","A_j","area_frac","AreaFrac","A_xi"),
                    z_cols_by_port: Optional[Dict[int, str]] = None
                   ) -> Tuple[Dict[int,float], pd.DataFrame, Dict]:
    """Aggregate q_s profiles by ξ and compute Aj-weighted means.

    Returns
    -------
    per_port_mean_qs : dict
        {port: Aj-weighted mean of median(q_s|xi)}
    profile_df : DataFrame
        Long-form per-port profile with columns ``[Port, xi, Aj, q_s_median, q_s_smoothed]``
    meta : dict
        {port: {n_xi, used_primary_leg, xi_col, aj_col}}

    Falls back to time-median if xi/Aj not available.
    """
    xi_col = _best_col(per_sample, xi_col_candidates)
    aj_col = _best_col(per_sample, aj_col_candidates)
    per_port_mean: Dict[int, float] = {}
    meta: Dict[int, Dict] = {}
    profile_rows = []
    for p, qcol in qcols_by_port.items():
        q = pd.to_numeric(per_sample[qcol], errors="coerce")
        m_keep = np.ones(len(q), dtype=bool)
        if z_cols_by_port and p in z_cols_by_port:
            zc = z_cols_by_port[p]
            if zc in per_sample.columns:
                z = pd.to_numeric(per_sample[zc], errors="coerce").to_numpy()
                m_keep = primary_leg_mask_from_z(z)
        if xi_col is None or aj_col is None:
            # robust time-median fallback
            per_port_mean[p] = float(np.nanmedian(q[m_keep]))
            meta[p] = {"n_xi": 0, "used_primary_leg": bool(m_keep.any()), "xi_col": None, "aj_col": None}
            continue
        df = pd.DataFrame({
            "xi": pd.to_numeric(per_sample[xi_col], errors="coerce"),
            "Aj": pd.to_numeric(per_sample[aj_col], errors="coerce"),
            "qs": q,
            "keep": m_keep,
        }).dropna(subset=["xi", "Aj", "qs"])
        if df.empty:
            per_port_mean[p] = float(np.nan)
            meta[p] = {"n_xi": 0, "used_primary_leg": False, "xi_col": xi_col, "aj_col": aj_col}
            continue
        # group by xi (rounded to 3 decimals to collapse repeats), take median qs per xi
        df["xi_b"] = df["xi"].round(3)
        g = df[df["keep"]].groupby("xi_b", as_index=False).agg(
            qs_med=("qs", "median"), Aj_med=("Aj", "median")
        )
        if g.empty:
            per_port_mean[p] = float(np.nan)
            meta[p] = {"n_xi": 0, "used_primary_leg": False, "xi_col": xi_col, "aj_col": aj_col}
            continue
        # Aj weight normalization across xi-bins
        w = g["Aj_med"].to_numpy(dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        w = w / (np.sum(w) + 1e-12)
        qs = g["qs_med"].to_numpy(dtype=float)
        per_port_mean[p] = float(np.nansum(w * qs))
        meta[p] = {
            "n_xi": int(len(g)),
            "used_primary_leg": bool(m_keep.any()),
            "xi_col": xi_col,
            "aj_col": aj_col,
        }
        # save profile rows for audit
        g = g.rename(columns={"xi_b": "xi", "Aj_med": "Aj", "qs_med": "q_s_median"})
        g["q_s_smoothed"] = g["q_s_median"].rolling(3, center=True, min_periods=1).mean()
        g["Port"] = p
        profile_rows.append(g[["Port", "xi", "Aj", "q_s_median", "q_s_smoothed"]])
    profile_df = (
        pd.concat(profile_rows, ignore_index=True)
        if profile_rows
        else pd.DataFrame(columns=["Port", "xi", "Aj", "q_s_median", "q_s_smoothed"])
    )
    return per_port_mean, profile_df, meta

