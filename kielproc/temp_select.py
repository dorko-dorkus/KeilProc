from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple

R_AIR = 287.05

def _nanmedian(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")

def _to_K(val: float, unit_hint: Optional[str] = None) -> Tuple[float, str]:
    """
    Convert candidate temperature to Kelvin with a best-effort unit guess.
    Returns (T_K, unit_used).
    """
    if val is None or not np.isfinite(val):
        return float("nan"), "unknown"
    if unit_hint:
        u = unit_hint.strip().upper()
        if u.startswith("K"):  return float(val), "K"
        if u.startswith("C"):  return float(val) + 273.15, "C"
    # Heuristic if no hint:
    #   - [120, 500] => degC typical for ducts
    #   - [500, 1500] => Kelvin
    #   - [200, 120] (nonsense) or <= 80C => likely degC but near ambient
    v = float(val)
    if 500.0 <= v <= 1500.0:
        return v, "K?"
    # Everything else assume degC
    return v + 273.15, "C?"

def pick_duct_temperature_K(per_port: Optional[pd.DataFrame],
                            workbook_hint_value: Optional[float] = None,
                            workbook_hint_unit: Optional[str] = None,
                            fallback_default_K: float = 540.0
                           ) -> Dict[str, Any]:
    """
    Choose a robust duct gas temperature in Kelvin with provenance.
    Precedence:
      1) Median of per-port T_K (or T_C + 273.15) if present and plausible.
      2) Workbook hint (with unit detection) if plausible.
      3) Fallback constant (logged as fallback).
    Plausibility window (can be tuned): 330 K .. 1100 K (≈ 57 .. 827 °C).
    """
    candidates = []
    # From per_port
    if per_port is not None and len(per_port):
        if "T_K" in per_port.columns:
            tk = _nanmedian(pd.to_numeric(per_port["T_K"], errors="coerce"))
            candidates.append(("per_port_T_K_median", tk, "K"))
        if "T_C_mean" in per_port.columns:
            tc = _nanmedian(pd.to_numeric(per_port["T_C_mean"], errors="coerce"))
            if np.isfinite(tc):
                candidates.append(("per_port_T_C_median", tc + 273.15, "C"))
    # Workbook hint
    if workbook_hint_value is not None:
        tk, u = _to_K(workbook_hint_value, workbook_hint_unit)
        candidates.append(("workbook_hint", tk, u))
    # Score candidates
    def score(TK: float) -> int:
        if not np.isfinite(TK): return -3
        if 330.0 <= TK <= 1100.0: return 3      # solid
        if 300.0 <= TK < 330.0:   return 2      # low but possibly valid
        if 1100.0 < TK <= 1500.0: return 1      # high but conceivable
        return -2                                # implausible (near-ambient or nonsense)
    best = max(candidates, key=lambda t: score(t[1]), default=None)
    if best and score(best[1]) >= 2:
        name, TK, unit = best
        return {"T_K": float(TK), "source": name, "unit_used": unit, "fallback": False}
    # Fallback
    return {"T_K": float(fallback_default_K), "source": "fallback_default", "unit_used": "K", "fallback": True}
