from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json, math
import numpy as np
import pandas as pd
import openpyxl

@dataclass
class FlowCalib:
    """Calibration constants for the 2 tracks."""
    K_uic_th_per_sqrt_mbar: float     # UIC: Flow_UIC = K * sqrt(DP_mbar)
    m_820_th_per_mbar: float          # 820: Flow_820 = m * DP_mbar + c
    c_820_th: float

def _fit_from_workbook(xlsx_path: Path) -> FlowCalib:
    """Extract UIC K and 820 linear (m,c) from the lookup table on Sheet1.
       Expects: col2=DP (mbar), col3=UIC flow (t/h), col4=820 flow (t/h)."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    ws = wb["Sheet1"]
    dps, uic, lin = [], [], []
    # The table starts ~row 19… and runs downward
    for r in range(19, 400):
        dp = ws.cell(row=r, column=2).value
        f3 = ws.cell(row=r, column=3).value
        f4 = ws.cell(row=r, column=4).value
        if not isinstance(dp, (int,float)):
            continue
        if isinstance(f3, (int,float)) and isinstance(f4, (int,float)) and dp >= 0:
            dps.append(float(dp)); uic.append(float(f3)); lin.append(float(f4))
    if not dps:
        raise ValueError("No DP rows found in Sheet1")
    dps = np.array(dps); uic = np.array(uic); lin = np.array(lin)
    # K from median of flow/sqrt(DP) over DP>0
    mask = dps > 0
    K = np.median(uic[mask] / np.sqrt(dps[mask]))
    # 820 linear via least-squares: flow = m*dp + c
    A = np.vstack([dps, np.ones_like(dps)]).T
    m, c = np.linalg.lstsq(A, lin, rcond=None)[0]
    return FlowCalib(K_uic_th_per_sqrt_mbar=float(K), m_820_th_per_mbar=float(m), c_820_th=float(c))

def _ensure_calib(K: Optional[float], m: Optional[float], c: Optional[float],
                  xlsx: Optional[Path]) -> FlowCalib:
    if xlsx:
        return _fit_from_workbook(xlsx)
    # If user provided all three, use them; else provide sensible defaults (sheet-like)
    if K is None or m is None or c is None:
        # Defaults inferred from the workbook you shared (summer set): K≈33.5, m≈8.4, c≈31.5
        K = 33.5 if K is None else K
        m = 8.4  if m is None else m
        c = 31.5 if c is None else c
    return FlowCalib(K_uic_th_per_sqrt_mbar=float(K), m_820_th_per_mbar=float(m), c_820_th=float(c))

def _to_mbar(dp_value: float, dp_unit: str) -> float:
    """Convert dp to mbar according to unit hint."""
    u = (dp_unit or "mbar").lower()
    if u in ("mbar","mb","milli-bar","millibar"): return float(dp_value)
    if u in ("pa","pascal","pascals"):           return float(dp_value) / 100.0
    if u in ("kpa",):                             return float(dp_value) * 10.0
    raise ValueError(f"Unsupported dp_unit '{dp_unit}'")

def compute_and_write_flow_lookup(
    csv_in: Path,
    out_json: Path,
    out_csv: Path,
    *,
    dp_col: str,
    T_col: Optional[str] = None,         # accepted but not used in pure lookup
    dp_unit: str = "mbar",
    K_uic: Optional[float] = None,
    m_820: Optional[float] = None,
    c_820: Optional[float] = None,
    calib_workbook: Optional[Path] = None,
) -> Dict[str, Any]:
    """Replicate the workbook's lookup: UIC=K*sqrt(DP_mbar), 820=m*DP_mbar+c.
       Writes a per-sample table and summary with the (K,m,c) used."""
    df = pd.read_csv(csv_in)
    if dp_col not in df.columns:
        raise ValueError(f"CSV is missing dp column '{dp_col}'")
    calib = _ensure_calib(K_uic, m_820, c_820, calib_workbook)
    dp_mbar = df[dp_col].apply(lambda v: _to_mbar(v, dp_unit)).astype(float)
    flow_uic = calib.K_uic_th_per_sqrt_mbar * np.sqrt(np.maximum(dp_mbar.values, 0.0))
    flow_820 = calib.m_820_th_per_mbar * dp_mbar.values + calib.c_820_th
    out = df.copy()
    out["DP_mbar"] = dp_mbar.values
    out["Flow_UIC_tph"] = flow_uic
    out["Flow_820_tph"] = flow_820
    out["Flow_err_820_minus_UIC_tph"] = out["Flow_820_tph"] - out["Flow_UIC_tph"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    meta = {
        "calib": {
            "K_uic_th_per_sqrt_mbar": calib.K_uic_th_per_sqrt_mbar,
            "m_820_th_per_mbar": calib.m_820_th_per_mbar,
            "c_820_th": calib.c_820_th,
            "source": ("workbook" if calib_workbook else "explicit_or_default"),
            "workbook": str(calib_workbook) if calib_workbook else None,
        },
        "inputs": {"csv": str(csv_in), "dp_col": dp_col, "T_col": T_col, "dp_unit": dp_unit},
        "outputs": {"csv": str(out_csv), "json": str(out_json)},
        "preview": out.head(12).to_dict(orient="list"),
    }
    out_json.write_text(json.dumps(meta, indent=2))
    return {"rows": int(out.shape[0]), "csv": str(out_csv), "json": str(out_json), "calib": meta["calib"]}
