from __future__ import annotations
import pandas as pd, numpy as np, re
from typing import Dict, Optional, List

UNIT_FACTORS = {
    "PA": 1.0,
    "KPA": 1_000.0,
    "MMH2O": 9.80665,   # common in older sheets
    "MMH₂O": 9.80665,
}

def _unit_factor(u: str, default: float = 1.0) -> float:
    if not u: return default
    u = re.sub(r"\s+", "", str(u)).upper()
    return UNIT_FACTORS.get(u, default)

def _parse_port_sheet_with_replicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly parse a legacy P# sheet that lays out 4 replicate blocks horizontally:
      [Time, Static, VP, Temp, Piccolo, (blank)] × 4
    Returns a long frame with columns:
      time(str), static_gauge_pa, vp_pa, temp_C, piccolo_mA, replicate(int)
    Fails safe: returns empty DataFrame if pattern not detected.
    """
    ncols = df.shape[1]
    blocks: List[pd.DataFrame] = []
    # Iterate possible 6-col blocks
    for start in range(0, ncols, 6):
        sub = df.iloc[:, start:start+6].copy()
        if sub.shape[1] < 4:
            continue
        # Header rows: 0=name, 1=unit, 2="Averages" row; data starts at row 3
        hdr = sub.iloc[0].astype(str).tolist()
        unt = sub.iloc[1].astype(str).tolist()
        # Map names in this block
        name2idx: Dict[str,int] = {}
        for j, nm in enumerate(hdr):
            nm_u = nm.strip().lower()
            if nm_u.startswith("time"):                      name2idx["time"] = j
            elif nm_u.startswith("static pressure"):         name2idx["static"] = j
            elif nm_u.startswith("velocity pressure"):       name2idx["vp"] = j
            elif nm_u.startswith("duct air temperature"):    name2idx["temp"] = j
            elif nm_u.startswith("piccolo tx current"):      name2idx["pic"] = j
        if "time" not in name2idx or "static" not in name2idx:
            # Not a valid block; skip
            continue
        # Unit factors
        uf_static = _unit_factor(unt[name2idx["static"]] if name2idx.get("static") is not None else "Pa", 1.0)
        uf_vp     = _unit_factor(unt[name2idx["vp"]]     if name2idx.get("vp")     is not None else "Pa", 1.0)
        # Slice data rows
        data = sub.iloc[3:, [name2idx[k] for k in sorted(name2idx)]].copy()
        data.columns = sorted(name2idx)  # ['pic','static','temp','time','vp'] depending on presence
        # Coerce numerics
        if "static" in data: data["static"] = pd.to_numeric(data["static"], errors="coerce") * uf_static
        if "vp"     in data: data["vp"]     = pd.to_numeric(data["vp"],     errors="coerce") * uf_vp
        if "temp"   in data: data["temp"]   = pd.to_numeric(data["temp"],   errors="coerce")
        if "pic"    in data: data["pic"]    = pd.to_numeric(data["pic"],    errors="coerce")
        data["replicate"] = start//6 + 1
        blocks.append(data)
    if not blocks:
        return pd.DataFrame()
    out = pd.concat(blocks, ignore_index=True)
    # Standardize column names for the integrator
    #   static_gauge_pa (explicit gauge), q_s is built later from vp
    out = out.rename(columns={
        "static": "static_gauge_pa",
        "vp": "VP_pa",
        "temp": "T_C",
        "pic": "piccolo_mA",
    })
    # Drop rows with no measurements
    meas = [c for c in ["static_gauge_pa","VP_pa","T_C","piccolo_mA"] if c in out.columns]
    out = out.dropna(subset=meas, how="all")
    # Ensure dtype
    for c in ["static_gauge_pa","VP_pa","T_C","piccolo_mA"]:
        if c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce")
    # Time stays as string (legacy has hh:mm:ss strings)
    return out
