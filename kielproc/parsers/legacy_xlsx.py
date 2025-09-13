from __future__ import annotations
from typing import Dict, Optional, List
import re
import numpy as np
import pandas as pd

UNIT_FACTORS = {"PA": 1.0, "KPA": 1_000.0, "MMH2O": 9.80665, "MMH₂O": 9.80665}

def _unit_factor(u: str, default: float = 1.0) -> float:
    if u is None: return default
    # strip whitespace and punctuation (so "Pa.", "Pa " → "PA")
    u = re.sub(r"[^A-Za-z0-9]+", "", str(u)).upper()
    return UNIT_FACTORS.get(u, default)

def _parse_port_sheet_with_replicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly parse a legacy P# sheet that may lay out 1 block (simple) or 4 replicate blocks horizontally:
      [Time, Static Pressure, Velocity Pressure, Duct Air Temperature, Piccolo Tx Current, (blank?)] × N
    Returns a long frame with columns:
      time(str), static_gauge_pa, vp_pa, temp_C, piccolo_mA, replicate(int)
    Fails safe: returns empty DataFrame if pattern not detected.
    """
    ncols = df.shape[1]
    blocks: List[pd.DataFrame] = []
    # Iterate possible 6-col blocks (works for single-block too)
    for start in range(0, ncols, 6):
        sub = df.iloc[:, start:start+6].copy()
        if sub.shape[1] < 3:
            continue
        # Header can be in row 0 or row 1 in some books; choose the first row that contains "Static Pressure"
        row0 = sub.iloc[0].astype(str).tolist()
        row1 = sub.iloc[1].astype(str).tolist() if len(sub) > 1 else []
        if any("Static Pressure" in s for s in row0):
            hdr = row0; unt = sub.iloc[1].astype(str).tolist() if len(sub) > 1 else []
            hdr_row_idx = 0; unit_row_idx = 1
        elif any("Static Pressure" in s for s in row1):
            hdr = row1; unt = sub.iloc[2].astype(str).tolist() if len(sub) > 2 else []
            hdr_row_idx = 1; unit_row_idx = 2
        else:
            continue
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
            continue
        # Unit factors (robust to "Pa.", "kPa ", etc.)
        uf_static = _unit_factor(unt[name2idx["static"]] if name2idx.get("static") is not None else "Pa", 1.0)
        uf_vp     = _unit_factor(unt[name2idx["vp"]]     if name2idx.get("vp")     is not None else "Pa", 1.0)
        # Find first data row: skip “Averages”, repeated headers, blanks; require static numeric
        first_data = max(unit_row_idx + 1, hdr_row_idx + 1)
        for r in range(first_data, min(first_data + 12, len(sub))):
            tcell = str(sub.iloc[r, name2idx["time"]]).strip().lower()
            if tcell in ("averages", "time", ""):
                continue
            try:
                float(str(sub.iloc[r, name2idx["static"]]).strip())
                first_data = r
                break
            except Exception:
                continue
        # Slice data rows from first_data
        data = sub.iloc[first_data:, [name2idx[k] for k in sorted(name2idx)]].copy()
        data.columns = sorted(name2idx)
        # Coerce numerics
        if "static" in data: data["static"] = pd.to_numeric(data["static"], errors="coerce")
        if "vp"     in data: data["vp"]     = pd.to_numeric(data["vp"],     errors="coerce")
        if "temp"   in data: data["temp"]   = pd.to_numeric(data["temp"],   errors="coerce")
        if "pic"    in data: data["pic"]    = pd.to_numeric(data["pic"],    errors="coerce")
        # Drop “Averages” rows (sometimes numeric in static/VP but time == 'Averages')
        if "time" in data.columns:
            data = data[~data["time"].astype(str).str.strip().str.lower().eq("averages")]
        # Unit convert
        if "static" in data: data["static"] *= uf_static
        if "vp" in data:     data["vp"]     *= uf_vp
        data["replicate"] = start//6 + 1
        blocks.append(data)
    if not blocks:
        return pd.DataFrame()
    out = pd.concat(blocks, ignore_index=True)
    # Standardize column names for the integrator
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
    return out
