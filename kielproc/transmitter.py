from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict, Any
import json, math
import pandas as pd
import numpy as np


@dataclass
class TxParams:
    fs_percent: float = 100.0   # worksheet J1
    ip_gain: float = 0.85       # worksheet G1
    ip_offset: float = 5.4      # worksheet G2
    uic_k: float = 4.26         # worksheet literal
    uic_c: float = 1.1843       # worksheet literal


def _temp_factor(T: float) -> float:
    # D12 = (T/500)*0.938 + 0.512
    return (T/500.0)*0.938 + 0.512


def uic_percent(dp: float, T: float, urv: float, p: TxParams = TxParams()) -> float:
    # A7..E7 (sheet-exact)
    A7 = ((dp/urv)*4.0) + 1.0
    B7 = ((T/500.0)*4.0) + 1.0
    D7 = (p.uic_k*(A7 - 1.0)) / (B7 + p.uic_c) + 1.0
    E7 = math.sqrt(max((D7 - 1.0)*4.0, 0.0)) + 1.0
    return ((E7 - 1.0)/4.0)*p.fs_percent


def y820_output(dp: float, T: float, urv: float, p: TxParams = TxParams()) -> float:
    # D12, E12, G12 (sheet-exact)
    D12 = _temp_factor(T)
    E12 = (((dp/urv)/D12)*20.0) - 10.0
    return ((((E12*p.ip_gain) + p.ip_offset) + 10.0)/20.0)*91.2


def derive_urv(dp: Iterable[float], T: Iterable[float], *, min_fraction: float = 0.60,
               quantile: float = 0.95) -> float:
    """URV so that p{quantile} of (dp/T-factor) uses ≥ min_fraction of span."""
    dp = np.asarray(list(dp), dtype=float)
    T  = np.asarray(list(T), dtype=float)
    if dp.size == 0 or T.size == 0 or dp.size != T.size:
        raise ValueError("dp and T arrays must be same nonzero length")
    D = (T/500.0)*0.938 + 0.512
    dp_eff = dp / D
    target = np.quantile(dp_eff, quantile)
    if target <= 0:
        raise ValueError("Nonpositive target dp_eff; cannot derive URV")
    if not (0.05 <= min_fraction <= 0.99):
        raise ValueError("min_fraction must be 0.05..0.99")
    return float(target/min_fraction)


def make_setpoint_table(df: pd.DataFrame, *, dp_col: str, T_col: str,
                        urv: float, params: TxParams = TxParams()) -> pd.DataFrame:
    """Vectorized: UIC% and 820 (sheet-exact)."""
    Tfac = (df[T_col]/500.0)*0.938 + 0.512
    # UIC%
    A7 = ((df[dp_col]/urv)*4.0) + 1.0
    B7 = ((df[T_col]/500.0)*4.0) + 1.0
    D7 = (params.uic_k*(A7 - 1.0))/(B7 + params.uic_c) + 1.0
    E7 = np.sqrt(np.maximum((D7 - 1.0)*4.0, 0.0)) + 1.0
    uic_pct = ((E7 - 1.0)/4.0)*params.fs_percent
    # 820
    E12 = (((df[dp_col]/urv)/Tfac)*20.0) - 10.0
    y820 = ((((E12*params.ip_gain) + params.ip_offset) + 10.0)/20.0)*91.2
    out = df.copy()
    out["UIC_percent"] = uic_pct
    out["Y820"] = y820
    return out


def compute_and_write_setpoints(csv_in: Path, out_json: Path, out_csv: Path,
                                *, dp_col: str, T_col: str,
                                min_fraction: float = 0.60, quantile: float = 0.95,
                                params: TxParams = TxParams()) -> Dict[str, Any]:
    """Load CSV, auto-derive URV, compute setpoints, write csv/json."""
    df = pd.read_csv(csv_in)
    if dp_col not in df or T_col not in df:
        raise ValueError(f"CSV must contain '{dp_col}' and '{T_col}' columns")
    urv = derive_urv(df[dp_col].values, df[T_col].values,
                     min_fraction=min_fraction, quantile=quantile)
    table = make_setpoint_table(df, dp_col=dp_col, T_col=T_col, urv=urv, params=params)
    meta = {
        "urv": urv, "lrv": 0.0,
        "min_fraction": min_fraction, "quantile": quantile,
        "params": vars(params),
        "inputs": {"csv": str(csv_in), "dp_col": dp_col, "T_col": T_col},
        "outputs": {"csv": str(out_csv), "json": str(out_json)}
    }
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps({"meta": meta, "preview": table.head(12).to_dict(orient="list")}, indent=2))
    return {"urv": urv, "rows": int(table.shape[0]), "csv": str(out_csv), "json": str(out_json)}
