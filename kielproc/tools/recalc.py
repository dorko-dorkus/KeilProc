from __future__ import annotations
from pathlib import Path
import json, math
import pandas as pd
import numpy as np


def recompute_duct_result_with_rho(outdir: Path, rho_kg_m3: float) -> Path | None:
    """
    Rebuild duct_result.json using a provided rho:
      v_i = sqrt(max(0, 2*q_s_i / rho)), v̄ = sum(w_i v_i), Q = A * v̄, m_dot = rho * Q
    Requires:
      • <outdir>/per_port.csv with columns q_s_pa and optionally weight (else equal weights)
      • <outdir>/duct_result.json (to get area_m2 and beta if present)
    Writes duct_result.json (overwrites coherent fields).
    """
    outdir = Path(outdir)
    per = outdir / "per_port.csv"
    djson = outdir / "duct_result.json"
    if not (per.exists() and djson.exists() and rho_kg_m3 and rho_kg_m3 > 0):
        return None
    df = pd.read_csv(per)
    if "q_s_pa" not in df.columns:
        return None
    w = df["weight"].to_numpy(float) if "weight" in df.columns else np.ones(len(df))
    w = np.clip(w, 0.0, None)
    if w.sum() == 0: w = np.ones_like(w)
    w = w / w.sum()
    qs = pd.to_numeric(df["q_s_pa"], errors="coerce").fillna(0.0).to_numpy(float)
    v = np.sqrt(np.clip(2.0 * qs / float(rho_kg_m3), 0.0, None))
    v_bar = float(np.sum(w * v))
    dj = json.loads(djson.read_text())
    A = float(dj.get("area_m2", 0.0))
    if A <= 0.0:
        # try per_port column
        if "area_m2" in df.columns:
            A = float(pd.to_numeric(df["area_m2"], errors="coerce").dropna().iloc[0])
    if A <= 0.0:
        return None
    Q = A * v_bar
    m_dot = float(rho_kg_m3) * Q
    # aggregate q_s and q_t
    q_s_mean = float(np.sum(w * qs))
    beta = dj.get("beta", None)
    q_t = (1.0 - float(beta)**4) * q_s_mean if beta is not None else None  # optional
    # update and write
    dj.update({
        "v_bar_m_s": v_bar,
        "Q_m3_s": Q,
        "m_dot_kg_s": m_dot,
        "q_s_pa": q_s_mean,
        "q_t_pa": q_t,
        "rho_kg_m3": float(rho_kg_m3),
        "rho_source": "ideal_gas_pT",
    })
    djson.write_text(json.dumps(dj, indent=2))
    return djson
