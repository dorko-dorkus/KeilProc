from __future__ import annotations
from pathlib import Path
import json, math
import numpy as np

R_SPECIFIC_AIR = 287.05  # J/(kg*K)

def _ensure_float(x):
    try: return float(x)
    except Exception: return None

def build_venturi_result(outdir: Path, *,
                         beta: float | None,
                         area_As_m2: float | None,
                         baro_pa: float | None,
                         T_K: float | None,
                         m_dot_hint_kg_s: float | None = None) -> Path | None:
    """
    Write <outdir>/venturi_result.json with fields:
      flow_kg_s[], dp_pa[], beta, A1_m2, At_m2, rho_kg_m3.
    Uses rho = baro_pa / (R * T_K). Requires beta and A1 (section area).
    If m_dot_hint is given, we sweep 0.25..2.0 × hint; else 0..100 t/h.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    if beta is None or area_As_m2 is None or baro_pa is None or T_K is None:
        return None
    beta = _ensure_float(beta); A1 = _ensure_float(area_As_m2)
    P   = _ensure_float(baro_pa); T  = _ensure_float(T_K)
    if None in (beta, A1, P, T) or beta <= 0 or A1 <= 0 or P <= 0 or T <= 0:
        return None
    rho = P / (R_SPECIFIC_AIR * T)
    At  = (beta**2) * A1
    # flow sweep
    if m_dot_hint_kg_s and m_dot_hint_kg_s > 0:
        m0 = float(m_dot_hint_kg_s)
        flow_kg_s = np.linspace(max(0.1, 0.25*m0), 2.0*m0, 200)
    else:
        flow_kg_s = np.linspace(0.1, 100.0/3.6, 250)  # 0.1..(100 t/h)
    # Δp = (1-β^4) * m^2 / (2 ρ A_t^2)
    dp_pa = (1.0 - beta**4) * (flow_kg_s**2) / (2.0 * rho * (At**2))
    out = {
        "flow_kg_s": flow_kg_s.tolist(),
        "dp_pa": dp_pa.tolist(),
        "beta": beta,
        "A1_m2": A1,
        "At_m2": At,
        "rho_kg_m3": rho,
    }
    p = outdir / "venturi_result.json"
    p.write_text(json.dumps(out, indent=2))
    return p
