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


def build_venturi_curve(*, beta: float | None, r: float | None, A1: float | None,
                        rho: float | None, m_dot_hint_kg_s: float | None = None) -> dict | None:
    """Return venturi curve data for the given geometry and density.

    Parameters
    ----------
    beta:
        Venturi diameter ratio :math:`\beta = d_2 / d_1`.
    r:
        Area ratio :math:`r = A_s / A_t` (verification plane over throat).
    A1:
        Verification plane area :math:`A_s` in square metres.
    rho:
        Fluid density in kg/m³.
    m_dot_hint_kg_s:
        Optional mass flow hint to set the sweep range.

    Returns
    -------
    dict | None
        Dictionary with ``flow_kg_s``, ``dp_pa``, ``beta``, ``A1_m2``, ``At_m2``,
        and ``rho_kg_m3`` or ``None`` if inputs are invalid.
    """

    beta = _ensure_float(beta)
    r = _ensure_float(r)
    A1 = _ensure_float(A1)
    rho = _ensure_float(rho)
    if None in (beta, r, A1, rho) or beta <= 0 or r <= 0 or A1 <= 0 or rho <= 0:
        return None
    At = A1 / r
    if m_dot_hint_kg_s and m_dot_hint_kg_s > 0:
        m0 = float(m_dot_hint_kg_s)
        flow_kg_s = np.linspace(max(0.1, 0.25 * m0), 2.0 * m0, 200)
    else:
        flow_kg_s = np.linspace(0.1, 100.0 / 3.6, 250)
    dp_pa = (1.0 - beta**4) * (flow_kg_s**2) / (2.0 * rho * (At**2))
    return {
        "flow_kg_s": flow_kg_s.tolist(),
        "dp_pa": dp_pa.tolist(),
        "beta": beta,
        "A1_m2": A1,
        "At_m2": At,
        "rho_kg_m3": rho,
    }
