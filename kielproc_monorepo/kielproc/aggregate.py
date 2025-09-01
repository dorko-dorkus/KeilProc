from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, re
import numpy as np
import pandas as pd

R = 287.05  # J/(kg·K)

# ——— Normalizer utilities ———

_VP_ALIASES = ["VP","Velocity Pressure","VelPress_Pa","q_dyn_Pa","VelPress_inH2O","VelPress_mmH2O"]
_T_ALIASES  = ["Temperature","Temp_C","Duct Air Temperature","T_C","Temp_F","Temperature_F"]
_S_ALIASES  = [
    "Static",
    "Static Pressure",
    "P_static_Pa",
    "Static_kPa",
    "P_abs_Pa",
    "Static_gauge",
    "Static_g",
    "Gauge_Pa",
]
_BARO_ALIASES = ["Baro", "Baro_Pa", "Barometric", "Atmos_Pa", "Ambient_kPa", "Barometric_kPa"]
_TS_ALIASES = ["Time","Timestamp","DateTime","t","time","epoch"]

def _pick(df: pd.DataFrame, names: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n and n.lower() in low:
            return low[n.lower()]
    return None

def _infer_unit_from_name(colname: str, default: str) -> str:
    nm = (colname or '').lower()
    if 'inh2o' in nm:
        return 'inH2O'
    if 'mmh2o' in nm:
        return 'mmH2O'
    if 'kpa' in nm:
        return 'kPa'
    if nm.endswith('_pa') or ' pa' in nm:
        return 'Pa'
    if nm.endswith('_f'):
        return 'F'
    if nm.endswith('_c'):
        return 'C'
    return default

def _coerce(kind: str, series: pd.Series, unit_hint: str) -> pd.Series:
    if kind == "VP" and unit_hint == "inH2O":
        return pd.to_numeric(series, errors="coerce") * 249.08891
    if kind == "VP" and unit_hint == "mmH2O":
        return pd.to_numeric(series, errors="coerce") * 9.80665
    if unit_hint == "kPa":
        return pd.to_numeric(series, errors="coerce") * 1000.0
    if kind == "Temperature" and unit_hint == "F":
        return (pd.to_numeric(series, errors="coerce") - 32.0) * (5.0/9.0)
    return pd.to_numeric(series, errors="coerce")

def _normalize_df(df_raw: pd.DataFrame, baro_cli_pa: float | None):
    # 1) project unify_schema if present
    try:
        from .io import unify_schema  # optional project helper
        cand = unify_schema(df_raw)
    except Exception:
        cand = None
    df = (cand if isinstance(cand, pd.DataFrame) else df_raw).copy()

    vp_col = _pick(df, _VP_ALIASES); t_col = _pick(df, _T_ALIASES)
    if not vp_col or not t_col:
        raise ValueError("CSV must contain VP and Temperature columns")

    st_col  = _pick(df, _S_ALIASES)
    bar_col = _pick(df, _BARO_ALIASES)
    ts_col  = _pick(df, _TS_ALIASES)

    vp_unit = _infer_unit_from_name(vp_col, "Pa")
    t_unit  = _infer_unit_from_name(t_col, "C")
    st_unit = _infer_unit_from_name(st_col, "Pa") if st_col else None
    bar_unit= _infer_unit_from_name(bar_col, "Pa") if bar_col else None

    out = pd.DataFrame()
    out["VP"] = _coerce("VP", df[vp_col], vp_unit)
    out["Temperature"] = _coerce("Temperature", df[t_col], t_unit)
    if ts_col:
        out["Time"] = pd.to_datetime(df[ts_col], errors="coerce")
    if "Replicate" in df.columns:
        out["Replicate"] = pd.to_numeric(df["Replicate"], errors="coerce").fillna(method="ffill").fillna(0).astype(int)

    # Absolute static: must exist or be reconstructable
    static_abs = None
    if st_col:
        st_series = _coerce("Static", df[st_col], st_unit or "Pa")
        # If the header suggests gauge (contains 'gauge' or '_g'), add barometric
        header_says_gauge = bool(re.search(r"gauge|\b_g\b", st_col, flags=re.I))
        if header_says_gauge:
            if bar_col:
                baro = _coerce("Static", df[bar_col], bar_unit or "Pa")
            elif baro_cli_pa is not None:
                baro = pd.Series(baro_cli_pa, index=st_series.index, dtype=float)
            else:
                raise ValueError("Gauge static supplied without barometric (column or --baro)")
            static_abs = st_series + baro
            p_src = "Static_gauge_plus_baro" if bar_col else "cli_baro_plus_gauge"
        else:
            # treat as absolute by contract
            static_abs = st_series
            p_src = "Static_absolute_column"
    elif baro_cli_pa is not None:
        # No static column → not allowed; we require static per the SoT
        raise ValueError("Static is required per-sample; provide Static (absolute) or Static_gauge + Baro")
    else:
        raise ValueError("Static is required per-sample; provide Static (absolute) or Static_gauge + Baro")

    out["Static_abs_Pa"] = static_abs
    meta = {
        "method": "repo_unify_schema" if cand is not None else "aliases_units",
        "vp_unit": vp_unit,
        "t_unit": t_unit,
        "static_unit": st_unit,
        "baro_unit": bar_unit,
        "p_abs_source": p_src,
    }
    return out, meta

# ——— Computation ———

@dataclass
class RunConfig:
    height_m: float
    width_m: float
    p_abs_pa: float | None = None              # barometric pressure [Pa] if Static is gauge
    weights: dict[str, float] | None = None    # keys like "PORT 1", must sum to 1.0 if provided
    replicate_strategy: str = "mean"           # "mean" or "last"
    emit_normalized: bool = False              # write normalized snapshots

def _rho(T_C: float, p_abs: float) -> float:
    return p_abs / (R * (T_C + 273.15))

def _reduce_port(df_norm: pd.DataFrame, cfg: RunConfig) -> tuple[float,float,float,dict]:
    """
    Returns (VP_mean_Pa, T_C_mean, p_abs_Pa, notes).
    Uses Static_abs_Pa if present; otherwise uses cfg.p_abs_pa.
    Replicates, if present, are reduced by cfg.replicate_strategy.
    """
    if "VP" not in df_norm or "Temperature" not in df_norm:
        raise ValueError("Normalized frame missing VP or Temperature")

    notes = {}
    if "Static_abs_Pa" in df_norm and df_norm["Static_abs_Pa"].notna().any():
        p_abs = float(pd.to_numeric(df_norm["Static_abs_Pa"], errors="coerce").median())
    elif cfg.p_abs_pa is not None:
        p_abs = float(cfg.p_abs_pa)
    else:
        raise ValueError("Absolute static pressure required: provide Static (absolute) or Static_gauge + Baro")

    frame = df_norm[["VP","Temperature"]].copy()
    if "Replicate" in df_norm.columns:
        g = df_norm.groupby("Replicate", as_index=False)[["VP","Temperature"]].mean()
        if cfg.replicate_strategy == "last":
            vp_mean = float(g["VP"].iloc[-1]); T_C = float(g["Temperature"].iloc[-1])
            notes["replicate_strategy"] = "last"
        else:
            vp_mean = float(g["VP"].mean());   T_C = float(g["Temperature"].mean())
            notes["replicate_strategy"] = "mean"
    else:
        vp_mean = float(pd.to_numeric(frame["VP"], errors="coerce").mean())
        T_C     = float(pd.to_numeric(frame["Temperature"], errors="coerce").mean())
        notes["replicate_strategy"] = "none"

    return vp_mean, T_C, p_abs, notes

def integrate_run(run_dir: Path, cfg: RunConfig, file_glob: str = "*.csv") -> dict:
    """
    Reads 'PORT *.csv' style files from run_dir, normalizes, reduces per port, integrates horizontally.
    Returns {'per_port': DataFrame, 'duct': dict, 'files': list, 'normalize_meta': dict_by_port}.
    """
    run_dir = Path(run_dir)
    # Prefer explicit PORT N files; fall back to any CSV with 'P[1-8]' in name.
    port_files = sorted([p for p in run_dir.glob(file_glob) if re.search(r"\bP([1-8])\b", p.stem, flags=re.I)])
    if not port_files:
        port_files = sorted([p for p in run_dir.glob("*.csv") if re.search(r"\bP([1-8])\b", p.stem, flags=re.I)])
    if not port_files:
        raise FileNotFoundError(f"No port CSVs matching 'P1..P8' in {run_dir}")

    rows, normalize_meta = [], {}
    normalized_outdir = run_dir / "_integrated" / "normalized"
    for pf in port_files:
        raw = pd.read_csv(pf)
        norm, meta = _normalize_df(raw, cfg.p_abs_pa)  # try repo unify; else aliases
        normalize_meta[pf.name] = meta
        if cfg.emit_normalized:
            normalized_outdir.mkdir(parents=True, exist_ok=True)
            norm.to_csv(normalized_outdir / f"{pf.stem}.normalized.csv", index=False)

        vp, T_C, p_abs, notes = _reduce_port(norm, cfg)
        notes["p_abs_source"] = meta.get("p_abs_source", "")
        rho = _rho(T_C, p_abs)
        v = np.sqrt(max(0.0, 2.0 * vp / rho)) if rho > 0 and vp >= 0 else float("nan")
        rows.append({"Port": pf.stem.upper(), "VP_pa": vp, "T_C": T_C, "rho_kg_m3": rho, "v_m_s": v, "p_abs_pa_used": p_abs, **notes})

    per = pd.DataFrame(rows).sort_values("Port").reset_index(drop=True)

    # weights
    if cfg.weights:
        w = np.array([cfg.weights.get(p, 0.0) for p in per["Port"]], float)
        if not np.isclose(w.sum(), 1.0):
            raise ValueError("Port weights must sum to 1.0")
    else:
        w = np.full(len(per), 1.0/len(per))

    # integration
    A = cfg.height_m * cfg.width_m
    v_bar = float(np.nansum(w * per["v_m_s"].to_numpy(float)))
    mflux = float(np.nansum(w * per["rho_kg_m3"].to_numpy(float) * per["v_m_s"].to_numpy(float)))
    Q = A * v_bar
    m_dot = A * mflux

    return {
        "per_port": per,
        "duct": {"v_bar_m_s": v_bar, "area_m2": A, "Q_m3_s": Q, "m_dot_kg_s": m_dot},
        "files": [str(p.name) for p in port_files],
        "normalize_meta": normalize_meta,
    }
