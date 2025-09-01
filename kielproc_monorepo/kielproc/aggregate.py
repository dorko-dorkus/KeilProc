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
_S_ALIASES  = ["Static","Static Pressure","P_static_Pa","Gauge_Pa","Static_kPa","P_abs_Pa","Baro_Pa"]
_TS_ALIASES = ["Time","Timestamp","DateTime","t","time","epoch"]

_UNIT_HINTS = {
    # (col, suffix_or_exact) -> converter to SI
    ("VP","Pa"):           lambda x: pd.to_numeric(x, errors="coerce"),
    ("VP","inH2O"):        lambda x: pd.to_numeric(x, errors="coerce") * 249.08891,
    ("VP","mmH2O"):        lambda x: pd.to_numeric(x, errors="coerce") * 9.80665,
    ("Static","Pa"):       lambda x: pd.to_numeric(x, errors="coerce"),
    ("Static","kPa"):      lambda x: pd.to_numeric(x, errors="coerce") * 1000.0,
    ("Temperature","C"):   lambda x: pd.to_numeric(x, errors="coerce"),
    ("Temperature","F"):   lambda x: (pd.to_numeric(x, errors="coerce") - 32.0) * (5.0/9.0),
}

def _pick(df: pd.DataFrame, names: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low: return low[n.lower()]
    return None

def _infer_unit_from_name(colname: str, default: str) -> str:
    nm = colname.lower()
    if "inh2o" in nm: return "inH2O"
    if "mmh2o" in nm: return "mmH2O"
    if nm.endswith("_kpa") or "kpa" in nm: return "kPa"
    if nm.endswith("_pa")  or re.search(r"(?<!k)pa\b", nm): return "Pa"
    if nm.endswith("_f")   or nm.endswith("temp_f"): return "F"
    if nm.endswith("_c"): return "C"
    return default

def _coerce(kind: str, series: pd.Series, unit_hint: str) -> pd.Series:
    key = (kind, unit_hint)
    if key not in _UNIT_HINTS:
        raise ValueError(f"Unsupported unit for {kind}: {unit_hint}")
    return _UNIT_HINTS[key](series)

def _try_repo_unifier(df: pd.DataFrame):
    """If the repo ships unify_schema/load_logger_csv, prefer that."""
    try:
        from .io import unify_schema  # type: ignore
        out = unify_schema(df)        # expected to return VP/Temperature/Static/Time if possible
        return out
    except Exception:
        return None

def _normalize_df(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Best-effort normalizer.
    1) Try repo unify_schema.
    2) Else map aliases + infer units from header names.
    Returns (normalized_df, meta) where normalized has VP [Pa], Temperature [C], optional Static [Pa], Time if present.
    """
    # Attempt 1: project-provided unifier
    un = _try_repo_unifier(df_raw)
    if isinstance(un, pd.DataFrame):
        meta = {"method": "repo_unify_schema"}
        return un.copy(), meta

    # Attempt 2: alias + unit inference
    df = df_raw.copy()
    vp_col = _pick(df, _VP_ALIASES)
    t_col  = _pick(df, _T_ALIASES)
    if vp_col is None or t_col is None:
        raise ValueError("CSV missing velocity pressure and/or temperature columns")

    st_col = _pick(df, _S_ALIASES)
    ts_col = _pick(df, _TS_ALIASES)

    vp_unit = _infer_unit_from_name(vp_col, "Pa")
    t_unit  = _infer_unit_from_name(t_col, "C")
    st_unit = _infer_unit_from_name(st_col, "Pa") if st_col else None

    out = pd.DataFrame()
    out["VP"] = _coerce("VP", df[vp_col], vp_unit)
    out["Temperature"] = _coerce("Temperature", df[t_col], t_unit)
    if st_col:
        out["Static"] = _coerce("Static", df[st_col], st_unit or "Pa")
    if ts_col:
        out["Time"] = pd.to_datetime(df[ts_col], errors="coerce")

    # If a Replicate column exists, carry it through; otherwise leave absent (caller may segment/aggregate).
    if "Replicate" in df.columns:
        out["Replicate"] = pd.to_numeric(df["Replicate"], errors="coerce").fillna(method="ffill").fillna(0).astype(int)

    meta = {"method": "aliases_units", "vp_unit": vp_unit, "t_unit": t_unit, "static_unit": st_unit}
    return out, meta

# ——— Computation ———

@dataclass
class RunConfig:
    height_m: float
    width_m: float
    p_abs_pa: float | None = None              # fallback absolute pressure if Static not absolute
    weights: dict[str, float] | None = None    # keys like "PORT 1", must sum to 1.0 if provided
    replicate_strategy: str = "mean"           # "mean" or "last"
    emit_normalized: bool = False              # write normalized snapshots

def _rho(T_C: float, p_abs: float) -> float:
    return p_abs / (R * (T_C + 273.15))

def _static_is_absolute(static_series: pd.Series) -> bool:
    """Heuristic: treat as absolute if median is in ~80–120 kPa."""
    s = pd.to_numeric(static_series, errors="coerce").dropna()
    if s.empty: return False
    med = float(s.median())
    return 80_000.0 <= med <= 120_000.0

def _reduce_port(df_norm: pd.DataFrame, cfg: RunConfig) -> tuple[float,float,float,dict]:
    """
    Returns (VP_mean_Pa, T_C_mean, p_abs_Pa, notes).
    Uses Static if present and absolute; otherwise uses cfg.p_abs_pa.
    Replicates, if present, are reduced by cfg.replicate_strategy.
    """
    if "VP" not in df_norm or "Temperature" not in df_norm:
        raise ValueError("Normalized frame missing VP or Temperature")

    # choose absolute pressure source
    notes = {}
    if "Static" in df_norm and df_norm["Static"].notna().any() and _static_is_absolute(df_norm["Static"]):
        p_abs = float(pd.to_numeric(df_norm["Static"], errors="coerce").median())
        notes["p_abs_source"] = "Static_column_absolute"
    elif cfg.p_abs_pa is not None:
        p_abs = float(cfg.p_abs_pa)
        notes["p_abs_source"] = "config_p_abs_pa"
    else:
        raise ValueError("Absolute pressure required: provide --p-abs or a Static column with absolute Pa")

    # aggregate by replicate if provided
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
        norm, meta = _normalize_df(raw)  # try repo unify; else aliases
        normalize_meta[pf.name] = meta
        if cfg.emit_normalized:
            normalized_outdir.mkdir(parents=True, exist_ok=True)
            norm.to_csv(normalized_outdir / f"{pf.stem}.normalized.csv", index=False)

        vp, T_C, p_abs, notes = _reduce_port(norm, cfg)
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
