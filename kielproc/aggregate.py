from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, re
import numpy as np
import pandas as pd
from .physics import rho_from_pT, rho_from_pT_scalar, R_SPECIFIC_AIR

# Matches P1, PORT 1, Port_1, Run07_P1, etc.  Avoids matching P10 etc.
_PORT_PAT = re.compile(r"(?i)(?<![0-9A-Za-z])P(?:ORT)?[ _]*([1-8])(?![0-9A-Za-z])")


def _port_id(label: str) -> str:
    """Return canonical port ID (``P1``..``P8``) from a filename or label."""
    m = _PORT_PAT.search(str(label))
    if not m:
        raise ValueError(f"Cannot parse port identifier from {label!r}")
    return f"P{m.group(1)}"


def _port_id_from_stem(stem: str) -> str | None:
    m = _PORT_PAT.search(stem)
    return f"P{m.group(1)}" if m else None


def _has_headers(df: pd.DataFrame) -> bool:
    """Quick sniff for acceptable VP/Temperature aliases without raising."""
    low = {c.lower() for c in df.columns}
    vp_ok = any(k.lower() in low for k in _VP_ALIASES)
    t_ok = any(k.lower() in low for k in _T_ALIASES)
    return bool(vp_ok and t_ok)


def _load_parse_summary_pairs(run_dir: Path) -> list[tuple[str, Path]] | None:
    """Use ``*__parse_summary.json`` if present to pick only vertical sheets P1..P8."""
    js = list(run_dir.glob("*__parse_summary.json"))
    if not js:
        return None
    try:
        data = json.loads(js[0].read_text())
        sheets = data.get("sheets", [])
    except Exception:
        return None
    pairs: list[tuple[str, Path]] = []
    for e in sheets:
        if e.get("mode") != "vertical":
            continue  # reject 'Data' and calc sheets outright
        sheet = str(e.get("sheet", "")).strip()
        m = re.match(r"(?i)^p([1-8])$", sheet)
        if not m:
            continue
        pid = f"P{m.group(1)}"
        p = Path(e.get("csv_path", ""))
        # if parse JSON holds an absolute path from another machine, resolve by filename in run_dir
        if not p.exists():
            local = run_dir / p.name
            if local.exists():
                p = local
        if p.exists():
            pairs.append((pid, p))
    if pairs:
        pairs.sort(key=lambda kv: int(kv[0][1:]))
        return pairs
    return None


def _discover_pairs(run_dir: Path, file_glob: str) -> tuple[list[tuple[str, Path]], list[tuple[str, str]]]:
    """
    Return ``(pairs, skipped)`` where ``pairs = [(PortId, Path)]`` and
    ``skipped = [(filename, reason)]``.
    Preference order:
      1) parse_summary vertical sheets P1..P8 (if present)
      2) header-sniffed CSVs in run_dir labeled P1.. in discovery order
    """
    pairs = _load_parse_summary_pairs(run_dir)
    skipped: list[tuple[str, str]] = []
    if pairs:
        # sanity: header sniff and drop any accidental non-data CSVs
        ok: list[tuple[str, Path]] = []
        for pid, p in pairs:
            try:
                df = pd.read_csv(p, nrows=50)
                if _has_headers(df):
                    ok.append((pid, p))
                else:
                    skipped.append((p.name, "no VP/Temperature headers"))
            except Exception as e:
                skipped.append((p.name, f"read error: {e}"))
        ok.sort(key=lambda kv: int(kv[0][1:]))
        return ok, skipped

    # fallback: glob, sniff, label as P1.. in sorted name order
    candidates = []
    for p in sorted(run_dir.glob(file_glob)):
        try:
            df = pd.read_csv(p, nrows=50)
        except Exception as e:
            skipped.append((p.name, f"read error: {e}"))
            continue
        if not _has_headers(df):
            skipped.append((p.name, "no VP/Temperature headers"))
            continue
        candidates.append(p)
    pairs: list[tuple[str, Path]] = []
    for idx, p in enumerate(candidates[:8], start=1):
        stem_id = _port_id_from_stem(p.stem)
        pid = stem_id if stem_id else f"P{idx}"
        pairs.append((pid, p))
    pairs.sort(key=lambda kv: int(kv[0][1:]))
    return pairs, skipped


def discover_pairs(run_dir: Path, file_glob: str) -> tuple[list[tuple[str, Path]], list[tuple[str, str]]]:
    """Public wrapper exposing :func:`_discover_pairs`.

    Parameters
    ----------
    run_dir:
        Directory containing port CSV files.
    file_glob:
        Glob pattern used to discover candidate CSV files.
    """
    return _discover_pairs(run_dir, file_glob)

R = R_SPECIFIC_AIR  # J/(kg·K)

# ——— Normalizer utilities ———

_VP_ALIASES = [
    "VP",
    "Velocity Pressure",
    "VelPress_Pa",
    "q_dyn_Pa",
    "VelPress_inH2O",
    "VelPress_mmH2O",
    "VelPress_cmH2O",
]
_T_ALIASES  = ["Temperature","Temp_C","Duct Air Temperature","T_C","Temp_F","Temperature_F"]
_S_ALIASES  = [
    "Static",
    "Static Pressure",
    "P_static_Pa",
    "Static_kPa",
    "P_abs_Pa",
    "Static_mbar",
    "P_abs_mbar",
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
    if 'cmh2o' in nm:
        return 'cmH2O'
    if 'mbar' in nm:
        return 'mbar'
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
    if kind == "VP" and unit_hint == "cmH2O":
        return pd.to_numeric(series, errors="coerce") * 98.0665
    if unit_hint == "kPa":
        return pd.to_numeric(series, errors="coerce") * 1000.0
    if unit_hint == "mbar":
        return pd.to_numeric(series, errors="coerce") * 100.0
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
    bar_unit= _infer_unit_from_name(bar_col, "Pa") if bar_col else None

    out = pd.DataFrame()
    out["VP"] = _coerce("VP", df[vp_col], vp_unit)
    out["Temperature"] = _coerce("Temperature", df[t_col], t_unit)
    if ts_col:
        out["Time"] = pd.to_datetime(df[ts_col], errors="coerce")
    if "Replicate" in df.columns:
        out["Replicate"] = pd.to_numeric(df["Replicate"], errors="coerce").fillna(method="ffill").fillna(0).astype(int)

    # Absolute static: prefer explicit absolute column, otherwise add barometric to gauge
    cols = {c.lower(): c for c in df.columns}
    def _col(name: str) -> pd.Series:
        return _coerce("Static", df[name], _infer_unit_from_name(name, "Pa"))

    static_abs: pd.Series
    p_src: str
    st_src: str | None = None
    if "static_abs_pa" in cols:
        st_src = cols["static_abs_pa"]
        raw_abs = _col(st_src)
        if np.nanmedian(raw_abs.to_numpy(float)) >= 8.0e4:
            static_abs = raw_abs
            p_src = "Static_abs_pa_column"
        else:
            if bar_col:
                baro = _coerce("Static", df[bar_col], bar_unit or "Pa")
            elif baro_cli_pa is not None:
                baro = pd.Series(baro_cli_pa, index=raw_abs.index, dtype=float)
            else:
                raise ValueError("Static_abs_pa present but <80 kPa and no baro_pa to correct it")
            static_abs = raw_abs + baro
            p_src = "Static_abs_pa_TOO_SMALL_treated_as_gauge_plus_baro"
    else:
        st_key = next((cols[k] for k in ("static_gauge_pa", "static_pa", "static") if k in cols), None)
        if st_key is None:
            raise KeyError("No static pressure column found (expected Static_abs_pa or Static[_gauge]_pa)")
        st_src = st_key
        static_g = _col(st_src).fillna(0.0)
        if bar_col:
            baro = _coerce("Static", df[bar_col], bar_unit or "Pa")
        elif baro_cli_pa is not None:
            baro = pd.Series(baro_cli_pa, index=static_g.index, dtype=float)
        else:
            raise ValueError("Barometric pressure required to reconstruct absolute plane static")
        static_abs = static_g + baro
        p_src = "Static_gauge_plus_baro"

    st_unit = _infer_unit_from_name(st_src, "Pa") if st_src else None
    out["Static_abs_Pa"] = static_abs
    meta = {
        "method": "repo_unify_schema" if cand is not None else "aliases_units",
        "vp_unit": vp_unit,
        "t_unit": t_unit,
        "static_unit": st_unit,
        "baro_unit": bar_unit,
        "p_abs_source": p_src,
        "baro_pa_used": float(baro_cli_pa) if baro_cli_pa is not None else None,
    }
    return out, meta

# ——— Computation ———

@dataclass
class RunConfig:
    height_m: float
    width_m: float
    weights: dict[str, float] | None = None    # keys like "P1"; normalized via ``_port_id``
    replicate_strategy: str = "mean"           # "mean" or "last"

    def __post_init__(self) -> None:
        if self.weights:
            self.weights = {_port_id(k): v for k, v in self.weights.items()}


def _port_scalars_from_samples(norm: pd.DataFrame, replicate_strategy: str) -> dict:
    """Compute scalar summaries for a port from sample-wise data.

    Returns a dict containing mean velocity, mass flux, dynamic pressure, and
    the mean static pressure and temperature (Kelvin) for the port. If a
    ``Replicate`` column is present, averaging respects the provided replicate
    strategy (``"mean"`` or ``"last"``).
    """
    tC = norm["Temperature"].to_numpy(float)
    if np.any(tC <= -273.15):
        raise ValueError("Temperature at or below -273.15°C encountered")
    if np.nanmedian(tC) < -30 or np.nanmedian(tC) > 650:
        raise ValueError("Temperature °C looks implausible for this campaign; check units/legacy parse")
    T_K = tC + 273.15
    p_s = norm["Static_abs_Pa"].to_numpy(float)
    vp  = norm["VP"].to_numpy(float).clip(min=0.0)
    rho = rho_from_pT(p_s, T_K, R=R_SPECIFIC_AIR)
    v   = np.sqrt(2.0 * vp / rho, where=rho > 0, out=np.full_like(rho, np.nan))
    rhov = rho * v
    qs  = 0.5 * rho * v * v

    df = pd.DataFrame({
        "v": v,
        "rhov": rhov,
        "qs": qs,
        "p_s_pa": p_s,
        "T_K": T_K,
        "rep": norm.get("Replicate", pd.Series(index=norm.index, data=np.nan)),
    })
    if "Replicate" in norm.columns:
        g = df.groupby("rep", dropna=True).mean(numeric_only=True)
        row = g.iloc[-1] if replicate_strategy == "last" else g.mean()
    else:
        row = df.mean(numeric_only=True)
    return {
        "v_m_s": float(row["v"]),
        "rho_v_kg_m2_s": float(row["rhov"]),
        "q_s_pa": float(row["qs"]),
        "p_s_pa": float(row["p_s_pa"]),
        "T_K": float(row["T_K"]),
    }

def integrate_run(
    run_dir: Path,
    cfg: RunConfig,
    file_glob: str = "*.csv",
    baro_cli_pa: float | None = None,
    area_ratio: float | None = None,
    beta: float | None = None,
) -> dict:
    """
    Discover port CSVs, normalize, reduce and integrate. Skips non-data CSVs
    safely. Returns
    ``{'per_port': DataFrame, 'duct': dict, 'files': list, 'normalize_meta': dict, 'pairs': list, 'skipped': list}``.
    """
    run_dir = Path(run_dir)
    pairs, skipped = _discover_pairs(run_dir, file_glob)
    if not pairs:
        raise FileNotFoundError(f"No usable port CSVs found in {run_dir}")

    rows, normalize_meta = [], {}
    for port_id, pf in pairs:
        raw = pd.read_csv(pf)
        norm, meta = _normalize_df(raw, baro_cli_pa)
        normalize_meta[pf.name] = meta
        scal = _port_scalars_from_samples(norm, cfg.replicate_strategy)
        rows.append({
            "Port": port_id,
            "FileStem": pf.stem,
            "VP_pa_mean": float(pd.to_numeric(norm["VP"]).mean()),
            "T_C_mean": float(pd.to_numeric(norm["Temperature"]).mean()),
            "Static_abs_pa_mean": float(pd.to_numeric(norm["Static_abs_Pa"]).mean()),
            **scal,
            "p_abs_source": meta.get("p_abs_source", ""),
            "replicate_strategy": (cfg.replicate_strategy if "Replicate" in norm.columns else "none"),
        })
    per = pd.DataFrame(rows).sort_values("Port").reset_index(drop=True)

    # weights
    if cfg.weights:
        # Start from provided weights (mapped to present ports), then renormalize
        # over the actually present set so that missing ports are dropped gracefully.
        w = np.array([cfg.weights.get(p, 0.0) for p in per["Port"]], float)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            raise ValueError("Provided weights are zero or invalid for the present ports")
        w = w / s
    else:
        w = np.full(len(per), 1.0 / len(per))

    A = cfg.height_m * cfg.width_m
    v_bar = float(np.nansum(w * per["v_m_s"].to_numpy(float)))
    mflux = float(np.nansum(w * per["rho_v_kg_m2_s"].to_numpy(float)))
    Q = A * v_bar
    m_dot = A * mflux

    p_s_mean = float(np.nanmean(per["p_s_pa"].to_numpy(float)))
    T_mean_K = float(np.nanmean(per["T_K"].to_numpy(float)))
    rho_scalar = None
    try:
        rho_scalar = rho_from_pT_scalar(p_s_mean, T_mean_K)
    except Exception:
        rho_scalar = None

    out = {
        "v_bar_m_s": v_bar,
        "area_m2": A,
        "Q_m3_s": Q,
        "m_dot_kg_s": m_dot,
        "p_s_pa_mean": p_s_mean,
        "temp_K_mean": T_mean_K,
        "temp_C_mean": T_mean_K - 273.15,
    }
    if rho_scalar is not None and np.isfinite(rho_scalar):
        out["rho_kg_m3"] = float(rho_scalar)
        out["rho_source"] = "ideal_gas_pT_plane_static"

    if area_ratio is not None:
        q_s = float(np.nansum(w * per["q_s_pa"].to_numpy(float)))
        out["q_s_pa"] = q_s
        out["q_t_pa"] = (area_ratio**2) * q_s
        if beta is not None:
            out["delta_p_vent_est_pa"] = (1.0 - beta**4) * out["q_t_pa"]
    return {
        "per_port": per,
        "duct": out,
        "files": [p.name for _, p in pairs],
        "normalize_meta": normalize_meta,
        "pairs": pairs,
        "skipped": skipped,
    }
