from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import json, math, re
import numpy as np
import pandas as pd

# ---------- Season presets ----------
# Source of truth for 820 linearization (flow = m*DP + c).
# You can override these via site.defaults["calib_820_summer"] / ["calib_820_winter"].
SEASON_PRESETS = {
    "summer": {"m": 8.40, "c": 31.50},   # t/h per mbar, t/h  ← set your final values
    "winter": {"m": 8.40, "c": 31.50},   # ← set your final winter values
}
# UIC constant (Flow_UIC = K * sqrt(DP_mbar))
K_UIC_DEFAULT = 33.50   # t/h per sqrt(mbar)  ← set your final K if needed

@dataclass
class SeasonCalib:
    season: str
    K_uic: float
    m_820: float
    c_820: float
    source: str

# ---------- Helpers ----------
def _calib_for_season(season: str,
                      site_defaults: Optional[dict] = None) -> SeasonCalib:
    s = (season or "summer").lower()
    # site.defaults override (explicit m/c/K) if you want deploy-time control
    if site_defaults:
        mck = site_defaults.get(f"calib_820_{s}")  # e.g., {"m": 8.4, "c": 31.5, "K_uic": 33.5}
        if isinstance(mck, dict) and "m" in mck and "c" in mck:
            return SeasonCalib(s, float(mck.get("K_uic", K_UIC_DEFAULT)), float(mck["m"]), float(mck["c"]), "site.defaults")
    # built-in presets
    preset = SEASON_PRESETS.get(s, SEASON_PRESETS["summer"])
    return SeasonCalib(s, K_UIC_DEFAULT, float(preset["m"]), float(preset["c"]), "preset")

def _autodetect_dp_col(df: pd.DataFrame) -> Tuple[str, str]:
    """Return (dp_col, inferred_unit). Infer unit by magnitude if needed."""
    name_candidates = ["dp_mbar", "dp (mbar)", "dp", "i/p", "diff", "differential"]
    for nm in df.columns:
        if any(re.fullmatch(pat.replace(" ", r"\s*"), nm.strip().lower()) for pat in name_candidates):
            series = df[nm].astype(float, errors="ignore")
            return nm, _infer_unit_from_series(series)
    # fallback: first numeric column
    for nm in df.columns:
        try:
            series = pd.to_numeric(df[nm], errors="coerce")
            if series.notna().sum() > 0:
                return nm, _infer_unit_from_series(series)
        except Exception:
            continue
    raise ValueError("No numeric DP-like column found in logger CSV.")

def _infer_unit_from_series(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return "mbar"
    mx = float(s.quantile(0.99))
    # crude heuristics: PA typically thousands, mbar typically < 100
    if mx >= 500: return "Pa"
    return "mbar"

def _to_mbar(values: pd.Series, unit_hint: str) -> pd.Series:
    u = (unit_hint or "mbar").lower()
    if u in ("mbar","mb","millibar","milli-bar"):
        return pd.to_numeric(values, errors="coerce")
    if u in ("pa","pascal","pascals"):
        return pd.to_numeric(values, errors="coerce") / 100.0
    if u in ("kpa",):
        return pd.to_numeric(values, errors="coerce") * 10.0
    return pd.to_numeric(values, errors="coerce")

# ---------- Builders ----------
def build_reference_table(cal: SeasonCalib, *, dp_max_mbar: float = 10.0, dp_step_mbar: float = 0.1) -> pd.DataFrame:
    grid = np.arange(0.0, dp_max_mbar + 1e-9, dp_step_mbar)
    flow_uic = cal.K_uic * np.sqrt(grid)
    flow_820 = cal.m_820 * grid + cal.c_820
    return pd.DataFrame({
        "ref_DP_mbar": grid,
        "ref_Flow_UIC_tph": flow_uic,
        "ref_Flow_820_tph": flow_820,
        "ref_Flow_err_820_minus_UIC_tph": flow_820 - flow_uic,
    })

def build_data_overlay(cal: SeasonCalib, df_logger: pd.DataFrame, dp_col: Optional[str] = None, dp_unit_hint: Optional[str] = None) -> pd.DataFrame:
    if dp_col is None:
        dp_col, inferred = _autodetect_dp_col(df_logger)
        dp_unit_hint = dp_unit_hint or inferred
    dp_mbar = _to_mbar(df_logger[dp_col], dp_unit_hint)
    flow_uic = cal.K_uic * np.sqrt(np.clip(dp_mbar.values, 0.0, None))
    flow_820 = cal.m_820 * dp_mbar.values + cal.c_820
    out = pd.DataFrame({
        "data_DP_mbar": dp_mbar.values,
        "data_Flow_UIC_tph": flow_uic,
        "data_Flow_820_tph": flow_820,
        "data_Flow_err_820_minus_UIC_tph": flow_820 - flow_uic,
    })
    return out

def write_lookup_outputs(
    outdir: Path,
    *,
    season: str,
    site_defaults: Optional[dict] = None,
    logger_csv: Optional[Path] = None,
    dp_col: Optional[str] = None,
    dp_unit_hint: Optional[str] = None,
    dp_max_mbar: Optional[float] = None,
    dp_step_mbar: float = 0.1,
) -> Dict[str, Any]:
    """Always write the constant reference table; add data overlay if logger present."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    cal = _calib_for_season(season, site_defaults=site_defaults)
    # sanity
    if (
        cal.m_820 is None
        or cal.c_820 is None
        or not all(math.isfinite(v) for v in (cal.m_820, cal.c_820))
    ):
        raise ValueError(
            f"Transmitter calibration missing or invalid for season '{season}' (m or c)."
        )
    span = float(dp_max_mbar or 0.0)
    if not (span > 0.0):
        raise ValueError("DP span (dp_max_mbar) must be > 0 to build the reference table.")
    # reference side (constant)
    ref = build_reference_table(cal, dp_max_mbar=span, dp_step_mbar=dp_step_mbar)
    if np.allclose(ref.drop(columns=["ref_DP_mbar"]).values, 0.0):
        raise ValueError("Reference lookup table is all zeros; check calibration inputs.")
    ref_csv = outdir / "transmitter_lookup_reference.csv"
    ref.to_csv(ref_csv, index=False)
    # data side (optional)
    overlay_csv = None
    overlay = pd.DataFrame(
        columns=[
            "data_DP_mbar",
            "data_Flow_UIC_tph",
            "data_Flow_820_tph",
            "data_Flow_err_820_minus_UIC_tph",
        ]
    )
    if logger_csv and Path(logger_csv).exists():
        df_log = pd.read_csv(logger_csv)
        overlay = build_data_overlay(cal, df_log, dp_col=dp_col, dp_unit_hint=dp_unit_hint)
        overlay_csv = outdir / "transmitter_lookup_data.csv"
        overlay.to_csv(overlay_csv, index=False)
    # combined vertical union with source flags
    ref_blk = ref.copy()
    ref_blk["source"] = "reference"
    ref_blk["is_reference"] = True
    ref_blk["range_mbar"] = span
    ov_blk = overlay.copy()
    ov_blk["source"] = "overlay"
    ov_blk["is_reference"] = False
    ov_blk["range_mbar"] = span
    combined = pd.concat([ref_blk, ov_blk], axis=0, ignore_index=True)
    combined_csv = outdir / "transmitter_lookup_combined.csv"
    combined.to_csv(combined_csv, index=False)

    # compute operating band from overlay DP percentiles if any data
    op_band = None
    try:
        dp_vals = pd.to_numeric(overlay.get("data_DP_mbar"), errors="coerce").dropna().to_numpy()
        if dp_vals.size > 0:
            op_band = {
                "p5_mbar": float(np.percentile(dp_vals, 5)),
                "p50_mbar": float(np.percentile(dp_vals, 50)),
                "p95_mbar": float(np.percentile(dp_vals, 95)),
            }
    except Exception:
        op_band = None
    meta = {
        "season": season,
        "calibration": {
            "K_uic": float(cal.K_uic),
            "m_820": float(cal.m_820),
            "c_820": float(cal.c_820),
            "range_mbar": float(span),
            "source": cal.source,
        },
        "reference_csv": str(ref_csv),
        "overlay_csv": (str(overlay_csv) if overlay_csv else None),
        "combined_csv": str(combined_csv),
        "operating_band_mbar": op_band,
    }
    (outdir / "transmitter_lookup_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# Backwards-compatible wrapper used by run_easy.py
def compute_and_write_flow_lookup(
    csv_in: Path,
    out_json: Path,
    out_csv: Path,
    *,
    dp_col: str,
    T_col: str,
    dp_unit: str = "mbar",
    K_uic: Optional[float] = None,
    m_820: Optional[float] = None,
    c_820: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute lookup table of UIC vs 820 flow and write csv/json.

    This lightweight wrapper preserves the older ``compute_and_write_flow_lookup``
    API expected by :mod:`run_easy`.  It converts the differential pressure to
    mbar, applies calibration constants (from presets or explicit overrides),
    and writes a CSV and JSON summary.  Returns metadata including the row
    count so callers can report results.
    """

    df = pd.read_csv(csv_in)
    if dp_col not in df:
        raise ValueError(f"CSV must contain '{dp_col}' column")

    # Base calibration from presets
    cal = _calib_for_season("summer")
    # Manual overrides
    if K_uic is not None:
        cal.K_uic = float(K_uic); cal.source = "manual"
    if m_820 is not None:
        cal.m_820 = float(m_820); cal.source = "manual"
    if c_820 is not None:
        cal.c_820 = float(c_820); cal.source = "manual"
    if (
        cal.m_820 is None
        or cal.c_820 is None
        or not all(math.isfinite(v) for v in (cal.m_820, cal.c_820))
    ):
        raise ValueError("Transmitter calibration requires finite m_820 and c_820 values.")

    dp_mbar = _to_mbar(df[dp_col], dp_unit)
    flow_uic = cal.K_uic * np.sqrt(np.clip(dp_mbar.values, 0.0, None))
    flow_820 = cal.m_820 * dp_mbar.values + cal.c_820
    table = pd.DataFrame({
        "DP_mbar": dp_mbar.values,
        "Flow_UIC_tph": flow_uic,
        "Flow_820_tph": flow_820,
    })
    if np.allclose(table.drop(columns=["DP_mbar"]).values, 0.0):
        raise ValueError("Lookup table is all zeros; check calibration inputs.")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    meta = {
        "calibration": {
            "K_uic": cal.K_uic,
            "m_820": cal.m_820,
            "c_820": cal.c_820,
            "source": cal.source,
        },
        "rows": int(table.shape[0]),
        "inputs": {"csv": str(csv_in), "dp_col": dp_col, "dp_unit": dp_unit},
        "outputs": {"csv": str(out_csv), "json": str(out_json)},
    }
    out_json.write_text(json.dumps(meta, indent=2))
    return {"rows": int(table.shape[0]), "csv": str(out_csv), "json": str(out_json)}


__all__ = [
    "write_lookup_outputs",
    "compute_and_write_flow_lookup",
]
