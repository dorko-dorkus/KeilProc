from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal
import math
import pandas as pd
import numpy as np

R_SPECIFIC_AIR = 287.05  # J/(kg·K)

@dataclass
class ResultsConfig:
    """Configuration for computing legacy-style results fields."""
    temp_col: str = "Temperature"          # °C
    vp_col: str = "VP"                     # dynamic pressure (Pa)
    static_col: Optional[str] = None       # absolute static pressure [Pa]
    piccolo_col: str = "Piccolo"           # 4–20 mA unless piccolo_units != 'mA'
    piccolo_units: Literal["mA", "mbar", "Pa"] = "mA"
    piccolo_range_mbar: float = 6.7        # transmitter range setting
    area_m2: Optional[float] = None        # duct plane area; if None, use height*width
    duct_height_m: Optional[float] = None  # used only if area_m2 is None
    duct_width_m: Optional[float] = None   # used only if area_m2 is None
    default_ps_pa: float = 101_325.0       # fallback absolute static pressure
    R: float = R_SPECIFIC_AIR              # specific gas constant for air

def _resolve_area(cfg: ResultsConfig) -> float:
    if cfg.area_m2 is not None:
        return float(cfg.area_m2)
    if cfg.duct_height_m and cfg.duct_width_m:
        return float(cfg.duct_height_m * cfg.duct_width_m)
    raise ValueError("Provide area_m2, or both duct_height_m and duct_width_m.")

def _piccolo_to_mbar(mean_val: float, units: str, rng_mbar: float) -> float:
    if units == "mA":
        return (mean_val - 4.0) / 16.0 * float(rng_mbar)
    if units == "mbar":
        return float(mean_val)
    if units == "Pa":
        return float(mean_val) / 100.0
    raise ValueError(f"Unsupported piccolo_units: {units}")

def compute_results(csv_path: Path | str, cfg: ResultsConfig) -> dict:
    """Compute legacy-style results fields from a raw logger CSV."""
    df = pd.read_csv(csv_path)
    A = _resolve_area(cfg)

    # Temperature (°C) and Kelvin for density calc
    tC = pd.to_numeric(df.get(cfg.temp_col, pd.Series(dtype=float)), errors="coerce")
    T_K = tC + 273.15
    T_mean_K = float(np.nanmean(T_K)) if T_K.notna().any() else 293.15
    tC_mean = float(np.nanmean(tC)) if tC.notna().any() else float("nan")

    # Density: prefer measured absolute static if provided, else default p_s
    ps_mean = cfg.default_ps_pa
    if cfg.static_col and cfg.static_col in df.columns:
        ps_series = pd.to_numeric(df[cfg.static_col], errors="coerce")
        ps_mean = float(np.nanmean(ps_series)) if ps_series.notna().any() else cfg.default_ps_pa
    rho = ps_mean / (cfg.R * T_mean_K)

    # Dynamic pressure & derived velocity
    vp = pd.to_numeric(df.get(cfg.vp_col, pd.Series(dtype=float)), errors="coerce")
    vp_mean = float(np.nanmean(vp)) if vp.notna().any() else float("nan")
    vp_std = float(np.nanstd(vp)) if vp.notna().any() else float("nan")
    v = math.sqrt(2 * vp_mean / rho) if vp_mean > 0 and rho > 0 else float("nan")
    q = v * A
    m_dot = q * rho

    # Piccolo translation
    piccolo_vals = pd.to_numeric(df.get(cfg.piccolo_col, pd.Series(dtype=float)), errors="coerce")
    piccolo_mean = float(np.nanmean(piccolo_vals)) if piccolo_vals.notna().any() else float("nan")
    piccolo_mbar = _piccolo_to_mbar(piccolo_mean, cfg.piccolo_units, cfg.piccolo_range_mbar)
    piccolo_pa = piccolo_mbar * 100.0

    return {
        "n_samples": int(len(df)),
        "temp_C_mean": tC_mean,
        "ps_pa": ps_mean,
        "rho_kg_m3": rho,
        "vp_pa_mean": vp_mean,
        "vp_pa_std": vp_std,
        "velocity_m_s": v,
        "volume_m3_s": q,
        "mass_kg_s": m_dot,
        "duct_height_m": float(cfg.duct_height_m) if cfg.duct_height_m is not None else None,
        "duct_width_m": float(cfg.duct_width_m) if cfg.duct_width_m is not None else None,
        "area_m2": A,
        "piccolo_mean": piccolo_mean,
        "piccolo_mbar": piccolo_mbar,
        "piccolo_pa": piccolo_pa,
    }
