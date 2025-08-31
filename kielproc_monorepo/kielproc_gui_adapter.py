
"""
Thin adapter for Tk GUI -> kielproc library
Provides simple functions with file-path I/O to avoid heavy refactors.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from kielproc.physics import map_qs_to_qt, venturi_dp_from_qt
from kielproc.translate import compute_translation_table, apply_translation
from kielproc.lag import estimate_lag_xcorr, advance_series, first_order_lag
from kielproc.report import write_summary_tables, plot_alignment

def map_verification_plane(csv_path: Path, qs_col: str, r: float, beta: float, sampling_hz: float|None, out_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    qt = map_qs_to_qt(df[qs_col].to_numpy(float), r=r, rho_t_over_rho_s=1.0)
    dpv = venturi_dp_from_qt(qt, beta=beta)
    out = df.copy()
    out["qt"] = qt
    out["dp_vent"] = dpv
    if sampling_hz and sampling_hz > 0:
        n = len(out); out["Sample"] = np.arange(n); out["Time_s"] = out["Sample"]/float(sampling_hz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path

def fit_alpha_beta(block_specs: Dict[str, Path], ref_col: str, piccolo_col: str, lambda_ratio: float, max_lag: int, outdir: Path) -> Dict[str, str]:
    blocks = {name: pd.read_csv(path) for name, path in block_specs.items()}
    per_block, pooled = compute_translation_table(blocks, ref_key=ref_col, picc_key=piccolo_col, lambda_ratio=lambda_ratio, max_lag=max_lag)
    outdir.mkdir(parents=True, exist_ok=True)
    files = write_summary_tables(outdir, per_block, pooled)
    # Make an alignment plot for the first block if present
    if blocks and not per_block.empty:
        name0 = list(blocks.keys())[0]
        d0 = blocks[name0]
        lag0 = int(per_block.loc[per_block["block"]==name0, "lag_samples"].iloc[0])
        picc_shift = advance_series(d0[piccolo_col].to_numpy(float), lag0)
        t = d0["Time_s"] if "Time_s" in d0 else np.arange(len(d0))
        png = plot_alignment(outdir, t, d0[ref_col], d0[piccolo_col], picc_shift, title=f"Alignment {name0}", stem=f"align_{name0}")
        files.append(png)
    return {"per_block_csv": str(outdir/"alpha_beta_by_block.csv"),
            "pooled_csv": str(outdir/"alpha_beta_pooled.csv"),
            "pooled_json": str(outdir/"alpha_beta_pooled.json") if (outdir/"alpha_beta_pooled.json").exists() else "",
            "align_png": str(outdir/"align_"+list(blocks.keys())[0]+".png") if blocks else ""}

def translate_piccolo(csv_path: Path, alpha: float, beta: float, piccolo_col: str, out_col: str, out_path: Path) -> Path:
    import pandas as pd
    df = pd.read_csv(csv_path)
    out = apply_translation(df, alpha, beta, src_col=piccolo_col, out_col=out_col)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


from kielproc.geometry import DiffuserGeometry, infer_geometry_from_table
from kielproc.report import plot_flow_map_unwrapped, compute_circumferential_static_deviation

def generate_flow_map_from_csv(data_csv: Path, theta_col: str, plane_col: str, ps_col: str,
                               outdir: Path, geom_csv: Path|None=None, agg: str="median",
                               normalize_by_col: str|None=None) -> dict:
    """
    Build an unwrapped flow map (θ vs z) scaled to diffuser length if available.
    - data_csv: long-form table with columns including theta_col, plane_col (z or plane index), ps_col (Pa)
    - geom_csv: optional CSV with D1,D2,L (m or mm). If absent, only unwrapped scaling is used.
    - normalize_by_col: optional column in data_csv used to normalize Δps (e.g., qt or dp_vent)
    """
    import pandas as pd, numpy as np
    df = pd.read_csv(data_csv)
    geom = None
    if geom_csv is not None and Path(geom_csv).exists():
        g = pd.read_csv(geom_csv)
        geom = infer_geometry_from_table(g) or None
    planes, thetas, dev = compute_circumferential_static_deviation(df, theta_col, plane_col, ps_col, agg=agg)
    z = planes.astype(float)
    # If planes are plane indices, try to parse to meters if a geometry table provides L and explicit z columns exist
    # Otherwise assume planes are already z in meters
    norm_val = None
    if normalize_by_col and normalize_by_col in df.columns:
        # Use plane-mean of the normalizer to avoid overweighting noisy points
        norm_tbl = df.groupby(plane_col)[normalize_by_col].median()
        # If scalar desired, take overall median
        norm_val = float(norm_tbl.median())
    png = plot_flow_map_unwrapped(outdir, z, thetas, dev, geom=geom, title="Circumferential static deviation Δps(θ,z)", stem="flowmap_unwrapped", norm_by=norm_val)
    return {"flowmap_png": png, "geom_used": bool(geom)}


def map_from_tot_and_static(csv_path: Path, total_col: str, static_col: str,
                            r: float, beta: float, sampling_hz: float|None, out_path: Path) -> Path:
    """
    Convenience: compute qs = p_t_kiel - p_s_avg mechanically averaged line,
    then map to qt and dp_vent.
    """
    import pandas as pd, numpy as np
    df = pd.read_csv(csv_path)
    if total_col not in df.columns or static_col not in df.columns:
        raise ValueError(f"Missing required columns: {total_col}, {static_col}")
    df = df.copy()
    df["qs_verif"] = df[total_col].astype(float) - df[static_col].astype(float)
    return map_verification_plane(csv_path, "qs_verif", r, beta, sampling_hz, out_path)


from kielproc.report import plot_polar_slice_wall
from kielproc.geometry import DiffuserGeometry

def generate_polar_slice_from_csv(data_csv: Path, theta_col: str, plane_col: str, ps_col: str,
                                  plane_value: float, outdir: Path, geom_csv: Path|None=None,
                                  normalize_by_col: str|None=None, band: tuple=(0.9,1.0)) -> dict:
    """
    Build a polar wall-static slice at a given plane (z or plane index).
    - plane_value: value to select in plane_col (exact match after float casting)
    - If geometry CSV provided with D1/D2/L, use linear radius at z to set R for scale.
    - If normalize_by_col is set, divide by its plane median to get dimensionless map.
    """
    import pandas as pd, numpy as np
    df = pd.read_csv(data_csv)
    # plane selection (tolerant exact float equality by string cast and float)
    pv = float(plane_value)
    # try numeric compare first
    try:
        df_plane = df[np.isclose(df[plane_col].astype(float), pv, rtol=0, atol=1e-9)]
    except Exception:
        df_plane = df[df[plane_col].astype(str) == str(plane_value)]
    if df_plane.empty:
        raise ValueError(f"No rows match {plane_col} == {plane_value}")
    # sort by theta
    d = df_plane[[theta_col, ps_col]].copy().dropna()
    d = d.sort_values(theta_col)
    vals = d[ps_col].to_numpy(float)
    th   = d[theta_col].to_numpy(float)
    # deviation from circumferential mean
    vals = vals - np.nanmean(vals)
    # optional normalization
    norm_val = None
    if normalize_by_col and normalize_by_col in df_plane.columns:
        norm_val = float(df_plane[normalize_by_col].median())
    # radius from geometry if available
    R = 1.0
    if geom_csv is not None and Path(geom_csv).exists():
        g = pd.read_csv(geom_csv)
        geo = infer_geometry_from_table(g)
        if geo is not None:
            # If plane_col is z (meters), use as-is; else if indexed 0..N, map to z via linear spacing
            try:
                z = float(plane_value)
            except Exception:
                z = 0.0
            R = float(geo.radius_at(np.array([z]))[0])
    png = plot_polar_slice_wall(outdir, th, vals, R=R, band=band,
                                title=f"Wall static deviation at {plane_col}={plane_value}", stem="polar_slice", norm_by=norm_val)
    return {"polar_slice_png": png, "R_m": R}
