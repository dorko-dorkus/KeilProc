
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def write_summary_tables(outdir: Path, per_block: pd.DataFrame, pooled: dict | None):
    outdir.mkdir(parents=True, exist_ok=True)
    per_block.to_csv(outdir / "alpha_beta_by_block.csv", index=False)
    files = [str(outdir / "alpha_beta_by_block.csv")]
    if pooled:
        import json
        pd.DataFrame([pooled]).to_csv(outdir / "alpha_beta_pooled.csv", index=False)
        files.append(str(outdir / "alpha_beta_pooled.csv"))
        (outdir / "alpha_beta_pooled.json").write_text(json.dumps(pooled, indent=2))
        files.append(str(outdir / "alpha_beta_pooled.json"))
    return files

def plot_alignment(outdir: Path, t_s, ref, piccolo, piccolo_shifted, title="Alignment", stem="alignment"):
    outdir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9,4.5))
    plt.plot(t_s, ref, label="Mapped reference")
    plt.plot(t_s, piccolo, label="Piccolo (raw)", alpha=0.6)
    plt.plot(t_s, piccolo_shifted, label="Piccolo (aligned)", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Δp (Pa)")
    plt.legend()
    plt.title(title)
    p = outdir / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def plot_flow_map_unwrapped(outdir: Path, z_m, theta_deg, values, geom=None, cmap=None, title="Flow map (unwrapped)", stem="flowmap_unwrapped", norm_by=None):
    """
    Create an unwrapped circumferential (θ) vs axial (z) heatmap of a scalar field (e.g., Δps deviation or Δps/qt).
    Inputs:
      - z_m: 1D array of axial positions (m) for columns
      - theta_deg: 1D array of circumferential angles (deg) for rows [0..360)
      - values: 2D array shape (len(theta), len(z)) of scalar values
      - geom: optional with .L to scale the x-axis limits
      - norm_by: optional scalar to divide values by (e.g., mean qt) to make it dimensionless
    """
    import numpy as np
    z_m = np.asarray(z_m, dtype=float)
    th  = np.asarray(theta_deg, dtype=float)
    V   = np.asarray(values, dtype=float)
    if norm_by not in (None, 0):
        V = V / float(norm_by)
    outdir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10,4.2))
    extent = [z_m.min(), z_m.max(), th.min(), th.max()]
    plt.imshow(V, aspect='auto', origin='lower', extent=extent, cmap=cmap)
    plt.colorbar(label="value" + (" (normalized)" if norm_by else ""))
    plt.xlabel("Axial position z (m)")
    plt.ylabel("Circumferential angle θ (deg)")
    if geom is not None:
        plt.xlim(0, getattr(geom, "L", z_m.max()))
    plt.title(title)
    p = outdir / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)

def compute_circumferential_static_deviation(df, theta_col:str, plane_col:str, ps_col:str, ref_col:str|None=None, agg:str="median"):
    """
    From a long table with columns [theta, plane, ps, (optional) ref metric like qt or dp_vent],
    compute per-plane circumferential deviation Δps(θ,plane) = ps(θ,plane) - mean_θ ps(θ,plane).
    Returns (planes_sorted, thetas_sorted, matrix Δps).
    """
    import numpy as np
    import pandas as pd
    g = df.groupby([plane_col, theta_col])[ps_col]
    if agg == "median":
        tbl = g.median().unstack(theta_col)
    elif agg == "mean":
        tbl = g.mean().unstack(theta_col)
    elif agg == "rms":
        tbl = (g.apply(lambda s: np.sqrt(np.mean((s - s.mean())**2)))).unstack(theta_col)
    else:
        tbl = g.mean().unstack(theta_col)
    tbl = tbl.sort_index().reindex(sorted(tbl.columns), axis=1)
    dev = tbl - tbl.mean(axis=1).values[:,None]
    planes = tbl.index.to_numpy()
    thetas = tbl.columns.to_numpy()
    return planes, thetas, dev.values.T  # shape (theta, plane)


def plot_polar_slice_wall(outdir: Path, theta_deg, values, R: float = 1.0, band: tuple=(0.9, 1.0),
                          title="Wall static deviation (polar slice)", stem="polar_slice", norm_by=None):
    """
    Render a polar cross-section slice at one axial plane, assuming wall-static values around the circumference.
    Draws a colored annulus from r=band[0]*R to r=band[1]*R where color varies with θ.
    - theta_deg: 1D array of θ in degrees (monotonic, 0..360 not necessarily closed)
    - values:    1D array of same length with the scalar to map (e.g., Δps or Δps / qt)
    - R:         radius in meters for visual scaling (optional)
    - norm_by:   optional scalar to normalize values for dimensionless plot
    """
    import numpy as np
    import matplotlib.pyplot as plt
    th = np.asarray(theta_deg, dtype=float)
    v  = np.asarray(values, dtype=float)
    if norm_by not in (None, 0):
        v = v / float(norm_by)
    # Ensure 0..360 closed for smooth pcolormesh
    if th[0] != 0.0:
        th = np.insert(th, 0, 0.0)
        v  = np.insert(v, 0, v[0])
    if th[-1] != 360.0:
        th = np.append(th, 360.0)
        v  = np.append(v, v[0])
    th_rad = np.deg2rad(th)
    # Build 2D grid for annulus
    r_inner, r_outer = band
    r_vals = np.array([r_inner*R, r_outer*R])
    TH, RR = np.meshgrid(th_rad, r_vals, indexing='xy')
    # replicate values along radial dimension
    VV = np.tile(v, (2,1))
    # Convert to Cartesian for pcolormesh
    X = RR * np.cos(TH)
    Y = RR * np.sin(TH)
    outdir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5.2,5.2))
    ax = fig.add_subplot(111)
    pcm = ax.pcolormesh(X, Y, VV, shading='auto')
    fig.colorbar(pcm, ax=ax, label="value" + (" (normalized)" if norm_by else ""))
    # draw outer circle for context
    circ = plt.Circle((0,0), R, fill=False, linewidth=1.0)
    ax.add_patch(circ)
    ax.set_aspect('equal', adjustable='datalim')
    pad = 0.1*R if R>0 else 0.1
    ax.set_xlim(-R-pad, R+pad); ax.set_ylim(-R-pad, R+pad)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(title)
    p = outdir / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)
