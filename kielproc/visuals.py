from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from .aggregate import _normalize_df  # reuse existing normalizer
from .aggregate import R              # 287.05 J kg^-1 K^-1

# accepted height column aliases (case-insensitive)
_H_ALIASES = ["Height_mm", "Height_m", "Height", "Z_mm", "Z_m"]

def _pick(df: pd.DataFrame, names: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None

def render_velocity_heatmap(
    outdir: Path,
    pairs: List[Tuple[str, Path]],
    baro_cli_pa: float | None,
    height_bins: int = 50,
    clip_percentiles: tuple[float, float] = (2.0, 98.0),
    interp: str = "nearest",
) -> Path:
    """
    Render a port x normalized-height heatmap of velocity [m/s].
    - pairs: list of (PortId 'P1'..'P8', csv_path)
    - height is normalized to [0,1]; if no height column, sample index is used
    - interpolation: 'nearest' | 'linear' | 'cubic'
    Output file: outdir / 'heatmap_velocity.png'
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    port_ids = [pid for pid, _ in pairs]
    if not port_ids:
        raise ValueError("No port CSVs provided to visuals")

    grid = np.full((height_bins, len(port_ids)), np.nan, dtype=float)

    for j, (pid, pf) in enumerate(pairs):
        raw = pd.read_csv(pf)
        norm, _ = _normalize_df(raw, baro_cli_pa)
        T_K = pd.to_numeric(norm["Temperature"], errors="coerce").to_numpy(float) + 273.15
        if np.any(T_K <= 0.0):
            raise ValueError("Unphysical temperature (T_K <= 0) in visuals")
        p_s = pd.to_numeric(norm["Static_abs_Pa"], errors="coerce").to_numpy(float)
        vp  = pd.to_numeric(norm["VP"], errors="coerce").to_numpy(float).clip(min=0.0)
        rho = p_s / (R * T_K)
        v   = np.sqrt(np.maximum(vp / rho, 0.0))
        # simple height binning
        h_col = next((c for c in raw.columns if c in _H_ALIASES), None)
        h = pd.to_numeric(raw[h_col], errors="coerce").to_numpy(float) if h_col else np.linspace(0, 1, len(v))
        bins = np.linspace(np.nanmin(h), np.nanmax(h), height_bins + 1)
        idx = np.digitize(h, bins) - 1
        for b in range(height_bins):
            sel = v[idx == b]
            grid[b, j] = float(np.nanmean(sel)) if np.any(np.isfinite(sel)) else np.nan

    # percentile clip to reduce outlier influence
    finite_vals = grid[np.isfinite(grid)]
    if finite_vals.size:
        lo, hi = np.nanpercentile(finite_vals, list(clip_percentiles))
    else:
        lo, hi = (0.0, 1.0)
    clipped = np.clip(grid, lo, hi)

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        clipped,
        aspect="auto",
        origin="lower",
        interpolation=interp,
    )
    ax.set_xlabel("Port")
    ax.set_xticks(range(len(port_ids)))
    ax.set_xticklabels(port_ids)
    ax.set_ylabel("Normalized height bin")
    cbar = fig.colorbar(im, ax=ax, label="v [m/s]")
    ax.set_title("Velocity heatmap")
    fig.tight_layout()
    out = outdir / "heatmap_velocity.png"
    fig.savefig(out, dpi=150, format="png")
    plt.close(fig)
    return out
