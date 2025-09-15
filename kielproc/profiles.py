from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, json, math
from typing import Dict, Any


def _running_median(y: np.ndarray, win: int) -> np.ndarray:
    win = max(3, int(win) | 1)  # odd
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = np.median(ypad[i : i + win])
    return out


def _edge_taper(xi: np.ndarray, xi0: float = 0.07, p: float = 2.0) -> np.ndarray:
    # 0..1 taper that down-weights edges; =1 in core.
    left = np.clip(xi / xi0, 0, 1) ** p
    right = np.clip((1 - xi) / xi0, 0, 1) ** p
    return np.minimum(np.minimum(left, right), 1.0)


def build_profiles(outdir: Path, cfg) -> Dict[str, Any]:
    """Emit per-port profiles (_integrated/profiles/Port*_profile.csv) and
    compute an Aj-weighted plane q_s if Aj is available; otherwise ports-equal.
    Returns a meta dict with weighting mode and q_s means."""
    outdir = Path(outdir)
    ts_path = outdir / "normalized_timeseries.csv"
    meta = {
        "profiles_dir": str(outdir / "profiles"),
        "weighting": "ports_equal",
        "q_s_pa_mean_ports_equal": None,
        "q_s_pa_mean_Aj": None,
    }
    if not ts_path.exists():
        return meta
    ts = pd.read_csv(ts_path)
    if "Port" not in ts or "VP_pa" not in ts or "Xi" not in ts:
        return meta

    # per-port equal average q_s (for comparison)
    pe = ts.groupby("Port")["VP_pa"].median().mean()  # robust central tendency
    meta["q_s_pa_mean_ports_equal"] = float(pe) if np.isfinite(pe) else None

    prof_dir = outdir / "profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)

    have_Aj = "Aj" in ts.columns and np.isfinite(ts["Aj"]).any()
    Aj_plane_num = 0.0
    Aj_plane_den = 0.0
    bins = int(getattr(cfg, "profile_bins", 101))
    span = float(getattr(cfg, "profile_loess_span", 0.15))
    xi0 = float(getattr(cfg, "profile_edge_xi0", 0.07))
    pwr = float(getattr(cfg, "profile_edge_p", 2.0))

    for port, dfp in ts.groupby("Port"):
        # Bin Xi to a fixed grid for stability
        xi = np.clip(pd.to_numeric(dfp["Xi"], errors="coerce").to_numpy(), 0, 1)
        qs = pd.to_numeric(dfp["VP_pa"], errors="coerce").to_numpy()
        m = np.isfinite(xi) & np.isfinite(qs)
        xi = xi[m]
        qs = qs[m]
        if not xi.size:
            continue
        # Fixed centers
        centers = np.linspace(0, 1, bins)
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.searchsorted(edges, xi, side="right") - 1, 0, bins - 1)
        # Aggregate by bin (median is robust)
        n = np.bincount(idx, minlength=bins)
        med = np.full(bins, np.nan)
        for k in range(bins):
            if n[k]:
                med[k] = float(np.median(qs[idx == k]))
        # Simple robust smoothing: running median with window ~ span*bins
        mask = np.isfinite(med)
        med_filled = (
            np.interp(np.arange(bins), np.where(mask)[0], med[mask]) if mask.any() else np.zeros(bins)
        )
        win = max(3, int(max(5, span * bins)))
        sm = _running_median(med_filled, win)
        # Edge taper
        taper = _edge_taper(centers, xi0, pwr)
        # Aj handling: if not present, use 1s (ports_equal); if present on samples, average per bin
        if have_Aj:
            Aj_vals = pd.to_numeric(dfp.get("Aj"), errors="coerce").to_numpy()[m]
            Aj_bin = np.zeros(bins)
            for k in range(bins):
                sel = idx == k
                if sel.any():
                    Aj_bin[k] = float(np.nanmedian(Aj_vals[sel]))
            Aj_eff = Aj_bin
        else:
            Aj_eff = np.ones(bins)
        w_eff = Aj_eff * taper
        # Plane accumulation
        Aj_plane_num += np.nansum(w_eff * sm)
        Aj_plane_den += np.nansum(w_eff)
        # Emit profile CSV
        prof = pd.DataFrame(
            {
                "xi": centers,
                "n": n,
                "q_s_pa_med": med,
                "q_s_pa_smooth": sm,
                "taper": taper,
                "Aj": Aj_eff,
                "w_eff": w_eff,
            }
        )
        prof.to_csv(prof_dir / f"Port{port}_profile.csv", index=False)

    if Aj_plane_den > 0:
        meta["q_s_pa_mean_Aj"] = float(Aj_plane_num / Aj_plane_den)
        meta["weighting"] = "Aj_edge_taper" if have_Aj else "ports_equal_edge_taper"

    with open(prof_dir / "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta

