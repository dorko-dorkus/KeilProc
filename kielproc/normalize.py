from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def normalize(
    frames: Iterable[Tuple[str, pd.DataFrame]],
    baro_pa: float,
    outdir_path: Path | str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Combine and normalize per-port sample data.

    Parameters
    ----------
    frames:
        Iterable of ``(stem, DataFrame)`` pairs where each frame contains at
        least ``static_gauge_pa`` (or ``static_abs_pa``) along with other sample
        data. ``stem`` is used for bookkeeping and appears in the returned
        metadata.
    baro_pa:
        Barometric pressure in Pascals used to reconstruct absolute static when
        only a gauge column is present.

    Returns
    -------
    DataFrame
        Concatenated per-sample data with a ``p_s_pa`` column representing the
        absolute static pressure.
    dict
        Metadata describing normalization, including a simple sanity check on
        the resulting static pressure.
    """

    out_rows: list[pd.DataFrame] = []
    stems: list[str] = []
    p_abs_source_summary: dict[str, str] = {}

    for stem, df in frames:
        stems.append(stem)
        if "static_gauge_pa" in df.columns:
            # Static absolute: explicitly sum gauge + baro (legacy sheets are gauge)
            p_g = pd.to_numeric(df["static_gauge_pa"], errors="coerce")
            p_abs = p_g + baro_pa
            p_abs_source = "Static(gauge)+baro"
        elif "static_abs_pa" in df.columns:
            p_abs = pd.to_numeric(df["static_abs_pa"], errors="coerce")
            p_abs_source = "Static(abs)"
        else:
            p_abs = pd.Series([np.nan] * len(df))
            p_abs_source = "missing"
        p_abs_source_summary[stem] = p_abs_source
        row = df.copy()
        row["p_s_pa"] = p_abs
        out_rows.append(row)

    out = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()

    if outdir_path is not None and isinstance(outdir_path, (str, Path)):
        outdir_path = Path(outdir_path)
        outdir_path.mkdir(parents=True, exist_ok=True)

        # Ensure canonical column names if present in source
        rename = {
            "VP_pa": "VP_pa",
            "T_C": "T_C",
            "static_gauge_pa": "static_gauge_pa",
            "piccolo_mA": "piccolo_mA",
            "Port": "Port",
            "p_s_pa": "p_s_pa",
        }
        ts_cols = [c for c in rename if c in out.columns]
        if ts_cols:
            ts = out[ts_cols].rename(columns=rename).copy()
            ts.to_csv(outdir_path / "normalized_timeseries.csv", index=False)

            grp = ts.groupby("Port", as_index=False)
            per_port = grp.agg({
                "VP_pa": "mean",
                "T_C": "mean",
                "p_s_pa": "mean",
            }).rename(columns={
                "VP_pa": "VP_pa_mean",
                "T_C": "T_C_mean",
                "p_s_pa": "Static_abs_pa_mean",
            })
            if "piccolo_mA" in ts.columns:
                per_port["piccolo_mA_mean"] = grp["piccolo_mA"].mean()["piccolo_mA"]
            if "static_gauge_pa" in ts.columns and "Static_abs_pa_mean" not in per_port.columns:
                # Preserve gauge info for completeness if abs not present
                pass
            per_port.to_csv(outdir_path / "per_port.csv", index=False)

    # Sanity check on plane static (useful to catch parsing off-by-header)
    p_med = float(np.nanmedian(pd.to_numeric(out["p_s_pa"], errors="coerce"))) if not out.empty else float("nan")
    if np.isfinite(p_med) and not (90_000.0 <= p_med <= 140_000.0):
        # out-of-family pressure, emit a loud hint in meta
        sanity_note = (
            f"WARNING: plane static median {p_med:.1f} Pa outside [90k, 140k] — check header/units."
        )
    else:
        sanity_note = "ok"

    meta = {
        "baro_pa_used": float(baro_pa),
        "p_abs_source": p_abs_source_summary,
        "files": stems,
        "replicate_layout": "auto",
        "sanity": {"p_s_pa_median": p_med, "note": sanity_note},
    }
    return out, meta
