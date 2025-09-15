from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import re


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

        # after building per-sample ts_all
        ts_all = out
        # Robust column discovery (handles label variants and mV→mA, plus Xi/Aj if present)
        cols = {"p_s_pa": "p_s_pa"}
        for c in ts_all.columns:
            lc = c.lower()
            if ("vp" in lc) and ("pa" in lc):
                cols["VP_pa"] = c
            if ("piccolo" in lc) and ("ma" in lc or "current" in lc):
                cols["piccolo_mA"] = c
            if ("piccolo" in lc) and ("mv" in lc):
                cols["piccolo_mV"] = c
            if lc in ("t_c", "temp_c", "temperature_c"):
                cols["T_C"] = c
            if ("static" in lc) and (("gauge" in lc) or ("sp_g" in lc) or ("g" == lc[-1:])):
                cols["static_gauge_pa"] = c
            if lc == "port":
                cols["Port"] = c
            # optional station index / area weight from legacy sheets (if parser exposed them)
            if re.search(r'(^|\b)(xi|ξ|station(_?index)?|grid(_?index)?)($|\b)', lc):
                cols["Xi"] = c
            if re.search(r'(^|\b)(aj|a_j|area_?weight)(|\b)', lc):
                cols["Aj"] = c
        ts = ts_all[[v for v in cols.values()]].rename(columns={v: k for k, v in cols.items()})
        # Convert mV→mA if needed (default 250 Ω)
        if "piccolo_mV" in ts.columns and "piccolo_mA" not in ts.columns:
            R = float(getattr(globals().get("cfg", object()), "piccolo_shunt_ohm", 250.0))
            ts["piccolo_mA"] = ts["piccolo_mV"] / R
        #
        # Ensure we have a station index Xi in [0,1]: if absent, synthesize by
        # normalized rank over time/sample order *within each port*.
        #
        if "Xi" not in ts.columns:
            # best effort ordering key: if original ts_all had a time-like column, use it
            time_col = None
            for cand in ts_all.columns:
                if re.search(r'(?i)time|timestamp|sample', str(cand)):
                    time_col = cand
                    break
            ts["_order_key"] = pd.to_numeric(ts_all.get(time_col, pd.Series(range(len(ts_all)))), errors="coerce")
            ts["_order_key"] = ts["_order_key"].fillna(method="ffill").fillna(method="bfill")
            ts["Xi"] = (
                ts.groupby("Port")["_order_key"]
                .rank(method="first", pct=True)
                .clip(0.0, 1.0)
            )
            ts.drop(columns=["_order_key"], errors="ignore", inplace=True)

        ts.to_csv(outdir_path / "normalized_timeseries.csv", index=False)

        per_port = ts.groupby("Port", as_index=False).agg({
            "VP_pa": "mean",
            "T_C": "mean",
            "p_s_pa": "mean",
        }).rename(columns={
            "VP_pa": "VP_pa_mean",
            "T_C": "T_C_mean",
            "p_s_pa": "Static_abs_pa_mean",
        })
        if "piccolo_mA" in ts.columns:
            per_port["piccolo_mA_mean"] = ts.groupby("Port")["piccolo_mA"].mean().to_numpy()
        # For visibility, if Aj was present on samples, compute a per-port Aj sum (diagnostic only)
        if "Aj" in ts.columns:
            s = ts.groupby("Port")["Aj"].sum().rename("Aj_sum")
            per_port = per_port.merge(s, on="Port", how="left")
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
