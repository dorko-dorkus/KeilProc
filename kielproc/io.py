
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

LEGACY_TIME_COLS = ["Time", "Timestamp", "t", "time"]

def load_legacy_excel(path: Path, sheet=None) -> dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(path)
    frames: dict[str, pd.DataFrame] = {}
    for nm in xls.sheet_names if sheet is None else [sheet]:
        try:
            frames[nm] = xls.parse(nm)
        except Exception:
            # Legacy workbooks occasionally contain charts or other objects
            # that ``pandas`` cannot parse into a ``DataFrame``.  These are
            # not fatal for the calling code, so simply skip over them rather
            # than swallowing the exception with ``pass`` which obscures the
            # control flow.
            continue
    return frames

def load_logger_csv(path: Path, sp_col: str, vp_col: str, time_col: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {sp_col: "SP", vp_col: "VP"}
    if time_col and time_col in df.columns:
        cols[time_col] = "Time"
    out = df.rename(columns=cols)[list(cols.values())].copy()
    return out

def unify_schema(df: pd.DataFrame, sampling_hz: float | None) -> pd.DataFrame:
    out = df.copy()
    if "SP" not in out or "VP" not in out:
        raise ValueError("Expect SP and VP columns after mapping.")
    n = len(out)
    if "Time" in out:
        t = pd.to_datetime(out["Time"], errors="coerce")
        if t.notna().sum() >= n//2:
            t0 = t.dropna().iloc[0]
            out["Time_s"] = (t - t0).dt.total_seconds()
        else:
            out["Time_s"] = np.nan
    else:
        out["Time_s"] = np.nan
    out["Sample"] = np.arange(n, dtype=int)
    if sampling_hz and sampling_hz > 0:
        out["Time_s"] = out["Sample"] / float(sampling_hz)
    return out
