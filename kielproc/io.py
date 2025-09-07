
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

def load_logger_csv(path: str | Path) -> pd.DataFrame:
    """Read a datalogger CSV file.

    Parameters
    ----------
    path : str or Path
        CSV file path.

    Returns
    -------
    pandas.DataFrame
        Parsed CSV contents.
    """
    return pd.read_csv(path)

def unify_schema(df: pd.DataFrame, sampling_hz: float | None = None) -> pd.DataFrame:
    """Add convenience columns such as ``Sample`` and ``Time_s``.

    This helper is intentionally lightweight and does not enforce the
    presence of any particular data columns.
    """
    out = df.copy()
    n = len(out)
    if "Time" in out:
        t = pd.to_datetime(out["Time"], errors="coerce")
        if t.notna().sum() >= n // 2:
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
