from __future__ import annotations

import pandas as pd

from .legacy_xlsx import _parse_port_sheet_with_replicates


def ingest_port_sheet(xls_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Returns a normalized long-form per-sample dataframe with:
      time, static_gauge_pa, VP_pa, T_C, piccolo_mA, replicate
    Works for both simple (single-block) and replicate-layout sheets.
    """
    df = pd.read_excel(xls_path, sheet_name=sheet_name, header=None)
    out = _parse_port_sheet_with_replicates(df)
    if out.empty:
        raise ValueError(
            f"Could not parse expected columns on sheet {sheet_name} (Time/Static/VP/Temp)"
        )
    return out
