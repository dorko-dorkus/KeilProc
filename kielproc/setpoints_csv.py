from __future__ import annotations
from pathlib import Path
import pandas as pd

from .io import load_logger_csv, unify_schema
from .setpoints import find_optimal_transmitter_span


def setpoints_from_logger_csv(path: str | Path, x_col: str = "Piccolo", y_col: str = "Reference", *, use_unify_schema: bool = True):
    df = load_logger_csv(path)
    if use_unify_schema:
        df = unify_schema(df)
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    return find_optimal_transmitter_span(x, y)


__all__ = ["setpoints_from_logger_csv"]
