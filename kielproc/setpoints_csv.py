from __future__ import annotations
"""
CSV adapter for computing transmitter setpoints from a datalogger export,
using :mod:`.io` and the span fitter.
"""

from typing import Any
import numpy as np
import pandas as pd

from .io import load_logger_csv, unify_schema
from .setpoints import find_optimal_transmitter_span, OptimalSpan


def setpoints_from_logger_csv(
    path: str,
    *,
    x_col: str = "i/p",      # datalogger input column (instrument input)
    y_col: str = "820",       # mapping output column to fit
    min_fraction_of_range: float = 0.6,
    slope_sign: int = +1,
    use_unify_schema: bool = False,
    **fit_kwargs: Any,
) -> OptimalSpan:
    """Load CSV via existing loader and compute optimal span.

    Parameters
    ----------
    path : str
        Path to datalogger CSV.
    x_col, y_col : str
        Column names to fit. Override if your schema differs.
    min_fraction_of_range : float
        Minimum fraction of observed x-range the span must cover.
    slope_sign : int
        +1 for increasing y with x; -1 decreasing; 0 unconstrained.
    use_unify_schema : bool
        Normalize common column names via ``unify_schema``.
    fit_kwargs : dict
        Passed through to ``find_optimal_transmitter_span`` (e.g., setpoint_ticks).
    """

    df = load_logger_csv(path)
    if use_unify_schema:
        df = unify_schema(df)

    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    return find_optimal_transmitter_span(
        x,
        y,
        min_fraction_of_range=min_fraction_of_range,
        slope_sign=slope_sign,
        **fit_kwargs,
    )


__all__ = ["setpoints_from_logger_csv"]
