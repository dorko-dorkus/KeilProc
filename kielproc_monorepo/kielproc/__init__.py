
"""
kielproc - Kiel + wall-static baseline processor & legacy piccolo translation.
"""

__version__ = "0.1.0"

from .physics import map_qs_to_qt, venturi_dp_from_qt, rho_from_pT
from .lag import estimate_lag_xcorr, shift_series, advance_series, delay_series
from .deming import deming_fit
from .pooling import pool_alpha_beta_random_effects, pool_alpha_beta_gls
from .setpoints import find_optimal_transmitter_span, OptimalSpan
from .setpoints_csv import setpoints_from_logger_csv
from .io import load_legacy_excel, load_logger_csv, unify_schema
from .translate import compute_translation_table, apply_translation
from .report import write_summary_tables, plot_alignment
from .qa import qa_indices
from .aggregate import RunConfig, integrate_run
from .geometry import (
    Geometry,
    plane_area,
    effective_upstream_area,
    throat_area,
    r_ratio,
    beta_from_geometry,
    geometry_summary,
)
from .legacy_results import ResultsConfig, compute_results as compute_legacy_results

__all__ = [
    "__version__",
    "map_qs_to_qt", "venturi_dp_from_qt", "rho_from_pT",
    "estimate_lag_xcorr", "shift_series", "advance_series", "delay_series",
    "deming_fit",
    "pool_alpha_beta_random_effects", "pool_alpha_beta_gls",
    "find_optimal_transmitter_span", "OptimalSpan", "setpoints_from_logger_csv",
    "load_legacy_excel", "load_logger_csv", "unify_schema",
    "compute_translation_table", "apply_translation",
    "write_summary_tables", "plot_alignment", "qa_indices",
    "Geometry", "plane_area", "effective_upstream_area", "throat_area", "r_ratio", "beta_from_geometry", "geometry_summary",
    "RunConfig", "integrate_run",
    "ResultsConfig", "compute_legacy_results",
]
