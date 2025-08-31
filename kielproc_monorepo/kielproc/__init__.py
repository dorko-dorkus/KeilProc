
"""
kielproc - Kiel + wall-static baseline processor & legacy piccolo translation.
"""
from .physics import map_qs_to_qt, venturi_dp_from_qt, rho_from_pT
from .lag import estimate_lag_xcorr, shift_series
from .deming import deming_fit
from .pooling import pool_alpha_beta_random_effects
from .io import load_legacy_excel, load_logger_csv, unify_schema
from .translate import compute_translation_table, apply_translation
from .report import write_summary_tables, plot_alignment

__all__ = [
    "map_qs_to_qt", "venturi_dp_from_qt", "rho_from_pT",
    "estimate_lag_xcorr", "shift_series",
    "deming_fit",
    "pool_alpha_beta_random_effects",
    "load_legacy_excel", "load_logger_csv", "unify_schema",
    "compute_translation_table", "apply_translation",
    "write_summary_tables", "plot_alignment",
]
