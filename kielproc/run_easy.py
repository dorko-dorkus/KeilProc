"""Minimal one-button pipeline orchestrator for GUI use.

This module intentionally avoids any command-line coupling.  The GUI
constructs a :class:`RunConfig` instance and passes it to :func:`run_all` which
sequences the core processing stages:

``parse -> integrate -> map -> fit -> translate``.

Each stage is represented by a function imported from elsewhere in the project
and can be monkeypatched in tests.  If a stage raises an exception the caller is
expected to handle it; this module performs only light validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SitePreset:
    """Bundle geometry/instrument defaults for a measurement site.

    The fields default to benign values so a ``SitePreset`` can be constructed
    without explicitly providing dictionaries.  The GUI may opt-in to supplying
    a preset by setting :attr:`RunConfig.enable_site` and providing an instance
    via :attr:`RunConfig.site`.
    """

    name: str = "UNSPECIFIED"
    geometry: Dict[str, Any] = field(default_factory=dict)
    instruments: Dict[str, Any] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)


# Null-object used when the GUI disables site handling entirely.
NULL_SITE = SitePreset(name="__NULL__")


@dataclass
class RunConfig:
    """Inputs required to run :func:`run_all`.

    The GUI populates this dataclass and passes it directly to ``run_all``.
    ``file_glob`` and unit fields have sensible defaults so the GUI may omit
    them.  ``baro_pa`` is validated by :func:`_resolve_baro`.
    """

    input_dir: str
    output_dir: str
    file_glob: str = "*__P[1-8].csv"
    baro_pa: Optional[float] = 101_325.0
    vp_unit: str = "Pa"
    temp_unit: str = "C"
    enable_site: bool = False
    site: Optional[SitePreset] = None


def _resolve_site(cfg: RunConfig) -> SitePreset:
    """Return the active :class:`SitePreset` for ``cfg``.

    If site presets are disabled or ``cfg.site`` is ``None`` a benign
    ``NULL_SITE`` placeholder is returned.
    """

    if cfg.enable_site and cfg.site:
        return cfg.site
    return NULL_SITE


def _resolve_baro(cfg: RunConfig) -> float:
    """Return a sanitized barometric pressure in Pascals.

    Any invalid value results in a fallback to the standard 101325 Pa and a
    warning is logged.
    """

    try:
        v = float(cfg.baro_pa) if cfg.baro_pa is not None else 101_325.0
        if v <= 0:
            raise ValueError("baro must be > 0 Pa")
        return v
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Invalid baro; falling back to 101325 Pa: %s", e)
        return 101_325.0


# ---------------------------------------------------------------------------
# Placeholder stage functions
# ---------------------------------------------------------------------------

def parse(input_dir: str, *, file_glob: str = "*"):
    """Parse raw input data.

    The real implementation lives elsewhere; this placeholder allows tests to
    monkeypatch the function without pulling in heavy dependencies.
    """

    raise NotImplementedError


def integrate(parsed, *, vp_unit: str = "Pa", temp_unit: str = "C"):
    """Integrate parsed data into duct aggregates."""

    raise NotImplementedError


def map_ports(integrated):
    """Map integrated results to velocities/pressures."""

    raise NotImplementedError


def fit(mapped, *, baro_pa: float):
    """Fit calibration models."""

    raise NotImplementedError


def translate(fitres, *, output_dir: str):
    """Translate fitted results into reports/tables."""

    raise NotImplementedError


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_all(cfg: RunConfig) -> Dict[str, Any]:
    """Run the full pipeline described by ``cfg``.

    Parameters
    ----------
    cfg:
        Configuration describing all user-supplied inputs.  The function makes
        a best effort to continue even if optional pieces (such as a site
        preset or barometric pressure) are missing or invalid.

    Returns
    -------
    dict
        Mapping of stage names to their respective results along with metadata
        such as the sanitized barometric pressure and chosen site name.
    """

    logger.info(
        "Run start: input=%s output=%s glob=%s", cfg.input_dir, cfg.output_dir, cfg.file_glob
    )
    site = _resolve_site(cfg)
    baro_pa = _resolve_baro(cfg)
    logger.info("Site: %s  Baro (Pa): %.1f", site.name, baro_pa)

    parsed = parse(cfg.input_dir, file_glob=cfg.file_glob)
    integrated = integrate(parsed, vp_unit=cfg.vp_unit, temp_unit=cfg.temp_unit)
    mapped = map_ports(integrated)
    fitres = fit(mapped, baro_pa=baro_pa)
    report = translate(fitres, output_dir=cfg.output_dir)
    return {
        "parsed": parsed,
        "integrated": integrated,
        "mapped": mapped,
        "fit": fitres,
        "report": report,
        "baro_pa": baro_pa,
        "site_name": site.name,
    }


__all__ = ["SitePreset", "RunConfig", "run_all"]

