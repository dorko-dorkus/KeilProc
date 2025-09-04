from __future__ import annotations

"""Lightweight façade for the "Run Easy" one-click processor.

This module defines a small orchestrator that sequences the legacy
processing pipeline steps.  It intentionally avoids pulling in the heavy
implementation details from the monorepo; the individual step methods are
thin placeholders which will be wired up to the real logic in future
iterations.  For now they allow the control-flow and error-handling to be
exercised in isolation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
import json
import time

NZT = "Pacific/Auckland"


@dataclass
class SitePreset:
    """Bundle geometry/instrument defaults for a measurement site."""

    name: str
    geometry: dict
    instruments: dict
    defaults: dict


@dataclass
class RunInputs:
    """Inputs required to run the one-click pipeline."""

    src: Path
    site: SitePreset
    baro_override_Pa: Optional[float] = None
    run_stamp: Optional[str] = None  # YYYYMMDD_HHMM in NZT


class OneClickError(Exception):
    """Raised when a hard failure occurs during the one-click run."""


class Orchestrator:
    """Coordinate parse → integrate → map → fit → translate → report."""

    def __init__(self, run: RunInputs):
        self.run = run
        self.artifacts: Dict[str, Path] = {}
        self.summary: Dict = {"warnings": [], "errors": []}

    # --- helpers -----------------------------------------------------
    def _stamp(self) -> str:
        if self.run.run_stamp:
            return self.run.run_stamp
        # Assume system clock already set to NZT to avoid tz deps.
        return time.strftime("%Y%m%d_%H%M")

    def _mkdirs(self, base: Path) -> Dict[str, Path]:
        base.mkdir(parents=True, exist_ok=True)
        ports = base / "ports_csv"
        ports.mkdir(exist_ok=True)
        return {"base": base, "ports": ports}

    # --- pipeline stages ---------------------------------------------
    def preflight(self) -> None:
        src = self.run.src
        if not src.exists():
            raise OneClickError(f"Input not found: {src}")
        if src.is_dir():
            wb = [p for p in src.iterdir() if p.suffix.lower() in {".xlsx", ".xls"}]
            if not wb:
                raise OneClickError("No legacy workbooks found in directory.")
        else:
            if src.suffix.lower() not in {".xlsx", ".xls"}:
                raise OneClickError("Legacy workbook must be .xlsx/.xls")
        if self.run.baro_override_Pa is not None and self.run.baro_override_Pa <= 0:
            raise OneClickError("Barometric override must be > 0 Pa")

    def parse(self, ports_dir: Path) -> None:  # pragma: no cover - placeholder
        """Parse legacy workbooks to per-port CSVs."""
        pass

    def integrate(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Integrate per-port files into duct aggregates."""
        pass

    def map(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Build velocity maps from integrated data."""
        pass

    def fit(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Fit calibration models."""
        pass

    def translate(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Generate control-system lookup tables."""
        pass

    def report(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Emit consolidated HTML/PDF reports."""
        pass

    # --- public ------------------------------------------------------
    def run_all(self) -> Path:
        """Execute the full one-click pipeline and return the run directory."""
        self.preflight()
        stamp = self._stamp()
        out = Path(f"RUN_{stamp}")
        dirs = self._mkdirs(out)
        # persist run context for audit
        (out / "run_context.json").write_text(
            json.dumps(
                {
                    "site": self.run.site.name,
                    "baro_override_Pa": self.run.baro_override_Pa,
                    "stamp": stamp,
                    "timezone": NZT,
                },
                indent=2,
            )
        )
        self.parse(dirs["ports"])
        self.integrate(out)
        self.map(out)
        self.fit(out)
        self.translate(out)
        self.report(out)
        return out


# Convenience entry point -----------------------------------------------------

def run_easy_legacy(
    src: Path,
    site: SitePreset,
    baro_override_Pa: Optional[float] = None,
    run_stamp: Optional[str] = None,
) -> Path:
    """Run the full pipeline for a legacy workbook using ``SitePreset`` defaults."""

    run = RunInputs(src=src, site=site, baro_override_Pa=baro_override_Pa, run_stamp=run_stamp)
    return Orchestrator(run).run_all()
