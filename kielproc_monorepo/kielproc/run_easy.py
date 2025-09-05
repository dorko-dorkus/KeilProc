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
        from tools.legacy_parser.legacy_parser.parser import parse_legacy_workbook

        src = self.run.src
        ports_dir.mkdir(parents=True, exist_ok=True)

        try:
            if src.is_dir():
                for wb in sorted(p for p in src.iterdir() if p.suffix.lower() in {".xlsx", ".xls"}):
                    parse_legacy_workbook(wb, out_dir=ports_dir, return_mode="files")
            else:
                parse_legacy_workbook(src, out_dir=ports_dir, return_mode="files")
            # record artifacts: csvs and summary jsons
            for p in ports_dir.glob("*.csv"):
                self.artifacts[p.name] = p
            for j in ports_dir.glob("*__parse_summary.json"):
                self.artifacts[j.name] = j
        except Exception as e:
            self.summary["errors"].append(f"parse: {e}")
            raise

    def integrate(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Integrate per-port files into duct aggregates."""
        from .aggregate import integrate_run, RunConfig
        import math

        ports_dir = base_dir / "ports_csv"

        geom = self.run.site.geometry or {}
        h = geom.get("duct_height_m")
        w = geom.get("duct_width_m")
        if h is None or w is None:
            d = geom.get("duct_diameter_m")
            if d:
                A = math.pi * (float(d) ** 2) / 4.0
                s = math.sqrt(A)
                h = w = s
        if h is None or w is None:
            h = w = 1.0  # fallback to keep pipeline moving

        cfg = RunConfig(height_m=float(h), width_m=float(w), weights=geom.get("weights"))

        res = integrate_run(
            ports_dir,
            cfg,
            baro_cli_pa=self.run.baro_override_Pa,
        )

        outdir = base_dir / "_integrated"
        outdir.mkdir(parents=True, exist_ok=True)

        per_csv = outdir / "per_port.csv"
        res["per_port"].to_csv(per_csv, index=False)
        duct_json = outdir / "duct_result.json"
        duct_json.write_text(json.dumps(res.get("duct", {}), indent=2))
        norm_json = outdir / "normalize_meta.json"
        norm_json.write_text(json.dumps(res.get("normalize_meta", {}), indent=2))

        ref_block = {
            "block_name": base_dir.name,
            "run_dir": str(base_dir),
            "outdir": str(outdir),
            "per_port_csv": str(per_csv),
            "duct_result_json": str(duct_json),
            "q_s_pa": res.get("duct", {}).get("q_s_pa"),
            "q_t_pa": res.get("duct", {}).get("q_t_pa"),
            "delta_p_vent_est_pa": res.get("duct", {}).get("delta_p_vent_est_pa"),
        }
        ref_json = outdir / "reference_block.json"
        ref_json.write_text(json.dumps(ref_block, indent=2))

        self.artifacts["per_port.csv"] = per_csv
        self.artifacts["duct_result.json"] = duct_json
        self.artifacts["normalize_meta.json"] = norm_json
        self.artifacts["reference_block.json"] = ref_json

        self._pairs = res.get("pairs", [])

    def map(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Build velocity maps from integrated data."""
        from kielproc_gui_adapter import process_legacy_parsed_csv
        from kielproc.visuals import render_velocity_heatmap
        from .aggregate import _port_id_from_stem
        from .geometry import Geometry

        ports_dir = base_dir / "ports_csv"
        mapped_dir = base_dir / "_mapped"
        mapped_dir.mkdir(parents=True, exist_ok=True)

        geom_dict = self.run.site.geometry or {}
        try:
            geom = Geometry(**geom_dict)
        except Exception:
            geom = None

        self._mapped_csvs: list[Path] = []
        pairs = []
        for csv in sorted(ports_dir.glob("*.csv")):
            try:
                pid = _port_id_from_stem(csv.stem) or csv.stem
            except Exception:
                pid = csv.stem
            pairs.append((pid, csv))
            if geom is not None:
                out_path = mapped_dir / f"{csv.stem}_mapped.csv"
                try:
                    process_legacy_parsed_csv(csv, geom, None, out_path)
                    self._mapped_csvs.append(out_path)
                    self.artifacts[out_path.name] = out_path
                except Exception as e:
                    self.summary["warnings"].append(f"map {csv.name}: {e}")

        try:
            png = render_velocity_heatmap(mapped_dir, pairs, self.run.baro_override_Pa)
            self.artifacts[png.name] = png
        except Exception as e:
            self.summary["warnings"].append(f"heatmap: {e}")

        self._pairs = pairs

    def fit(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Fit calibration models."""
        from kielproc_gui_adapter import fit_alpha_beta

        mapped = getattr(self, "_mapped_csvs", [])
        if not mapped:
            self.summary["warnings"].append("fit: no mapped CSVs available")
            return

        block_specs = {Path(p).stem: p for p in mapped}
        outdir = base_dir / "_fit"
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            res = fit_alpha_beta(
                block_specs,
                ref_col="deltpVent",
                piccolo_col="Piccolo",
                lambda_ratio=1.0,
                max_lag=300,
                outdir=outdir,
            )
            for k in ("per_block_csv", "per_block_json", "pooled_csv", "pooled_json", "align_png"):
                v = res.get(k)
                if v:
                    p = Path(v)
                    self.artifacts[p.name] = p
            if res.get("blocks_info"):
                b0 = res["blocks_info"][0]
                self._alpha_beta = {"alpha": b0.get("alpha"), "beta": b0.get("beta")}
        except Exception as e:
            self.summary["warnings"].append(f"fit: {e}")

    def translate(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Generate control-system lookup tables."""
        from kielproc_gui_adapter import translate_piccolo

        if not getattr(self, "_alpha_beta", None) or not getattr(self, "_mapped_csvs", None):
            self.summary["warnings"].append("translate: missing fit results or mapped data")
            return

        alpha = self._alpha_beta.get("alpha")
        beta = self._alpha_beta.get("beta")
        if alpha is None or beta is None:
            self.summary["warnings"].append("translate: alpha/beta unavailable")
            return

        src_csv = self._mapped_csvs[0]
        outdir = base_dir / "_translated"
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / f"{Path(src_csv).stem}_translated.csv"
        try:
            translate_piccolo(src_csv, alpha, beta, "Piccolo", "Piccolo_translated", out_path)
            self.artifacts[out_path.name] = out_path
            self._translated_csv = out_path
        except Exception as e:
            self.summary["warnings"].append(f"translate: {e}")

    def report(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Emit consolidated HTML/PDF reports."""
        from kielproc_gui_adapter import legacy_results_from_csv
        from kielproc.legacy_results import ResultsConfig
        import math

        csv = getattr(self, "_translated_csv", None)
        if not csv:
            self.summary["warnings"].append("report: no translated CSV available")
            return

        geom = self.run.site.geometry or {}
        h = geom.get("duct_height_m")
        w = geom.get("duct_width_m")
        if h is None or w is None:
            d = geom.get("duct_diameter_m")
            if d:
                A = math.pi * (float(d) ** 2) / 4.0
                s = math.sqrt(A)
                h = w = s

        cfg = ResultsConfig(duct_height_m=h, duct_width_m=w)
        outdir = base_dir / "_report"
        outdir.mkdir(parents=True, exist_ok=True)
        csv_out = outdir / "legacy_results.csv"
        try:
            res = legacy_results_from_csv(csv, cfg, csv_out)
            self.artifacts["legacy_results.csv"] = csv_out
            json_out = outdir / "legacy_results.json"
            json_out.write_text(json.dumps(res, indent=2))
            self.artifacts["legacy_results.json"] = json_out
        except Exception as e:
            self.summary["warnings"].append(f"report: {e}")

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
