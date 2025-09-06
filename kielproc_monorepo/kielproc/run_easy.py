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
from typing import Optional, Dict, List, Callable
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
    output_base: Optional[Path] = None  # If set, RUN_* will be created here


class OneClickError(Exception):
    """Raised when a hard failure occurs during the one-click run."""


class Orchestrator:
    """Coordinate parse → integrate → map → fit → translate → report."""

    def __init__(self, run: RunInputs, progress_cb: Optional[Callable[[str], None]] = None):
        self.run = run
        self.artifacts: list[Path] = []
        self.summary: Dict = {"warnings": [], "errors": []}
        self._progress_cb = progress_cb
        # strict mode: escalate critical-path issues to hard failures
        # enabled via RunInputs defaults (see run_easy_legacy signature)
        self.strict: bool = getattr(run, "strict", False)

    def _progress(self, msg: str) -> None:
        if self._progress_cb:
            self._progress_cb(msg)

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
                self.artifacts.append(p)
            for j in ports_dir.glob("*__parse_summary.json"):
                self.artifacts.append(j)
        except Exception as e:
            self.summary["errors"].append(f"parse: {e}")
            if self.strict:
                raise

    def integrate(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Integrate per-port files into duct aggregates."""
        from .aggregate import integrate_run, RunConfig
        import math
        from .geometry import Geometry, r_ratio, beta_from_geometry
        from dataclasses import fields as dataclass_fields

        ports_dir = base_dir / "ports_csv"

        geom_dict = self.run.site.geometry or {}
        h = geom_dict.get("duct_height_m")
        w = geom_dict.get("duct_width_m")
        if h is None or w is None:
            d = geom_dict.get("duct_diameter_m")
            if d:
                A = math.pi * (float(d) ** 2) / 4.0
                s = math.sqrt(A)
                h = w = s
        if h is None or w is None:
            h = w = 1.0  # fallback to keep pipeline moving

        cfg = RunConfig(height_m=float(h), width_m=float(w), weights=geom_dict.get("weights"))

        valid = {f.name for f in dataclass_fields(Geometry)}
        geom = None
        try:
            geom = Geometry(**{k: v for k, v in geom_dict.items() if k in valid})
        except Exception:
            geom = None

        area_ratio = beta = None
        if geom is not None:
            try:
                area_ratio = r_ratio(geom)
                beta = beta_from_geometry(geom)
            except Exception:
                pass

        res = integrate_run(
            ports_dir,
            cfg,
            baro_cli_pa=self.run.baro_override_Pa,
            area_ratio=area_ratio,
            beta=beta,
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

        self.artifacts.append(per_csv)
        self.artifacts.append(duct_json)
        self.artifacts.append(norm_json)
        self.artifacts.append(ref_json)

        self._pairs = res.get("pairs", [])

    
    def map(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Build velocity maps from integrated data."""
        from kielproc_gui_adapter import process_legacy_parsed_csv
        from kielproc.physics import map_qs_to_qt
        from kielproc.visuals import render_velocity_heatmap
        from .aggregate import _discover_pairs
        from .geometry import Geometry, r_ratio

        ports_dir = base_dir / "ports_csv"
        mapped_dir = base_dir / "_mapped"
        mapped_dir.mkdir(parents=True, exist_ok=True)

        geom_dict = self.run.site.geometry or {}
        # Filter out any unexpected geometry fields so presets with extra
        # keys (e.g. ``duct_diameter_m``) don't cause ``Geometry``
        # initialization to fail.
        from dataclasses import fields as dataclass_fields

        valid_keys = {f.name for f in dataclass_fields(Geometry)}
        filtered = {k: v for k, v in geom_dict.items() if k in valid_keys}

        try:
            geom = Geometry(**filtered)
        except Exception:
            geom = None

        self._mapped_csvs: list[Path] = []
        pairs: list[tuple[str, Path]] = []

        if geom is None:
            self.summary["warnings"].append("map: geometry unavailable; skipping mapping")
            self._pairs = pairs
            return

        r = r_ratio(geom)
        if r is None:
            self.summary["warnings"].append(
                "map: geometry missing port area ratio; skipping mapping"
            )
            self._pairs = pairs
            return

        pairs, skipped = _discover_pairs(ports_dir, "*.csv")
        for name, reason in skipped:
            self.summary["warnings"].append(f"map {name}: {reason}")
        if not pairs:
            msg = "map: no port CSVs available"
            if self.strict:
                raise OneClickError(msg)
            self.summary["warnings"].append(msg)
            self._pairs = []
            return

        for pid, csv in pairs:
            out_path = mapped_dir / f"{csv.stem}_mapped.csv"
            try:
                # Full mapping (qp + deltpVent) if Geometry has a throat
                process_legacy_parsed_csv(csv, geom, None, out_path)  # qs_col defaults to "VP"
                self._mapped_csvs.append(out_path)
                self.artifacts.append(out_path)
            except Exception as e:
                # Graceful fallback: produce qp only (no venturi Δp) if beta cannot be derived
                try:
                    import pandas as pd
                    df = pd.read_csv(csv)
                    if "VP" not in df.columns:
                        raise RuntimeError("no VP column after normalization")
                    qs = pd.to_numeric(df["VP"], errors="coerce").to_numpy(float)
                    qp = map_qs_to_qt(qs, r=r, rho_t_over_rho_s=1.0)
                    out = df.copy()
                    out["qp"] = qp
                    out.to_csv(out_path, index=False)
                    self._mapped_csvs.append(out_path)
                    self.artifacts.append(out_path)
                    self.summary["warnings"].append(
                        f"map {csv.name}: throat missing; wrote qp only (no deltpVent)."
                    )
                except Exception as e2:
                    if self.strict:
                        raise OneClickError(f"map: {csv.name}: {e2}")
                    self.summary["warnings"].append(f"map: {csv.name}: {e2}")

        try:
            png = render_velocity_heatmap(mapped_dir, pairs, self.run.baro_override_Pa)
            self.artifacts.append(png)
        except Exception as e:
            self.summary["warnings"].append(f"heatmap: {e}")

        self._pairs = pairs

        if not getattr(self, "_mapped_csvs", None):
            msg = "map: no mapped results produced"
            if self.strict:
                raise OneClickError(msg)
            self.summary["warnings"].append(msg)


    def fit(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Fit calibration models."""
        from kielproc_gui_adapter import fit_alpha_beta

        mapped = getattr(self, "_mapped_csvs", [])
        if not mapped:
            msg = "fit: no mapped CSVs available"
            if self.strict:
                raise OneClickError(msg)
            self.summary["warnings"].append(msg)
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
                    self.artifacts.append(p)
            if res.get("blocks_info"):
                b0 = res["blocks_info"][0]
                self._alpha_beta = {"alpha": b0.get("alpha"), "beta": b0.get("beta")}
        except Exception as e:
            if self.strict:
                raise OneClickError(f"fit: {e}")
            self.summary["warnings"].append(f"fit: {e}")

    def translate(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Generate control-system lookup tables."""
        from kielproc_gui_adapter import translate_piccolo

        if not getattr(self, "_alpha_beta", None) or not getattr(self, "_mapped_csvs", None):
            msg = "translate: missing fit results or mapped data"
            if self.strict:
                raise OneClickError(msg)
            self.summary["warnings"].append(msg)
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
            self.artifacts.append(out_path)
            self._translated_csv = out_path
        except Exception as e:
            if self.strict:
                raise OneClickError(f"translate: {e}")
            self.summary["warnings"].append(f"translate: {e}")

    def report(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Emit consolidated HTML/PDF reports."""
        from kielproc_gui_adapter import legacy_results_from_csv
        from kielproc.legacy_results import ResultsConfig
        import math

        csv = getattr(self, "_translated_csv", None)
        if not csv:
            msg = "report: no translated CSV available"
            if self.strict:
                raise OneClickError(msg)
            self.summary["warnings"].append(msg)
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
            self.artifacts.append(csv_out)
            json_out = outdir / "legacy_results.json"
            json_out.write_text(json.dumps(res, indent=2))
            self.artifacts.append(json_out)
        except Exception as e:
            if self.strict:
                raise OneClickError(f"report: {e}")
            self.summary["warnings"].append(f"report: {e}")

    # --- public ------------------------------------------------------
    def run_all(self) -> Path:
        """Execute the full one-click pipeline and return the run directory."""
        self._progress("Preflight…")
        self.preflight()
        stamp = self._stamp()
        base = (self.run.output_base or Path.cwd())
        out = base / f"RUN_{stamp}"
        dirs = self._mkdirs(out)
        # persist run context for audit
        context_path = out / "run_context.json"
        context_path.write_text(
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
        self.artifacts.append(context_path)

        self._progress("Parsing…")
        self.parse(dirs["ports"])
        self._progress("Integrating…")
        self.integrate(out)
        self._progress("Mapping…")
        self.map(out)
        self._progress("Fitting…")
        self.fit(out)
        self._progress("Translating…")
        self.translate(out)
        self._progress("Reporting…")
        self.report(out)

        # build manifest of outputs
        tables = [str(p) for p in self.artifacts if p.suffix in {".csv", ".json"}]
        plots = [str(p) for p in self.artifacts if p.suffix in {".png", ".pdf", ".svg"}]
        key_vals: Dict[str, float] = {}
        ref = next((p for p in self.artifacts if p.name == "reference_block.json"), None)
        if ref and ref.exists():
            try:
                ref_data = json.loads(ref.read_text())
                for k in ("q_s_pa", "q_t_pa", "delta_p_vent_est_pa"):
                    v = ref_data.get(k)
                    if v is not None:
                        key_vals[k] = v
            except Exception:
                pass
        if getattr(self, "_alpha_beta", None):
            key_vals.update({k: v for k, v in self._alpha_beta.items() if v is not None})

        manifest = {"tables": tables, "plots": plots, "key_values": key_vals}
        manifest_path = out / "summary.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        self.artifacts.append(manifest_path)

        # optional strict postcondition: ensure we produced something non-trivial
        if self.strict and not (tables or plots):
            raise OneClickError("report: no tables/plots produced")
        self._progress("Done")
        return out


# Convenience entry point -----------------------------------------------------

def run_easy_legacy(
    src: Path,
    site: SitePreset,
    baro_override_Pa: Optional[float] = None,
    run_stamp: Optional[str] = None,
    output_base: Optional[Path] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    strict: bool = False,
) -> tuple[Path, Dict, List[str]]:
    """Run the full pipeline for a legacy workbook using ``SitePreset`` defaults.

    Returns
    -------
    tuple
        ``(run_dir, summary, artifacts)`` where ``run_dir`` is the output
        directory path, ``summary`` contains warning/error details, and
        ``artifacts`` is a list of generated tables/plots.
    """

    run = RunInputs(
        src=src,
        site=site,
        baro_override_Pa=baro_override_Pa,
        run_stamp=run_stamp,
        output_base=output_base,
    )
    # stash strict flag onto RunInputs for Orchestrator to read
    setattr(run, "strict", strict)
    orch = Orchestrator(run, progress_cb=progress_cb)
    out = orch.run_all()
    # return triple so app GUI can display warnings/plots/tables
    return out, orch.summary, [str(p) for p in orch.artifacts]
