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
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
import dataclasses
import json
from zoneinfo import ZoneInfo

from .aggregate import integrate_run, RunConfig, discover_pairs
from .gui_adapter import (
    process_legacy_parsed_csv,
    fit_alpha_beta,
    translate_piccolo,
    legacy_results_from_csv,
)

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


from .presets import PRESETS

class OneClickError(Exception):
    """Raised when a hard failure occurs during the one-click run."""


class Orchestrator:
    """Coordinate parse → integrate → map → fit → translate → report."""

    def __init__(self, run: RunInputs, progress_cb: Optional[Callable[[str], None]] = None):
        self.run = run
        self.artifacts: list[Path] = []
        self.summary: dict[str, list[str]] = {"warnings": [], "errors": []}
        self._pairs: list[tuple[str, Path]] = []
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
        # Localize to NZT explicitly to avoid relying on system timezone.
        return datetime.now(ZoneInfo(NZT)).strftime("%Y%m%d_%H%M")

    def _mkdirs(self, base: Path) -> dict[str, Path]:
        base.mkdir(parents=True, exist_ok=True)
        ports = base / "ports_csv"
        ports.mkdir(exist_ok=True)
        return {"base": base, "ports": ports}

    def _placeholder(self, path: Path, note: str) -> Path:
        """Create a placeholder file noting why a stage did not run."""
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(f"placeholder: {note}\n")
        except Exception:
            # swallow any failure; this is best-effort
            pass
        self.artifacts.append(path)
        return path

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
        from .tools.legacy_parser.parser import parse_legacy_workbook

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

        area_ratio = geom_dict.get("r") or geom_dict.get("area_ratio")
        beta = geom_dict.get("beta")
        if geom is not None:
            try:
                if area_ratio is None:
                    area_ratio = r_ratio(geom)
                if beta is None:
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
        if area_ratio is not None:
            ref_block["r"] = area_ratio
        if beta is not None:
            ref_block["beta"] = beta
        self._venturi = {"r": area_ratio, "beta": beta}
        ref_json = outdir / "reference_block.json"
        ref_json.write_text(json.dumps(ref_block, indent=2))

        self.artifacts.append(per_csv)
        self.artifacts.append(duct_json)
        self.artifacts.append(norm_json)
        self.artifacts.append(ref_json)

        self._pairs = res.get("pairs", [])
        self._skipped = res.get("skipped", [])
        if self._skipped:
            for name, reason in self._skipped:
                self.summary["warnings"].append(f"integrate {name}: {reason}")

    
    def map(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Build velocity maps from integrated data."""
        from .physics import map_qs_to_qt
        from .visuals import render_velocity_heatmap
        from .geometry import Geometry, r_ratio

        ports_dir = base_dir / "ports_csv"
        mapped_dir = base_dir / "_mapped"
        mapped_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = mapped_dir / "heatmap_velocity.png"

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
            note = "map: geometry unavailable; skipping mapping"
            self.summary["warnings"].append(note)
            self._placeholder(heatmap_path, note)
            self._pairs = pairs
            return

        r = r_ratio(geom)
        if r is None:
            note = "map: geometry missing port area ratio; skipping mapping"
            self.summary["warnings"].append(note)
            self._placeholder(heatmap_path, note)
            self._pairs = pairs
            return

        pairs, skipped = discover_pairs(ports_dir, "*.csv")
        for name, reason in skipped:
            self.summary["warnings"].append(f"map {name}: {reason}")
        if not pairs:
            msg = "map: no port CSVs available"
            if self.strict:
                raise OneClickError(msg)
            self.summary["warnings"].append(msg)
            self._placeholder(heatmap_path, msg)
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
                    note = f"map: {csv.name}: {e2}"
                    if self.strict:
                        raise OneClickError(note)
                    self.summary["warnings"].append(note)
                    self._placeholder(out_path, note)

        try:
            png = render_velocity_heatmap(mapped_dir, pairs, self.run.baro_override_Pa)
            self.artifacts.append(png)
        except Exception as e:
            note = f"heatmap: {e}"
            self.summary["warnings"].append(note)
            self._placeholder(heatmap_path, note)

        self._pairs = pairs

        if not getattr(self, "_mapped_csvs", None):
            msg = "map: no mapped results produced"
            if self.strict:
                raise OneClickError(msg)
            self.summary["warnings"].append(msg)
            self._placeholder(heatmap_path, msg)


    def fit(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Fit calibration models."""

        mapped = getattr(self, "_mapped_csvs", [])
        outdir = base_dir / "_fit"
        outdir.mkdir(parents=True, exist_ok=True)
        placeholders = [
            outdir / "alpha_beta_by_block.csv",
            outdir / "alpha_beta_by_block.json",
            outdir / "alpha_beta_pooled.csv",
            outdir / "alpha_beta_pooled.json",
            outdir / "align.png",
        ]
        if not mapped:
            msg = "fit: no mapped CSVs available"
            if self.strict:
                raise OneClickError(msg)
            self.summary["warnings"].append(msg)
            for p in placeholders:
                self._placeholder(p, msg)
            return

        block_specs = {Path(p).stem: p for p in mapped}
        beta_override = self.run.site.defaults.get("beta_translate")
        try:
            res = fit_alpha_beta(
                block_specs,
                ref_col="deltpVent",
                piccolo_col="Piccolo",
                lambda_ratio=1.0,
                max_lag=300,
                outdir=outdir,
                beta_override=beta_override,
            )
            for k in ("per_block_csv", "per_block_json", "pooled_csv", "pooled_json", "align_png"):
                v = res.get(k)
                if v:
                    p = Path(v)
                    self.artifacts.append(p)
            if res.get("blocks_info"):
                b0 = res["blocks_info"][0]
                self._alpha_beta = {"alpha": b0.get("alpha"), "beta": b0.get("beta")}
                self._lag = b0.get("lag_samples")
        except Exception as e:
            note = f"fit: {e}"
            if self.strict:
                raise OneClickError(note)
            self.summary["warnings"].append(note)
            for p in placeholders:
                self._placeholder(p, note)

    def translate(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Generate control-system lookup tables."""

        outdir = base_dir / "_translated"
        outdir.mkdir(parents=True, exist_ok=True)
        default_out = outdir / "translated.csv"
        if not getattr(self, "_alpha_beta", None) or not getattr(self, "_mapped_csvs", None):
            msg = "translate: missing fit results or mapped data"
            if self.strict:
                raise OneClickError(msg)
            self.summary["warnings"].append(msg)
            self._placeholder(default_out, msg)
            return

        src_csv = self._mapped_csvs[0]
        out_path = outdir / f"{Path(src_csv).stem}_translated.csv"
        alpha = self._alpha_beta.get("alpha")
        beta = self._alpha_beta.get("beta")
        if alpha is None or beta is None:
            note = "translate: alpha/beta unavailable"
            self.summary["warnings"].append(note)
            self._placeholder(out_path, note)
            return

        try:
            translate_piccolo(src_csv, alpha, beta, "Piccolo", "Piccolo_translated", out_path)
            self.artifacts.append(out_path)
            self._translated_csv = out_path
        except Exception as e:
            note = f"translate: {e}"
            if self.strict:
                raise OneClickError(note)
            self.summary["warnings"].append(note)
            self._placeholder(out_path, note)

    def report(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        """Emit consolidated HTML/PDF reports."""
        from .legacy_results import ResultsConfig
        import math

        outdir = base_dir / "_report"
        outdir.mkdir(parents=True, exist_ok=True)
        csv_out = outdir / "legacy_results.csv"
        json_out = outdir / "legacy_results.json"
        sp_path = outdir / "setpoints.json"
        csv = getattr(self, "_translated_csv", None)
        if not csv:
            msg = "report: no translated CSV available"
            if self.strict:
                raise OneClickError(msg)
            self.summary["warnings"].append(msg)
            self._placeholder(csv_out, msg)
            self._placeholder(json_out, msg)
            self._placeholder(sp_path, msg)
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
        try:
            res = legacy_results_from_csv(csv, cfg, csv_out)
            self.artifacts.append(csv_out)
            json_out.write_text(json.dumps(res, indent=2))
            self.artifacts.append(json_out)
            try:
                from .setpoints_csv import setpoints_from_logger_csv
                sp = setpoints_from_logger_csv(str(csv), x_col="Piccolo_translated", y_col="Piccolo")
                sp_path.write_text(json.dumps(dataclasses.asdict(sp), indent=2))
                self.artifacts.append(sp_path)
            except Exception as e2:
                note = f"setpoints: {e2}"
                if self.strict:
                    raise OneClickError(note)
                self.summary["warnings"].append(note)
                self._placeholder(sp_path, note)
        except Exception as e:
            note = f"report: {e}"
            if self.strict:
                raise OneClickError(note)
            self.summary["warnings"].append(note)
            self._placeholder(csv_out, note)
            self._placeholder(json_out, note)
            self._placeholder(sp_path, note)

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
        key_vals: dict[str, object] = {}
        pooled = next((p for p in self.artifacts if p.name == 'alpha_beta_pooled.json'), None)
        if pooled and pooled.exists():
            try:
                data = json.loads(pooled.read_text())
                for k in ('alpha', 'beta'):
                    v = data.get(k)
                    if v is not None:
                        key_vals[k] = v
            except Exception:
                pass
        elif getattr(self, '_alpha_beta', None):
            key_vals.update({k: v for k, v in self._alpha_beta.items() if v is not None})
        if getattr(self, '_lag', None) is not None:
            key_vals['lag_samples'] = self._lag
        ref = next((p for p in self.artifacts if p.name == 'reference_block.json'), None)
        if ref and ref.exists():
            try:
                ref_data = json.loads(ref.read_text())
                mapping = {
                    'q_s_pa': 'q_s_pa',
                    'q_t_pa': 'q_t_pa',
                    'delta_p_vent_est_pa': 'delta_p_vent_est_pa',
                    'r': 'venturi_r',
                    'beta': 'venturi_beta',
                }
                for src_key, dst_key in mapping.items():
                    v = ref_data.get(src_key)
                    if v is not None:
                        key_vals[dst_key] = v
            except Exception:
                pass
        span_file = next((
            p
            for p in self.artifacts
            if p.suffix == '.json' and ('setpoint' in p.name.lower() or 'span' in p.name.lower())
        ), None)
        if span_file and span_file.exists():
            try:
                span_data = json.loads(span_file.read_text())
                if isinstance(span_data, dict):
                    container = span_data.get('setpoints') if 'setpoints' in span_data else span_data
                    if isinstance(container, dict):
                        span = container.get('span')
                        if span is not None:
                            key_vals['transmitter_span'] = span
                        mapping = container.get('mapping')
                        if mapping is not None:
                            key_vals['transmitter_setpoints'] = mapping
            except Exception:
                pass
        required_keys = [
            "alpha",
            "beta",
            "lag_samples",
            "venturi_r",
            "venturi_beta",
            "transmitter_span",
            "transmitter_setpoints",
        ]
        key_vals = {k: key_vals.get(k) for k in required_keys}

        qa_gates = {"delta_opp_max": 0.01, "w_max": 0.002}
        venturi = getattr(self, "_venturi", {})
        inputs = {
            "baro_override_Pa": self.run.baro_override_Pa,
            "site": self.run.site.name,
            "r": venturi.get("r"),
            "beta": venturi.get("beta"),
            "reference": None,
        }
        if ref and ref.exists():
            try:
                ref_data = json.loads(ref.read_text())
                for candidate in (
                    "delta_p_vent_est_pa",
                    "q_t_pa",
                    "q_s_pa",
                ):
                    if ref_data.get(candidate) is not None:
                        inputs["reference"] = candidate
                        break
            except Exception:
                pass

        manifest = {
            "tables": tables,
            "plots": plots,
            "key_values": key_vals,
            "qa_gates": qa_gates,
            "inputs": inputs,
        }
        skipped = getattr(self, "_skipped", [])
        if skipped:
            manifest["skipped_files"] = [
                {"file": f, "reason": r} for f, r in skipped
            ]

        manifest_path = out / "summary.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        self.artifacts.append(manifest_path)
        try:
            import shutil
            bundle_base = out.with_name(f"{out.name}__bundle")
            shutil.make_archive(str(bundle_base), "zip", root_dir=out)
            bundle_zip = bundle_base.with_suffix(".zip")
            self.artifacts.append(bundle_zip)
            manifest["bundle_zip"] = str(bundle_zip)
            manifest_path.write_text(json.dumps(manifest, indent=2))
        except Exception as e:
            self.summary["warnings"].append(f"bundle: {e}")

        # optional strict postcondition: ensure we produced something non-trivial
        if self.strict and not (tables or plots):
            raise OneClickError("report: no tables/plots produced")
        self._progress("Done")
        return out


# Public API ---------------------------------------------------------------


def run_all(
    src: str | Path,
    site: str | SitePreset = "DefaultSite",
    baro_override: float | None = None,
    run_stamp: str | None = None,
    output_base: str | Path | None = None,
    *,
    strict: bool = False,
):
    """Convenience wrapper around :class:`Orchestrator`.

    Parameters are forwarded to :class:`RunInputs`.  ``site`` may be a preset
    name or a ``SitePreset`` instance.  Returns the run directory ``Path``.
    """

    if isinstance(site, str):
        site = PRESETS[site]
    run = RunInputs(
        Path(src),
        site,
        baro_override,
        run_stamp,
        Path(output_base) if output_base else None,
    )
    setattr(run, "strict", strict)
    return Orchestrator(run).run_all()


# Convenience entry point -----------------------------------------------------

def run_easy_legacy(
    src: Path,
    site: SitePreset,
    baro_override_Pa: Optional[float] = None,
    run_stamp: Optional[str] = None,
    output_base: Optional[Path] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    strict: bool = False,
) -> tuple[Path, dict, list[str]]:
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
