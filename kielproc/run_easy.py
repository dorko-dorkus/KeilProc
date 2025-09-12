"""One-click GUI pipeline orchestrator (integrate → SoT → transmitter)."""


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json, math, logging
import hashlib, time, os
import pandas as pd
from .aggregate import RunConfig as IntegratorConfig, integrate_run
from .geometry import Geometry, r_ratio, beta_from_geometry, duct_area
from .transmitter import compute_and_write_setpoints, TxParams
from .transmitter_flow import write_lookup_outputs
from .report_pdf import build_run_report_pdf
from .tools.legacy_parser import parse_legacy_workbook
from .tools.legacy_overlay import (
    extract_piccolo_overlay_from_workbook,
    extract_baro_from_workbook,
    extract_piccolo_range_and_avg_current,
    extract_process_temperature_from_workbook,
)
from .tools.venturi_builder import build_venturi_result
from .tools.recalc import recompute_duct_result_with_rho

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
    # Logger CSV (optional; if provided we overlay data on the constant lookup)
    setpoints_csv: Optional[str] = None
    setpoints_min_frac: float = 0.6
    setpoints_slope_sign: int = +1
    # Season selector (sole user input for flow lookup)
    season: str = "summer"              # "summer" | "winter"
    # Static source mode for density at traverse plane
    static_source_mode: str = "ring_gauge"  # "ring_gauge" | "wall_gauge" | "wall_abs" | "baro_only"
    # Optional site defaults: calib_820_summer / calib_820_winter
    lookup_dp_max_mbar: float = 10.0


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


def _resolve_geometry(site: SitePreset) -> Tuple[Optional[Geometry], Optional[float], Optional[float], Optional[Tuple[float,float]], Optional[float], Optional[float]]:
    """Return (Geometry, r, beta, (H,W) for integrator, As, At) from site inputs."""
    if not site or not site.geometry:
        return None, None, None, None, None, None
    g = site.geometry
    # compute As (duct area) from width/height/area, prefer explicit dims
    H = float(g["duct_height_m"]) if "duct_height_m" in g else None
    W = float(g["duct_width_m"])  if "duct_width_m"  in g else None
    As = None
    if H and W:
        As = H * W
    elif "duct_area_m2" in g:
        As = float(g["duct_area_m2"])
    # throat area
    At = None
    if "throat_area_m2" in g:
        At = float(g["throat_area_m2"])
    # beta direct or from areas
    beta = float(g["beta"]) if "beta" in g else None
    if (beta is None) and (As is not None) and (At is not None) and At > 0:
        beta = math.sqrt(At/As)  # β = sqrt(At/A1)
    if (At is None) and (beta is not None) and (As is not None):
        At = beta*beta*As
    # r = As/At when both known
    r = (As/At) if (As and At and At > 0) else None
    # Build Geometry if we have any duct info (As or H/W)
    geom = Geometry(
        duct_width_m=W, duct_height_m=H, duct_area_m2=(As if (As and not (H and W)) else None),
        throat_area_m2=At,
    ) if (As or (H and W)) else None
    # For integrator: prefer real H,W; else use (1, As) so A = H*W = As
    dims = (H, W) if (H and W) else ((1.0, As) if As else None)
    return geom, r, beta, dims, As, At


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_all(cfg: RunConfig) -> Dict[str, Any]:
    """Execute integration and transmitter setpoints; write SoT artifacts."""
    logger.info("Run start: input=%s output=%s glob=%s", cfg.input_dir, cfg.output_dir, cfg.file_glob)
    site = _resolve_site(cfg)
    baro_source: Dict[str, Any] = {"source": "gui_or_default"}
    thermo_source: Dict[str, Any] = {"source": "absent"}
    T_K: Optional[float] = None
    geom, r, beta, dims, As, At = _resolve_geometry(site)
    if dims is None:
        raise ValueError("Geometry required: provide duct width+height or duct area.")
    H, W = dims
    int_cfg = IntegratorConfig(height_m=float(H), width_m=float(W))

    # ---------------------- Input prep: folder or workbook -------------------
    in_path = Path(cfg.input_dir)
    prepared_dir: Path
    input_mode: str
    piccolo_info: Dict[str, Any] = {}
    if in_path.is_dir():
        # Pre-formatted CSV folder
        prepared_dir = in_path
        input_mode = "csv_folder"
    elif in_path.is_file() and in_path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        # Legacy workbook → stage to CSVs
        prepared_dir = Path(cfg.output_dir) / "_staging_legacy"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        # writes per-port CSVs + summary.json into prepared_dir
        parse_legacy_workbook(in_path, out_dir=prepared_dir, return_mode="files")
        input_mode = "legacy_workbook"
        # Piccolo range & avg current (for debug/report)
        try:
            piccolo_info = extract_piccolo_range_and_avg_current(in_path)
        except Exception as _e:  # pragma: no cover - defensive
            piccolo_info = {"status": "error", "error": str(_e)}
        # Barometric pressure from workbook (Data!H15:I19), overrides GUI value
        try:
            _baro = extract_baro_from_workbook(in_path)
        except Exception as _e:  # pragma: no cover - defensive
            _baro = {"status": "error", "error": str(_e)}
        if _baro.get("status") == "ok" and _baro.get("baro_pa", 0) > 0:
            cfg.baro_pa = float(_baro["baro_pa"])
            baro_source = {
                "source": "workbook",
                "cell": _baro.get("cell"),
                "unit_raw": _baro.get("unit_raw"),
            }
        else:
            baro_source = {"source": "gui_or_default", "detail": _baro}

        # Process temperature from workbook (°C → K)
        try:
            _therm = extract_process_temperature_from_workbook(in_path)
        except Exception as _e:  # pragma: no cover - defensive
            _therm = {"status": "error", "error": str(_e)}
        if _therm.get("status") == "ok" and _therm.get("T_K", 0) > 0:
            T_K = float(_therm["T_K"])
            thermo_source = {"source": "workbook", "cell": _therm.get("cell")}
        else:
            # Fallback: no workbook temp — leave unset and let downstream decide
            T_K = None
            thermo_source = {"source": "absent", "detail": _therm}
    else:
        raise FileNotFoundError(
            f"Input path must be a folder of CSVs or an .xlsx workbook: {in_path}"
        )

    baro_pa = _resolve_baro(cfg)
    logger.info("Site: %s  Baro (Pa): %.1f", site.name, baro_pa)

    # ---------------------- Integrate on prepared_dir ------------------------
    res = integrate_run(
        prepared_dir,
        int_cfg,
        file_glob=cfg.file_glob,
        baro_cli_pa=baro_pa,
        area_ratio=r,
        beta=beta,
    )
    outdir = Path(cfg.output_dir) / "_integrated"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "per_port.csv").write_text(res["per_port"].to_csv(index=False))
    (outdir / "duct_result.json").write_text(json.dumps(res.get("duct", {}), indent=2))
    (outdir / "normalize_meta.json").write_text(json.dumps(res.get("normalize_meta", {}), indent=2))
    piccolo_overlay_csv = None
    overlay_expected = False
    if input_mode == "legacy_workbook":
        # Try to auto-extract Piccolo DP for overlay from the same workbook
        try:
            # NOTE: extractor returns a dict; use its 'csv' field (don’t assume location)
            _meta = extract_piccolo_overlay_from_workbook(in_path, prepared_dir / "piccolo_overlay.csv")
            if _meta.get("status") == "ok":
                piccolo_overlay_csv = Path(_meta.get("csv", prepared_dir / "piccolo_overlay.csv"))
                overlay_expected = True
            else:
                piccolo_overlay_csv = None
        except Exception:
            piccolo_overlay_csv = None
    
    # --- Use plane static for density; build Venturi curve; recompute duct totals coherently ---
    venturi = {}
    venturi_path: Optional[Path] = None
    rho_used: Optional[float] = None
    # plane static mean from per_port (preferred)
    p_s_mean = None
    try:
        pp = pd.read_csv(outdir / "per_port.csv")
        if "p_s_pa" in pp.columns:
            p_s_mean = float(pd.to_numeric(pp["p_s_pa"], errors="coerce").dropna().mean())
    except Exception:
        p_s_mean = None
    if (beta is not None) and (As is not None) and (p_s_mean or baro_pa):
        # If workbook didn't supply temp, try duct result mean temp
        if T_K is None:
            duct = res.get("duct", {}) or {}
            if isinstance(duct.get("temp_K_mean"), (int, float)):
                T_K = float(duct.get("temp_K_mean"))
                thermo_source = {"source": "duct_result", "key": "temp_K_mean"}
            elif isinstance(duct.get("temp_C_mean"), (int, float)):
                T_K = float(duct.get("temp_C_mean")) + 273.15
                thermo_source = {"source": "duct_result", "key": "temp_C_mean"}
        if T_K is None:
            pp = res.get("per_port")
            if pp is not None:
                try:
                    col = "T_C_mean"
                    if col in pp.columns:
                        T_K = float(pp[col].mean()) + 273.15
                        thermo_source = {"source": "per_port", "column": col}
                except Exception:
                    pass
        if T_K and T_K > 0:
            # Ideal-gas density at traverse plane (prefer plane static)
            P_use = float(p_s_mean) if p_s_mean else float(baro_pa)
            rho_used = P_use / (287.05 * float(T_K))
            # sanity: density must be in [0.2, 2.0] kg/m^3 for hot PA; if not, ABORT
            if not (0.2 <= rho_used <= 2.0):
                msg = (
                    f"FATAL: implausible density {rho_used:.6f} kg/m^3 "
                    f"(P_use={P_use:.1f} Pa, T_K={T_K:.2f}). "
                    f"Likely using GAUGE static as absolute. "
                    f"Ensure p_s = baro + Static(gauge)."
                )
                logger.error(msg)
                raise SystemExit(msg)
            try:
                venturi_path = build_venturi_result(
                    outdir,
                    beta=beta,
                    area_As_m2=As,
                    baro_pa=P_use,
                    T_K=T_K,
                    m_dot_hint_kg_s=(res.get("duct", {}) or {}).get("m_dot_kg_s"),
                )
                if venturi_path:
                    venturi = json.loads(Path(venturi_path).read_text())
                # Recompute duct totals so v̄, Q, m_dot match this rho; pass r (A_s/A_t)
                r_ar = (1.0 / (beta * beta)) if beta else None
                recompute_duct_result_with_rho(outdir, rho_used, r_area_ratio=r_ar)
            except Exception:  # pragma: no cover - defensive
                logger.error("FATAL: Venturi curve build/recompute failed", exc_info=True)
                raise
    # Optional transmitter setpoints
    sp_csv = cfg.setpoints_csv or (site.defaults.get("setpoints_csv") if site and site.defaults else None)
    # Use the extracted overlay (workbook mode) by default
    if piccolo_overlay_csv and piccolo_overlay_csv.exists():
        sp_csv = str(piccolo_overlay_csv)
    if sp_csv:
        try:
            sp = compute_and_write_setpoints(
                Path(sp_csv),
                out_json=outdir / "transmitter_setpoints.json",
                out_csv=outdir / "transmitter_setpoints.csv",
                dp_col="DP_mbar",
                T_col="T_C",
                min_fraction=float(cfg.setpoints_min_frac or 0.6),
                quantile=0.95,
                params=TxParams(),
            )
        except Exception as e:
            sp = {"error": str(e)}
    else:
        sp = {}

    # ---------------------- Flow lookup (always) -----------------------------
    # Fail loud if we expected an overlay but ended up with none
    if overlay_expected and not (sp_csv and Path(sp_csv).exists()):
        raise RuntimeError("Overlay expected from workbook, but piccolo_overlay.csv not found.")

    tx_meta = write_lookup_outputs(
        outdir,
        season=cfg.season,
        site_defaults=(site.defaults or {}),
        logger_csv=(Path(sp_csv) if sp_csv else None),
        dp_max_mbar=float(cfg.lookup_dp_max_mbar or 10.0),
    )

    # --- Build input manifest & hashes for audit/repro ---
    def _file_meta(p: Path):
        try:
            st = p.stat()
            h = hashlib.sha256()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            return {
                "path": str(p),
                "bytes": st.st_size,
                "mtime_utc": int(st.st_mtime),
                "sha256": h.hexdigest(),
            }
        except Exception:
            return {"path": str(p), "error": "stat_or_hash_failed"}

    input_manifest = []
    # include legacy workbook or prepared CSVs; also any logger/overlay CSV we produced
    try:
        if input_mode == "legacy_workbook":
            input_manifest.append(_file_meta(in_path))
        # staged/parsed inputs (if present)
        for p in sorted(Path(cfg.input_dir).glob(cfg.file_glob or "*.csv")):
            input_manifest.append(_file_meta(p))
    except Exception:
        pass

    # overlay/lookup files if present in output
    for name in [
        "transmitter_lookup_combined.csv",
        "transmitter_lookup_data.csv",
        "transmitter_lookup_reference.csv",
        "per_port.csv",
    ]:
        p = outdir / name
        if p.exists():
            input_manifest.append(_file_meta(p))

    # acceptance thresholds (can be overridden later by cfg)
    acceptance = {"mean_abs_tph_max": 0.5, "worst_abs_tph_max": 1.0}

    # reproducibility block
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    git_sha = None
    try:
        # best effort: read .git/HEAD
        head = Path(__file__).resolve().parents[2] / ".git" / "HEAD"
        if head.exists():
            ref = head.read_text().strip()
            if ref.startswith("ref:"):
                ref_path = (
                    Path(__file__).resolve().parents[2] / ".git" / ref.split(":", 1)[1].strip()
                )
                if ref_path.exists():
                    git_sha = ref_path.read_text().strip()[:12]
            else:
                git_sha = ref[:12]
    except Exception:
        git_sha = None
    git_sha = os.environ.get("GIT_SHA", git_sha)
    cfg_hash_src = json.dumps(
        {
            "inputs": [m.get("path") for m in input_manifest],
            "site": site.name,
            "baro_pa": baro_pa,
            "beta": beta,
            "T_K": T_K if "T_K" in locals() else None,
        },
        sort_keys=True,
    ).encode("utf-8")
    cfg_hash = hashlib.sha256(cfg_hash_src).hexdigest()[:12]

    summary = {
        "baro_pa": baro_pa,
        "baro_source": baro_source,
        "piccolo_info": piccolo_info,
        "site_name": site.name,
        "r": r,
        "beta": beta,
        "thermo_source": thermo_source,
        "T_K": T_K,
        "rho_kg_m3": rho_used,
        "rho_source": (
            "ideal_gas_pT_plane_static" if (rho_used and p_s_mean) else
            "ideal_gas_pT_baro" if rho_used else "unknown"
        ),
        # audit / repro metadata
        "inputs_manifest": input_manifest,
        "acceptance": acceptance,
        "run_id": run_id,
        "git_sha": git_sha,
        "config_hash": cfg_hash,
        "input_mode": input_mode,
        "prepared_input_dir": str(prepared_dir),
        "per_port_csv": str(outdir / "per_port.csv"),
        "duct_result_json": str(outdir / "duct_result.json"),
        "normalize_meta_json": str(outdir / "normalize_meta.json"),
        # Record exactly what we used
        "piccolo_overlay": {
            "status": "ok" if (piccolo_overlay_csv and Path(piccolo_overlay_csv).exists()) else "absent",
            "csv": (str(piccolo_overlay_csv) if piccolo_overlay_csv else None),
        },
        "setpoints": sp,
        "venturi_result_json": (str(venturi_path) if venturi_path else None),
    }
    # capture flow lookup meta paths for audit
    summary["flow_lookup"] = tx_meta
    # ALSO expose transmitter details at top-level for legacy report readers
    cal = (tx_meta or {}).get("calibration", {})
    summary["season"] = (tx_meta or {}).get("season")
    summary["m_820"] = cal.get("m_820")
    summary["c_820"] = cal.get("c_820")
    summary["dp_range_mbar"] = cal.get("range_mbar")
    summary["operating_band_mbar"] = (tx_meta or {}).get("operating_band_mbar")
    # Also expose K for any legacy readers
    if cal.get("K_uic") is not None:
        summary["K_uic"] = cal["K_uic"]
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---------------------- Single PDF report ------------------------
    try:
        report_pdf = build_run_report_pdf(
            outdir=outdir,
            summary_path=outdir / "summary.json",
            filename="RunReport.pdf",
        )
        summary["report_pdf"] = str(report_pdf)
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    except Exception as e:
        summary["report_pdf_error"] = str(e)
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    return summary


__all__ = ["SitePreset", "RunConfig", "run_all"]

