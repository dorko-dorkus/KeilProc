"""One-click GUI pipeline orchestrator (integrate → SoT → transmitter)."""


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging, json, math
from .aggregate import RunConfig as IntegratorConfig, integrate_run
from .geometry import Geometry, r_ratio, beta_from_geometry, duct_area
from .transmitter import compute_and_write_setpoints, TxParams
from .tools.legacy_parser import parse_legacy_workbook

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
    # Optional transmitter inputs (map to your logger CSV)
    setpoints_csv: Optional[str] = None
    setpoints_x_col: str = "i/p"   # dp column name
    setpoints_y_col: str = "820"   # temperature column name
    setpoints_min_frac: float = 0.6
    setpoints_slope_sign: int = +1


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
    baro_pa = _resolve_baro(cfg)
    geom, r, beta, dims, As, At = _resolve_geometry(site)
    if dims is None:
        raise ValueError("Geometry required: provide duct width+height or duct area.")
    H, W = dims
    int_cfg = IntegratorConfig(height_m=float(H), width_m=float(W))

    # ---------------------- Input prep: folder or workbook -------------------
    in_path = Path(cfg.input_dir)
    prepared_dir: Path
    input_mode: str
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
    else:
        raise FileNotFoundError(
            f"Input path must be a folder of CSVs or an .xlsx workbook: {in_path}"
        )

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

    # ---------------------- Venturi mapping (optional) -----------------------
    venturi = {}
    try:
        if (As is not None) and (At is not None) and (beta is not None) and (r is not None) and (r > 1.0):
            duct = res.get("duct", {}) or {}
            # Try to recover Qs (verification-plane volumetric flow, m^3/s)
            Qs_candidates = [
                duct.get("Qs_m3s"), duct.get("q_s_m3s"),
                duct.get("Q_m3s"), duct.get("q_m3s"),
                duct.get("volumetric_m3s"),
            ]
            Qs = next((x for x in Qs_candidates if isinstance(x, (int, float))), None)
            # Fallback from mass flow if present
            rho = duct.get("rho_kg_m3")
            T_C = duct.get("temp_C_mean")
            if rho is None:
                # Ideal gas fallback for air: rho = P / (R T)
                R = 287.05  # J/kg/K
                T_K = (float(T_C) + 273.15) if isinstance(T_C, (int, float)) else 293.15
                rho = float(cfg.baro_pa or 101325.0) / (R * T_K)
            if Qs is None and isinstance(duct.get("m_dot_kg_s"), (int, float)) and rho:
                Qs = float(duct["m_dot_kg_s"]) / float(rho)

            if isinstance(Qs, (int, float)) and Qs > 0:
                # Venturi Δp using standard equation (low Mach), with an optional Cd
                Cd = 0.98
                if site and site.defaults and isinstance(site.defaults.get("venturi_Cd"), (int, float)):
                    Cd = float(site.defaults["venturi_Cd"])
                # Δp = (Q / (Cd*At))^2 * (ρ/2) * (1 - β^4)
                dp_vent = (Qs / (Cd * At)) ** 2 * (rho / 2.0) * (1.0 - beta ** 4)
                vt = Qs / At
                # Build a small curve across the normalized flow range (10%..100%)
                curve = []
                for frac in [x/10 for x in range(1, 11)]:  # 0.1..1.0
                    Q = Qs * frac
                    dp = (Q / (Cd * At)) ** 2 * (rho / 2.0) * (1.0 - beta ** 4)
                    curve.append({"frac_of_Qs": frac, "Q_m3s": Q, "dp_vent_Pa": dp})
                venturi = {
                    "As_m2": As, "At_m2": At, "beta": beta, "r": r,
                    "rho_kg_m3": rho, "Qs_m3s": Qs, "vt_m_s": vt,
                    "Cd": Cd, "dp_vent_Pa_at_Qs": dp_vent, "curve": curve,
                }
                (outdir / "venturi_result.json").write_text(json.dumps(venturi, indent=2))
                # Write curve CSV for quick plotting
                try:
                    import csv
                    with (outdir / "venturi_curve.csv").open("w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=["frac_of_Qs","Q_m3s","dp_vent_Pa"])
                        w.writeheader()
                        w.writerows(curve)
                except Exception:
                    pass
    except Exception as e:
        venturi = {"error": str(e)}
    # Optional transmitter setpoints
    sp_csv = cfg.setpoints_csv or (site.defaults.get("setpoints_csv") if site and site.defaults else None)
    if sp_csv:
        try:
            sp = compute_and_write_setpoints(
                Path(sp_csv),
                out_json=outdir / "transmitter_setpoints.json",
                out_csv=outdir / "transmitter_setpoints.csv",
                dp_col=(cfg.setpoints_x_col or "i/p"),
                T_col=(cfg.setpoints_y_col or "820"),
                min_fraction=float(cfg.setpoints_min_frac or 0.6),
                quantile=0.95,
                params=TxParams(),
            )
        except Exception as e:
            sp = {"error": str(e)}
    else:
        sp = {}
    summary = {
        "baro_pa": baro_pa,
        "site_name": site.name,
        "r": r,
        "beta": beta,
        "input_mode": input_mode,
        "prepared_input_dir": str(prepared_dir),
        "per_port_csv": str(outdir / "per_port.csv"),
        "duct_result_json": str(outdir / "duct_result.json"),
        "normalize_meta_json": str(outdir / "normalize_meta.json"),
        "setpoints": sp,
        "venturi_result_json": (str(outdir / "venturi_result.json") if venturi else None),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


__all__ = ["SitePreset", "RunConfig", "run_all"]

