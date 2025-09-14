"""One-click GUI pipeline orchestrator (integrate → SoT → transmitter)."""


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json, math, logging
import hashlib, time, os
import pandas as pd
import numpy as np
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
from .tools.recalc import recompute_duct_result_with_rho
from .profile_xi import aggregate_by_xi
from .temp_select import pick_duct_temperature_K
from .piccolo import current_to_dp_raw_mbar, fit_current_to_dp, build_pred_dp_from_qs_mbar

R_AIR = 287.05  # J/(kg*K)

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
    # Optional correction factor plane→throat (loss/recovery). Default 1.0
    plane_to_throat_coeff: float = 1.0
    # Optional Venturi ISO curve parameters
    venturi_C: float = 0.98
    venturi_eps: float = 1.0
    At_m2: Optional[float] = None


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
    thermo_meta: Dict[str, Any] = {}
    wb_T_K: Optional[float] = None
    T_K: Optional[float] = None
    geom, r, beta, dims, As, At = _resolve_geometry(site)
    A1 = As
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

        # Process temperature from workbook (°C → K) used as a hint
        try:
            _therm = extract_process_temperature_from_workbook(in_path)
        except Exception as _e:  # pragma: no cover - defensive
            _therm = {"status": "error", "error": str(_e)}
        if _therm.get("status") == "ok" and _therm.get("T_K", 0) > 0:
            wb_T_K = float(_therm["T_K"])
            th_hint = {"T_K": wb_T_K, "cell": _therm.get("cell"), "source": "workbook"}
            if _therm.get("value") is not None:
                th_hint["value"] = _therm.get("value")
            if _therm.get("unit"):
                th_hint["unit"] = _therm.get("unit")
            thermo_meta["workbook_hint"] = th_hint
        else:
            wb_T_K = None
            thermo_meta["workbook_hint"] = {"detail": _therm}
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
    per_port_df = res.get("per_port")
    ts_df = res.get("per_sample")
    (outdir / "per_port.csv").write_text(per_port_df.to_csv(index=False))
    if ts_df is not None:
        ts = ts_df.rename(
            columns={
                "VP": "VP_pa",
                "Temperature": "T_C",
                "Static_abs_Pa": "p_s_pa",
            }
        )
        rename = {
            "VP_pa": "VP_pa",
            "T_C": "T_C",
            "static_gauge_pa": "static_gauge_pa",
            "piccolo_mA": "piccolo_mA",
            "p_s_pa": "p_s_pa",
            "Port": "Port",
        }
        ts_cols = [c for c in rename if c in ts.columns]
        ts[ts_cols].to_csv(outdir / "normalized_timeseries.csv", index=False)
    else:
        (outdir / "normalized_timeseries.csv").write_text("")
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

    # --- Robust duct temperature selection ---
    t_pick = pick_duct_temperature_K(
        per_port_df,
        workbook_hint_value=wb_T_K,
        workbook_hint_unit=("K" if wb_T_K is not None else None),
    )
    T_K = float(t_pick["T_K"])
    thermo_meta["thermo_choice"] = t_pick

    # --- Use plane static for density; recompute duct totals coherently ---
    rho_used: Optional[float] = None
    rho_src: str = "unknown"
    # ---- Plane static selection & reconstruction ----
    mode = getattr(cfg, "static_source_mode", "ring_gauge")
    ps_abs_col = []
    ps_gauge_col = []
    if per_port_df is not None:
        ps_abs_col = [c for c in per_port_df.columns if c.lower() in ("static_abs_pa", "static_absolute_pa")]
        ps_gauge_col = [c for c in per_port_df.columns if c.lower() in ("static_gauge_pa", "static_pa", "static")]
    p_plane_static_pa: float | None = None
    p_static_source_note = ""
    if mode == "wall_abs" and ps_abs_col:
        p_plane_static_pa = float(np.nanmean(per_port_df[ps_abs_col[0]].to_numpy()))
        p_static_source_note = "wall_tap_absolute"
    elif mode in ("wall_gauge", "ring_gauge") and ps_gauge_col:
        p_plane_static_pa = float(baro_pa + np.nanmean(per_port_df[ps_gauge_col[0]].to_numpy()))
        p_static_source_note = (
            "wall_tap_gauge+baro" if mode == "wall_gauge" else "probe_ring_gauge+baro"
        )
    elif ps_abs_col:
        p_plane_static_pa = float(np.nanmean(per_port_df[ps_abs_col[0]].to_numpy()))
        p_static_source_note = "auto:absolute_static_column"
    elif ps_gauge_col:
        p_plane_static_pa = float(baro_pa + np.nanmean(per_port_df[ps_gauge_col[0]].to_numpy()))
        p_static_source_note = "auto:gauge_static_column+baro"
    else:
        p_plane_static_pa = float(baro_pa)
        p_static_source_note = "barometric_only (no static column)"

    # Density convention (plane static vs baro) as configured
    if getattr(cfg, "rho_convention", "plane_static") == "baro_T":
        rho_used = baro_pa / (R_AIR * T_K)
        rho_src = "ideal_gas_baro_T"
    else:
        rho_used = float(p_plane_static_pa if p_plane_static_pa is not None else baro_pa) / (R_AIR * T_K)
        rho_src = f"ideal_gas_TK @ {p_static_source_note}"
    if not (0.2 <= rho_used <= 2.0):
        msg = (
            f"FATAL: implausible density {rho_used:.6f} kg/m^3 "
            f"(P_use={p_plane_static_pa:.1f} Pa, T_K={T_K:.2f}). "
            f"Likely using GAUGE static as absolute. "
            f"Ensure p_s = baro + Static(gauge)."
        )
        logger.error(msg)
        raise SystemExit(msg)
    try:
        r_ar = (1.0 / (beta * beta)) if beta else None
        recompute_duct_result_with_rho(outdir, rho_used, r_area_ratio=r_ar)
    except Exception:  # pragma: no cover - defensive
        logger.error("FATAL: density recompute failed", exc_info=True)
        raise
    # Update duct_result with unified temperature and density
    duct_json = outdir / "duct_result.json"
    try:
        dj = json.loads(duct_json.read_text()) if duct_json.exists() else {}
    except Exception:
        dj = {}
    dj.update(
        {
            "temp_K_mean": T_K,
            "temp_C_mean": T_K - 273.15,
            "rho_kg_m3": rho_used,
            "rho_source": rho_src,
        }
    )
    duct_json.write_text(json.dumps(dj, indent=2))
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

    # --- If ξ and Aj exist, build robust Aj-weighted qs per port (z-gated where present) ---
    profile_xi_meta = None
    qc_info: Dict[str, Any] = {}
    try:
        per_port_path = outdir / "per_port.csv"
        per_port = pd.read_csv(per_port_path)
        per_sample = None
        candidates = [
            outdir / "normalized_timeseries.csv",
            outdir / "per_sample.csv",
            outdir / "staging_legacy.csv",
        ]
        for p in candidates:
            if p.exists():
                try:
                    per_sample = pd.read_csv(p)
                    break
                except Exception:
                    per_sample = None
        if per_sample is not None and len(per_sample) >= 50:
            qmap = {}
            for idx, row in per_port.reset_index().itertuples():
                pnum = int(getattr(row, "Port", idx + 1)) if "Port" in per_port.columns else (idx + 1)
                for c in (f"q_s_P{pnum}_Pa", f"VP_P{pnum}_Pa", f"q_s_p{pnum}_pa", f"VP_p{pnum}_pa"):
                    if c in per_sample.columns:
                        qmap[pnum] = c
                        break
            zmap = {}
            for pnum in qmap.keys():
                for zc in (f"z_P{pnum}_cm", f"height_P{pnum}_cm", f"pos_P{pnum}_cm"):
                    if zc in per_sample.columns:
                        zmap[pnum] = zc
                        break
            qs_mean_by_port, xi_profile_df, profile_xi_meta = aggregate_by_xi(
                per_sample, qmap, z_cols_by_port=zmap
            )
            if "Port" in per_port.columns:
                for i in range(len(per_port)):
                    pnum = int(per_port["Port"].iloc[i])
                    if pnum in qs_mean_by_port and np.isfinite(qs_mean_by_port[pnum]):
                        per_port.at[i, "q_s_pa_mean"] = qs_mean_by_port[pnum]
            else:
                for i, pnum in enumerate(sorted(qs_mean_by_port.keys())):
                    if np.isfinite(qs_mean_by_port[pnum]):
                        per_port.at[i, "q_s_pa_mean"] = qs_mean_by_port[pnum]
            per_port.to_csv(per_port_path, index=False)
        else:
            profile_xi_meta = {}
            qc_info["note"] = "Timeseries not found \u2192 QC & \u03be aggregation skipped."
    except Exception:
        profile_xi_meta = {}
        qc_info["note"] = "Timeseries not found \u2192 QC & \u03be aggregation skipped."

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

    # Load duct_result to make summary authoritative and consistent
    duct_path = outdir / "duct_result.json"
    duct_obj: Dict[str, Any] | None = None
    if duct_path.exists():
        try:
            duct_obj = json.loads(duct_path.read_text())
        except Exception:
            duct_obj = None

    # Prefer plane static mean from per_port (p_s_pa) if present
    p_plane: float | None = None
    try:
        if "p_s_pa" in per_port.columns:
            p_plane = float(np.nanmean(pd.to_numeric(per_port["p_s_pa"], errors="coerce")))
    except Exception:
        p_plane = None
    if p_plane is None and duct_obj and isinstance(duct_obj.get("p_s_pa_mean"), (int, float)):
        try:
            p_plane = float(duct_obj.get("p_s_pa_mean"))
        except Exception:
            p_plane = None

    # Adopt rho/Q/v/m_dot from duct_result when available
    rho_final = (duct_obj or {}).get("rho_kg_m3", rho_used)
    rho_src_final = (duct_obj or {}).get("rho_source", rho_src)
    v_bar_final = (duct_obj or {}).get("v_bar_m_s")
    Q_final = (duct_obj or {}).get("Q_m3_s")
    m_dot_final = (duct_obj or {}).get("m_dot_kg_s")
    q_s_mean = (duct_obj or {}).get("q_s_pa")

    # --- Geometry mapping to throat (global constants) ---
    q_t_mean = (duct_obj or {}).get("q_t_pa")
    delta_p_vent = (duct_obj or {}).get("delta_p_vent_est_pa")
    if q_s_mean is None and per_port is not None:
        try:
            q_s_mean = float(
                np.nanmean(pd.to_numeric(per_port["q_s_pa_mean"], errors="coerce"))
            )
        except Exception:
            q_s_mean = None
    if (q_t_mean is None) and (q_s_mean is not None) and (r is not None) and np.isfinite(r):
        q_t_mean = float((r ** 2) * q_s_mean)
    if (delta_p_vent is None) and (q_t_mean is not None) and (beta is not None) and np.isfinite(beta):
        delta_p_vent = float((1.0 - (beta ** 4)) * q_t_mean)  # Pa (geometry-only)
    if duct_obj is None:
        duct_obj = {}
    if q_s_mean is not None:
        duct_obj["q_s_pa"] = float(q_s_mean)
    if q_t_mean is not None:
        duct_obj["q_t_pa"] = float(q_t_mean)
    if delta_p_vent is not None:
        duct_obj["delta_p_vent_est_pa"] = float(delta_p_vent)
    duct_path.write_text(json.dumps(duct_obj, indent=2))

    # Optional correction factor plane→throat (loss/recovery). Default 1.0
    C_f = float(getattr(cfg, "plane_to_throat_coeff", 1.0))

    # ---------------- ξ profiles export (used for calibration & audit) ----------------
    # Expect xi_profile_df with per-port rows: ['Port','xi','Aj','q_s_median','q_s_smoothed',...]
    profiles_dir = outdir / "profiles"; profiles_dir.mkdir(exist_ok=True)
    try:
        for p in sorted(xi_profile_df["Port"].unique()):
            sub = xi_profile_df[xi_profile_df["Port"] == p].copy()
            sub.to_csv(profiles_dir / f"{p}_profile.csv", index=False)
    except Exception:
        pass

    # ---------------- Piccolo current → DP (raw + corrected) ----------------
    piccolo_fit = {}
    df_ts = None
    ts_path = outdir / "normalized_timeseries.csv"
    if ts_path.exists():
        df_ts = pd.read_csv(ts_path)

    # Build predicted DP series from q_s (timeseries primary; per-port means fallback)
    dp_pred_mbar_series = None; I_series = None
    if df_ts is not None and "VP_pa" in df_ts.columns:
        dp_pred_mbar_series = build_pred_dp_from_qs_mbar(
            pd.to_numeric(df_ts["VP_pa"], errors="coerce"), r=r, beta=beta, Cf=C_f
        )
    if df_ts is not None and "piccolo_mA" in df_ts.columns:
        I_series = pd.to_numeric(df_ts["piccolo_mA"], errors="coerce").to_numpy()
    if (I_series is None or not np.isfinite(I_series).any()) and "piccolo_mA_mean" in per_port.columns:
        I_series = pd.to_numeric(per_port["piccolo_mA_mean"], errors="coerce").to_numpy()
        if "q_s_pa" in per_port.columns:
            dp_pred_mbar_series = build_pred_dp_from_qs_mbar(
                pd.to_numeric(per_port["q_s_pa"], errors="coerce").to_numpy(),
                r=r,
                beta=beta,
                Cf=C_f,
            )

    lrv_mbar = float(getattr(cfg, "piccolo_lrv_mbar", 0.0))
    urv_mbar = float(getattr(cfg, "piccolo_urv_mbar", getattr(cfg, "transmitter_range_mbar", 8.5)))
    overlay_df = pd.DataFrame()
    if I_series is not None:
        overlay_df["data_DP_mbar_raw"] = current_to_dp_raw_mbar(I_series, lrv_mbar, urv_mbar)
    if dp_pred_mbar_series is not None:
        overlay_df["dp_pred_mbar_from_qs"] = np.asarray(dp_pred_mbar_series)
    if I_series is not None:
        # Optional regression for inspection only; DO NOT replace the overlay
        if "dp_pred_mbar_from_qs" in overlay_df.columns:
            a, b = fit_current_to_dp(I_series, overlay_df["dp_pred_mbar_from_qs"].to_numpy())
            overlay_df["data_DP_mbar_corr"] = a * I_series + b
            piccolo_fit = {
                "a_mbar_per_mA": float(a),
                "b_mbar": float(b),
                "lrv_mbar": lrv_mbar,
                "urv_mbar": urv_mbar,
                "n_points": int(np.isfinite(I_series).sum()),
            }
    if not overlay_df.empty:
        overlay_df.to_csv(outdir / "transmitter_lookup_data.csv", index=False)
        (outdir / "piccolo_cal.json").write_text(json.dumps(piccolo_fit or {}, indent=2))

    # ---------------- Reconciliation (works with or without overlay) ----------------
    dp_geom_mbar = float(((1.0 - beta**4) * (r**2) * q_s_mean) / 100.0) if ((beta is not None) and np.isfinite(beta) and (r is not None) and np.isfinite(r) and (q_s_mean is not None) and np.isfinite(q_s_mean)) else None
    p5 = p50 = p95 = None
    C_f_star = None
    dp_corr_mbar = None
    # --- predicted band from qs series (if any) ---
    pred_band_geom = pred_band_corr = None
    try:
        # Prefer timeseries qs; fallback to per_port means if needed
        qs_series = None
        if df_ts is not None and "VP_pa" in df_ts.columns:
            qs_series = pd.to_numeric(df_ts["VP_pa"], errors="coerce").to_numpy()
        elif "q_s_pa" in per_port.columns:
            qs_series = pd.to_numeric(per_port["q_s_pa"], errors="coerce").to_numpy()
        if qs_series is not None:
            qs_series = qs_series[np.isfinite(qs_series) & (qs_series > 0)]
            if qs_series.size:
                # DP_geom = (1-β^4) r^2 * qs / 100
                k = (1.0 - beta**4) * (r**2) / 100.0
                g5, g95 = np.percentile(k * qs_series, [5, 95])
                pred_band_geom = [float(g5), float(g95)]
    except Exception:
        pass
    reconcile = {
        "dp_overlay_p5_mbar": None, "dp_overlay_p50_mbar": None, "dp_overlay_p95_mbar": None,
        "dp_pred_geom_mbar": dp_geom_mbar,
        "dp_pred_corr_mbar": None,
        "dp_error_geom_mbar": None, "dp_error_corr_mbar": None,
        "dp_error_geom_pct_vs_p50": None, "dp_error_corr_pct_vs_p50": None,
        "C_f": C_f, "C_f_fit": None, "piccolo_fit": piccolo_fit,
        "pred_band_geom_mbar": pred_band_geom,
        "pred_band_corr_mbar": None,
    }
    data_csv = outdir / "transmitter_lookup_data.csv"
    if data_csv.exists():
        df_overlay = pd.read_csv(data_csv)
        # Use RAW overlay for percentiles & Cf fit
        col_raw = "data_DP_mbar_raw" if "data_DP_mbar_raw" in df_overlay.columns else None
        if col_raw is not None:
            dp = pd.to_numeric(df_overlay[col_raw], errors="coerce").dropna().to_numpy()
            if dp.size:
                p5, p50, p95 = np.percentile(dp, [5, 50, 95])
                reconcile["dp_overlay_p5_mbar"] = float(p5)
                reconcile["dp_overlay_p50_mbar"] = float(p50)
                reconcile["dp_overlay_p95_mbar"] = float(p95)
        # Fit Cf from RAW overlay vs predicted series
        if {"data_DP_mbar_raw", "dp_pred_mbar_from_qs"}.issubset(df_overlay.columns):
            xv = pd.to_numeric(df_overlay["dp_pred_mbar_from_qs"], errors="coerce").to_numpy()
            yv = pd.to_numeric(df_overlay["data_DP_mbar_raw"], errors="coerce").to_numpy()
            m = np.isfinite(xv) & np.isfinite(yv) & (xv > 0)
            if m.any():
                ratios = (yv[m] / xv[m])
                ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
                if ratios.size:
                    C_f_star = float(np.median(ratios))
                    # Guard rail: ignore absurd fits
                    if 0.5 <= C_f_star <= 20.0 and (dp_geom_mbar is not None):
                        dp_corr_mbar = C_f_star * dp_geom_mbar
                        reconcile["dp_pred_corr_mbar"] = float(dp_corr_mbar)
                        reconcile["C_f_fit"] = float(C_f_star)
                        if p50:
                            reconcile["dp_error_corr_mbar"] = float(dp_corr_mbar - p50)
                            reconcile["dp_error_corr_pct_vs_p50"] = float(
                                100.0 * (dp_corr_mbar - p50) / p50
                            )
                # Predicted bands (5–95%) from series
                if pred_band_geom and C_f_star:
                    reconcile["pred_band_corr_mbar"] = [
                        float(C_f_star * pred_band_geom[0]),
                        float(C_f_star * pred_band_geom[1]),
                    ]
        if p50 is not None and dp_geom_mbar is not None:
            reconcile["dp_error_geom_mbar"] = float(dp_geom_mbar - p50)
            reconcile["dp_error_geom_pct_vs_p50"] = float(100.0 * (dp_geom_mbar - p50) / p50) if p50 else None

    # Static-source labeling: derive from per_port "p_abs_source" if we can
    static_mode = mode
    try:
        srcs = per_port["p_abs_source"].dropna().astype(str).unique().tolist()
        if srcs:
            static_mode = srcs[0]
    except Exception:
        pass

    summary = {
        "baro_pa": baro_pa,
        "baro_source": baro_source,
        "piccolo_info": piccolo_info,
        "site_name": site.name,
        "r": r,
        "beta": beta,
        "thermo_source": thermo_meta,
        "A1_m2": A1,
        "At_m2": At,
        "T_K": T_K,
        "rho_kg_m3": rho_final,
        "rho_source": rho_src_final,
        "static_source_mode": static_mode,
        "p_plane_static_pa_mean": p_plane if p_plane is not None else baro_pa,
        "q_s_pa_mean": q_s_mean,
        "q_t_pa_mean": q_t_mean,
        "delta_p_vent_est_pa": delta_p_vent,
        "v_bar_m_s": v_bar_final,
        "Q_m3_s": Q_final,
        "m_dot_kg_s": m_dot_final,
        "reconcile": reconcile,
        # audit / repro metadata
        "inputs_manifest": input_manifest,
        "acceptance": acceptance,
        "run_id": run_id,
        "git_sha": git_sha,
        "config_hash": cfg_hash,
        "input_mode": input_mode,
        "prepared_input_dir": str(prepared_dir),
        "per_port_csv": str(outdir / "per_port.csv"),
        "normalized_timeseries_csv": str(outdir / "normalized_timeseries.csv"),
        "duct_result_json": str(outdir / "duct_result.json"),
        "normalize_meta_json": str(outdir / "normalize_meta.json"),
        # Record exactly what we used
        "piccolo_overlay": {
            "status": "ok" if (piccolo_overlay_csv and Path(piccolo_overlay_csv).exists()) else "absent",
            "csv": (str(piccolo_overlay_csv) if piccolo_overlay_csv else None),
        },
        "setpoints": sp,
    }
    summary.setdefault("profile_xi", {})["meta"] = profile_xi_meta or {}
    summary["qc"] = qc_info
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

    summary["plane_qs_weighting"] = "ports_equal"  # will flip to 'Aj' when profiles are available

    # --- Venturi curve (mass-flow domain) ---
    # Geometry-only curve and an ISO-style curve with configurable C and epsilon
    venturi_json_path = outdir / "venturi_result.json"
    vent_C = float(getattr(cfg, "venturi_C", 0.98))
    vent_eps = float(getattr(cfg, "venturi_eps", 1.0))
    curve: Dict[str, Any] = {}
    try:
        At_val = float(getattr(cfg, "At_m2", None) or (At if At is not None else 1.8))
        if (
            beta is not None
            and np.isfinite(beta)
            and np.isfinite(At_val)
            and (rho_final is not None)
        ):
            m_dot_ref = float(m_dot_final) if m_dot_final else 0.0
            m_grid = np.linspace(0.0, max(m_dot_ref * 1.6, 10.0), 200)
            k_geom = (1.0 - beta**4) / (2.0 * rho_final * (At_val**2))
            dp_pa_geom = k_geom * (m_grid**2)
            k_iso = k_geom / max((vent_C * vent_eps) ** 2, 1e-9)
            dp_pa_iso = k_iso * (m_grid**2)
            curve = {
                "beta": float(beta),
                "At_m2": float(At_val),
                "rho_kg_m3": float(rho_final),
                "C": vent_C,
                "epsilon": vent_eps,
                "m_dot_kg_s_grid": m_grid.tolist(),
                "dp_pa_geom_grid": dp_pa_geom.tolist(),
                "dp_pa_iso_grid": dp_pa_iso.tolist(),
                "op_point": {
                    "m_dot_kg_s": float(m_dot_ref) if m_dot_ref else None,
                    "dp_pa_est": float(delta_p_vent) if delta_p_vent is not None else None,
                },
            }
    except Exception:
        curve = {}
    if curve:
        venturi_json_path.write_text(json.dumps(curve, indent=2))
        summary["venturi_result_json"] = str(venturi_json_path)
    else:
        summary["venturi_result_json"] = None
        summary["venturi_note"] = "Skipped: require beta, At, and density."
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

