from __future__ import annotations
from matplotlib.backends.backend_pdf import PdfPages
import json, math, textwrap, hashlib, os, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

# -------- Visual style (no change to data/logic) ----------------------------
# Neutral, readable defaults using Matplotlib’s bundled DejaVu fonts.
BRAND = {
    "font":  "DejaVu Sans",
    "mono":  "DejaVu Sans Mono",
    "accent":"#0F766E",   # teal-700
    "muted": "#6B7280",   # gray-500
    "grid":  "#E5E7EB",   # gray-200
    "text":  "#111827",   # gray-900
}

def _apply_matplotlib_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "font.family": BRAND["font"],
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.color": BRAND["grid"],
        "grid.linestyle": "-",
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(
            color=["#0F766E", "#2563EB", "#9333EA", "#EA580C", "#DC2626"]
        ),
        "legend.frameon": False,
    })

_apply_matplotlib_style()


def _load_json(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}


def _load_tx_tables(outdir: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Return (combined, data_only, ref_only) if present (any may be None)."""
    outdir = Path(outdir)
    comb_p = outdir / "transmitter_lookup_combined.csv"
    data_p = outdir / "transmitter_lookup_data.csv"
    ref_p  = outdir / "transmitter_lookup_reference.csv"
    comb = pd.read_csv(comb_p) if comb_p.exists() else None
    data = pd.read_csv(data_p) if data_p.exists() else None
    ref  = pd.read_csv(ref_p)  if ref_p.exists()  else None
    return comb, data, ref


def _overlay_dp_series(comb: Optional[pd.DataFrame], data: Optional[pd.DataFrame]) -> pd.Series:
    """Extract overlay DP vector robustly from either combined or data table.

    Prefers corrected DP (``*_corr``) if present, otherwise falls back to
    legacy column names.
    """
    if comb is not None:
        # Side-by-side shape with explicit data_ prefix
        for col in ["data_DP_mbar_corr", "data_DP_mbar"]:
            if col in comb.columns:
                s = pd.to_numeric(comb[col], errors="coerce").dropna()
                if s.size > 0:
                    return s
        # Vertical union shape
        for col in ["DP_mbar_corr", "DP_mbar"]:
            if col in comb.columns:
                df = comb
                if "source" in df.columns:
                    df = df[df["source"].astype(str).str.lower() == "overlay"]
                elif "is_reference" in df.columns:
                    df = df[df["is_reference"] == False]
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if s.size > 0:
                    return s
    if data is not None:
        for col in ["data_DP_mbar_corr", "data_DP_mbar"]:
            if col in data.columns:
                s = pd.to_numeric(data[col], errors="coerce").dropna()
                if s.size > 0:
                    return s
    return pd.Series(dtype=float)


def _recommended_band(outdir: Path, summary: dict) -> tuple[float | None, float | None, float | None]:
    """Return (lo, hi, coverage_frac) where lo/hi are mbar.

    Precedence:
      1) summary.json: ["transmitter"]["recommended_band_dp_mbar"] = [lo, hi]
      2) transmitter_lookup_data.csv DP percentiles P5–P95
    coverage_frac is fraction of overlay samples within [lo,hi] if data present.
    """

    lo = hi = cov = None
    try:
        band = (summary or {}).get("transmitter", {}).get("recommended_band_dp_mbar", None)
        if isinstance(band, (list, tuple)) and len(band) == 2:
            lo, hi = float(band[0]), float(band[1])
    except Exception:
        pass

    if lo is None or hi is None:
        data_csv = Path(outdir) / "transmitter_lookup_data.csv"
        if data_csv.exists():
            df = pd.read_csv(data_csv)
            if "data_DP_mbar" in df.columns and df["data_DP_mbar"].notna().any():
                dp = pd.to_numeric(df["data_DP_mbar"], errors="coerce").dropna().to_numpy()
                if dp.size >= 20:
                    lo = float(np.percentile(dp, 5))
                    hi = float(np.percentile(dp, 95))
                    cov = float(np.mean((dp >= lo) & (dp <= hi)))

    if cov is None and (lo is not None and hi is not None):
        try:
            df = pd.read_csv(Path(outdir) / "transmitter_lookup_data.csv")
            dp = pd.to_numeric(df["data_DP_mbar"], errors="coerce").dropna().to_numpy()
            cov = float(np.mean((dp >= lo) & (dp <= hi))) if dp.size else None
        except Exception:
            cov = None

    return lo, hi, cov


def _shade_recommended_band(ax: plt.Axes, outdir: Path, summary: dict, label_prefix: str = "Recommended") -> None:
    """Shade vertical DP band on current axes if available."""

    lo, hi, _ = _recommended_band(outdir, summary)
    if lo is None or hi is None or not np.isfinite([lo, hi]).all():
        return
    ax.axvspan(lo, hi, alpha=0.12, ec="none", zorder=0, label=f"{label_prefix} band {lo:.3g}–{hi:.3g} mbar")


def _fig_text(title: str, lines: list[str]) -> plt.Figure:
    """
    Single-page text with auto-fit:
      - tries (font, wrap) = (10pt,80) -> (9pt,88) -> (8pt,96)
      - ASCII-normalizes risky glyphs to avoid missing characters
    """
    # Pre-normalize once
    norm = []
    for raw in lines:
        safe = (raw.replace("√", "sqrt")
                    .replace("·", "*")
                    .replace("Δ", "d")
                    .replace("ρ", "rho")
                    .replace("²", "^2")
                    .replace("³", "^3")
                    .replace("μ", "u"))
        norm.append(safe)

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0,0,1,1]); ax.axis("off")
    # Header
    fig.text(0.08, 0.98, title, ha="left", va="top",
             fontsize=14, weight="bold", color=BRAND["accent"])
    fig.lines.append(plt.Line2D([0.08, 0.92], [0.955, 0.955],
                                lw=0.6, color="#CBD5E1"))
    left_margin = 0.07
    top_y, bot_y = 0.94, 0.08

    # Try a few (font, wrap, spacing) combos until it fits
    trials = [
        {"fs": 10, "wrap": 80, "step": 0.033, "extra": 0.012},
        {"fs":  9, "wrap": 88, "step": 0.030, "extra": 0.011},
        {"fs":  8, "wrap": 96, "step": 0.028, "extra": 0.010},
    ]
    chosen = None
    wrapped_variants = []
    for t in trials:
        ww = t["wrap"]
        wrapped = [textwrap.fill(s, width=ww) for s in norm]
        # estimate vertical space needed
        nlines = sum(max(1, w.count("\n")+1) for w in wrapped)
        # header consumes ~0.044; remaining height:
        avail = (top_y - 0.044) - bot_y
        need  = nlines * (t["step"] + t["extra"])  # conservative estimate
        wrapped_variants.append(wrapped)
        if need <= avail:
            chosen = t
            break
    if chosen is None:
        # fall back to last trial; it may still overrun slightly, but far less
        chosen = trials[-1]
    wrapped = wrapped_variants[trials.index(chosen)]

    y = top_y - 0.044
    for w in wrapped:
        ax.text(left_margin, y, w, va="top", ha="left",
                fontsize=chosen["fs"])
        # reduce step a touch if paragraph is a single short line
        lines_here = max(1, w.count("\n")+1)
        y -= chosen["step"] + chosen["extra"] * lines_here
        if y < bot_y:
            break
    return fig


def _fig_cover(outdir: Path, summary_path: Path) -> plt.Figure:
    s = {}
    try:
        s = json.loads(Path(summary_path).read_text())
    except Exception:
        pass
    meta_path = Path(outdir) / "transmitter_lookup_meta.json"
    flow_meta = {}
    try:
        flow_meta = json.loads(meta_path.read_text())
    except Exception:
        pass
    norm_meta = _load_json(Path(outdir) / "normalize_meta.json")

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    ax = fig.add_axes([0,0,1,1]); ax.axis("off")
    lines = []
    lines.append("KielProc — Mill PA Differential Validation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    if s:
        lines.append(f"Site: {s.get('site_name','')}")
        bp = s.get("baro_pa", None)
        if isinstance(bp, (int, float)): lines.append(f"Barometric pressure: {bp:.0f} Pa")
        lines.append(f"Input mode: {s.get('input_mode','')}")
        lines.append(f"Prepared input: {s.get('prepared_input_dir','')}")
        if s.get("beta") is not None or s.get("r") is not None:
            lines.append(f"β: {s.get('beta')}    r: {s.get('r')}")
    if norm_meta:
        lines.append(f"Barometric pressure used: {norm_meta.get('baro_pa_used','n/a')} Pa")
        srcs = norm_meta.get("p_abs_source", {})
        if srcs:
            lines.append("Static abs source: " + ", ".join(f"{k}:{v}" for k,v in srcs.items()))
        if norm_meta.get("replicate_layout"):
            lines.append(f"Ingest: {norm_meta['replicate_layout']}")
    if flow_meta:
        cal = flow_meta.get("calibration", {})
        lines.append("")
        lines.append(f"Season: {flow_meta.get('season','')}")
        if cal:
            lines.append(f"UIC K (t/h per sqrt(mbar)): {cal.get('K_uic','')}")
            lines.append(f"820 m (t/h/mbar): {cal.get('m_820','')}   c (t/h): {cal.get('c_820','')}")
            lines.append(f"Calibration source: {cal.get('source','')}")
    ax.text(0.08, 0.92, "\n".join(lines), va="top", ha="left", fontsize=12)
    ax.text(0.08, 0.06, "Generated by kielproc.run_easy.run_all()",
            fontsize=9, color=BRAND["muted"])
    return fig


def _summary_merged(outdir: Path, summary_path: Path) -> plt.Figure:
    """
    One compact page: Summary + Context & Method + Recommendations.
    Also fixes Piccolo mapping to multi-line bullets so it never overflows.
    """
    s = _load_json(summary_path)
    meta = _load_json(Path(outdir) / "transmitter_lookup_meta.json")

    # Calibration / season (prefer meta, fallback to summary)
    season = meta.get("season") or s.get("season") or ""
    cal = meta.get("calibration", {}) or {}
    K = cal.get("K_uic", s.get("K_uic"))
    m = cal.get("m_820", s.get("m_820"))
    c = cal.get("c_820", s.get("c_820"))
    rng = cal.get("range_mbar", s.get("dp_range_mbar"))
    baro = s.get("baro_pa", None)
    baro_line = f"{baro:.0f} Pa" if isinstance(baro, (int,float)) else "n/a"

    # ---------- Overlay stats ----------
    comb, data, ref = _load_tx_tables(outdir)
    overlay_dp = _overlay_dp_series(comb, data)
    n = int(overlay_dp.size)
    dp_min = float(overlay_dp.min()) if n > 0 else None
    dp_max = float(overlay_dp.max()) if n > 0 else None
    mean_abs = worst_abs = None
    df_err = comb if comb is not None else data
    if df_err is not None and {"data_Flow_UIC_tph","data_Flow_820_tph"}.issubset(df_err.columns):
        err = pd.to_numeric(df_err["data_Flow_820_tph"], errors="coerce") - pd.to_numeric(df_err["data_Flow_UIC_tph"], errors="coerce")
        err = err.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if err.size > 0:
            mean_abs = float(np.nanmean(np.abs(err)))
            worst_abs = float(np.nanmax(np.abs(err)))

    # crossover DP (informative)
    dp_cross = None
    try:
        if all(isinstance(v,(int,float)) for v in [K,m,c]) and K>0 and m>0:
            A = (m*m); B = (2*m*c - K*K); Cq = (c*c)
            disc = B*B - 4*A*Cq
            if disc >= 0:
                r1 = (-B + math.sqrt(disc)) / (2*A)
                r2 = (-B - math.sqrt(disc)) / (2*A)
                for r in (r1, r2):
                    if r and r > 0: dp_cross = float(r); break
    except Exception:
        pass

    # ---------- Recommendations (ideal local linearization over band) ----------
    def _fit_linear_L2(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        if x.size < 2:
            xm = float(np.nanmedian(x)) if x.size else 1.0
            ym = float(np.nanmedian(y)) if x.size else 0.0
            m_ = 0.0 if xm <= 0 else (ym/(2.0*xm)); c_ = ym - m_*xm
            return float(max(m_,0.0)), float(max(c_,0.0))
        X = np.c_[x, np.ones_like(x)]
        m_, c_ = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(max(m_,0.0)), float(max(c_,0.0))

    # operating band: prefer meta/summary, else compute from overlay
    ob = meta.get("operating_band_mbar") or s.get("operating_band_mbar") or {}
    p5 = ob.get("p5_mbar"); p95 = ob.get("p95_mbar")
    if not (isinstance(p5, (int, float)) and isinstance(p95, (int, float))):
        if n > 0:
            p5 = float(np.percentile(overlay_dp, 5.0))
            p95 = float(np.percentile(overlay_dp, 95.0))
            if p95 - p5 < 1e-6:
                p95 = p5 + 0.2
        else:
            p5 = p95 = None
    # fallbacks for downstream computations (use defaults if absent)
    lo = float(p5) if isinstance(p5, (int, float)) else 2.0
    hi = float(p95) if isinstance(p95, (int, float)) else 6.0
    lo = max(1e-6, lo); hi = max(lo + 1e-6, hi)
    mid = 0.5*(lo+hi)

    # reference function and helpers
    K0 = float(K or 0.0)
    def f(dp): return K0 * np.sqrt(np.clip(dp, 0.0, None))
    def fprime(dp): return (0.0 if dp <= 0 else K0 / (2.0*np.sqrt(dp)))

    # Tangent at DP_mid
    m_tan = fprime(mid)
    c_tan = f(mid) - m_tan*mid

    # Secant across [lo,hi]
    m_sec = (f(hi) - f(lo)) / (hi - lo)
    c_sec = f(lo) - m_sec*lo

    # L2 over band (uniform)
    gx = np.linspace(lo, hi, 800)
    yx = f(gx)
    m_l2, c_l2 = _fit_linear_L2(gx, yx)

    # Minimax (L∞) over band: 1D search over slope; optimal c centers sup error
    m_min = fprime(hi)  # smallest slope in band
    m_max = fprime(lo)  # largest slope in band
    m_grid = np.linspace(m_min, m_max, 400)
    def extremal_g(mv):
        # g(x) = m*x - f(x); extrema at lo, hi, and where f'(x)=m
        xstar = (K0/(2.0*mv))**2 if (mv > 0 and K0 > 0) else None
        xs = [lo, hi]
        if xstar is not None and lo <= xstar <= hi:
            xs.append(xstar)
        gvals = [mv*x - f(x) for x in xs]
        return xs[int(np.argmax(gvals))], xs[int(np.argmin(gvals))], float(np.max(gvals)), float(np.min(gvals))
    best = {"t": np.inf, "m": None, "c": None, "x_pos": None, "x_neg": None}
    for mv in m_grid:
        xp, xn, gp, gn = extremal_g(mv)
        # choose c to center sup error: max(e)= -min(e) -> c = - (gmax + gmin)/2
        cv = -0.5*(gp + gn)
        t = 0.5*(gp - gn)  # minimized sup |e|
        if t < best["t"]:
            best.update({"t": t, "m": float(mv), "c": float(cv), "x_pos": xp, "x_neg": xn})
    # refine around best slope
    if best["m"] is not None:
        m_lo = max(m_min, best["m"] - 0.1*(m_max-m_min))
        m_hi = min(m_max, best["m"] + 0.1*(m_max-m_min))
        for mv in np.linspace(m_lo, m_hi, 200):
            xp, xn, gp, gn = extremal_g(mv)
            cv = -0.5*(gp + gn); t = 0.5*(gp - gn)
            if t < best["t"]:
                best.update({"t": t, "m": float(mv), "c": float(cv), "x_pos": xp, "x_neg": xn})
    m_inf = float(best["m"]) if best["m"] is not None else m_l2
    c_inf = float(best["c"]) if best["c"] is not None else c_l2

    # error metrics (uniform grid on band)
    def _band_err(K_, m_, c_, lo_, hi_):
        gx = np.linspace(max(0.0, lo_), max(lo_, hi_), 1000)
        e = (m_*gx + c_) - (K_*np.sqrt(gx))
        return float(np.nanmean(np.abs(e))), float(np.nanmax(np.abs(e))), float(gx[int(np.nanargmax(np.abs(e)))])
    cur_mean, cur_worst, cur_wdp = _band_err(K0, float(m or 0.0), float(c or 0.0), lo, hi)
    tan_mean, tan_worst, tan_wdp = _band_err(K0, m_tan, c_tan, lo, hi)
    sec_mean, sec_worst, sec_wdp = _band_err(K0, m_sec, c_sec, lo, hi)
    l2_mean,  l2_worst,  l2_wdp  = _band_err(K0, m_l2,  c_l2,  lo, hi)
    inf_mean, inf_worst, inf_wdp = _band_err(K0, m_inf, c_inf, lo, hi)
    # recommended = minimax
    m_rec, c_rec = m_inf, c_inf

    def _cross(K_, m_, c_):
        try:
            A = m_*m_; B = 2*m_*c_ - K_*K_; Cq = c_*c_
            disc = B*B - 4*A*Cq
            if disc < 0: return None
            r1 = (-B + math.sqrt(disc))/(2*A); r2 = (-B - math.sqrt(disc))/(2*A)
            for r in (r1, r2):
                if r and r > 0: return float(r)
        except Exception: return None
        return None
    cross_rec = _cross(K0, m_rec, c_rec)

    # ---------- Build one compact page ----------
    # (header "Summary" provided separately by _fig_text)
    L: list[str] = []
    Ktxt = f"{K:.4f} t/h per sqrt(mbar)" if isinstance(K,(int,float)) else "n/a"
    prof = s.get("profile_meta") or {}
    fs  = prof.get("full_scale_tph")
    bu  = prof.get("bias_unit")
    extra = []
    if isinstance(fs,(int,float)):
        extra.append(f"FS={fs:.6g} t/h")
    if bu:
        extra.append(f"bias unit={bu}")
    extra_txt = ("   " + " ".join(extra)) if extra else ""
    L.append("Season: {}   K(UIC)={}   820 (configured): m={}  c={}   span={} mbar{}"
             .format(season or "n/a", Ktxt,
                     (f"{m:.6g}" if isinstance(m,(int,float)) else "n/a"),
                     (f"{c:.6g}" if isinstance(c,(int,float)) else "n/a"),
                     (f"{rng:.6g}" if isinstance(rng,(int,float)) else "n/a"),
                     extra_txt))
    L.append(f"Site: {s.get('site_name','')}")
    L.append(f"Barometric pressure: {baro_line}")
    label = "overlay"  # avoid ambiguity with 'current' (ohmic/time)
    if p5 is not None and p95 is not None:
        L.append(f"Operating band ({label}): {p5:.5g}–{p95:.5g} mbar (P5–P95)")
    else:
        L.append(f"Operating band ({label}): n/a (no overlay DP present)")
    rb_lo, rb_hi, rb_cov = _recommended_band(outdir, s)
    if rb_lo is not None and rb_hi is not None:
        cov_txt = f" — coverage {rb_cov*100:.1f}%" if rb_cov is not None else ""
        L.append(f"Recommended DP band: {rb_lo:.5g}–{rb_hi:.5g} mbar{cov_txt}")

    meta = _load_json(Path(outdir) / "normalize_meta.json")
    if meta and isinstance(meta.get("sanity"), dict):
        snt = meta["sanity"]
        L.append(
            f"Static sanity: median plane static = {snt.get('p_s_pa_median','n/a')} Pa ({snt.get('note','')})"
        )

    # Temperature & density (if present)
    rho = s.get("rho_kg_m3", None); rho_src = s.get("rho_source", None)
    if rho is None:
        try:
            duct = json.load(open(Path(outdir) / "duct_result.json"))
            rho = duct.get("rho_kg_m3", None); rho_src = duct.get("rho_source", rho_src)
        except Exception:
            pass
    tmeta = (s.get("thermo_source") or {}).get("thermo_choice", {})
    TK = s.get("T_K", None)
    t_src = tmeta.get("source", "unknown")
    fallback = bool(tmeta.get("fallback", False))
    L.append("Temperature: {} K ({})".format(f"{TK:.2f}" if isinstance(TK,(int,float)) else "n/a", t_src))
    if fallback or (isinstance(TK,(int,float)) and TK < 320.0):
        L.append("  Note: temperature selection fallback or near-ambient; verify thermo channel.")
    L.append("Density: {} kg/m³ ({})".format(f"{rho:.6g}" if isinstance(rho,(int,float)) else "n/a",
                                             rho_src if rho_src else "n/a"))
    if isinstance(rho, (int, float)) and (rho < 0.2 or rho > 2.0):
        L.append("WARNING: implausible density — check baro/T units.")
    if n > 0:
        L.append("")  # spacer
        L.append("Overlay (Piccolo-derived DP):")
        L.append(f"  • Samples: n={n}")
        L.append(f"  • DP band: {dp_min:.3f}–{dp_max:.3f} mbar")
        L.append(f"  • 820 vs UIC: mean abs error = {mean_abs:.3f} t/h; worst abs = {worst_abs:.3f} t/h")
        L.append("Piccolo mapping:")
        pic = (s.get('piccolo_info') or {})
        rng = pic.get('range_mbar', None)
        avgI = pic.get('avg_current_mA', None)
        impliedI = None
        if rng and dp_min is not None and dp_max is not None and rng > 0:
            I_lo = 4.0 + 16.0 * (dp_min / float(rng))
            I_hi = 4.0 + 16.0 * (dp_max / float(rng))
            impliedI = (I_lo, I_hi)
        if rng is not None:
            L.append(f"  • Range: {rng:.3f} mbar")
        if avgI is not None:
            L.append(f"  • Average current (workbook): {avgI:.4f} mA")
        if impliedI:
            L.append(f"  • Implied current from overlay DP: {impliedI[0]:.4f}–{impliedI[1]:.4f} mA")
    else:
        L.append("Overlay: not present (reference curves only).")
    # --- Plane↔Throat reconciliation ---
    rec = s.get("reconcile", {}) or {}
    if rec:
        dp_geom = rec.get("dp_pred_geom_mbar", None)
        dp_corr = rec.get("dp_pred_corr_mbar", None)
        dp_p50  = rec.get("dp_overlay_p50_mbar", None)
        dp_p5   = rec.get("dp_overlay_p5_mbar", None)
        dp_p95  = rec.get("dp_overlay_p95_mbar", None)
        C_f     = rec.get("C_f", 1.0)
        if isinstance(dp_geom, (int, float)) or isinstance(dp_corr, (int, float)):
            if all(isinstance(x, (int, float)) for x in [dp_p5, dp_p50, dp_p95]):
                L.append(f"Overlay Δp: P50 = {dp_p50:.4g} mbar; band {dp_p5:.4g}–{dp_p95:.4g} mbar")
        if isinstance(dp_geom, (int, float)):
            errg = rec.get("dp_error_geom_pct_vs_p50", None)
            L.append(
                f"Pred Δp (geom) = {dp_geom:.4g} mbar" +
                (f"  (error {errg:+.2f}%)" if isinstance(errg, (int, float)) else "")
            )
        if isinstance(dp_corr, (int, float)):
            errc = rec.get("dp_error_corr_pct_vs_p50", None)
            L.append(
                f"Pred Δp (reconciled, C_f={C_f:.3f}) = {dp_corr:.4g} mbar" +
                (f"  (error {errc:+.2f}%)" if isinstance(errc, (int, float)) else "")
            )
    if dp_cross is not None:
        L.append(f"Crossover (configured 820 = UIC): DP ~= {dp_cross:.3f} mbar")
    # Piccolo fit/residuals
    try:
        rec2 = (s or {}).get("reconcile", {}) or {}
        fit = rec2.get("piccolo_fit", {}) or {}
        if fit:
            L.append(
                f"Piccolo calibration: DP = {fit.get('a_mbar_per_mA','?'):.5g}·I + {fit.get('b_mbar','?'):.5g}  (mbar, mA)"
            )
            L.append(
                f"  Fit points: {fit.get('n_points','?')}, LRV={fit.get('lrv_mbar','?')} mbar, URV={fit.get('urv_mbar','?')} mbar"
            )
    except Exception:
        pass
    L.append("")  # spacer
    L.append("Context & Method:")
    L.append("  • UIC: Flow_UIC = K*sqrt(DP)  (K in t/h per sqrt(mbar), DP in mbar)")
    L.append("  • 820: Flow_820 = m*DP + c   (m in t/h/mbar, c in t/h)")
    L.append("  • Overlay DP from Piccolo 4-20 mA; baro/PA T from workbook when available.")

    blocks = [L]

    rec = [
        "Recommendations (local linearization over operating band):",
        f"  Band: {lo:.3f}–{hi:.3f} mbar   (mid {mid:.3f})",
        f"  820 (configured): m={(f'{m:.6g}' if isinstance(m,(int,float)) else 'n/a')}  c={(f'{c:.6g}' if isinstance(c,(int,float)) else 'n/a')}",
        f"  Proposed 820 (minimax L_inf): m*={m_rec:.4f}  c*={c_rec:.4f}  (dm={(m_rec-(m or 0.0)):+.4f}  dc={(c_rec-(c or 0.0)):+.4f})",
        f"    Errors — Configured: mean={cur_mean:.3f}, worst={cur_worst:.3f}@{cur_wdp:.3f} mbar",
        f"              L_inf:  mean={inf_mean:.3f}, worst={inf_worst:.3f}@{inf_wdp:.3f} mbar",
        f"    Alternatives — Tangent@mid: m={m_tan:.4f}  c={c_tan:.4f}  (worst={tan_worst:.3f})",
        f"                   Secant:      m={m_sec:.4f}  c={c_sec:.4f}  (worst={sec_worst:.3f})",
        f"                   L2:          m={m_l2:.4f}  c={c_l2:.4f}    (worst={l2_worst:.3f})",
    ]
    if cross_rec is not None:
        rec.append(f"  Proposed crossover (820=UIC): DP ~= {cross_rec:.3f} mbar")
    blocks.append(rec)

    # flattened lines go into one auto-fit page
    flat = blocks and sum(([b] if isinstance(b, str) else [*b] for b in blocks), [])
    return _fig_text("Summary", flat)


# ---------- NEW: Inputs snapshot ----------
def _page_inputs_snapshot(outdir: Path, summary_path: Path) -> plt.Figure:
    s = json.loads(Path(summary_path).read_text())
    L = []
    L.append("Inputs snapshot:")
    man = s.get("inputs_manifest") or []
    if not man:
        L.append("  • No manifest recorded.")
    else:
        for m in man:
            line = f"  • {m.get('path','?')}"
            if "bytes" in m:
                line += f"  ({m['bytes']} bytes)"
            if "mtime_utc" in m:
                ts = datetime.datetime.utcfromtimestamp(int(m["mtime_utc"]))\
                     .strftime("%Y-%m-%d %H:%M:%SZ")
                line += f"  mtime_utc={ts}"
            if "sha256" in m:
                line += f"  sha256={m['sha256'][:12]}"
            if "error" in m:
                line += f"  [{m['error']}]"
            L.append(line)
    beta = s.get("beta"); A1=None; At=None; rho=s.get("rho_kg_m3")
    duct = Path(outdir) / "duct_result.json"
    if duct.exists():
        dj = json.loads(duct.read_text())
        A1 = dj.get("area_m2")
        At = dj.get("At_m2") or ((beta**2)*A1 if beta and A1 else None)
    L.append("")
    L.append("Geometry & density:")
    if beta is not None:
        L.append(f"  • beta={beta:.4f}")
    if A1   is not None:
        L.append(f"  • A1={A1:.4f} m^2")
    if At   is not None:
        L.append(f"  • At={At:.4f} m^2")
    if rho  is not None:
        L.append(f"  • rho={rho:.4f} kg/m^3")
    return _fig_text("Inputs", L)


# ---------- NEW: Data quality & exclusions ----------
def _page_data_quality(outdir: Path, summary_path: Path) -> plt.Figure:
    s = json.loads(Path(summary_path).read_text())
    qc = s.get("qc", {}) or {}
    xi_meta = (s.get("profile_xi", {}) or {}).get("meta", {})

    L = []
    L.append("Data quality and exclusions:")
    comb = Path(outdir) / "transmitter_lookup_combined.csv"
    if comb.exists():
        df = pd.read_csv(comb)
        L.append(
            f"Overlay CSV: rows={len(df)}  NaNs per column: " +
            ", ".join(f"{c}:{df[c].isna().sum()}" for c in df.columns)
        )
    else:
        L.append("Overlay CSV not present.")
    per = Path(outdir) / "per_port.csv"
    if per.exists():
        d = pd.read_csv(per)
        L.append(
            f"Per-port CSV: rows={len(d)}  NaNs per column: " +
            ", ".join(f"{c}:{d[c].isna().sum()}" for c in d.columns)
        )
        if "weight" in d.columns:
            disabled = d.index[(d["weight"].fillna(0) <= 0)].tolist()
            if disabled:
                L.append(f"Disabled/zero-weight ports: {disabled}")
    else:
        L.append("Per-port CSV not present.")

    if qc.get("enabled", False):
        L.append("")
        L.append("QC summary:")
        L.append(
            "  Adjacency co-variation: {} (min r = {:.2f})".format(
                "non-uniform" if qc.get("nonuniform_adjacency", False) else "ok",
                float(qc.get("adjacency_min_r", 0.0)),
            )
        )
        trav = qc.get("traverse", {}) or {}
        if trav:
            L.append(
                "  Traverse bias: avg={:+.3f}   max_port={:+.3f}".format(
                    float(trav.get("bias_avg", 0.0)),
                    float(trav.get("max_port_bias", 0.0)),
                )
            )
            bad = trav.get("bad_ports")
            if bad:
                L.append(f"    Bad ports: {bad}")
        if xi_meta:
            used = [
                f"P{k}:{('xi' if v.get('n_xi', 0) > 0 else 'time')}"
                for k, v in xi_meta.items()
            ]
            L.append("  ξ-profile aggregation: " + ", ".join(used))

    return _fig_text("Data quality", L)


# ---------- NEW: Operating band & recommendations table + verdict ----------
def _page_band_table_and_verdict(outdir: Path, summary_path: Path) -> plt.Figure:
    s = json.loads(Path(summary_path).read_text())
    meta = _load_json(Path(outdir) / "transmitter_lookup_meta.json")
    cal = (meta.get("calibration") or {})
    # Prefer meta; fallback to summary for all Tx params
    K = float(cal.get("K_uic") or s.get("K") or s.get("K_uic") or 0.0)
    m = (cal.get("m_820") or s.get("m_820"))
    c = (cal.get("c_820") or s.get("c_820"))
    # operating band: prefer stored, else compute from overlay, else default
    ob = meta.get("operating_band_mbar") or s.get("operating_band_mbar") or {}
    lo = ob.get("p5_mbar"); hi = ob.get("p95_mbar")
    if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
        comb, data, ref = _load_tx_tables(outdir)
        overlay_dp = _overlay_dp_series(comb, data)
        if overlay_dp.size > 0:
            lo = float(np.percentile(overlay_dp, 5.0))
            hi = float(np.percentile(overlay_dp, 95.0))
        else:
            lo, hi = 2.0, 6.0
    lo = max(1e-6, lo); hi = max(lo + 1e-6, hi); mid = 0.5*(lo+hi)
    def f(dp):
        return K*np.sqrt(np.clip(dp,0,None))
    def fprime(dp):
        return (0.0 if dp<=0 else K/(2.0*np.sqrt(dp)))
    m_tan = fprime(mid); c_tan = f(mid)-m_tan*mid
    m_sec = (f(hi)-f(lo))/(hi-lo); c_sec = f(lo)-m_sec*lo
    gx = np.linspace(lo, hi, 1000); yx = f(gx)
    X = np.c_[gx, np.ones_like(gx)]; m_l2, c_l2 = np.linalg.lstsq(X, yx, rcond=None)[0]
    m_min, m_max = fprime(hi), fprime(lo)
    def extremal_g(mv):
        xstar = (K/(2.0*mv))**2 if (mv>0 and K>0) else None
        xs = [lo, hi] + ([xstar] if (xstar is not None and lo<=xstar<=hi) else [])
        g = [mv*x - f(x) for x in xs]
        iM, im = int(np.argmax(g)), int(np.argmin(g))
        return xs[iM], xs[im], float(max(g)), float(min(g))
    best = {"t": np.inf, "m": None, "c": None}
    for mv in np.linspace(m_min, m_max, 400):
        xp, xn, gp, gn = extremal_g(mv)
        cv = -0.5*(gp+gn); t = 0.5*(gp-gn)
        if t < best["t"]:
            best.update({"t": t, "m": float(mv), "c": float(cv)})
    m_inf = float(best["m"]) if best["m"] is not None else float(m_l2)
    c_inf = float(best["c"]) if best["c"] is not None else float(c_l2)
    def band_err(mv, cv):
        e = (mv*gx + cv) - yx
        return float(np.mean(np.abs(e))), float(np.max(np.abs(e)))
    cur_mean, cur_worst = band_err(float(m or 0.0), float(c or 0.0))
    tan_mean, tan_worst = band_err(m_tan, c_tan)
    sec_mean, sec_worst = band_err(m_sec, c_sec)
    l2_mean,  l2_worst  = band_err(m_l2,  c_l2)
    inf_mean, inf_worst = band_err(m_inf, c_inf)
    acc = s.get("acceptance") or {}
    thr_mean = float(acc.get("mean_abs_tph_max", 0.5))
    thr_worst = float(acc.get("worst_abs_tph_max", 1.0))
    verdict_ok = (inf_mean <= thr_mean) and (inf_worst <= thr_worst)
    L = []
    L.append("Operating band and recommendations (table):")
    L.append(f"  Band: {lo:.3f}–{hi:.3f} mbar   mid={mid:.3f} mbar")
    L.append(
        f"  Thresholds: mean|e|<= {thr_mean:.3f} t/h, worst|e|<= {thr_worst:.3f} t/h  ->  Verdict: {'PASS' if verdict_ok else 'FAIL'}"
    )
    # Use unambiguous wording: 'Configured'
    L.append("  820 (configured):  m={:.4f}  c={:.4f}    mean|e|={:.3f}  worst|e|={:.3f}".format(
        float(m or 0.0), float(c or 0.0), cur_mean, cur_worst))
    L.append("  Tangent:  m={:.4f}  c={:.4f}    mean|e|={:.3f}  worst|e|={:.3f}".format(
        m_tan, c_tan, tan_mean, tan_worst))
    L.append("  Secant:   m={:.4f}  c={:.4f}    mean|e|={:.3f}  worst|e|={:.3f}".format(
        m_sec, c_sec, sec_mean, sec_worst))
    L.append("  L2:       m={:.4f}  c={:.4f}    mean|e|={:.3f}  worst|e|={:.3f}".format(
        m_l2,  c_l2,  l2_mean,  l2_worst))
    L.append("  L_inf*:   m*={:.4f} c*={:.4f}   mean|e|={:.3f}  worst|e|={:.3f}   [Proposed]".format(
        m_inf, c_inf, inf_mean, inf_worst))
    return _fig_text("Operating band & recommendations", L)


# ---------- NEW: Before/after error bars ----------
def _fig_error_bars(outdir: Path, summary_path: Path) -> plt.Figure:
    s = json.loads(Path(summary_path).read_text())
    meta = _load_json(Path(outdir) / "transmitter_lookup_meta.json")
    cal = (meta.get("calibration") or {})
    K = float(cal.get("K_uic") or s.get("K") or s.get("K_uic") or 0.0)
    m = (cal.get("m_820") or s.get("m_820"))
    c = (cal.get("c_820") or s.get("c_820"))
    # operating band: prefer stored, else compute from overlay, else default
    ob = meta.get("operating_band_mbar") or s.get("operating_band_mbar") or {}
    lo = ob.get("p5_mbar"); hi = ob.get("p95_mbar")
    if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
        comb, data, ref = _load_tx_tables(outdir)
        overlay_dp = _overlay_dp_series(comb, data)
        if overlay_dp.size > 0:
            lo = float(np.percentile(overlay_dp, 5.0))
            hi = float(np.percentile(overlay_dp, 95.0))
        else:
            lo, hi = 2.0, 6.0
    lo = max(1e-6, lo); hi = max(lo + 1e-6, hi)
    gx = np.linspace(lo, hi, 1000); yx = K*np.sqrt(gx)
    X = np.c_[gx, np.ones_like(gx)]; m_l2, c_l2 = np.linalg.lstsq(X, yx, rcond=None)[0]
    m_min, m_max = K/(2.0*np.sqrt(hi)), K/(2.0*np.sqrt(lo))
    best = {"t": np.inf, "m": None, "c": None}
    def extremal(mv):
        xstar = (K/(2.0*mv))**2 if (mv>0 and K>0) else None
        xs = [lo, hi] + ([xstar] if (xstar is not None and lo<=xstar<=hi) else [])
        g = [mv*x - K*np.sqrt(x) for x in xs]
        gmax, gmin = max(g), min(g)
        return -0.5*(gmax+gmin), 0.5*(gmax-gmin)
    for mv in np.linspace(m_min, m_max, 400):
        cv, t = extremal(mv)
        if t < best["t"]:
            best.update({"t": t, "m": float(mv), "c": float(cv)})
    def stats(mv, cv):
        e = (mv*gx + cv) - yx
        return float(np.mean(np.abs(e))), float(np.max(np.abs(e)))
    cur = stats(float(m or 0.0), float(c or 0.0))
    l2  = stats(float(m_l2), float(c_l2))
    linf= stats(float(best["m"]), float(best["c"]))
    labels = ["820 (configured)", "L2", "L_inf (Proposed)"]
    means  = [cur[0], l2[0], linf[0]]
    worsts = [cur[1], l2[1], linf[1]]
    fig = plt.figure(figsize=(11.69, 8.27)); ax = fig.add_subplot(111)
    x = np.arange(len(labels))
    ax.bar(x-0.18, means,  width=0.36, label="mean|e| (t/h)",  edgecolor="#111827", linewidth=0.5)
    ax.bar(x+0.18, worsts, width=0.36, label="worst|e| (t/h)", edgecolor="#111827", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Error (t/h)"); ax.set_title("Error over operating band")
    ax.set_axisbelow(True); ax.grid(True, alpha=0.35); ax.legend()
    return fig


# ---------- NEW: Overlay coverage histogram ----------
def _fig_overlay_hist(outdir: Path) -> plt.Figure | None:
    p = Path(outdir) / "transmitter_lookup_combined.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "data_DP_mbar" not in df.columns:
        return None
    vals = pd.to_numeric(df["data_DP_mbar"], errors="coerce").dropna().to_numpy(float)
    fig = plt.figure(figsize=(11.69, 8.27)); ax = fig.add_subplot(111)
    ax.hist(vals, bins=40, alpha=0.9, edgecolor="#111827", linewidth=0.5)
    ax.set_xlabel("DP (mbar)"); ax.set_ylabel("Count")
    ax.set_title("Overlay DP coverage (samples)")
    ax.set_axisbelow(True); ax.margins(x=0.02); ax.grid(True, alpha=0.35)
    return fig


# ---------- NEW: Density & geometry provenance ----------
def _page_density_geometry(outdir: Path, summary_path: Path) -> plt.Figure:
    s = json.loads(Path(summary_path).read_text())
    L = []
    L.append("Density & geometry provenance:")
    L.append(f"  • baro_pa = {s.get('baro_pa','n/a')}")
    T_K = s.get("T_K"); thermo = s.get("thermo_source") or {}
    if isinstance(T_K, (int,float)):
        src = thermo.get("source","unknown"); cell = thermo.get("cell",""); label = thermo.get("label","")
        L.append(f"  • T_K = {T_K:.2f} K ({T_K-273.15:.1f} C)  [source={src} {cell} '{label}']")
    rho = s.get("rho_kg_m3"); rsrc = s.get("rho_source","unknown")
    if isinstance(rho,(int,float)):
        L.append(f"  • rho = {rho:.4f} kg/m^3  [{rsrc}]")
        if rho < 0.2 or rho > 2.0:
            L.append("  • WARNING: implausible density — check baro/T units.")
    beta = s.get("beta")
    duct = Path(outdir) / "duct_result.json"
    A1=At=None
    if duct.exists():
        dj = json.loads(duct.read_text())
        A1 = dj.get("area_m2"); At = dj.get("At_m2") or ((beta**2)*A1 if beta and A1 else None)
    if beta is not None:
        L.append(f"  • beta = {beta:.4f}")
    if A1   is not None:
        L.append(f"  • A1 = {A1:.4f} m^2")
    if At   is not None:
        L.append(f"  • At = {At:.4f} m^2")
    # plane static provenance from normalization meta
    norm_meta = Path(outdir) / "normalize_meta.json"
    if norm_meta.exists():
        nm = json.loads(norm_meta.read_text())
        L.append(
            f"  • plane static source: {nm.get('p_abs_source','n/a')}  (baro_pa_used={nm.get('baro_pa_used','n/a')})"
        )
        snty = nm.get("sanity", {})
        if snty:
            L.append(
                f"  • sanity: median p_s={snty.get('p_s_pa_median','n/a')} Pa; median T={snty.get('T_K_median','n/a')} K"
            )
    return _fig_text("Density & Geometry", L)


# ---------- NEW: Port map & weights ----------
def _page_port_weights(outdir: Path) -> plt.Figure:
    per = Path(outdir) / "per_port.csv"
    L = ["Port map & weights:"]
    if not per.exists():
        L.append("  • per_port.csv not present.")
        return _fig_text("Port map", L)
    df = pd.read_csv(per)
    cols = [c for c in df.columns if c.lower() in ("port","id","name")]
    ids = df[cols[0]].astype(str).tolist() if cols else [str(i) for i in range(len(df))]
    w = df["weight"].fillna(1.0/len(df)) if "weight" in df.columns else pd.Series([1.0/len(df)]*len(df))
    qs = df["q_s_pa"] if "q_s_pa" in df.columns else pd.Series([float("nan")]*len(df))
    for i,(pid, wi, qi) in enumerate(zip(ids, w, qs)):
        L.append(f"  • port {pid:>3}: weight={wi:.4f}   q_s_pa={('%.2f'%qi) if pd.notna(qi) else 'n/a'}")
    return _fig_text("Port map", L)


# ---------- NEW: Transmitter details ----------
def _page_tx_details(outdir: Path, summary_path: Path) -> plt.Figure:
    s = json.loads(Path(summary_path).read_text())
    meta = _load_json(Path(outdir) / "transmitter_lookup_meta.json")
    cal = (meta.get("calibration") or {})
    m = (cal.get("m_820") or s.get("m_820"))
    c = (cal.get("c_820") or s.get("c_820"))
    span = cal.get("range_mbar")
    if span is None:
        for name in ["transmitter_lookup_combined.csv", "transmitter_lookup_reference.csv"]:
            for base in [Path(outdir) / "_integrated", Path(outdir)]:
                p = base / name
                if p.exists():
                    df = pd.read_csv(p)
                    if "range_mbar" in df.columns:
                        vals = pd.to_numeric(df["range_mbar"], errors="coerce").dropna()
                        if len(vals):
                            span = float(vals.iloc[0])
                            break
            if span is not None:
                break
    L = []
    L.append("Transmitter details:")
    L.append(f"  • 820 (configured): m={m if m is not None else 'n/a'}  c={c if c is not None else 'n/a'}")
    L.append(f"  • DP span (range_mbar): {span if span is not None else 'n/a'}")
    L.append("  • Proposed: see Operating band & recommendations page (L_inf row).")
    return _fig_text("Transmitter", L)


# ---------- NEW: Repro & version ----------
def _page_repro_version(outdir: Path, summary_path: Path) -> plt.Figure:
    s = json.loads(Path(summary_path).read_text())
    L = []
    L.append("Reproducibility & version:")
    L.append(f"  • Run ID: {s.get('run_id','n/a')}")
    L.append(f"  • Pipeline git SHA: {s.get('git_sha','n/a')}")
    L.append(f"  • Config hash: {s.get('config_hash','n/a')}")
    acc = s.get("acceptance") or {}
    L.append(
        f"  • Acceptance thresholds: mean|e| <= {acc.get('mean_abs_tph_max',0.5)} t/h; worst|e| <= {acc.get('worst_abs_tph_max',1.0)} t/h"
    )
    return _fig_text("Repro & version", L)


def _fig_per_port_table(per_port_csv: Path) -> plt.Figure | None:
    p = Path(per_port_csv)
    if not p.exists(): return None
    df = pd.read_csv(p)
    # Choose a compact subset if available
    prefer = [c for c in ["Port","VP_pa_mean","T_C_mean","Static_abs_pa_mean","q_s_pa","w_s","w_t"] if c in df.columns]
    view = df[prefer] if prefer else df
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.92]); ax.axis("off")
    ax.set_title("Per-port summary", loc="left")
    tbl = ax.table(cellText=view.values[:20], colLabels=view.columns, loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.2)
    return fig


def _fig_flow_reference_with_overlay(outdir: Path) -> plt.Figure | None:
    ref = Path(outdir) / "transmitter_lookup_reference.csv"
    if not ref.exists():
        return None
    dref = pd.read_csv(ref)
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.plot(dref["ref_DP_mbar"], dref["ref_Flow_UIC_tph"], label="UIC (√DP) – reference")
    ax.plot(dref["ref_DP_mbar"], dref["ref_Flow_820_tph"], label="820 (linear) – reference")

    s_all = _load_json(Path(outdir) / "summary.json")
    cal = (s_all or {}).get("calibration", {}) or {}
    K = cal.get("K_uic", (s_all or {}).get("K_uic"))

    def _uic(dp: np.ndarray) -> np.ndarray:
        if not isinstance(K, (int, float)):
            return np.full_like(np.asarray(dp, float), np.nan)
        return float(K) * np.sqrt(np.clip(np.asarray(dp, float), 0.0, None))

    data = Path(outdir) / "transmitter_lookup_data.csv"
    if data.exists():
        dd = pd.read_csv(data)
        dp_raw = pd.to_numeric(dd.get("data_DP_mbar_raw", pd.Series([])), errors="coerce").dropna().to_numpy()
        dp_corr = pd.to_numeric(dd.get("data_DP_mbar_corr", pd.Series([])), errors="coerce").dropna().to_numpy()
        if dp_corr.size == 0 and "data_DP_mbar" in dd.columns:
            dp_corr = pd.to_numeric(dd["data_DP_mbar"], errors="coerce").dropna().to_numpy()
        if dp_raw.size:
            ax.scatter(dp_raw, _uic(dp_raw), s=6, alpha=0.25, marker="o", label="Overlay (raw)")
        if dp_corr.size:
            ax.scatter(dp_corr, _uic(dp_corr), s=9, alpha=0.85, marker=".", label="Overlay (corrected)")

    rec = (s_all or {}).get("reconcile", {}) or {}
    dp_geom = rec.get("dp_pred_geom_mbar", None)
    dp_corr = rec.get("dp_pred_corr_mbar", None)
    if isinstance(dp_geom, (int, float)):
        ax.axvline(dp_geom, linestyle=":", linewidth=1.1, alpha=0.9, label=f"Pred Δp (geom) {dp_geom:.3g} mbar")
    if isinstance(dp_corr, (int, float)) and (dp_corr != dp_geom):
        ax.axvline(dp_corr, linestyle="--", linewidth=1.1, alpha=0.9, label=f"Pred Δp (reconciled) {dp_corr:.3g} mbar")
    ax.set_xlabel("DP (mbar)"); ax.set_ylabel("Flow (t/h)")
    ax.set_title("Flow lookup: reference (constant) with data overlay")
    ax.set_axisbelow(True); ax.grid(True, alpha=0.35)
    _shade_recommended_band(ax, outdir, s_all, label_prefix="Recommended")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best")
    return fig


def _fig_flow_reference_zoom(outdir: Path) -> plt.Figure | None:
    """Same as the full plot, but X-zoomed to the overlay DP band (+ padding).
    Skips if no overlay data is present."""
    ref = Path(outdir) / "transmitter_lookup_reference.csv"
    data = Path(outdir) / "transmitter_lookup_data.csv"
    if not (ref.exists() and data.exists()):
        return None
    dref = pd.read_csv(ref)
    dd = pd.read_csv(data)
    dp_raw = pd.to_numeric(dd.get("data_DP_mbar_raw", pd.Series([])), errors="coerce").dropna()
    dp_corr = pd.to_numeric(dd.get("data_DP_mbar_corr", pd.Series([])), errors="coerce").dropna()
    if dp_corr.empty and "data_DP_mbar" in dd.columns:
        dp_corr = pd.to_numeric(dd["data_DP_mbar"], errors="coerce").dropna()
    if dp_corr.empty and dp_raw.empty:
        return None
    dp_all = pd.concat([dp_corr, dp_raw]) if not dp_raw.empty else dp_corr
    dp_min = float(dp_all.min()); dp_max = float(dp_all.max()); span = dp_max - dp_min
    pad = max(0.10 * span, 0.05)  # at least ±0.05 mbar
    lo = max(0.0, dp_min - pad)
    ref_max = float(dref["ref_DP_mbar"].max())
    hi = min(ref_max, dp_max + pad)
    if hi <= lo:
        return None
    zr = dref[(dref["ref_DP_mbar"] >= lo) & (dref["ref_DP_mbar"] <= hi)]
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.plot(zr["ref_DP_mbar"], zr["ref_Flow_UIC_tph"], label="UIC (√DP) – reference", zorder=1)
    ax.plot(zr["ref_DP_mbar"], zr["ref_Flow_820_tph"], label="820 (linear) – reference", zorder=1)

    s_all: dict = {}
    cal: dict = {}
    try:
        s_all = _load_json(Path(outdir) / "summary.json")
        cal = (s_all or {}).get("calibration", {}) or {}
    except Exception:
        s_all = {}
        cal = {}
    K = cal.get("K_uic", (s_all or {}).get("K_uic"))

    def _uic(dp: np.ndarray) -> np.ndarray:
        if not isinstance(K, (int, float)):
            return np.full_like(np.asarray(dp, float), np.nan)
        return float(K) * np.sqrt(np.clip(np.asarray(dp, float), 0.0, None))

    if dp_raw.size:
        ax.scatter(dp_raw, _uic(dp_raw), s=20, alpha=0.25, marker="o", label="Overlay (raw)", zorder=5)
    if dp_corr.size:
        ax.scatter(dp_corr, _uic(dp_corr), s=20, alpha=0.85, marker=".", label="Overlay (corrected)", zorder=5)

    ax.set_xlim(lo, hi)
    ax.set_xlabel("DP (mbar)"); ax.set_ylabel("Flow (t/h)")
    title = f"Flow lookup — overlay zoom (n={len(dp_all)}, DP {dp_min:.3f}–{dp_max:.3f} mbar)"
    rng = I_lo = I_hi = None
    try:
        rng = (s_all.get("piccolo_info") or {}).get("range_mbar", None)
        if rng and rng > 0:
            I_lo = 4.0 + 16.0 * (dp_min / float(rng))
            I_hi = 4.0 + 16.0 * (dp_max / float(rng))
            title += f"\n(Piccolo range {rng:.3f} mbar → I {I_lo:.4f}–{I_hi:.4f} mA)"
    except Exception:
        pass
    rec = (s_all or {}).get("reconcile", {}) or {}
    dp_geom = rec.get("dp_pred_geom_mbar", None)
    dp_corr_pred = rec.get("dp_pred_corr_mbar", None)
    if isinstance(dp_geom, (int, float)):
        ax.axvline(dp_geom, linestyle=":", linewidth=1.1, alpha=0.9, label=f"Pred Δp (geom) {dp_geom:.3g} mbar")
    if isinstance(dp_corr_pred, (int, float)) and (dp_corr_pred != dp_geom):
        ax.axvline(dp_corr_pred, linestyle="--", linewidth=1.1, alpha=0.9, label=f"Pred Δp (reconciled) {dp_corr_pred:.3g} mbar")
    ax.set_title(title); ax.set_axisbelow(True); ax.grid(True, alpha=0.35)
    _shade_recommended_band(ax, outdir, s_all, label_prefix="Recommended")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best")
    if I_lo is not None and I_hi is not None:
        fig.text(0.5, 0.03, f"Implied Piccolo I from overlay DP: {I_lo:.4f}–{I_hi:.4f} mA", ha="center", fontsize=10)
    return fig


def _fig_venturi_curve(outdir: Path) -> plt.Figure | None:
    """
    Venturi Δp vs Mass Flow with explicit units and on-plot metadata.
    """
    outdir = Path(outdir)
    vr = outdir / "venturi_result.json"
    dr = outdir / "duct_result.json"
    flow_kg_s = None; dp_pa = None
    beta = None; A1 = None; At = None; rho = None
    if vr.exists():
        d = json.loads(vr.read_text())
        flow_kg_s = np.asarray(d.get("flow_kg_s") or d.get("flow"), float)
        dp_pa     = np.asarray(d.get("dp_pa") or d.get("delta_p_pa"), float)
        beta = d.get("beta"); A1 = d.get("A1_m2") or d.get("As_m2"); At = d.get("At_m2"); rho = d.get("rho_kg_m3")
    elif dr.exists():
        d = json.loads(dr.read_text())
        # Try to reconstruct curve if geometry is present
        beta = d.get("beta"); A1 = d.get("area_m2"); At = d.get("At_m2")
        if (beta is not None) and (A1 is not None) and (At is None):
            try: At = (float(beta)**2) * float(A1)
            except: pass
        rho = d.get("rho_kg_m3")
        m0  = d.get("m_dot_kg_s"); dp0 = d.get("delta_p_vent_est_pa")
        if (m0 and rho and At and beta and dp0):
            m0 = float(m0); rho = float(rho); At = float(At); beta = float(beta)
            flow_kg_s = np.linspace(max(0.1, 0.25*m0), 2.0*m0, 200)
            dp_pa = (1.0 - beta**4) * (flow_kg_s**2) / (2.0 * rho * (At**2))
        else:
            return None
    else:
        return None
    if flow_kg_s is None or dp_pa is None:
        return None
    flow_tph = flow_kg_s * 3.6
    fig = plt.figure(figsize=(11.69, 8.27)); ax = fig.add_subplot(111)
    ax.plot(flow_tph, dp_pa, label="Model: Δp = (1−β⁴)·ṁ²/(2·ρ·Aₜ²)")
    ax.set_title("Venturi Δp vs Mass Flow")
    ax.set_xlabel("Mass flow (t/h)")
    ax.set_ylabel("Venturi Δp (Pa)")
    ax.grid(True, linestyle="--", alpha=0.4); ax.legend()
    # On-plot metadata (geometry & density)
    meta = []
    if beta is not None: meta.append(f"β = {float(beta):.4f}")
    if A1   is not None: meta.append(f"A₁ = {float(A1):.4f} m²")
    if At   is not None: meta.append(f"Aₜ = {float(At):.4f} m²")
    if rho  is not None: meta.append(f"ρ = {float(rho):.4f} kg/m³")
    if meta:
        ax.text(0.98, 0.98, "\n".join(meta), ha="right", va="top", transform=ax.transAxes,
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9))
    if (rho is not None) and (rho < 0.2 or rho > 2.0):
        ax.text(0.02, 0.02, "WARNING: ρ looks implausible — check baro/T units",
                ha="left", va="bottom", transform=ax.transAxes, fontsize=9, color="crimson")
    return fig


def _profiles_page(outdir: Path) -> Optional[plt.Figure]:
    import glob
    files = sorted(glob.glob(str(Path(outdir) / "profiles" / "P*_profile.csv")))
    if not files:
        return None
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 6.5), constrained_layout=True)
    axes = axes.ravel()
    for ax, fn in zip(axes, files[:8]):
        df = pd.read_csv(fn)
        xi = pd.to_numeric(df.get("xi"), errors="coerce")
        qs0 = pd.to_numeric(df.get("q_s_median"), errors="coerce")
        qss = pd.to_numeric(df.get("q_s_smoothed"), errors="coerce")
        if xi.notna().any() and qs0.notna().any():
            ax.plot(xi, qs0, linewidth=1.0, label="median q_s(ξ)")
        if xi.notna().any() and qss.notna().any():
            ax.plot(xi, qss, linewidth=1.0, linestyle="--", label="smoothed")
        ax.set_title(Path(fn).stem.replace('_profile',''), fontsize=9)
        ax.set_xlabel("ξ"); ax.set_ylabel("q_s [Pa]")
        ax.grid(True, alpha=0.2); ax.legend(fontsize=7)
    return fig


def _fig_setpoints(outdir: Path) -> plt.Figure | None:
    csv = Path(outdir) / "transmitter_setpoints.csv"
    if not csv.exists(): return None
    df = pd.read_csv(csv)
    needed = {"UIC_percent","Y820"}
    if not needed.issubset(df.columns): return None
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.plot(df.index, df["UIC_percent"], label="UIC %")
    ax.plot(df.index, df["Y820"], label="820")
    ax.set_xlabel("Sample index"); ax.set_ylabel("Output")
    ax.set_title("Transmitter outputs (from logger)"); ax.grid(True, linestyle="--", alpha=0.4); ax.legend()
    return fig


def build_run_report_pdf(
    outdir: Path,
    summary_path: Path,
    filename: str = "RunReport.pdf",
) -> Path:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / filename
    with PdfPages(pdf_path) as pdf:
        # Cover
        pdf.savefig(_fig_cover(outdir, summary_path)); plt.close()
        # Single merged page (Summary + Context & Method + Recommendations)
        f = _summary_merged(outdir, summary_path)
        if f: pdf.savefig(f); plt.close()
        summary_path = Path(summary_path)
        # 1) Inputs snapshot
        pdf.savefig(_page_inputs_snapshot(outdir, summary_path)); plt.close()
        # 2) Data quality & exclusions
        pdf.savefig(_page_data_quality(outdir, summary_path)); plt.close()
        # 3) Operating band table & verdict
        pdf.savefig(_page_band_table_and_verdict(outdir, summary_path)); plt.close()
        # 4) Before/after error bars
        pdf.savefig(_fig_error_bars(outdir, summary_path)); plt.close()
        # 5) Overlay coverage histogram
        f = _fig_overlay_hist(outdir)
        if f is not None: pdf.savefig(f); plt.close()
        # 6) Density & geometry provenance
        pdf.savefig(_page_density_geometry(outdir, summary_path)); plt.close()
        # 7) Port map & weights
        pdf.savefig(_page_port_weights(outdir)); plt.close()
        # 8) Transmitter details
        pdf.savefig(_page_tx_details(outdir, summary_path)); plt.close()
        # 9) Repro & version
        pdf.savefig(_page_repro_version(outdir, summary_path)); plt.close()
        # Per-port table
        f = _fig_per_port_table(outdir / "per_port.csv")
        if f: pdf.savefig(f); plt.close()
        # Flow reference (constant) + data overlay (optional)
        f = _fig_flow_reference_with_overlay(outdir)
        if f: pdf.savefig(f); plt.close()
        # Flow reference + overlay (zoomed to overlay region)
        f = _fig_flow_reference_zoom(outdir)
        if f: pdf.savefig(f); plt.close()
        # Venturi curve (optional)
        f = _fig_venturi_curve(outdir)
        if f: pdf.savefig(f); plt.close()
        # Profiles page (optional)
        f = _profiles_page(outdir)
        if f: pdf.savefig(f); plt.close()
        # Setpoints plot (optional)
        f = _fig_setpoints(outdir)
        if f: pdf.savefig(f); plt.close()
    return pdf_path


__all__ = ["build_run_report_pdf"]

