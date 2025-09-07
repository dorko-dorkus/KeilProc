
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import pandas as pd
import numpy as np
from .io import load_legacy_excel, load_logger_csv, unify_schema
from .physics import rho_from_pT, map_qs_to_qt, venturi_dp_from_qt
from .aggregate import integrate_run, RunConfig
from .translate import compute_translation_table, apply_translation
from .lag import shift_series
from .report import write_summary_tables, plot_alignment
from .legacy_results import ResultsConfig, compute_results as compute_legacy_results
from .run_easy import run_easy_legacy
from .presets import PRESETS

def build_parser():
    p = argparse.ArgumentParser(prog="kielproc", description="Kiel + wall-static baseline & legacy translation")
    sub = p.add_subparsers(dest="cmd", required=True)

    oc = sub.add_parser(
        "one-click",
        help="Parse → Integrate → Map → Fit → Translate → Report legacy worksheets",
    )
    oc.add_argument("src", type=Path, help="Workbook file or directory of workbooks")
    oc.add_argument("--site", default="DefaultSite", choices=PRESETS.keys())
    oc.add_argument("--baro", type=float, default=None, help="Override barometric pressure in Pa")
    oc.add_argument("--stamp", type=str, default=None, help="Override run stamp YYYYMMDD_HHMM (NZT)")
    oc.add_argument("--out", type=Path, default=None, help="Output base directory (RUN_* will be created here)")
    oc.add_argument("--bundle", action="store_true", help="Zip run directory into RUN_*__bundle.zip")
    oc.add_argument("--strict", action="store_true", help="Fail fast on missing/invalid stages")

    i0 = sub.add_parser("results", help="Report legacy-style results from a logger CSV")
    i0.add_argument("--csv", required=True, help="Input logger CSV path")
    i0.add_argument("--config", help="JSON file with ResultsConfig fields")
    i0.add_argument("--json-out", help="Optional path to write results as JSON")
    i0.add_argument("--csv-out", help="Optional path to write results as CSV")

    i1 = sub.add_parser(
        "map",
        help="Map verification-plane static to throat Δp_vent for comparison",
    )
    i1.add_argument("--csv", required=True)
    i1.add_argument("--qs-col", required=True, help="Verification-plane dynamic qs column (Pa)")
    i1.add_argument("--r", required=True, type=float, help="Area ratio r = As/At")
    i1.add_argument("--beta", required=True, type=float, help="Venturi beta = dt/D1")
    i1.add_argument("--sampling-hz", required=False, type=float, help="DCS/logger sampling rate (Hz)")
    i1.add_argument("--out", required=True)

    i2 = sub.add_parser("fit", help="Fit α/β translation from CSV blocks")
    i2.add_argument("--blocks", nargs="+", required=True, help="Pairs: name=path.csv")
    i2.add_argument("--ref-col", default="mapped_ref")
    i2.add_argument("--piccolo-col", default="piccolo")
    i2.add_argument("--lambda-ratio", type=float, default=1.0)
    i2.add_argument("--max-lag", type=int, default=300, help="Lag scan bound (samples)")
    i2.add_argument("--sampling-hz", type=float, default=None,
                    help="Sampling rate [Hz]; enables lag_seconds = lag_samples/sampling_hz and plateau windows in seconds")
    # Optional steady-plateau gating
    i2.add_argument("--plateau-window-s", type=float, default=None,
                    help="Sliding-window size [s] for steady-plateau SD/mean check")
    i2.add_argument("--plateau-sd-frac", type=float, default=None,
                    help="Max allowed (SD/mean) within the window for both reference and piccolo")
    # Optional Piccolo scaling/QA from mA
    i2.add_argument("--piccolo-ma-col", type=str, default=None, help="If provided, convert this 4–20 mA column to Pa")
    i2.add_argument("--piccolo-range-mbar", type=float, default=6.7, help="Transmitter range plate [mbar]; default 6.7")
    i2.add_argument("--pN-col", default="pN")
    i2.add_argument("--pS-col", default="pS")
    i2.add_argument("--pE-col", default="pE")
    i2.add_argument("--pW-col", default="pW")
    i2.add_argument("--q-mean-col", default="q_mean")
    i2.add_argument("--qa-gate-opp", type=float, default=None,
                    help="Max allowed Δ_opp; gate if exceeded")
    i2.add_argument("--qa-gate-w", type=float, default=None,
                    help="Max allowed W index; gate if exceeded")
    i2.add_argument("--outdir", required=True)

    i3 = sub.add_parser(
        "translate",
        help="Translate legacy piccolo using α/β to overlay on mapped reference",
    )
    i3.add_argument("--csv", required=True)
    i3.add_argument("--alpha", type=float, required=True)
    i3.add_argument("--beta", type=float, required=True)
    i3.add_argument("--in-col", default="piccolo")
    i3.add_argument("--out-col", default="piccolo_translated")
    i3.add_argument("--out", required=True)

    p4 = sub.add_parser(
        "integrate-ports",
        help="Integrate port CSVs into duct results",
    )
    p4.add_argument(
        "--run-dir",
        required=True,
        help="Directory containing port CSVs (filenames should include P1..P8 or variants like PORT 1)",
    )
    p4.add_argument("--duct-height", type=float, required=True)
    p4.add_argument("--duct-width", type=float, required=True)
    p4.add_argument("--baro", type=float, default=None, help="Barometric [Pa] to combine with gauge Static if Baro column absent")
    p4.add_argument("--weights-json", type=str, default=None,
                    help='Optional JSON mapping, e.g. {"P1":0.125,...}. '
                         'If omitted, auto-uses <run-dir>/weights.json when present.')
    p4.add_argument("--replicate-strategy", choices=["mean","last"], default="mean")
    p4.add_argument("--area-ratio", type=float, default=None, help="Downstream-to-throat area ratio r = A_s/A_t for q_t mapping")
    p4.add_argument("--beta", type=float, default=None, help="Venturi diameter ratio β=d_t/D1 for Δp_vent estimate")
    p4.add_argument("--file-glob", default="*.csv", help="Custom glob if needed (default *.csv)")
    p4.add_argument("--viz", action="store_true", help="Render velocity heatmap png")
    p4.add_argument("--viz-height-bins", type=int, default=50)
    p4.add_argument("--viz-clip", default="2,98", help="percentiles low,high e.g. 2,98")
    p4.add_argument("--viz-interp", choices=["nearest", "linear", "cubic"], default="nearest")
    return p

def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.cmd == "one-click":
        out, summary, artifacts = run_easy_legacy(
            Path(a.src),
            PRESETS[a.site],
            baro_override_Pa=a.baro,
            run_stamp=a.stamp,
            output_base=a.out,
            strict=a.strict,
        )
        bundle_zip = None
        if a.bundle:
            bundle_zip = shutil.make_archive(str(out) + "__bundle", "zip", root_dir=out)
        res = {
            "ok": True,
            "out_dir": str(out),
            "summary": summary,
            "artifacts": artifacts,
        }
        if bundle_zip:
            res["bundle_zip"] = bundle_zip
        print(json.dumps(res))
    elif a.cmd == "results":
        cfg_dict = {}
        if a.config:
            with open(a.config) as fh:
                cfg_dict = json.load(fh)
        cfg = ResultsConfig(**cfg_dict)
        res = compute_legacy_results(a.csv, cfg)
        if a.json_out:
            Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
            with open(a.json_out, "w") as fh:
                json.dump(res, fh, indent=2)
        if a.csv_out:
            Path(a.csv_out).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([res]).to_csv(a.csv_out, index=False)
        print(json.dumps(res, indent=2))
    elif a.cmd == "map":
        df = pd.read_csv(a.csv)
        qt = map_qs_to_qt(df[a.qs_col].to_numpy(float), r=a.r, rho_t_over_rho_s=1.0)
        dpv = venturi_dp_from_qt(qt, beta=a.beta)
        out = df.copy()
        out["qt"] = qt
        out["dp_vent"] = dpv
        if a.sampling_hz:
            n = len(out); out["Sample"] = np.arange(n); out["Time_s"] = out["Sample"]/float(a.sampling_hz)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False)
        print(json.dumps({"mapped_csv": str(Path(a.out))}))
    elif a.cmd == "fit":
        blocks = {}
        for spec in a.blocks:
            name, p = spec.split("=", 1)
            blocks[name] = pd.read_csv(p)
        # Optional Piccolo mA → Pa conversion & QA marks
        if a.piccolo_ma_col:
            for name, df in blocks.items():
                if a.piccolo_ma_col not in df.columns:
                    raise SystemExit(f"--piccolo-ma-col={a.piccolo_ma_col} not found in {name}")
                ma = pd.to_numeric(df[a.piccolo_ma_col], errors="coerce").to_numpy(float)
                bad = (ma <= 0.0) | (ma < 4.0) | (ma > 20.0)
                df["piccolo_ma_bad"] = bad.astype(int)
                piccolo_mbar = (ma - 4.0) / 16.0 * float(a.piccolo_range_mbar)
                df[a.piccolo_col] = 100.0 * piccolo_mbar
        # Optional steady-plateau gate (sliding window SD/mean)
        if a.plateau_window_s and a.plateau_sd_frac and a.sampling_hz:
            win = max(int(round(a.plateau_window_s * a.sampling_hz)), 1)
            for name, df in list(blocks.items()):
                def max_sd_over_mean(x):
                    x = pd.to_numeric(x, errors="coerce").to_numpy(float)
                    n = len(x)
                    if n < win:
                        return float("inf")
                    best = 0.0
                    for i in range(0, n - win + 1):
                        w = x[i:i+win]
                        m = np.nanmean(w)
                        s = np.nanstd(w)
                        if m != 0:
                            best = max(best, s/abs(m))
                    return best
                mref = max_sd_over_mean(df[a.ref_col])
                mpic = max_sd_over_mean(df[a.piccolo_col])
                df["plateau_sd_over_mean_ref"] = mref
                df["plateau_sd_over_mean_pic"] = mpic
                if (mref > a.plateau_sd_frac) or (mpic > a.plateau_sd_frac):
                    del blocks[name]
        per_block, pooled = compute_translation_table(blocks, ref_key=a.ref_col, picc_key=a.piccolo_col,
                                                      lambda_ratio=a.lambda_ratio, max_lag=a.max_lag)
        if a.sampling_hz:
            per_block["lag_seconds"] = per_block["lag_samples"] / float(a.sampling_hz)
        # Compute QA indices for each block
        from .qa import qa_indices
        qa_rows = []
        for name, df in blocks.items():
            pN = df[a.pN_col].mean()
            pS = df[a.pS_col].mean()
            pE = df[a.pE_col].mean()
            pW = df[a.pW_col].mean()
            q = df[a.q_mean_col].mean()
            d_opp, W = qa_indices(pN, pS, pE, pW, q)
            ok = True
            if a.qa_gate_opp is not None and d_opp > a.qa_gate_opp:
                ok = False
            if a.qa_gate_w is not None and W > a.qa_gate_w:
                ok = False
            qa_rows.append(dict(block=name, delta_opp=d_opp, W=W, qa_pass=ok))
        qa_df = pd.DataFrame(qa_rows)

        if not qa_df["qa_pass"].all():
            outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
            write_summary_tables(outdir, qa_df, None)
            raise SystemExit("Ring QA failed; aborting fit")

        # Merge QA with translation results
        per_block = per_block.merge(qa_df, on="block", how="left")
        outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
        write_summary_tables(outdir, per_block, pooled)
        png = ""
        if blocks and not per_block.empty:
            name0 = list(blocks.keys())[0]
            d0 = blocks[name0]
            lag0 = int(per_block.loc[per_block["block"]==name0, "lag_samples"].iloc[0])
            # Advance piccolo so that it aligns with reference for plotting
            # Positive lag -> piccolo lags the reference.  Shift piccolo forward
            # (left) by ``lag0`` samples for overlay.
            picc_shift = shift_series(d0[a.piccolo_col].to_numpy(float), -lag0)
            t = d0["Time_s"] if "Time_s" in d0 else np.arange(len(d0))
            png = plot_alignment(outdir, t, d0[a.ref_col], d0[a.piccolo_col], picc_shift, title=f"Alignment {name0}", stem=f"align_{name0}")
        result = {
            "per_block_csv": str(outdir/"alpha_beta_by_block.csv"),
            "per_block_json": str(outdir/"alpha_beta_by_block.json"),
            "pooled_csv": str(outdir/"alpha_beta_pooled.csv") if (outdir/"alpha_beta_pooled.csv").exists() else "",
            "pooled_json": str(outdir/"alpha_beta_pooled.json") if (outdir/"alpha_beta_pooled.json").exists() else "",
            "align_png": png,
            "blocks_info": per_block.to_dict(orient="records"),
        }
        print(json.dumps(result, indent=2))
        for row in per_block.itertuples():
            print(f"block={row.block} τ={row.lag_samples} r_peak={row.r_peak:.3f}")
    elif a.cmd == "integrate-ports":
        # Auto-discover per-run artifacts
        weights: dict[str, float] | None = None
        if a.weights_json:
            weights = json.loads(Path(a.weights_json).read_text())
        else:
            cand = Path(a.run_dir) / "weights.json"
            if cand.exists():
                weights = json.loads(cand.read_text())
        area_ratio = a.area_ratio
        beta = a.beta
        gpath = Path(a.run_dir) / "geometry.json"
        if gpath.exists() and (area_ratio is None or beta is None):
            g = json.loads(gpath.read_text())
            area_ratio = area_ratio if area_ratio is not None else (g.get("r") or g.get("area_ratio"))
            beta = beta if beta is not None else (g.get("\u03b2") or g.get("beta"))
        cfg = RunConfig(
            height_m=a.duct_height,
            width_m=a.duct_width,
            weights=weights,
            replicate_strategy=a.replicate_strategy,
        )
        res = integrate_run(
            Path(a.run_dir),
            cfg,
            file_glob=a.file_glob,
            baro_cli_pa=a.baro,
            area_ratio=area_ratio,
            beta=beta,
        )
        outdir = Path(a.run_dir) / "_integrated"
        outdir.mkdir(parents=True, exist_ok=True)
        res["per_port"].to_csv(outdir / "per_port.csv", index=False)
        (outdir / "duct_result.json").write_text(json.dumps(res["duct"], indent=2))
        (outdir / "normalize_meta.json").write_text(json.dumps(res["normalize_meta"], indent=2))
        summary = {
            "per_port_csv": str(outdir / "per_port.csv"),
            "duct_result_json": str(outdir / "duct_result.json"),
            "normalize_meta_json": str(outdir / "normalize_meta.json"),
            "files_used": res["files"],
            "pairs_used": [(pid, pf.name) for pid, pf in res.get("pairs", [])],
        }
        if res.get("skipped"):
            summary["skipped_files"] = [{"file": f, "reason": r} for f, r in res["skipped"]]
        if a.viz:
            from .visuals import render_velocity_heatmap
            lo, hi = (float(x) for x in a.viz_clip.split(","))
            heatmap_path = render_velocity_heatmap(
                outdir=outdir,
                pairs=res.get("pairs", []),
                baro_cli_pa=a.baro,
                height_bins=a.viz_height_bins,
                clip_percentiles=(lo, hi),
                interp=a.viz_interp,
            )
            summary["heatmap_velocity_png"] = str(heatmap_path)
        print(json.dumps(summary, indent=2))
    elif a.cmd == "translate":
        df = pd.read_csv(a.csv)
        out = apply_translation(df, a.alpha, a.beta, src_col=a.in_col, out_col=a.out_col)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False)
        print(json.dumps({"translated_csv": str(Path(a.out))}))

if __name__ == "__main__":
    main()
