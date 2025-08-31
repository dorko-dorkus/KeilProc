
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from .io import load_legacy_excel, load_logger_csv, unify_schema
from .physics import rho_from_pT, map_qs_to_qt, venturi_dp_from_qt
from .translate import compute_translation_table, apply_translation
from .lag import estimate_lag_xcorr, advance_series
from .report import write_summary_tables, plot_alignment

def build_parser():
    p = argparse.ArgumentParser(prog="kielproc", description="Kiel + wall-static baseline & legacy translation")
    sub = p.add_subparsers(dest="cmd", required=True)

    i1 = sub.add_parser("map", help="Map verification-plane Kiel/static to throat Δp_vent for comparison")
    i1.add_argument("--csv", required=True)
    i1.add_argument("--qs-col", required=True, help="Verification-plane dynamic qs column (Pa)")
    i1.add_argument("--r", required=True, type=float, help="Area ratio r = As/At")
    i1.add_argument("--beta", required=True, type=float, help="Venturi beta = dt/D1")
    i1.add_argument("--sampling-hz", required=False, type=float, help="DCS/logger sampling rate (Hz)")
    i1.add_argument("--out", required=True)

    i2 = sub.add_parser("fit", help="Fit alpha/beta translation from blocks of CSVs")
    i2.add_argument("--blocks", nargs="+", required=True, help="Pairs: name=path.csv")
    i2.add_argument("--ref-col", default="mapped_ref")
    i2.add_argument("--piccolo-col", default="piccolo")
    i2.add_argument("--lambda-ratio", type=float, default=1.0)
    i2.add_argument("--max-lag", type=int, default=300)
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

    i3 = sub.add_parser("translate", help="Apply alpha/beta to legacy piccolo to overlay on mapped reference")
    i3.add_argument("--csv", required=True)
    i3.add_argument("--alpha", type=float, required=True)
    i3.add_argument("--beta", type=float, required=True)
    i3.add_argument("--in-col", default="piccolo")
    i3.add_argument("--out-col", default="piccolo_translated")
    i3.add_argument("--out", required=True)
    return p

def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.cmd == "map":
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
        print(f"Wrote {a.out}")
    elif a.cmd == "fit":
        blocks = {}
        for spec in a.blocks:
            name, p = spec.split("=", 1)
            blocks[name] = pd.read_csv(p)
        per_block, pooled = compute_translation_table(blocks, ref_key=a.ref_col, picc_key=a.piccolo_col,
                                                      lambda_ratio=a.lambda_ratio, max_lag=a.max_lag)
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
            files = write_summary_tables(outdir, qa_df, None)
            print("\n".join(str(f) for f in files))
            raise SystemExit("Ring QA failed; aborting fit")

        # Merge QA with translation results
        per_block = per_block.merge(qa_df, on="block", how="left")
        outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
        files = write_summary_tables(outdir, per_block, pooled)
        if blocks and not per_block.empty:
            name0 = list(blocks.keys())[0]
            d0 = blocks[name0]
            lag0 = int(per_block.loc[per_block["block"]==name0, "lag_samples"].iloc[0])
            # Advance piccolo so that it aligns with reference for plotting
            picc_shift = advance_series(d0[a.piccolo_col].to_numpy(float), lag0)
            t = d0["Time_s"] if "Time_s" in d0 else np.arange(len(d0))
            png = plot_alignment(outdir, t, d0[a.ref_col], d0[a.piccolo_col], picc_shift, title=f"Alignment {name0}", stem=f"align_{name0}")
            files.append(png)
        print("\n".join(str(f) for f in files))
    elif a.cmd == "translate":
        df = pd.read_csv(a.csv)
        out = apply_translation(df, a.alpha, a.beta, src_col=a.in_col, out_col=a.out_col)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False)
        print(f"Wrote {a.out}")

if __name__ == "__main__":
    main()
