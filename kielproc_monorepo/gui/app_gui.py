#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated GUI for Kiel + Wall-Static Baseline & Legacy Translation
- Runs standalone; no patching of existing files required.
- Provides Physics/Translation tab (mapping, α/β, flow map, polar slice).
- Keeps the door open to embed/launch your legacy explorer on a separate tab.
"""

import sys
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog

# Ensure repo root is importable when running as "python gui/app_gui.py"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Adapter functions into the kielproc backend
from kielproc_gui_adapter import (
    map_verification_plane, map_from_tot_and_static,
    fit_alpha_beta, translate_piccolo,
    generate_flow_map_from_csv, generate_polar_slice_from_csv
)
from kielproc.qa import DEFAULT_W_MAX, DEFAULT_DELTA_OPP_MAX

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kiel/Piccolo Processor — Integrated GUI")
        self.geometry("980x720")
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        self._build()

    def _build(self):
        self.nb = ttk.Notebook(self); self.nb.pack(fill="both", expand=True)
        self.tab_phys = ttk.Frame(self.nb); self.nb.add(self.tab_phys, text="Physics / Translation")
        self._build_phys(self.tab_phys)

    def _build_phys(self, frm):
        pad = {"padx": 6, "pady": 4}
        row = 0

        # Mapping section
        ttk.Label(frm, text="Mapping: qs→qt→Δp_vent").grid(row=row, column=0, sticky="w", **pad); row+=1
        self.var_qs_csv = tk.StringVar()
        ttk.Label(frm, text="Verification CSV").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_qs_csv, width=56).grid(row=row, column=1, **pad)
        ttk.Button(frm, text="Browse", command=lambda: self._pick(self.var_qs_csv, [("CSV","*.csv")])).grid(row=row, column=2, **pad)
        row+=1

        self.var_use_tot_stat = tk.BooleanVar(value=True)
        self.var_tot_col = tk.StringVar(value="p_t_kiel")
        self.var_ps_col_avg = tk.StringVar(value="p_s_avg")
        self.var_qs_col = tk.StringVar(value="qs_verif")

        ttk.Checkbutton(frm, text="Compute qs from (total − static_avg)", variable=self.var_use_tot_stat).grid(row=row, column=0, columnspan=2, sticky="w", **pad); row+=1
        ttk.Label(frm, text="total col").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_tot_col, width=20).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="static_avg col").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_ps_col_avg, width=20).grid(row=row, column=3, sticky="w", **pad); row+=1
        ttk.Label(frm, text="(or) qs column").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_qs_col, width=20).grid(row=row, column=1, sticky="w", **pad); row+=1

        self.var_r = tk.DoubleVar(value=1.10)
        self.var_beta = tk.DoubleVar(value=0.55)
        self.var_fs = tk.DoubleVar(value=10.0)
        self.var_outdir = tk.StringVar(value=str(Path.cwd()/ "outputs"))

        ttk.Label(frm, text="r = As/At").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_r, width=10).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="beta = dt/D1").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_beta, width=10).grid(row=row, column=3, sticky="w", **pad); row+=1
        ttk.Label(frm, text="Sampling (Hz)").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_fs, width=10).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="Output dir").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_outdir, width=36).grid(row=row, column=3, sticky="w", **pad); row+=1
        ttk.Button(frm, text="Map qs→qt→Δp_vent", command=self._do_map).grid(row=row, column=1, sticky="w", **pad); row+=1

        ttk.Separator(frm, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", **pad); row+=1

        # Translation section
        ttk.Label(frm, text="Legacy piccolo translation (lag removal + Deming α,β)").grid(row=row, column=0, sticky="w", **pad); row+=1
        self.var_blocks = tk.StringVar()
        self.var_refcol = tk.StringVar(value="mapped_ref")
        self.var_piccol = tk.StringVar(value="piccolo")
        self.var_lambda = tk.DoubleVar(value=1.0)
        self.var_maxlag = tk.IntVar(value=300)
        self.var_pN = tk.StringVar(value="pN")
        self.var_pS = tk.StringVar(value="pS")
        self.var_pE = tk.StringVar(value="pE")
        self.var_pW = tk.StringVar(value="pW")
        self.var_qmean = tk.StringVar(value="q_mean")
        self.var_gate_opp = tk.StringVar(value=str(DEFAULT_DELTA_OPP_MAX))
        self.var_gate_w = tk.StringVar(value=str(DEFAULT_W_MAX))
        ttk.Label(frm, text="Replicate blocks (name=path.csv, comma-separated)").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_blocks, width=64).grid(row=row, column=1, columnspan=3, sticky="w", **pad); row+=1
        ttk.Label(frm, text="ref col").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_refcol, width=16).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="piccolo col").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_piccol, width=16).grid(row=row, column=3, sticky="w", **pad); row+=1
        ttk.Label(frm, text="λ ratio").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_lambda, width=10).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="max lag (samples)").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_maxlag, width=10).grid(row=row, column=3, sticky="w", **pad); row+=1
        ttk.Label(frm, text="pN col").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_pN, width=10).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="pS col").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_pS, width=10).grid(row=row, column=3, sticky="w", **pad); row+=1
        ttk.Label(frm, text="pE col").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_pE, width=10).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="pW col").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_pW, width=10).grid(row=row, column=3, sticky="w", **pad); row+=1
        ttk.Label(frm, text="q_mean col").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_qmean, width=10).grid(row=row, column=1, sticky="w", **pad); row+=1

        qa_frm = ttk.LabelFrame(frm, text="QA thresholds")
        qa_frm.grid(row=row, column=0, columnspan=4, sticky="ew", **pad); row+=1
        ttk.Label(qa_frm, text="Δ_opp max").grid(row=0, column=0, sticky="e", **pad)
        ttk.Entry(qa_frm, textvariable=self.var_gate_opp, width=10).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(qa_frm, text="W max").grid(row=1, column=0, sticky="e", **pad)
        ttk.Entry(qa_frm, textvariable=self.var_gate_w, width=10).grid(row=1, column=1, sticky="w", **pad)

        ttk.Button(frm, text="Fit α,β (with lag removal)", command=self._do_fit).grid(row=row, column=1, sticky="w", **pad); row+=1

        # Apply translation
        ttk.Label(frm, text="Apply translation to a legacy CSV").grid(row=row, column=0, sticky="w", **pad); row+=1
        self.var_tr_csv = tk.StringVar()
        self.var_alpha  = tk.DoubleVar(value=1.0)
        self.var_beta   = tk.DoubleVar(value=0.0)
        ttk.Label(frm, text="Legacy CSV").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_tr_csv, width=56).grid(row=row, column=1, **pad)
        ttk.Button(frm, text="Browse", command=lambda: self._pick(self.var_tr_csv, [("CSV","*.csv")])).grid(row=row, column=2, **pad); row+=1
        ttk.Label(frm, text="alpha").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_alpha, width=10).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="beta").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_beta, width=10).grid(row=row, column=3, sticky="w", **pad); row+=1
        ttk.Button(frm, text="Apply translation → CSV", command=self._do_translate).grid(row=row, column=1, sticky="w", **pad); row+=1

        ttk.Separator(frm, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", **pad); row+=1

        # Flow map
        ttk.Label(frm, text="Flow map (legacy-style static deviations)").grid(row=row, column=0, sticky="w", **pad); row+=1
        self.var_flow_csv = tk.StringVar()
        self.var_geom_csv = tk.StringVar()
        self.var_theta_col = tk.StringVar(value="theta_deg")
        self.var_plane_col = tk.StringVar(value="z_m")
        self.var_ps_col    = tk.StringVar(value="ps")
        self.var_norm_col  = tk.StringVar(value="")

        ttk.Label(frm, text="Data CSV").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_flow_csv, width=56).grid(row=row, column=1, **pad)
        ttk.Button(frm, text="Browse", command=lambda: self._pick(self.var_flow_csv, [("CSV","*.csv")])).grid(row=row, column=2, **pad); row+=1
        ttk.Label(frm, text="Geometry CSV (opt)").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_geom_csv, width=56).grid(row=row, column=1, **pad)
        ttk.Button(frm, text="Browse", command=lambda: self._pick(self.var_geom_csv, [("CSV","*.csv")])).grid(row=row, column=2, **pad); row+=1
        ttk.Label(frm, text="θ column").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_theta_col, width=16).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="z/plane column").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_plane_col, width=16).grid(row=row, column=3, sticky="w", **pad); row+=1
        ttk.Label(frm, text="ps column").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_ps_col, width=16).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="normalize by (opt)").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_norm_col, width=16).grid(row=row, column=3, sticky="w", **pad); row+=1
        ttk.Button(frm, text="Generate Flow Map", command=self._do_flowmap).grid(row=row, column=1, sticky="w", **pad); row+=1

        # Polar slice
        ttk.Separator(frm, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", **pad); row+=1
        ttk.Label(frm, text="Polar cross-section (single plane)").grid(row=row, column=0, sticky="w", **pad); row+=1
        self.var_polar_plane = tk.StringVar(value="0.0")
        ttk.Label(frm, text="Plane value (z or index)").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_polar_plane, width=16).grid(row=row, column=1, sticky="w", **pad)
        ttk.Button(frm, text="Generate Polar Slice", command=self._do_polar).grid(row=row, column=2, sticky="w", **pad); row+=1

        # Log box
        self.txt = tk.Text(frm, height=12)
        self.txt.grid(row=row, column=0, columnspan=4, sticky="nsew", **pad)
        frm.grid_rowconfigure(row, weight=1); frm.grid_columnconfigure(1, weight=1)

    def _pick(self, var, patterns):
        p = filedialog.askopenfilename(title="Choose file", filetypes=patterns)
        if p: var.set(p)

    def log(self, msg):
        self.txt.insert("end", str(msg)+"\n"); self.txt.see("end")

    # Handlers
    def _do_map(self):
        try:
            outdir = Path(self.var_outdir.get()); outdir.mkdir(parents=True, exist_ok=True)
            out = outdir/"mapped_ref.csv"
            if self.var_use_tot_stat.get():
                mapped = map_from_tot_and_static(
                    Path(self.var_qs_csv.get()),
                    self.var_tot_col.get(),
                    self.var_ps_col_avg.get(),
                    float(self.var_r.get()), float(self.var_beta.get()),
                    float(self.var_fs.get()) if self.var_fs.get()>0 else None,
                    out
                )
            else:
                mapped = map_verification_plane(
                    Path(self.var_qs_csv.get()), self.var_qs_col.get(),
                    float(self.var_r.get()), float(self.var_beta.get()),
                    float(self.var_fs.get()) if self.var_fs.get()>0 else None,
                    out
                )
            self.log(f"[OK] Wrote {mapped}")
        except Exception as e:
            self.log(f"[ERROR] map: {e}\n{traceback.format_exc()}")

    def _do_fit(self):
        try:
            block_specs = {}
            txt = self.var_blocks.get().strip()
            for item in txt.split(","):
                if not item.strip(): continue
                name, path = item.split("=", 1)
                block_specs[name.strip()] = Path(path.strip())
            out = Path(self.var_outdir.get()); out.mkdir(parents=True, exist_ok=True)
            qa_opp = float(self.var_gate_opp.get()) if self.var_gate_opp.get().strip() else None
            qa_w = float(self.var_gate_w.get()) if self.var_gate_w.get().strip() else None
            res = fit_alpha_beta(
                block_specs,
                self.var_refcol.get(),
                self.var_piccol.get(),
                float(self.var_lambda.get()),
                int(self.var_maxlag.get()),
                out,
                pN_col=self.var_pN.get(),
                pS_col=self.var_pS.get(),
                pE_col=self.var_pE.get(),
                pW_col=self.var_pW.get(),
                q_mean_col=self.var_qmean.get(),
                qa_gate_opp=qa_opp,
                qa_gate_w=qa_w,
            )
            self.log("[OK] Fitted α,β. Outputs: " + "; ".join([f"{k}={v}" for k, v in sorted(res.items()) if k != "blocks_info" and v]))
            for info in res.get("blocks_info", []):
                self.log(f"block={info['block']} τ={info['lag_samples']} r_peak={info['r_peak']:.3f}")
        except Exception as e:
            self.log(f"[ERROR] fit: {e}\n{traceback.format_exc()}")

    def _do_translate(self):
        try:
            out = Path(self.var_outdir.get())/"translated.csv"
            res = translate_piccolo(Path(self.var_tr_csv.get()), float(self.var_alpha.get()),
                                    float(self.var_beta.get()), self.var_piccol.get(), "piccolo_translated", out)
            self.log(f"[OK] Wrote {res}")
        except Exception as e:
            self.log(f"[ERROR] translate: {e}\n{traceback.format_exc()}")

    def _do_flowmap(self):
        try:
            outdir = Path(self.var_outdir.get()); outdir.mkdir(parents=True, exist_ok=True)
            res = generate_flow_map_from_csv(
                Path(self.var_flow_csv.get()),
                self.var_theta_col.get(),
                self.var_plane_col.get(),
                self.var_ps_col.get(),
                outdir,
                Path(self.var_geom_csv.get()) if self.var_geom_csv.get() else None,
                agg="median",
                normalize_by_col=self.var_norm_col.get() if self.var_norm_col.get() else None
            )
            self.log("[OK] Flow map: " + "; ".join(f"{k}={v}" for k,v in res.items()))
        except Exception as e:
            self.log(f"[ERROR] flowmap: {e}\n{traceback.format_exc()}")

    def _do_polar(self):
        try:
            outdir = Path(self.var_outdir.get()); outdir.mkdir(parents=True, exist_ok=True)
            res = generate_polar_slice_from_csv(
                Path(self.var_flow_csv.get()),
                self.var_theta_col.get(),
                self.var_plane_col.get(),
                self.var_ps_col.get(),
                float(self.var_polar_plane.get()),
                outdir,
                Path(self.var_geom_csv.get()) if self.var_geom_csv.get() else None,
                normalize_by_col=self.var_norm_col.get() if self.var_norm_col.get() else None
            )
            self.log("[OK] Polar slice: " + "; ".join(f"{k}={v}" for k,v in res.items()))
        except Exception as e:
            self.log(f"[ERROR] polar: {e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    App().mainloop()
