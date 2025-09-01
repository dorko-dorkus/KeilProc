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
import json, re
from pathlib import Path
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd

# Ensure repo root is importable when running as "python gui/app_gui.py"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Adapter functions into the kielproc backend
from kielproc_gui_adapter import (
    map_verification_plane, map_from_tot_and_static,
    fit_alpha_beta, translate_piccolo,
    generate_flow_map_from_csv, generate_polar_slice_from_csv,
    legacy_results_from_csv, ResultsConfig
)
from kielproc.aggregate import integrate_run, RunConfig
from kielproc.legacy_results import compute_results as compute_results_cli
from kielproc.qa import DEFAULT_W_MAX, DEFAULT_DELTA_OPP_MAX
from kielproc.geometry import (
    Geometry,
    plane_area,
    throat_area,
    effective_upstream_area,
    r_ratio,
    beta_from_geometry,
)


class ScrollableFrame(ttk.Frame):
    """A simple scrollable frame container using a canvas and vertical scrollbar."""

    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>", lambda e: self.canvas.itemconfig(self._win, width=e.width)
        )

        def _on_mousewheel(event):
            delta = event.delta
            if delta == 0 and event.num in (4, 5):
                delta = 120 if event.num == 4 else -120
            self.canvas.yview_scroll(int(-delta / 120), "units")

        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas.bind_all("<Button-4>", _on_mousewheel)
        self.canvas.bind_all("<Button-5>", _on_mousewheel)

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
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)

        self.tab_phys = ScrollableFrame(self.nb)
        self.nb.add(self.tab_phys, text="Physics / Translation")
        self._build_phys(self.tab_phys.inner)

        self.tab_results = ScrollableFrame(self.nb)
        self.nb.add(self.tab_results, text="Results")
        self._build_results(self.tab_results.inner)

        self.tab_integrate = ScrollableFrame(self.nb)
        self.nb.add(self.tab_integrate, text="8-Port Integration")
        self._build_integrate(self.tab_integrate.inner)

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

        # Geometry inputs
        self.var_height = tk.DoubleVar(value=1.0)
        self.var_width = tk.DoubleVar(value=2.3)
        self.var_dt = tk.DoubleVar(value=0.0)
        self.var_at = tk.DoubleVar(value=0.0)
        self.var_A1 = tk.DoubleVar(value=0.0)
        self.var_use_plane = tk.BooleanVar(value=True)
        self.var_rho = tk.DoubleVar(value=0.7)

        self.var_As = tk.StringVar()
        self.var_At = tk.StringVar()
        self.var_r_geom = tk.StringVar()
        self.var_beta_geom = tk.StringVar()
        self.var_D1 = tk.StringVar()

        for v in [self.var_height, self.var_width, self.var_dt, self.var_at, self.var_A1, self.var_use_plane]:
            v.trace_add('write', lambda *args: self._update_geometry())

        ttk.Label(frm, text="Duct height [m]").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_height, width=10).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="Duct width [m]").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_width, width=10).grid(row=row, column=3, sticky="w", **pad); row+=1

        ttk.Label(frm, text="Throat dia [m]").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_dt, width=10).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="Throat area [m²]").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_at, width=10).grid(row=row, column=3, sticky="w", **pad); row+=1

        ttk.Checkbutton(frm, text="Use plane area for beta (A1 = As)", variable=self.var_use_plane, command=self._update_geometry).grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        ttk.Label(frm, text="Upstream area A1 [m²]").grid(row=row, column=2, sticky="e", **pad)
        self.ent_A1 = ttk.Entry(frm, textvariable=self.var_A1, width=10)
        self.ent_A1.grid(row=row, column=3, sticky="w", **pad); row+=1

        ttk.Label(frm, text="ρ default [kg/m³]").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_rho, width=10).grid(row=row, column=1, sticky="w", **pad); row+=1

        ttk.Label(frm, text="As [m²]").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_As, width=10, state="readonly").grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="At [m²]").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_At, width=10, state="readonly").grid(row=row, column=3, sticky="w", **pad); row+=1

        ttk.Label(frm, text="r = As/At").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_r_geom, width=10, state="readonly").grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="beta").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_beta_geom, width=10, state="readonly").grid(row=row, column=3, sticky="w", **pad); row+=1

        ttk.Label(frm, text="D1 equiv [m]").grid(row=row, column=0, sticky="e", **pad)
        self.ent_D1 = ttk.Entry(frm, textvariable=self.var_D1, width=10, state="readonly")
        self.ent_D1.grid(row=row, column=1, sticky="w", **pad); row+=1

        self._update_geometry()

        self.var_fs = tk.DoubleVar(value=10.0)
        self.var_outdir = tk.StringVar(value=str(Path.cwd()/ "outputs"))

        ttk.Label(frm, text="Sampling (Hz)").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_fs, width=10).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="Output dir").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_outdir, width=36).grid(row=row, column=3, sticky="w", **pad); row+=1
        ttk.Button(frm, text="Map qs→qt→Δp_vent", command=self._do_map).grid(row=row, column=1, sticky="w", **pad); row+=1

        ttk.Separator(frm, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", **pad); row+=1

        # Translation section
        ttk.Label(frm, text="Legacy piccolo translation (lag removal + Deming α,β)").grid(row=row, column=0, sticky="w", **pad); row+=1
        # expose an ingest function for Integration tab
        self._ingest_reference_block_on_translate_tab = self._translate_ingest_reference_block
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

        # Legacy results
        ttk.Label(frm, text="Legacy results from raw CSV").grid(row=row, column=0, sticky="w", **pad); row+=1
        self.var_res_csv = tk.StringVar()
        self.var_res_static = tk.StringVar(value="Static")
        self.var_res_pic_units = tk.StringVar(value="mA")
        self.var_res_pic_rng = tk.DoubleVar(value=6.7)
        ttk.Label(frm, text="CSV input").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_res_csv, width=56).grid(row=row, column=1, **pad)
        ttk.Button(frm, text="Browse", command=lambda: self._pick(self.var_res_csv, [("CSV","*.csv")])).grid(row=row, column=2, **pad); row+=1
        ttk.Label(frm, text="Static col").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_res_static, width=16).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="Piccolo units").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_res_pic_units, width=10).grid(row=row, column=3, sticky="w", **pad); row+=1
        ttk.Label(frm, text="Piccolo range mbar").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_res_pic_rng, width=10).grid(row=row, column=1, sticky="w", **pad); row+=1
        ttk.Button(frm, text="Compute legacy results", command=self._do_results).grid(row=row, column=1, sticky="w", **pad); row+=1

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
        self.txt = scrolledtext.ScrolledText(frm, height=12)
        self.txt.grid(row=row, column=0, columnspan=4, sticky="nsew", **pad)
        frm.grid_rowconfigure(row, weight=1); frm.grid_columnconfigure(1, weight=1)

    def _build_results(self, frm):
        pad = {"padx": 6, "pady": 4}
        row = 0
        ttk.Label(frm, text="Compute legacy-style results from a logger CSV").grid(row=row, column=0, columnspan=3, sticky="w", **pad); row+=1

        self.var_res_csv_in = tk.StringVar()
        ttk.Label(frm, text="Logger CSV").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_res_csv_in, width=56).grid(row=row, column=1, **pad)
        ttk.Button(frm, text="Browse", command=lambda: self._pick(self.var_res_csv_in, [("CSV","*.csv")])).grid(row=row, column=2, **pad); row+=1

        self.var_res_cfg_json = tk.StringVar()
        ttk.Label(frm, text="Config JSON (opt)").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_res_cfg_json, width=56).grid(row=row, column=1, **pad)
        ttk.Button(frm, text="Browse", command=lambda: self._pick(self.var_res_cfg_json, [("JSON","*.json")])).grid(row=row, column=2, **pad); row+=1

        self.var_res_json_out = tk.StringVar()
        ttk.Label(frm, text="JSON out (opt)").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_res_json_out, width=56).grid(row=row, column=1, **pad)
        ttk.Button(frm, text="Browse", command=lambda: self._save_as(self.var_res_json_out, [("JSON","*.json")])).grid(row=row, column=2, **pad); row+=1

        self.var_res_csv_out = tk.StringVar()
        ttk.Label(frm, text="CSV out (opt)").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_res_csv_out, width=56).grid(row=row, column=1, **pad)
        ttk.Button(frm, text="Browse", command=lambda: self._save_as(self.var_res_csv_out, [("CSV","*.csv")])).grid(row=row, column=2, **pad); row+=1

        ttk.Button(frm, text="Compute Results", command=self._compute_results).grid(row=row, column=1, sticky="w", **pad); row+=1

        self.txt_results = scrolledtext.ScrolledText(frm, height=20)
        self.txt_results.grid(row=row, column=0, columnspan=3, sticky="nsew", **pad)
        frm.grid_rowconfigure(row, weight=1); frm.grid_columnconfigure(1, weight=1)

    _PORT_PAT = re.compile(r"(?i)\bP(?:ORT)?\s*([1-8])\b")

    def _port_id_from_stem(self, stem: str):
        m = self._PORT_PAT.search(stem)
        return f"P{m.group(1)}" if m else None

    def _build_integrate(self, frm):
        pad = {"padx": 6, "pady": 4}
        row = 0

        # Run folder
        self.var_run_dir = tk.StringVar()
        ttk.Label(frm, text="Run folder (PORT CSVs)").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_run_dir, width=56).grid(row=row, column=1, columnspan=2, sticky="we", **pad)
        ttk.Button(frm, text="Browse...", command=self._pick_run_dir).grid(row=row, column=3, sticky="w", **pad); row += 1

        # Geometry
        self.var_height = tk.StringVar(value="1.150")
        self.var_width  = tk.StringVar(value="1.932")
        ttk.Label(frm, text="Duct height [m]").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_height, width=12).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="Duct width [m]").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_width, width=12).grid(row=row, column=3, sticky="w", **pad); row += 1

        # Static & baro reconciliation
        self.var_baro = tk.StringVar()  # empty means "use CSV absolute or CSV baro"
        ttk.Label(frm, text="Baro [Pa] (only if Static_gauge)").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_baro, width=12).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="Replicate strategy").grid(row=row, column=2, sticky="e", **pad)
        self.var_rep = tk.StringVar(value="mean")
        ttk.Combobox(frm, textvariable=self.var_rep, values=["mean","last"], width=10, state="readonly").grid(row=row, column=3, sticky="w", **pad); row += 1

        # Weights
        self.var_weights_path = tk.StringVar()
        ttk.Label(frm, text="Weights JSON (optional)").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_weights_path, width=56).grid(row=row, column=1, columnspan=2, sticky="we", **pad)
        ttk.Button(frm, text="Browse...", command=self._pick_weights).grid(row=row, column=3, sticky="w", **pad); row += 1

        # DCS comparison (optional)
        self.var_r = tk.StringVar()
        self.var_beta = tk.StringVar()
        ttk.Label(frm, text="Area ratio r = As/At (optional)").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_r, width=12).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="Venturi beta β (optional)").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_beta, width=12).grid(row=row, column=3, sticky="w", **pad); row += 1

        # Visuals
        self.var_viz = tk.BooleanVar(value=False)
        self.var_viz_bins = tk.StringVar(value="50")
        self.var_viz_clip = tk.StringVar(value="2,98")
        self.var_viz_interp = tk.StringVar(value="nearest")
        ttk.Checkbutton(frm, text="Render velocity heatmap", variable=self.var_viz).grid(row=row, column=0, sticky="w", **pad)
        ttk.Label(frm, text="Height bins").grid(row=row, column=1, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_viz_bins, width=6).grid(row=row, column=2, sticky="w", **pad)
        ttk.Label(frm, text="Clip pcts").grid(row=row, column=3, sticky="e", **pad); row += 1
        ttk.Entry(frm, textvariable=self.var_viz_clip, width=8).grid(row=row, column=0, sticky="w", **pad)
        ttk.Label(frm, text="Interp").grid(row=row, column=1, sticky="e", **pad)
        ttk.Combobox(frm, textvariable=self.var_viz_interp, values=["nearest","linear","cubic"], width=10, state="readonly").grid(row=row, column=2, sticky="w", **pad)
        self.btn_open_heatmap = ttk.Button(frm, text="Open Heatmap", command=self._open_heatmap, state="disabled")
        self.btn_open_heatmap.grid(row=row, column=3, sticky="we", **pad); row += 1

        # Actions
        ttk.Button(frm, text="Discover Ports", command=self._discover_ports).grid(row=row, column=0, sticky="we", **pad)
        ttk.Button(frm, text="Run Integration", command=self._run_integration).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(frm, text="Open Output Folder", command=self._open_out).grid(row=row, column=2, sticky="we", **pad)
        ttk.Button(frm, text="Send to Translation", command=self._send_to_translation).grid(row=row, column=3, sticky="we", **pad); row += 1

        # Discovery and results panes
        self.txt_discovery = scrolledtext.ScrolledText(frm, height=6, width=100)
        self.txt_discovery.grid(row=row, column=0, columnspan=4, sticky="nsew", **pad); row += 1

        cols = ("PortId","FileStem","VP_pa_mean","T_C_mean","Static_abs_pa_mean","v_m_s","rho_v_kg_m2_s","q_s_pa","p_abs_source","replicate_strategy")
        self.tree = ttk.Treeview(frm, columns=cols, show="headings", height=10)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor="center")
        self.tree.grid(row=row, column=0, columnspan=4, sticky="nsew", **pad); row += 1

        self.var_summary = tk.StringVar()
        ttk.Label(frm, textvariable=self.var_summary, justify="left").grid(row=row, column=0, columnspan=4, sticky="w", **pad)

        frm.grid_columnconfigure(1, weight=1)
        frm.grid_rowconfigure(row, weight=1)

    def _pick_run_dir(self):
        d = filedialog.askdirectory(title="Select run folder with PORT CSVs")
        if d: self.var_run_dir.set(d)

    def _pick_weights(self):
        p = filedialog.askopenfilename(title="Weights JSON", filetypes=[("JSON","*.json"), ("All","*.*")])
        if p: self.var_weights_path.set(p)

    def _discover_ports(self):
        run = Path(self.var_run_dir.get())
        if not run.exists():
            messagebox.showerror("Error", "Run folder not found"); return
        csvs = sorted(run.glob("*.csv"))
        pairs = []
        for p in csvs:
            pid = self._port_id_from_stem(p.stem)
            if pid: pairs.append((pid, p.name))
        present = [pid for pid,_ in pairs]
        missing = [f"P{i}" for i in range(1,9) if f"P{i}" not in present]
        self.txt_discovery.delete("1.0", "end")
        self.txt_discovery.insert("end", f"Found: {present}\nMissing: {missing}\nFiles: {pairs}\n")

    def _load_weights(self):
        wp = self.var_weights_path.get().strip()
        if not wp: return None
        try:
            jd = json.loads(Path(wp).read_text())
        except Exception as e:
            messagebox.showerror("Weights JSON", f"Failed to load JSON: {e}"); return None
        # normalize keys like PORT 1, p1, Run_P1 -> P1
        out = {}
        for k,v in jd.items():
            pid = self._port_id_from_stem(Path(k).stem) or str(k).upper()
            out[pid] = out.get(pid, 0.0) + float(v)
        s = sum(out.values())
        if abs(s - 1.0) > 1e-6:
            messagebox.showerror("Weights JSON", f"Weights must sum to 1.0 (got {s:.6f})"); return None
        return out

    def _run_integration(self):
        try:
            run = Path(self.var_run_dir.get())
            H = float(self.var_height.get()); W = float(self.var_width.get())
            baro = self.var_baro.get().strip()
            baro_pa = float(baro) if baro else None
            rep = self.var_rep.get()
            weights = self._load_weights()
            r = float(self.var_r.get()) if self.var_r.get().strip() else None
            beta = float(self.var_beta.get()) if self.var_beta.get().strip() else None

            cfg = RunConfig(height_m=H, width_m=W, weights=weights, replicate_strategy=rep)
            res = integrate_run(run, cfg, file_glob="*.csv", baro_cli_pa=baro_pa, area_ratio=r, beta=beta)

            # populate table
            for i in self.tree.get_children(): self.tree.delete(i)
            per = res["per_port"]
            # DataFrame may be returned or path; handle both
            if hasattr(per, "to_dict"):
                rows = per.to_dict(orient="records")
            else:
                import pandas as pd
                rows = pd.read_csv(run / "_integrated" / "per_port.csv").to_dict(orient="records")
            for row in rows:
                vals = [row.get(k, "") for k in ("PortId","FileStem","VP_pa_mean","T_C_mean","Static_abs_pa_mean","v_m_s","rho_v_kg_m2_s","q_s_pa","p_abs_source","replicate_strategy")]
                self.tree.insert("", "end", values=vals)

            duct = res["duct"]
            summary = [f"area_m2 = {duct.get('area_m2'):.6f}",
                       f"v_bar_m_s = {duct.get('v_bar_m_s'):.6f}",
                       f"Q_m3_s = {duct.get('Q_m3_s'):.6f}",
                       f"m_dot_kg_s = {duct.get('m_dot_kg_s'):.6f}"]
            if "q_s_pa" in duct: summary.append(f"q_s_pa = {duct.get('q_s_pa'):.3f}")
            if "q_t_pa" in duct: summary.append(f"q_t_pa = {duct.get('q_t_pa'):.3f}")
            if "delta_p_vent_est_pa" in duct: summary.append(f"Δp_vent_est_pa = {duct.get('delta_p_vent_est_pa'):.3f}")
            outdir = run / "_integrated"
            self._last_outdir = outdir
            self.var_summary.set("  ".join(summary) + f"   ->  outputs: {outdir}")

            # build a light-weight reference block adapter for Translation
            try:
                # load duct_result so we can include q_s/q_t/delta_p if present
                import json
                duct = json.loads((outdir / "duct_result.json").read_text())
                ref_block = {
                    "block_name": run.name,
                    "run_dir": str(run),
                    "outdir": str(outdir),
                    "per_port_csv": str(outdir / "per_port.csv"),
                    "duct_result_json": str(outdir / "duct_result.json"),
                    "q_s_pa": duct.get("q_s_pa"),
                    "q_t_pa": duct.get("q_t_pa"),
                    "delta_p_vent_est_pa": duct.get("delta_p_vent_est_pa"),
                }
                (outdir / "reference_block.json").write_text(json.dumps(ref_block, indent=2))
                self._last_reference_block = outdir / "reference_block.json"
            except Exception as e:
                # non-fatal: Translation can still browse to the files manually
                self._last_reference_block = None

            # visuals
            if self.var_viz.get():
                try:
                    from kielproc.visuals import render_velocity_heatmap
                    # rebuild pairs from res if available; else rediscover
                    pairs = res.get("pairs")
                    if not pairs:
                        # rediscover using same pattern as integrator
                        csvs = sorted(run.glob("*.csv"))
                        pairs = []
                        for p in csvs:
                            pid = self._port_id_from_stem(p.stem)
                            if pid: pairs.append((pid, p))
                        pairs.sort(key=lambda kv: int(kv[0][1:]))
                    bins = int(self.var_viz_bins.get() or "50")
                    lo, hi = (float(x) for x in (self.var_viz_clip.get() or "2,98").split(","))
                    hm = render_velocity_heatmap(
                        outdir=outdir,
                        pairs=pairs,
                        baro_cli_pa=baro_pa,
                        height_bins=bins,
                        clip_percentiles=(lo, hi),
                        interp=self.var_viz_interp.get(),
                    )
                    self._heatmap_path = hm
                    self.btn_open_heatmap.config(state="normal")
                except Exception as e:
                    traceback.print_exc()
                    messagebox.showerror("Heatmap error", str(e))

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Integration error", str(e))

    def _open_out(self):
        p = getattr(self, "_last_outdir", None)
        if not p:
            messagebox.showinfo("Open output", "Run integration first")
            return
        try:
            import os, platform, subprocess
            if platform.system() == "Windows":
                os.startfile(str(p))
            elif platform.system() == "Darwin":
                subprocess.check_call(["open", str(p)])
            else:
                subprocess.check_call(["xdg-open", str(p)])
        except Exception as e:
            messagebox.showerror("Open output", str(e))

    def _send_to_translation(self):
        try:
            rb = getattr(self, "_last_reference_block", None)
            if not rb or not Path(rb).exists():
                messagebox.showinfo("Send to Translation", "Run Integration first so a reference block can be created.")
                return
            # require Translation tab to expose an ingest method
            if not hasattr(self, "_ingest_reference_block_on_translate_tab"):
                # locate a peer method if Translation tab is built as part of this class
                if hasattr(self, "_translate_ingest_reference_block"):
                    ingest = self._translate_ingest_reference_block
                else:
                    messagebox.showerror("Send to Translation", "Translation tab does not expose an ingest method.")
                    return
            else:
                ingest = self._ingest_reference_block_on_translate_tab
            # perform ingestion, then switch tabs
            ingest(Path(rb))
            self.nb.select(self.tab_phys)
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Send to Translation", str(e))

    def _open_heatmap(self):
        try:
            p = getattr(self, "_heatmap_path", None)
            if not p:
                messagebox.showinfo("Open heatmap", "No heatmap rendered yet")
                return
            import os, platform, subprocess
            if platform.system() == "Windows":
                os.startfile(str(p))
            elif platform.system() == "Darwin":
                subprocess.check_call(["open", str(p)])
            else:
                subprocess.check_call(["xdg-open", str(p)])
        except Exception as e:
            messagebox.showerror("Open heatmap", str(e))

    def _pick(self, var, patterns):
        p = filedialog.askopenfilename(title="Choose file", filetypes=patterns)
        if p: var.set(p)

    def _save_as(self, var, patterns):
        p = filedialog.asksaveasfilename(title="Choose destination", defaultextension=patterns[0][1], filetypes=patterns)
        if p: var.set(p)

    def log(self, msg):
        self.txt.insert("end", str(msg)+"\n"); self.txt.see("end")

    # Geometry helpers
    def _fmt(self, val):
        try:
            return f"{val:.3g}"
        except Exception:
            return ""

    def _update_geometry(self, *args):
        try:
            h = float(self.var_height.get())
            w = float(self.var_width.get())
        except Exception:
            return
        As = h * w
        self.var_As.set(self._fmt(As))
        if self.var_use_plane.get():
            self.ent_A1.configure(state="disabled")
            self.var_A1.set(As)
            A1 = As
        else:
            self.ent_A1.configure(state="normal")
            try:
                A1 = float(self.var_A1.get())
            except Exception:
                A1 = None
        At = None
        try:
            dt = float(self.var_dt.get())
            if dt > 0:
                At = math.pi * (dt ** 2) / 4
        except Exception:
            pass
        try:
            at_in = float(self.var_at.get())
            if at_in > 0:
                if At is None or abs(at_in - At) / At <= 0.005:
                    At = at_in
        except Exception:
            pass
        if At is not None:
            self.var_At.set(self._fmt(At))
            if A1 is None:
                beta = float("nan")
                r = float("nan")
            else:
                r = As / At
                beta = math.sqrt(At / A1)
            self.var_r_geom.set(self._fmt(r))
            self.var_beta_geom.set(self._fmt(beta))
        else:
            self.var_At.set("")
            self.var_r_geom.set("")
            self.var_beta_geom.set("")
        if not self.var_use_plane.get() and A1 is not None:
            D1 = math.sqrt(4 * A1 / math.pi)
            self.var_D1.set(self._fmt(D1))
        else:
            self.var_D1.set("")

    def _get_geometry(self) -> Geometry:
        dt = float(self.var_dt.get()) if self.var_dt.get() else None
        at = float(self.var_at.get()) if self.var_at.get() else None
        if dt and at:
            At_dt = math.pi * (dt ** 2) / 4
            if abs(At_dt - at) / At_dt > 0.005:
                raise ValueError("Throat diameter and area mismatch >0.5%")
            at = At_dt  # favor diameter
        if dt is None and at is None:
            raise ValueError("Provide throat diameter or area")
        A1 = None if self.var_use_plane.get() else (float(self.var_A1.get()) if self.var_A1.get() else None)
        g = Geometry(
            duct_height_m=float(self.var_height.get()),
            duct_width_m=float(self.var_width.get()),
            upstream_area_m2=A1,
            throat_diameter_m=dt if dt else None,
            throat_area_m2=at if at and (dt is None) else None,
            rho_default_kg_m3=float(self.var_rho.get()),
        )
        As = plane_area(g)
        At = throat_area(g)
        A1_eff = effective_upstream_area(g)
        warn = []
        if At >= As:
            warn.append("At >= As")
        if At >= A1_eff:
            warn.append("At >= A1")
        if warn:
            msg = ", ".join(warn) + ". Continue?"
            if not messagebox.askokcancel("Geometry warning", msg):
                raise RuntimeError("Mapping aborted by user")
        return g

    # Handlers
    def _do_map(self):
        try:
            geom = self._get_geometry()
            outdir = Path(self.var_outdir.get()); outdir.mkdir(parents=True, exist_ok=True)
            out = outdir/"mapped_ref.csv"
            fs = float(self.var_fs.get()) if self.var_fs.get() > 0 else None
            if self.var_use_tot_stat.get():
                mapped = map_from_tot_and_static(
                    Path(self.var_qs_csv.get()),
                    self.var_tot_col.get(),
                    self.var_ps_col_avg.get(),
                    geom,
                    fs,
                    out,
                )
            else:
                mapped = map_verification_plane(
                    Path(self.var_qs_csv.get()),
                    self.var_qs_col.get(),
                    geom,
                    fs,
                    out,
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

    def _do_results(self):
        try:
            outdir = Path(self.var_outdir.get()); outdir.mkdir(parents=True, exist_ok=True)
            cfg = ResultsConfig(
                static_col=self.var_res_static.get().strip() or None,
                piccolo_units=self.var_res_pic_units.get().strip() or "mA",
                piccolo_range_mbar=float(self.var_res_pic_rng.get()),
                duct_height_m=float(self.var_height.get()) if self.var_height.get() else None,
                duct_width_m=float(self.var_width.get()) if self.var_width.get() else None,
            )
            res = legacy_results_from_csv(Path(self.var_res_csv.get()), cfg, outdir/"legacy_results.csv")
            self.log("[OK] Results: " + "; ".join(f"{k}={v}" for k,v in res.items()))
        except Exception as e:
            self.log(f"[ERROR] results: {e}\n{traceback.format_exc()}")

    def _compute_results(self):
        try:
            cfg_dict = {}
            if self.var_res_cfg_json.get():
                with open(self.var_res_cfg_json.get()) as fh:
                    cfg_dict = json.load(fh)
            cfg = ResultsConfig(**cfg_dict)
            res = compute_results_cli(self.var_res_csv_in.get(), cfg)
            self.txt_results.delete("1.0", "end")
            for k, v in res.items():
                self.txt_results.insert("end", f"{k}: {v}\n")
            if self.var_res_json_out.get():
                p = Path(self.var_res_json_out.get())
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "w") as fh:
                    json.dump(res, fh, indent=2)
            if self.var_res_csv_out.get():
                p = Path(self.var_res_csv_out.get())
                p.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([res]).to_csv(p, index=False)
            self.log("[OK] Results computed")
        except Exception as e:
            self.log(f"[ERROR] compute results: {e}\n{traceback.format_exc()}")

    def _translate_ingest_reference_block(self, ref_json_path: Path):
        """
        Add a reference block row to the Translation tab using the integration outputs.
        The JSON is produced by Integration and contains per_port.csv and duct_result.json paths.
        """
        import json
        data = json.loads(Path(ref_json_path).read_text())
        name = data.get("block_name") or Path(data.get("outdir", "")).parent.name
        per_port = data.get("per_port_csv", "")
        duct_json = data.get("duct_result_json", "")
        q_s = data.get("q_s_pa", None)
        q_t = data.get("q_t_pa", None)
        dpv = data.get("delta_p_vent_est_pa", None)

        if not hasattr(self, "blocks"):
            self.blocks = []
        self.blocks.append({
            "name": name,
            "reference_per_port_csv": per_port,
            "reference_duct_json": duct_json,
            "q_s_pa": q_s,
            "q_t_pa": q_t,
            "delta_p_vent_est_pa": dpv,
        })

        # update blocks entry string
        entry = f"{name}={per_port}"
        cur = self.var_blocks.get().strip()
        if cur:
            parts = [p.strip() for p in cur.split(",") if p.strip()]
            if entry not in parts:
                parts.append(entry)
            self.var_blocks.set(", ".join(parts))
        else:
            self.var_blocks.set(entry)

        tv = getattr(self, "tr_blocks", None)
        if tv:
            try:
                tv.insert("", "end", values=(name, per_port, q_s if q_s is not None else "", q_t if q_t is not None else "", dpv if dpv is not None else ""))
            except Exception:
                try:
                    tv.insert("", "end", values=(name, per_port))
                except Exception:
                    pass

        if hasattr(self, "var_ref_path"):
            self.var_ref_path.set(per_port)
        if hasattr(self, "var_ref_meta"):
            self.var_ref_meta.set(duct_json)

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
