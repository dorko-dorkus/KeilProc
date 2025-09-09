import os
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from kielproc.run_easy import RunConfig, SitePreset, run_all

APP_TITLE = "KielProc – One-Button Run"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("880x640")
        self._build()

    # ------------------------------------------------------------------ UI
    def _build(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        # Paths --------------------------------------------------------
        row = 0
        ttk.Label(frm, text="Input path (folder or .xlsx)").grid(column=0, row=row, sticky="w")
        self.in_dir = ttk.Entry(frm, width=70)
        self.in_dir.grid(column=1, row=row, sticky="ew", padx=6)
        btns = ttk.Frame(frm)
        btns.grid(column=2, row=row, sticky="w")
        ttk.Button(btns, text="Folder…", command=self._pick_in).grid(column=0, row=0, padx=(0, 4))
        ttk.Button(
            btns,
            text="Workbook…",
            command=lambda: self._browse_file(
                self.in_dir, filetypes=[("Excel", "*.xlsx *.xlsm *.xls")]
            ),
        ).grid(column=1, row=0)
        row += 1

        ttk.Label(frm, text="Output folder").grid(column=0, row=row, sticky="w")
        self.out_dir = ttk.Entry(frm, width=70)
        self.out_dir.grid(column=1, row=row, sticky="ew", padx=6)
        ttk.Button(frm, text="Browse…", command=self._pick_out).grid(column=2, row=row)
        row += 1

        # File glob and units ------------------------------------------
        ttk.Label(frm, text="File glob").grid(column=0, row=row, sticky="w")
        self.file_glob = ttk.Entry(frm, width=30)
        self.file_glob.insert(0, "*__P[1-8].csv")
        self.file_glob.grid(column=1, row=row, sticky="w", padx=6)
        row += 1

        ttk.Label(frm, text="Barometric pressure (Pa)").grid(column=0, row=row, sticky="w")
        self.baro_pa = ttk.Entry(frm, width=20)
        self.baro_pa.insert(0, "101325")
        self.baro_pa.grid(column=1, row=row, sticky="w", padx=6)
        row += 1

        ttk.Label(frm, text="Velocity pressure unit").grid(column=0, row=row, sticky="w")
        self.vp_unit = ttk.Combobox(frm, values=["Pa", "inH2O"], width=10, state="readonly")
        self.vp_unit.set("Pa")
        self.vp_unit.grid(column=1, row=row, sticky="w", padx=6)
        row += 1

        ttk.Label(frm, text="Temperature unit").grid(column=0, row=row, sticky="w")
        self.temp_unit = ttk.Combobox(frm, values=["C", "K"], width=10, state="readonly")
        self.temp_unit.set("C")
        self.temp_unit.grid(column=1, row=row, sticky="w", padx=6)
        row += 1

        # Optional site preset ----------------------------------------
        self.enable_site = tk.BooleanVar(value=False)
        self.chk_site = ttk.Checkbutton(
            frm,
            text="Enable Site preset (advanced)",
            variable=self.enable_site,
            command=self._toggle_site,
        )
        self.chk_site.grid(column=0, row=row, sticky="w", pady=(6, 2))
        row += 1

        site = ttk.LabelFrame(frm, text="Site preset")
        site.grid(column=0, row=row, columnspan=3, sticky="ew", pady=(0, 8))
        self.site_frame = site

        srow = 0
        ttk.Label(site, text="Name").grid(column=0, row=srow, sticky="w")
        self.site_name = ttk.Entry(site, width=20)
        self.site_name.insert(0, "MySite")
        self.site_name.grid(column=1, row=srow, sticky="w", padx=6)
        srow += 1

        ttk.Label(site, text="Duct width (m)").grid(column=0, row=srow, sticky="w")
        self.duct_w = ttk.Entry(site, width=10)
        self.duct_w.grid(column=1, row=srow, sticky="w", padx=6)
        srow += 1

        ttk.Label(site, text="Duct height (m)").grid(column=0, row=srow, sticky="w")
        self.duct_h = ttk.Entry(site, width=10)
        self.duct_h.grid(column=1, row=srow, sticky="w", padx=6)
        srow += 1
        ttk.Label(site, text="Throat area At (m², optional)").grid(column=0, row=srow, sticky="w")
        self.throat_area = ttk.Entry(site, width=12)
        self.throat_area.grid(column=1, row=srow, sticky="w", padx=6)
        srow += 1
        ttk.Label(site, text="β (venturi ratio, optional)").grid(column=0, row=srow, sticky="w")
        self.beta_entry = ttk.Entry(site, width=12)
        self.beta_entry.grid(column=1, row=srow, sticky="w", padx=6)
        srow += 1
        ttk.Label(site, text="Venturi Cd (optional, default 0.98)").grid(column=0, row=srow, sticky="w")
        self.cd_entry = ttk.Entry(site, width=12)
        self.cd_entry.grid(column=1, row=srow, sticky="w", padx=6)
        srow += 1

        ttk.Label(site, text="Piccolo present").grid(column=0, row=srow, sticky="w")
        self.piccolo = ttk.Combobox(site, values=["no", "yes"], width=6, state="readonly")
        self.piccolo.set("no")
        self.piccolo.grid(column=1, row=srow, sticky="w", padx=6)
        srow += 1

        ttk.Label(site, text="Instruments").grid(column=0, row=srow, sticky="w")
        self.site_vp = ttk.Combobox(site, values=["Pa", "inH2O"], width=10, state="readonly")
        self.site_vp.set("Pa")
        self.site_vp.grid(column=1, row=srow, sticky="w", padx=(6, 12))
        self.site_temp = ttk.Combobox(site, values=["C", "K"], width=10, state="readonly")
        self.site_temp.set("C")
        self.site_temp.grid(column=2, row=srow, sticky="w")
        srow += 1

        self._toggle_site()  # start collapsed

        # Transmitter setpoints (optional) ---------------------------
        ttk.Separator(frm, orient="horizontal").grid(column=0, row=row, columnspan=3, sticky="ew", pady=6)
        row += 1
        ttk.Label(frm, text="Logger CSV (dp & temperature)").grid(column=0, row=row, sticky="w")
        self.sp_csv = ttk.Entry(frm, width=40)
        self.sp_csv.grid(column=1, row=row, sticky="w", padx=6)
        ttk.Button(frm, text="Browse…", command=lambda: self._browse_file(self.sp_csv)).grid(column=2, row=row, sticky="w")
        row += 1
        ttk.Label(frm, text="dp column name (default: i/p)").grid(column=0, row=row, sticky="w")
        self.sp_x = ttk.Entry(frm, width=20); self.sp_x.insert(0, "i/p")
        self.sp_x.grid(column=1, row=row, sticky="w", padx=6); row += 1
        ttk.Label(frm, text="Temperature column (default: 820)").grid(column=0, row=row, sticky="w")
        self.sp_y = ttk.Entry(frm, width=20); self.sp_y.insert(0, "820")
        self.sp_y.grid(column=1, row=row, sticky="w", padx=6); row += 1
        ttk.Label(frm, text="Min fraction of span at p95 (0.60)").grid(column=0, row=row, sticky="w")
        self.sp_min = ttk.Entry(frm, width=10); self.sp_min.insert(0, "0.6")
        self.sp_min.grid(column=1, row=row, sticky="w", padx=6); row += 1
        ttk.Label(frm, text="Slope sign (+1 or -1)").grid(column=0, row=row, sticky="w")
        self.sp_sign = ttk.Entry(frm, width=10); self.sp_sign.insert(0, "+1")
        self.sp_sign.grid(column=1, row=row, sticky="w", padx=6); row += 1

        # Run button and status ---------------------------------------
        self.run_btn = ttk.Button(frm, text="Run", command=self._on_run)
        self.run_btn.grid(column=0, row=row, pady=8, sticky="w")
        self.status = ttk.Label(frm, text="Idle")
        self.status.grid(column=1, row=row, sticky="w")
        row += 1

        self.log = ScrolledText(frm, height=18)
        self.log.grid(column=0, row=row, columnspan=3, sticky="nsew")
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(row, weight=1)

    # ----------------------------------------------------------------- helpers
    def _pick_in(self):
        d = filedialog.askdirectory()
        if d:
            self.in_dir.delete(0, tk.END)
            self.in_dir.insert(0, d)

    def _pick_out(self):
        d = filedialog.askdirectory()
        if d:
            self.out_dir.delete(0, tk.END)
            self.out_dir.insert(0, d)

    def _browse_file(self, entry: ttk.Entry, **kwargs):
        f = filedialog.askopenfilename(**kwargs)
        if f:
            entry.delete(0, tk.END)
            entry.insert(0, f)

    def _toggle_site(self):
        state = "normal" if self.enable_site.get() else "disabled"
        for child in self.site_frame.winfo_children():
            child.configure(state=state)
        self.site_frame.configure(labelanchor="nw")

    def _append_log(self, text: str):
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.update_idletasks()

    # ------------------------------------------------------------------ run
    def _on_run(self):
        in_path = self.in_dir.get().strip()
        out_dir = self.out_dir.get().strip()
        valid_file = os.path.isfile(in_path) and in_path.lower().endswith(
            (".xlsx", ".xlsm", ".xls")
        )
        if not (os.path.isdir(in_path) or valid_file):
            messagebox.showerror(
                APP_TITLE, "Input path must be a folder or .xlsx workbook."
            )
            return
        if not os.path.isdir(out_dir):
            messagebox.showerror(APP_TITLE, "Output folder does not exist.")
            return

        try:
            baro = float(self.baro_pa.get().strip() or "0")
        except Exception:
            messagebox.showerror(APP_TITLE, "Barometric pressure must be a number in Pa.")
            return
        if baro <= 0:
            messagebox.showerror(APP_TITLE, "Barometric pressure must be > 0 Pa.")
            return

        cfg = RunConfig(
            input_dir=in_path,
            output_dir=out_dir,
            file_glob=self.file_glob.get().strip() or "*__P[1-8].csv",
            baro_pa=baro,
            vp_unit=self.vp_unit.get(),
            temp_unit=self.temp_unit.get(),
            enable_site=self.enable_site.get(),
            site=None,
        )

        if self.enable_site.get():
            geom = {}
            if self.duct_w.get().strip():
                geom["duct_width_m"] = float(self.duct_w.get().strip())
            if self.duct_h.get().strip():
                geom["duct_height_m"] = float(self.duct_h.get().strip())
            if self.throat_area.get().strip():
                geom["throat_area_m2"] = float(self.throat_area.get().strip())
            if self.beta_entry.get().strip():
                geom["beta"] = float(self.beta_entry.get().strip())
            instr = {"vp_unit": self.site_vp.get(), "temp_unit": self.site_temp.get()}
            defs = {"fallback_baro_Pa": baro}
            cfg.site = SitePreset(
                name=self.site_name.get().strip() or "MySite",
                geometry=geom,
                instruments=instr,
                defaults=defs,
            )
            if self.cd_entry.get().strip():
                try:
                    cfg.site.defaults["venturi_Cd"] = float(self.cd_entry.get().strip())
                except Exception:
                    pass
            # Optional transmitter block
            if self.sp_csv.get().strip():
                cfg.setpoints_csv = self.sp_csv.get().strip()
                cfg.setpoints_x_col = (self.sp_x.get().strip() or "i/p")
                cfg.setpoints_y_col = (self.sp_y.get().strip() or "820")
                try:
                    cfg.setpoints_min_frac = float(self.sp_min.get().strip() or "0.6")
                except Exception:
                    cfg.setpoints_min_frac = 0.6
                try:
                    cfg.setpoints_slope_sign = int(self.sp_sign.get().strip() or "+1")
                except Exception:
                    cfg.setpoints_slope_sign = +1

        self.run_btn.configure(state="disabled")
        self.status.configure(text="Running…")
        self.log.delete("1.0", tk.END)
        threading.Thread(target=self._run_thread, args=(cfg,), daemon=True).start()

    def _run_thread(self, cfg: RunConfig):
        try:
            self._append_log("Parsing…")
            res = run_all(cfg)
            self._append_log("Integrated.")
            self._append_log("Mapped.")
            self._append_log(
                f"Fitted. Baro used: {res.get('baro_pa'):.1f} Pa  Site: {res.get('site_name')}"
            )
            self._append_log("Translated / reported.")
            self.status.configure(text="Done")
        except Exception:
            self.status.configure(text="Failed")
            tb = traceback.format_exc()
            self._append_log(tb)
            messagebox.showerror(APP_TITLE, "Run failed. See log for details.")
        finally:
            self.run_btn.configure(state="normal")


if __name__ == "__main__":
    App().mainloop()

