# runeasy_gui.py — Operator-first UI (text boxes by default), optional presets in Engineering mode
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from kielproc.run_easy import run_easy_legacy
from kielproc.presets import SitePreset, PRESETS  # presets available but NOT required


class RunEasyApp(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="Workbook").grid(row=0, column=0, sticky="w")
        self.input_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.input_var, width=56).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(top, text="Browse…", command=self._pick_input).grid(
            row=0, column=2, padx=6
        )

        ttk.Label(top, text="Run label (site tag)").grid(row=1, column=0, sticky="w")
        self.site_tag_var = tk.StringVar(value="AdHoc")
        ttk.Entry(top, textvariable=self.site_tag_var, width=20).grid(
            row=1, column=1, sticky="w", padx=6
        )

        ttk.Label(top, text="Baro override (Pa)").grid(row=1, column=2, sticky="e")
        self.baro_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.baro_var, width=14).grid(
            row=1, column=3, sticky="w", padx=6
        )

        self.strict_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Strict (fail fast)", variable=self.strict_var).grid(
            row=1, column=4, sticky="w"
        )

        # Engineering mode toggle (optional presets)
        eng = ttk.Frame(self)
        eng.pack(fill="x", padx=8, pady=(0, 8))
        self.use_preset_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            eng,
            text="Engineering mode: use preset geometry",
            variable=self.use_preset_var,
            command=self._toggle_preset_mode,
        ).pack(side="left")

        # Preset picker (hidden until engineering mode enabled)
        self.preset_row = ttk.Frame(self)
        self.preset_row.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(self.preset_row, text="Preset site").grid(row=0, column=0, sticky="w")
        self.preset_site_var = tk.StringVar()
        names = sorted(PRESETS.keys())
        self.preset_site_var.set(names[0] if names else "")
        self.preset_menu = ttk.OptionMenu(
            self.preset_row, self.preset_site_var, self.preset_site_var.get(), *names
        )
        self.preset_menu.grid(row=0, column=1, sticky="w", padx=6)
        self.preset_row.grid_remove()  # hidden by default

        # Geometry text boxes (default path)
        geo = ttk.LabelFrame(self, text="Geometry — enter areas OR width×height")
        geo.pack(fill="x", padx=8, pady=8)
        self.geo_frame = geo

        ttk.Label(geo, text="Duct area As (m²)").grid(row=0, column=0, sticky="w")
        self.As_var = tk.StringVar()
        ttk.Entry(geo, textvariable=self.As_var, width=12).grid(
            row=0, column=1, padx=6
        )
        ttk.Label(geo, text="or Width (m)").grid(row=0, column=2, sticky="e")
        self.Ws_var = tk.StringVar()
        ttk.Entry(geo, textvariable=self.Ws_var, width=10).grid(
            row=0, column=3, padx=4
        )
        ttk.Label(geo, text="Height (m)").grid(row=0, column=4, sticky="e")
        self.Hs_var = tk.StringVar()
        ttk.Entry(geo, textvariable=self.Hs_var, width=10).grid(
            row=0, column=5, padx=4
        )

        ttk.Label(geo, text="Throat area At (m²)").grid(row=1, column=0, sticky="w")
        self.At_var = tk.StringVar()
        ttk.Entry(geo, textvariable=self.At_var, width=12).grid(
            row=1, column=1, padx=6
        )
        ttk.Label(geo, text="or Width (m)").grid(row=1, column=2, sticky="e")
        self.Wt_var = tk.StringVar()
        ttk.Entry(geo, textvariable=self.Wt_var, width=10).grid(
            row=1, column=3, padx=4
        )
        ttk.Label(geo, text="Height (m)").grid(row=1, column=4, sticky="e")
        self.Ht_var = tk.StringVar()
        ttk.Entry(geo, textvariable=self.Ht_var, width=10).grid(
            row=1, column=5, padx=4
        )

        ttk.Label(geo, text="Venturi inlet D1 (m, optional)").grid(
            row=2, column=0, sticky="w"
        )
        self.D1_var = tk.StringVar()
        ttk.Entry(geo, textvariable=self.D1_var, width=12).grid(
            row=2, column=1, padx=6
        )
        ttk.Label(geo, text="Throat dt (m, optional)").grid(row=2, column=2, sticky="e")
        self.dt_var = tk.StringVar()
        ttk.Entry(geo, textvariable=self.dt_var, width=10).grid(
            row=2, column=3, padx=4
        )

        # Controls
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=8, pady=8)
        ttk.Button(bottom, text="RUN", command=self._run).pack(side="left")
        self.log = tk.Text(self, height=14)
        self.log.pack(fill="both", expand=True, padx=8, pady=4)

    def _toggle_preset_mode(self):
        use = bool(self.use_preset_var.get())
        # Show/hide preset picker
        if use:
            self.preset_row.grid()
        else:
            self.preset_row.grid_remove()
        # Enable/disable geometry text fields
        state = "disabled" if use else "normal"
        for w in self.geo_frame.winfo_children():
            if isinstance(w, ttk.Entry):
                w.configure(state=state)

    def _pick_input(self):
        p = filedialog.askopenfilename(
            title="Select legacy workbook", filetypes=[("Excel", "*.xlsx *.xls")]
        )
        if p:
            self.input_var.set(p)

    def _f(self, s: str):
        s = (s or "").strip()
        return None if s == "" else float(s)

    def _collect_geometry_from_textboxes(self) -> dict:
        As = self._f(self.As_var.get())
        Ws = self._f(self.Ws_var.get())
        Hs = self._f(self.Hs_var.get())
        At = self._f(self.At_var.get())
        Wt = self._f(self.Wt_var.get())
        Ht = self._f(self.Ht_var.get())
        D1 = self._f(self.D1_var.get())
        dt = self._f(self.dt_var.get())

        g = {}
        if As is not None:
            g["duct_area_m2"] = As
        elif Ws is not None and Hs is not None:
            g["duct_width_m"] = Ws
            g["duct_height_m"] = Hs
        else:
            raise ValueError("Provide DUCT area (As) or width & height.")

        if At is not None:
            g["throat_area_m2"] = At
        elif Wt is not None and Ht is not None:
            g["throat_width_m"] = Wt
            g["throat_height_m"] = Ht
        else:
            raise ValueError("Provide THROAT area (At) or width & height.")

        if D1 is not None:
            g["D1_m"] = D1
        if dt is not None:
            g["dt_m"] = dt
        return g

    def _run(self):
        try:
            wb = Path(self.input_var.get().strip())
            if not wb.exists():
                messagebox.showerror("Missing workbook", "Choose a valid workbook file.")
                return

            baro = self._f(self.baro_var.get())
            strict = bool(self.strict_var.get())
            site_label = (self.site_tag_var.get() or "AdHoc").strip()

            if self.use_preset_var.get():
                # Engineering mode: use preset geometry
                name = self.preset_site_var.get()
                if name not in PRESETS:
                    raise ValueError(f"Unknown preset '{name}'")
                site = PRESETS[name]
                if not site.geometry:
                    raise ValueError(f"Preset '{name}' has no geometry set.")
                setattr(site, "_geometry_source", f"gui:preset:{name}")
            else:
                # Default path: text boxes
                geom = self._collect_geometry_from_textboxes()
                site = SitePreset(
                    name=site_label,
                    geometry=geom,
                    instruments={"vp_unit": "Pa", "temp_unit": "C"},
                )
                setattr(site, "_geometry_source", "gui:textbox")

            out_dir, summary, artifacts = run_easy_legacy(
                wb, site, baro_override_Pa=baro, strict=strict
            )
            self._log(f"OK → {out_dir}\nArtifacts: {len(artifacts)}\n")
        except Exception as e:
            messagebox.showerror("Run failed", str(e))
            self._log(f"ERROR: {e}\n")

    def _log(self, msg: str):
        self.log.insert("end", msg)
        self.log.see("end")


def main():
    root = tk.Tk()
    root.title("KeilProc — One-Click")
    root.geometry("860x580")
    try:
        root.tk.call("tk", "scaling", 1.25)
    except Exception:
        pass
    app = RunEasyApp(master=root)
    root.mainloop()


if __name__ == "__main__":
    main()

