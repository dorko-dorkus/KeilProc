
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Duct ΔP Visualizer — Tk GUI (free, stdlib only)
- Wraps the existing analyzer in duct_dp_visualizer(.py/.v12.py)
- Lets operators pick Excel or CSV, set params, and run
- Emits CSV/PNG outputs; optional ZIP

Author: Brandon's helper (MIT)
"""

import sys, os, threading, traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Ensure repo root is importable when running as "python gui/duct_dp_visualizer_tk_original.py"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_polish import bind_mousewheel

# Try to import the analysis functions from your existing module
Analyzer = None
_module_err = None
for modname in ("duct_dp_visualizer", "duct_dp_visualizer_v12"):
    try:
        _m = __import__(modname)
        # ensure required callables exist
        if hasattr(_m, "analyze_excel") and hasattr(_m, "analyze_csv"):
            Analyzer = _m
            break
    except Exception as e:
        _module_err = e

if Analyzer is None:
    # Fallback: tell user exactly what to do
    raise ImportError(
        "Could not import analysis backend (duct_dp_visualizer.py).\n"
        "Place duct_dp_visualizer.py (or duct_dp_visualizer_v12.py) in the same folder as this script.\n"
        f"Last error: {_module_err}"
    )

class ScrollableFrame(ttk.Frame):
    """A simple scrollable frame with vertical scrollbar."""

    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self._win, width=e.width))
        bind_mousewheel(self.canvas)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Duct ΔP Visualizer — Tk")
        self.geometry("840x520")
        self.resizable(True, True)

        self.var_in_excel = tk.BooleanVar(value=True)
        self.var_in_csv   = tk.BooleanVar(value=False)
        self.var_zip      = tk.BooleanVar(value=True)

        self.var_xlsx = tk.StringVar()
        self.var_csv  = tk.StringVar()
        self.var_sp   = tk.StringVar()
        self.var_vp   = tk.StringVar()
        self.var_time = tk.StringVar()
        self.var_out  = tk.StringVar(value=str(Path.cwd() / "dp_viz_outputs"))
        self.var_frac = tk.DoubleVar(value=0.15)
        self.var_kmin = tk.IntVar(value=10)

        self._build()

    def _build(self):
        pad = {"padx": 6, "pady": 4}

        frm = ScrollableFrame(self)
        frm.pack(fill="both", expand=True)

        # Input type
        f1 = ttk.LabelFrame(frm.inner, text="Input")
        f1.pack(fill="x", **pad)

        ttk.Radiobutton(f1, text="Excel", variable=self.var_in_excel, value=True,
                        command=self._set_modes).grid(row=0, column=0, sticky="w", **pad)
        ttk.Radiobutton(f1, text="CSV", variable=self.var_in_excel, value=False,
                        command=self._set_modes).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(f1, text="Excel file:").grid(row=1, column=0, sticky="e", **pad)
        ttk.Entry(f1, textvariable=self.var_xlsx, width=70).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(f1, text="Browse…", command=self._browse_xlsx).grid(row=1, column=2, **pad)

        ttk.Label(f1, text="CSV file:").grid(row=2, column=0, sticky="e", **pad)
        ttk.Entry(f1, textvariable=self.var_csv, width=70).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(f1, text="Browse…", command=self._browse_csv).grid(row=2, column=2, **pad)

        ttk.Label(f1, text="CSV SP column:").grid(row=3, column=0, sticky="e", **pad)
        ttk.Entry(f1, textvariable=self.var_sp, width=24).grid(row=3, column=1, sticky="w", **pad)
        ttk.Label(f1, text="CSV VP column:").grid(row=3, column=1, sticky="e", padx=(220,4), pady=4)
        ttk.Entry(f1, textvariable=self.var_vp, width=24).grid(row=3, column=1, sticky="e", padx=(380,4), pady=4)
        ttk.Label(f1, text="CSV Time (opt):").grid(row=3, column=2, sticky="e", **pad)
        ttk.Entry(f1, textvariable=self.var_time, width=18).grid(row=3, column=2, sticky="w", padx=(110,4), pady=4)

        # Params
        f2 = ttk.LabelFrame(frm.inner, text="Parameters")
        f2.pack(fill="x", **pad)

        ttk.Label(f2, text="Bottom/Top fraction (0.05–0.40):").grid(row=0, column=0, sticky="e", **pad)
        ttk.Scale(f2, from_=0.05, to=0.40, orient="horizontal", variable=self.var_frac, length=280).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(f2, textvariable=self.var_frac, width=6).grid(row=0, column=2, sticky="w", **pad)

        ttk.Label(f2, text="Minimum segment length:").grid(row=1, column=0, sticky="e", **pad)
        ttk.Entry(f2, textvariable=self.var_kmin, width=8).grid(row=1, column=1, sticky="w", **pad)

        ttk.Checkbutton(f2, text="Zip outputs after run", variable=self.var_zip).grid(row=1, column=2, sticky="w", **pad)

        # Output + run
        f3 = ttk.LabelFrame(frm.inner, text="Run")
        f3.pack(fill="both", expand=True, **pad)

        ttk.Label(f3, text="Output folder:").grid(row=0, column=0, sticky="e", **pad)
        ttk.Entry(f3, textvariable=self.var_out, width=70).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(f3, text="Browse…", command=self._browse_out).grid(row=0, column=2, **pad)

        self.btn_run = ttk.Button(f3, text="Run", command=self._run)
        self.btn_run.grid(row=1, column=0, **pad)
        ttk.Button(f3, text="Open Output", command=self._open_out).grid(row=1, column=1, sticky="w", **pad)

        self.txt = scrolledtext.ScrolledText(f3, height=14, wrap="word")
        self.txt.grid(row=2, column=0, columnspan=3, sticky="nsew", **pad)
        f3.rowconfigure(2, weight=1); f3.columnconfigure(1, weight=1)

        self._set_modes()

    def _set_modes(self):
        is_excel = self.var_in_excel.get()
        # radio hack: var_in_excel True means Excel; False means CSV
        for child in self.children.values():
            pass
        # enable/disable relevant widgets
        # We will just enable both file boxes; but CSV columns only make sense in CSV mode.
        # The UX is simpler than micro-disabling each control.
        return

    def _browse_xlsx(self):
        p = filedialog.askopenfilename(filetypes=[("Excel files","*.xlsx;*.xls")])
        if p:
            self.var_xlsx.set(p)

    def _browse_csv(self):
        p = filedialog.askopenfilename(filetypes=[("CSV","*.csv")])
        if p:
            self.var_csv.set(p)

    def _browse_out(self):
        p = filedialog.askdirectory()
        if p:
            self.var_out.set(p)

    def _open_out(self):
        out = Path(self.var_out.get()).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(out)  # Windows
        except Exception:
            import subprocess
            for cmd in (["open", str(out)], ["xdg-open", str(out)]):
                try:
                    subprocess.Popen(cmd); break
                except Exception:
                    pass

    def log(self, msg):
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def _run(self):
        # Disable button, run in thread
        self.btn_run.config(state="disabled")
        threading.Thread(target=self._run_task, daemon=True).start()

    def _run_task(self):
        try:
            out_dir = Path(self.var_out.get()).expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            from duct_dp_visualizer import SliceParams, zip_outputs  # prefer base; fallback below
        except Exception:
            try:
                from duct_dp_visualizer_v12 import SliceParams, zip_outputs
            except Exception as e:
                messagebox.showerror("Import error", f"Missing backend: {e}")
                self.btn_run.config(state="normal")
                return

        params = SliceParams(frac=float(self.var_frac.get()), kmin=int(self.var_kmin.get()))
        files = []

        try:
            if self.var_in_excel.get():
                xlsx = self.var_xlsx.get().strip()
                if not xlsx:
                    raise ValueError("Choose an Excel file.")
                self.log(f"Analyzing Excel: {xlsx}")
                files += Analyzer.analyze_excel(Path(xlsx), out_dir, params)
            else:
                csv = self.var_csv.get().strip()
                sp  = self.var_sp.get().strip()
                vp  = self.var_vp.get().strip()
                tm  = self.var_time.get().strip() or None
                if not (csv and sp and vp):
                    raise ValueError("CSV mode requires file + SP/VP column names.")
                self.log(f"Analyzing CSV: {csv} [SP={sp}, VP={vp}, Time={tm or '—'}]")
                files += Analyzer.analyze_csv(Path(csv), out_dir, params, sp, vp, tm)

            if self.var_zip.get():
                zip_path = out_dir / "dp_viz_outputs.zip"
                Analyzer.zip_outputs([Path(f) for f in files], zip_path)
                self.log(f"Zipped -> {zip_path}")
            self.log("Done.")
            messagebox.showinfo("Done", "Analysis complete.")
        except Exception as e:
            self.log("ERROR: " + str(e))
            self.log(traceback.format_exc())
            messagebox.showerror("Error", str(e))
        finally:
            self.btn_run.config(state="normal")


if __name__ == "__main__":
    app = App()
    app.mainloop()
