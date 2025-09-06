from __future__ import annotations

from pathlib import Path
import json
import threading
import queue
import inspect
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText

from kielproc.run_easy import run_easy_legacy, SitePreset
from kielproc.cli import PRESETS as kp_presets


def _available_presets():
    presets = dict(kp_presets)
    if not presets:
        presets = {
            "DefaultSite": SitePreset(
                name="DefaultSite", geometry={}, instruments={}, defaults={}
            )
        }
    return presets


class _Runner(threading.Thread):
    """Background runner for the pipeline."""

    def __init__(
        self,
        src: Path,
        preset,
        baro: float | None,
        stamp: str | None,
        out_dir: Path | None,
        queue: queue.Queue,
    ):
        super().__init__(daemon=True)
        self.src = src
        self.preset = preset
        self.baro = baro
        self.stamp = stamp
        self.out_dir = out_dir
        self.queue = queue

    def run(self):
        self.queue.put(("started",))
        try:
            if self.out_dir:
                self.out_dir.mkdir(parents=True, exist_ok=True)

            kwargs = {
                "output_base": self.out_dir,
                "progress_cb": lambda s: self.queue.put(("progress", s)),
            }
            if "strict" in inspect.signature(run_easy_legacy).parameters:
                kwargs["strict"] = True
            res = run_easy_legacy(
                self.src,
                self.preset,
                self.baro,
                self.stamp,
                **kwargs,
            )

            summary = {}
            artifacts: list[str] = []
            out = res
            if isinstance(res, tuple):
                out = res[0]
                if len(res) > 1 and isinstance(res[1], dict):
                    summary = res[1]
                if len(res) > 2 and isinstance(res[2], list):
                    artifacts = [str(p) for p in res[2]]
            elif hasattr(res, "summary"):
                summary = getattr(res, "summary", {})
                artifacts = [str(p) for p in getattr(res, "artifacts", [])]
                out = getattr(res, "run_dir", out)

            out_path = Path(out)
            if not out_path.is_absolute():
                out_path = (self.out_dir or Path.cwd()) / out_path
            self.queue.put(("finished", str(out_path), summary, artifacts))
        except Exception as e:
            self.queue.put(("failed", str(e)))


class RunEasyPanel(ttk.Frame):
    """Minimal operator UI: preset → input → Process."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._presets = _available_presets()
        self._runner: _Runner | None = None
        self._queue: queue.Queue | None = None
        self._build()

    def _build(self):
        """Construct the UI with separate tabs for inputs and geometry."""

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="x", pady=2)

        tab_inputs = ttk.Frame(self.nb)
        tab_geom = ttk.Frame(self.nb)
        self.nb.add(tab_inputs, text="Inputs")
        self.nb.add(tab_geom, text="Geometry")

        # --- Inputs tab -------------------------------------------------
        row1 = ttk.Frame(tab_inputs)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Site preset:").pack(side="left")
        self.preset_var = tk.StringVar(value=next(iter(self._presets.keys())))
        self.preset = ttk.Combobox(
            row1,
            textvariable=self.preset_var,
            values=sorted(self._presets.keys()),
            state="readonly",
            width=20,
        )
        self.preset.pack(side="left")

        row2 = ttk.Frame(tab_inputs)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="Workbook or folder:").pack(side="left")
        self.path_var = tk.StringVar()
        self.path = ttk.Entry(row2, textvariable=self.path_var, width=40)
        self.path.pack(side="left")
        ttk.Button(row2, text="Browse…", command=self._browse).pack(side="left")

        row3 = ttk.Frame(tab_inputs)
        row3.pack(fill="x", pady=2)
        ttk.Label(row3, text="Output directory:").pack(side="left")
        self.outdir_var = tk.StringVar()
        self.outdir = ttk.Entry(row3, textvariable=self.outdir_var, width=40)
        self.outdir.pack(side="left")
        ttk.Button(row3, text="Browse…", command=self._browse_outdir).pack(side="left")

        row4 = ttk.Frame(tab_inputs)
        row4.pack(fill="x", pady=2)
        ttk.Label(row4, text="Baro override (Pa, optional):").pack(side="left")
        self.baro_var = tk.StringVar()
        self.baro = ttk.Entry(row4, textvariable=self.baro_var, width=10)
        self.baro.pack(side="left")

        row5 = ttk.Frame(tab_inputs)
        row5.pack(fill="x", pady=2)
        ttk.Label(row5, text="Run stamp (optional):").pack(side="left")
        self.stamp_var = tk.StringVar()
        self.stamp = ttk.Entry(row5, textvariable=self.stamp_var, width=15)
        self.stamp.pack(side="left")

        row6 = ttk.Frame(tab_inputs)
        row6.pack(fill="x", pady=2)
        ttk.Label(row6, text="Translation β (optional):").pack(side="left")
        self.beta_fit_var = tk.StringVar()
        self.beta_fit = ttk.Entry(row6, textvariable=self.beta_fit_var, width=10)
        self.beta_fit.pack(side="left")

        row9 = ttk.Frame(tab_inputs)
        row9.pack(fill="x", pady=2)
        self.btn = ttk.Button(row9, text="Process", command=self._process)
        self.btn.pack()

        # --- Geometry tab ----------------------------------------------
        g1 = ttk.Frame(tab_geom)
        g1.pack(fill="x", pady=2)
        ttk.Label(g1, text="Duct height (m, optional):").pack(side="left")
        self.height_var = tk.StringVar()
        self.height = ttk.Entry(g1, textvariable=self.height_var, width=10)
        self.height.pack(side="left")

        g2 = ttk.Frame(tab_geom)
        g2.pack(fill="x", pady=2)
        ttk.Label(g2, text="Duct width (m, optional):").pack(side="left")
        self.width_var = tk.StringVar()
        self.width = ttk.Entry(g2, textvariable=self.width_var, width=10)
        self.width.pack(side="left")

        g3 = ttk.Frame(tab_geom)
        g3.pack(fill="x", pady=2)
        ttk.Label(g3, text="Throat diameter (m, optional):").pack(side="left")
        self.throat_var = tk.StringVar()
        self.throat = ttk.Entry(g3, textvariable=self.throat_var, width=10)
        self.throat.pack(side="left")

        g4 = ttk.Frame(tab_geom)
        g4.pack(fill="x", pady=2)
        ttk.Label(g4, text="Duct diameter (m, optional):").pack(side="left")
        self.duct_diam_var = tk.StringVar()
        self.duct_diam = ttk.Entry(g4, textvariable=self.duct_diam_var, width=10)
        self.duct_diam.pack(side="left")

        g5 = ttk.Frame(tab_geom)
        g5.pack(fill="x", pady=2)
        ttk.Label(g5, text="Throat width (m, optional):").pack(side="left")
        self.throat_width_var = tk.StringVar()
        self.throat_width = ttk.Entry(g5, textvariable=self.throat_width_var, width=10)
        self.throat_width.pack(side="left")

        g6 = ttk.Frame(tab_geom)
        g6.pack(fill="x", pady=2)
        ttk.Label(g6, text="Throat height (m, optional):").pack(side="left")
        self.throat_height_var = tk.StringVar()
        self.throat_height = ttk.Entry(g6, textvariable=self.throat_height_var, width=10)
        self.throat_height.pack(side="left")

        g7 = ttk.Frame(tab_geom)
        g7.pack(fill="x", pady=2)
        ttk.Label(g7, text="Static Kiel area As (m², optional):").pack(side="left")
        self.static_port_area_var = tk.StringVar()
        self.static_port_area = ttk.Entry(g7, textvariable=self.static_port_area_var, width=10)
        self.static_port_area.pack(side="left")

        g8 = ttk.Frame(tab_geom)
        g8.pack(fill="x", pady=2)
        ttk.Label(g8, text="Total Kiel area At_ports (m², optional):").pack(side="left")
        self.total_port_area_var = tk.StringVar()
        self.total_port_area = ttk.Entry(g8, textvariable=self.total_port_area_var, width=10)
        self.total_port_area.pack(side="left")

        g9 = ttk.Frame(tab_geom)
        g9.pack(fill="x", pady=2)
        ttk.Label(g9, text="Venturi β (optional):").pack(side="left")
        self.beta_var = tk.StringVar()
        self.beta = ttk.Entry(g9, textvariable=self.beta_var, width=10)
        self.beta.pack(side="left")

        # --- Log --------------------------------------------------------
        self.log = ScrolledText(self, height=10, state="normal")
        self.log.pack(fill="both", expand=True, pady=2)

    # UI helpers
    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select workbook", filetypes=[("Excel", "*.xlsx *.xls")]
        )
        if not path:
            path = filedialog.askdirectory(title="Select directory") or ""
        if path:
            self.path_var.set(path)

    def _browse_outdir(self):
        d = filedialog.askdirectory(title="Select output directory") or ""
        if d:
            self.outdir_var.set(d)

    def _append_log(self, text: str):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _set_busy(self, busy: bool):
        state = ["disabled"] if busy else ["!disabled"]
        self.btn.state(state)
        widgets = [
            self.path,
            self.preset,
            self.baro,
            self.outdir,
            self.stamp,
            self.beta_fit,
            self.height,
            self.width,
            self.throat,
            getattr(self, "duct_diam", None),
            getattr(self, "throat_width", None),
            getattr(self, "throat_height", None),
            getattr(self, "static_port_area", None),
            getattr(self, "total_port_area", None),
            getattr(self, "beta", None),
        ]
        for widget in widgets:
            if widget is not None:
                widget.state(state)

    # Orchestration
    def _process(self):
        p = self.path_var.get().strip()
        if not p:
            self._append_log("⚠️ Provide a workbook or directory path.")
            return
        src = Path(p)
        name = self.preset_var.get()
        preset = self._presets[name]
        baro_text = self.baro_var.get().strip()
        baro = float(baro_text) if baro_text else None
        out_dir_text = self.outdir_var.get().strip()
        out_dir = Path(out_dir_text).expanduser().resolve() if out_dir_text else None
        stamp = self.stamp_var.get().strip() or None
        beta_fit_text = (
            (self.beta_fit_var.get() if hasattr(self, "beta_fit_var") else "")
        )
        beta_fit_text = (beta_fit_text or "").strip()

        # Optional geometry overrides
        geom_override = {}
        defaults_override = {}
        h_text = (self.height_var.get() or "").strip()
        w_text = (self.width_var.get() or "").strip()
        t_text = (self.throat_var.get() or "").strip()
        dd_text = (self.duct_diam_var.get() or "").strip() if hasattr(self, "duct_diam_var") else ""
        tw_text = (self.throat_width_var.get() or "").strip() if hasattr(self, "throat_width_var") else ""
        th_text = (self.throat_height_var.get() or "").strip() if hasattr(self, "throat_height_var") else ""
        as_text = (self.static_port_area_var.get() or "").strip() if hasattr(self, "static_port_area_var") else ""
        at_text = (self.total_port_area_var.get() or "").strip() if hasattr(self, "total_port_area_var") else ""
        beta_text = (self.beta_var.get() or "").strip() if hasattr(self, "beta_var") else ""
        try:
            if h_text:
                geom_override["duct_height_m"] = float(h_text)
            if w_text:
                geom_override["duct_width_m"] = float(w_text)
            if t_text:
                # convert diameter → area for downstream beta_from_geometry
                import math
                dt = float(t_text)
                geom_override["throat_area_m2"] = math.pi * (dt**2) / 4.0
            if dd_text:
                import math
                d = float(dd_text)
                geom_override["duct_area_m2"] = math.pi * (d**2) / 4.0
            if tw_text and th_text:
                geom_override["throat_width_m"] = float(tw_text)
                geom_override["throat_height_m"] = float(th_text)
            if as_text:
                geom_override["static_port_area_m2"] = float(as_text)
            if at_text:
                geom_override["total_port_area_m2"] = float(at_text)
            if beta_text:
                geom_override["beta"] = float(beta_text)
            if beta_fit_text:
                defaults_override["beta_translate"] = float(beta_fit_text)
        except ValueError:
            self._append_log("⚠️ Geometry values must be numeric.")
            return

        # Convert diameter-only values in the PRESET too (so beta/r can be derived)
        import math
        pgeom = {**preset.geometry}
        if ("duct_area_m2" not in pgeom) and (pgeom.get("duct_diameter_m")):
            try:
                d = float(pgeom["duct_diameter_m"])
                pgeom["duct_area_m2"] = math.pi * (d**2) / 4.0
            except Exception:
                pass
        if ("throat_area_m2" not in pgeom) and (pgeom.get("throat_diameter_m")):
            try:
                dt = float(pgeom["throat_diameter_m"])
                pgeom["throat_area_m2"] = math.pi * (dt**2) / 4.0
            except Exception:
                pass

        if geom_override or defaults_override:
            pgeom.update(geom_override)
            defaults = dict(preset.defaults)
            defaults.update(defaults_override)
            preset = SitePreset(
                name=preset.name,
                geometry=pgeom,
                instruments=preset.instruments,
                defaults=defaults,
            )

        # Preflight gate for one-click completeness: need r and beta
        A1 = pgeom.get("duct_area_m2") or (
            (float(h_text) * float(w_text)) if (h_text and w_text) else None
        )
        At = pgeom.get("throat_area_m2") or (
            (float(tw_text) * float(th_text)) if (tw_text and th_text) else None
        )
        As = pgeom.get("static_port_area_m2")
        At_ports = pgeom.get("total_port_area_m2")
        beta_override = pgeom.get("beta")
        if not (As and At_ports and (beta_override or (A1 and At))):
            self._append_log(
                "⚠️ Need duct area A1 and throat area At (or β) and Kiel areas As & At_ports for full pipeline."
            )
            return
        try:
            r = float(As) / float(At_ports)
            beta = float(beta_override) if beta_override is not None else (float(At) / float(A1)) ** 0.5
            self._append_log(f"Derived: r={r:.4g}, β={beta:.4g}")
        except Exception:
            pass

        self._queue = queue.Queue()
        self._runner = _Runner(src, preset, baro, stamp, out_dir, self._queue)
        self._runner.start()
        self._set_busy(True)
        self._append_log("▶︎ Starting…")
        self.after(100, self._poll_queue)

    def _poll_queue(self):
        assert self._queue is not None
        try:
            while True:
                msg = self._queue.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    self._append_log(msg[1])
                elif kind == "failed":
                    self._on_failed(msg[1])
                    self._set_busy(False)
                    return
                elif kind == "finished":
                    self._on_finished(msg[1], msg[2], msg[3])
                    self._set_busy(False)
                    return
        except queue.Empty:
            if self._runner and self._runner.is_alive():
                self.after(100, self._poll_queue)

    def _on_failed(self, msg: str):
        friendly = msg.replace("deltpVent", "Δp (Venturi)")
        self._append_log(f"❌ {friendly}")

    def _on_finished(self, out_dir: str, summary: dict, artifacts: list[str]):
        self._append_log("✅ Done.")
        self._append_log(f"Open results directory: {out_dir}")

        manifest = Path(out_dir) / "summary.json"
        try:
            data = json.loads(manifest.read_text())
            tables = data.get("tables", [])
            plots = data.get("plots", [])
            self._append_log(f"Tables: {len(tables)}, Plots: {len(plots)}")
            kv = data.get("key_values", {})
            if kv:
                kv_text = ", ".join(f"{k}={v}" for k, v in kv.items())
                self._append_log(kv_text)
        except Exception:
            self._append_log("Summary unavailable.")

        warns = list(summary.get("warnings", [])) if summary else []
        errs = list(summary.get("errors", [])) if summary else []
        for p in artifacts:
            if p.endswith("__parse_summary.json"):
                try:
                    j = json.loads(Path(p).read_text())
                    warns.extend(j.get("warnings", []))
                    errs.extend(j.get("errors", []))
                except Exception:
                    continue
        for e in errs:
            self._append_log(f"❌ {e}")
        for w in warns:
            self._append_log(f"⚠️ {w}")

