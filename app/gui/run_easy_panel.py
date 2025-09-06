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
        # Preset row
        row1 = ttk.Frame(self)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Site preset:").pack(side="left")
        self.preset_var = tk.StringVar(value=next(iter(self._presets.keys())))
        self.preset = ttk.Combobox(
            row1,
            textvariable=self.preset_var,
            values=sorted(self._presets.keys()),
            state="readonly",
        )
        self.preset.pack(side="left", fill="x", expand=True)

        # Input row
        row2 = ttk.Frame(self)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="Workbook or folder:").pack(side="left")
        self.path_var = tk.StringVar()
        self.path = ttk.Entry(row2, textvariable=self.path_var)
        self.path.pack(side="left", fill="x", expand=True)
        ttk.Button(row2, text="Browse…", command=self._browse).pack(side="left")

        # Output directory
        row3 = ttk.Frame(self)
        row3.pack(fill="x", pady=2)
        ttk.Label(row3, text="Output directory:").pack(side="left")
        self.outdir_var = tk.StringVar()
        self.outdir = ttk.Entry(row3, textvariable=self.outdir_var)
        self.outdir.pack(side="left", fill="x", expand=True)
        ttk.Button(row3, text="Browse…", command=self._browse_outdir).pack(side="left")

        # Baro override
        row4 = ttk.Frame(self)
        row4.pack(fill="x", pady=2)
        ttk.Label(row4, text="Baro override (Pa, optional):").pack(side="left")
        self.baro_var = tk.StringVar()
        self.baro = ttk.Entry(row4, textvariable=self.baro_var)
        self.baro.pack(side="left", fill="x", expand=True)

        # Run stamp
        row5 = ttk.Frame(self)
        row5.pack(fill="x", pady=2)
        ttk.Label(row5, text="Run stamp (optional):").pack(side="left")
        self.stamp_var = tk.StringVar()
        self.stamp = ttk.Entry(row5, textvariable=self.stamp_var)
        self.stamp.pack(side="left", fill="x", expand=True)

        # Geometry fields
        row6 = ttk.Frame(self)
        row6.pack(fill="x", pady=2)
        ttk.Label(row6, text="Duct height (m, optional):").pack(side="left")
        self.height_var = tk.StringVar()
        self.height = ttk.Entry(row6, textvariable=self.height_var)
        self.height.pack(side="left", fill="x", expand=True)

        row7 = ttk.Frame(self)
        row7.pack(fill="x", pady=2)
        ttk.Label(row7, text="Duct width (m, optional):").pack(side="left")
        self.width_var = tk.StringVar()
        self.width = ttk.Entry(row7, textvariable=self.width_var)
        self.width.pack(side="left", fill="x", expand=True)

        row8 = ttk.Frame(self)
        row8.pack(fill="x", pady=2)
        ttk.Label(row8, text="Throat diameter (m, optional):").pack(side="left")
        self.throat_var = tk.StringVar()
        self.throat = ttk.Entry(row8, textvariable=self.throat_var)
        self.throat.pack(side="left", fill="x", expand=True)

        # Process button
        row9 = ttk.Frame(self)
        row9.pack(fill="x", pady=2)
        self.btn = ttk.Button(row9, text="Process", command=self._process)
        self.btn.pack()

        # Log
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
        for widget in [
            self.path,
            self.preset,
            self.baro,
            self.outdir,
            self.stamp,
            self.height,
            self.width,
            self.throat,
        ]:
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

        # Optional geometry overrides
        geom_override = {}
        h_text = self.height_var.get().strip()
        w_text = self.width_var.get().strip()
        t_text = self.throat_var.get().strip()
        dd_text = self.duct_diam_var.get().strip() if hasattr(self, "duct_diam_var") else ""
        tw_text = self.throat_width_var.get().strip() if hasattr(self, "throat_width_var") else ""
        th_text = self.throat_height_var.get().strip() if hasattr(self, "throat_height_var") else ""
        as_text = self.static_port_area_var.get().strip() if hasattr(self, "static_port_area_var") else ""
        at_text = self.total_port_area_var.get().strip() if hasattr(self, "total_port_area_var") else ""
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

        if geom_override:
            pgeom.update(geom_override)
            preset = SitePreset(
                name=preset.name,
                geometry=pgeom,
                instruments=preset.instruments,
                defaults=preset.defaults,
            )

        # Preflight gate for one-click completeness: need A1, At, As, At_ports
        A1 = pgeom.get("duct_area_m2") or (
            (float(h_text) * float(w_text)) if (h_text and w_text) else None
        )
        At = pgeom.get("throat_area_m2") or (
            (float(tw_text) * float(th_text)) if (tw_text and th_text) else None
        )
        As = pgeom.get("static_port_area_m2")
        At_ports = pgeom.get("total_port_area_m2")
        if not (A1 and At and As and At_ports):
            self._append_log(
                "⚠️ Need duct area A1, throat area At, and Kiel areas As & At_ports for full pipeline."
            )
            return
        try:
            r = float(As) / float(At_ports)
            beta = (float(At) / float(A1)) ** 0.5
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

