from __future__ import annotations

from pathlib import Path
import json
import threading
import queue
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

            res = run_easy_legacy(
                self.src,
                self.preset,
                self.baro,
                self.stamp,
                output_base=self.out_dir,
                progress_cb=lambda s: self.queue.put(("progress", s)),
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
        try:
            if h_text:
                geom_override["duct_height_m"] = float(h_text)
            if w_text:
                geom_override["duct_width_m"] = float(w_text)
            if t_text:
                geom_override["throat_diameter_m"] = float(t_text)
        except ValueError:
            self._append_log("⚠️ Geometry values must be numeric.")
            return
        if geom_override:
            preset = SitePreset(
                name=preset.name,
                geometry={**preset.geometry, **geom_override},
                instruments=preset.instruments,
                defaults=preset.defaults,
            )

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
        self._append_log(f"❌ Failed: {msg}")

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

