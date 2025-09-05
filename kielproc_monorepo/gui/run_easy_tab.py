from __future__ import annotations
import json
import threading
from pathlib import Path
import os

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

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


class RunEasyTab(ttk.Frame):
    """Minimal operator UI: preset → input → Process.

    Insert as the first tab of the main ttk.Notebook to shift others to the right.
    """

    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self._presets = _available_presets()
        self._build()

    # ------------------------------------------------------------------ UI helpers
    def _build(self):
        pad = {"padx": 4, "pady": 4}
        row = 0

        ttk.Label(self, text="Site preset:").grid(row=row, column=0, sticky="w", **pad)
        self.preset = ttk.Combobox(
            self, values=sorted(self._presets.keys()), state="readonly"
        )
        self.preset.current(0)
        self.preset.grid(row=row, column=1, sticky="ew", **pad)
        row += 1

        ttk.Label(self, text="Workbook or folder:").grid(row=row, column=0, sticky="w", **pad)
        self.path = ttk.Entry(self)
        self.path.grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(self, text="Browse…", command=self._browse).grid(
            row=row, column=2, sticky="ew", **pad
        )
        row += 1

        ttk.Label(self, text="Output directory:").grid(row=row, column=0, sticky="w", **pad)
        self.outdir = ttk.Entry(self)
        self.outdir.grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(self, text="Browse…", command=self._browse_outdir).grid(
            row=row, column=2, sticky="ew", **pad
        )
        row += 1

        ttk.Label(self, text="Baro override (Pa, optional):").grid(
            row=row, column=0, sticky="w", **pad
        )
        self.baro = ttk.Spinbox(self, from_=0.0, to=300000.0, increment=0.1)
        self.baro.set("0.0")
        self.baro.grid(row=row, column=1, sticky="ew", **pad)
        row += 1

        ttk.Label(self, text="Run stamp (optional):").grid(row=row, column=0, sticky="w", **pad)
        self.stamp = ttk.Entry(self)
        self.stamp.grid(row=row, column=1, sticky="ew", **pad)
        row += 1

        self.btn = ttk.Button(self, text="Process", command=self._process)
        self.btn.grid(row=row, column=0, columnspan=3, **pad)
        row += 1

        self.log = scrolledtext.ScrolledText(self, height=12, state="disabled")
        self.log.grid(row=row, column=0, columnspan=3, sticky="nsew", **pad)

        for c in range(3):
            self.columnconfigure(c, weight=1)
        self.rowconfigure(row, weight=1)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select workbook", filetypes=[("Excel", "*.xlsx *.xls")]
        )
        if path:
            self.path.delete(0, tk.END)
            self.path.insert(0, path)

    def _browse_outdir(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.outdir.delete(0, tk.END)
            self.outdir.insert(0, d)

    def _append_log(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert(tk.END, msg + "\n")
        self.log.configure(state="disabled")
        self.log.see(tk.END)

    def _log(self, msg: str):
        self.after(0, self._append_log, msg)

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        widgets = [self.btn, self.path, self.preset, self.baro, self.outdir, self.stamp]
        for w in widgets:
            w.configure(state=state)

    # ------------------------------------------------------------------ orchestration
    def _process(self):
        src_text = self.path.get().strip()
        if not src_text:
            self._log("⚠️ Provide a workbook or directory path.")
            return
        src = Path(src_text)
        name = self.preset.get()
        preset = self._presets[name]
        baro = float(self.baro.get()) or None
        out_dir_text = self.outdir.get().strip()
        out_dir = Path(out_dir_text) if out_dir_text else None
        stamp = self.stamp.get().strip() or None

        def runner():
            try:
                cwd = Path.cwd()
                if out_dir:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    os.chdir(out_dir)
                self._log("▶︎ Starting…")
                res = run_easy_legacy(
                    src,
                    preset,
                    baro,
                    stamp,
                    progress_cb=self._log,
                )
                summary: dict = {}
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
                    out_path = (out_dir or cwd) / out_path
                self._log("✅ Done.")
                self._log(f"Results: {out_path}")

                manifest = out_path / "summary.json"
                try:
                    data = json.loads(manifest.read_text())
                    tables = data.get("tables", [])
                    plots = data.get("plots", [])
                    self._log(f"Tables: {len(tables)}, Plots: {len(plots)}")
                    kv = data.get("key_values", {})
                    if kv:
                        self._log(", ".join(f"{k}={v}" for k, v in kv.items()))
                except Exception:
                    self._log("Summary unavailable.")

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
                    self._log(f"❌ {e}")
                for w in warns:
                    self._log(f"⚠️ {w}")
            except Exception as e:  # pragma: no cover - exercised via unit tests
                self._log(f"❌ Failed: {e}")
            finally:
                if out_dir:
                    os.chdir(cwd)
                self.after(0, self._set_busy, False)

        self._set_busy(True)
        threading.Thread(target=runner, daemon=True).start()


def insert_run_easy_tab(nb: ttk.Notebook):
    """Insert the Run Easy tab into *nb* at index 0 and select it.

    Returns the created :class:`RunEasyTab` instance.
    """

    tab = RunEasyTab(nb)
    nb.insert(0, tab, text="Run Easy")
    try:
        nb.select(tab)
    except Exception:
        pass
    return tab
