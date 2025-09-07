#!/usr/bin/env python3
"""
RunEasy — Tk GUI wrapper for kielproc.run_easy

Purpose
  • Single-window, operator-proof GUI to run the full SOP pipeline ("Run-Easy") with zero terminal use.
  • Shows key values (α, β, lag, transmitter span & setpoints) and lists plots/tables from summary.json.
  • Double-click to open artifacts; preview PNGs inline; open run folder / bundle zip.
  • API‑first (kielproc.run_easy.run_all); silent fallback to CLI `kielproc one-click`.

Nice-to-haves
  • Remembers last paths/site/baro in a per-user config file.
  • Optional modern theme if `sv_ttk` is installed; otherwise ttk default.
  • No non‑stdlib hard deps (Pillow preview is optional if present).

Packaging
  • `pip install .` (your package) then `python runeasy_gui.py` OR expose as console_script: kielproc-gui
  • One-file app: `pyinstaller --noconsole --onefile --name RunEasy-GUI runeasy_gui.py`

This file is standalone. Drop it anywhere that can import `kielproc`.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import queue
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------- Optional extras ----------
try:  # optional modern ttk theme
    import sv_ttk  # type: ignore
except Exception:  # pragma: no cover
    sv_ttk = None  # noqa: N816

try:  # optional image preview (falls back to tk only)
    from PIL import Image, ImageTk  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageTk = None  # type: ignore

APP_NAME = "RunEasy-GUI"
CONFIG_DIR = Path.home() / ".kielproc_gui"
CONFIG_FILE = CONFIG_DIR / "config.json"
SITE_PRESETS_TXT = Path(__file__).with_name("site_presets.txt")
SITE_PRESETS_JSON = Path(__file__).with_name("site_presets.json")

PNG_EXTS = {".png", ".gif", ".ppm"}

# ----------------- helpers -----------------

def _open_path(p: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(p)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(p)], check=False)
        else:
            subprocess.run(["xdg-open", str(p)], check=False)
    except Exception as e:
        messagebox.showerror("Open failed", f"{e}")


def _newest_run_dir(base: Path) -> Optional[Path]:
    runs = [p for p in base.glob("RUN_*") if p.is_dir()]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def _read_text_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _try_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None


@dataclass
class RunResult:
    run_dir: Path
    summary: dict


# ----------------- core calls -----------------

def call_runeasy_api_or_cli(
    input_path: Path, site: Optional[str], baro: Optional[str], logfn, strict: bool = False
) -> RunResult:
    """Run via API first; fall back to CLI. Return RunResult."""
    # 1) API path
    try:
        logfn("Using kielproc.run_easy API…")
        from kielproc.run_easy import run_all  # now valid
        rd = run_all(
            str(input_path),
            site=(site or "DefaultSite"),
            baro_override=_try_float(baro),
            strict=strict,
        )
        if isinstance(rd, (str, Path)):
            run_dir = Path(rd)
        elif isinstance(rd, dict):
            p = rd.get("run_dir") or rd.get("output_dir") or rd.get("path")
            run_dir = Path(p) if p else (_newest_run_dir(Path.cwd()) or None)
        else:
            run_dir = _newest_run_dir(Path.cwd())
        if not run_dir:
            raise RuntimeError("API finished but no RUN_* directory was found.")
        smry = _read_text_json(Path(run_dir) / "summary.json")
        return RunResult(run_dir=Path(run_dir), summary=smry)
    except Exception as e:
        logfn(f"API path unavailable: {e}. Falling back to CLI…")

    # 2) CLI path
    cmd = ["kielproc", "one-click", str(input_path)]
    if site:
        cmd += ["--site", site]
    if baro:
        cmd += ["--baro", baro]
    if strict:
        cmd.append("--strict")
    logfn("Running CLI: " + " ".join(cmd))
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        if proc.stdout:
            logfn(proc.stdout.rstrip())
    except FileNotFoundError:
        raise RuntimeError("kielproc CLI not found and API import failed. Ensure kielproc is installed.")

    run_dir = _newest_run_dir(Path.cwd())
    if not run_dir:
        raise RuntimeError("Could not find a RUN_* directory after CLI execution.")
    smry = _read_text_json(run_dir / "summary.json")
    return RunResult(run_dir=run_dir, summary=smry)


# ----------------- config -----------------

def load_config() -> dict:
    try:
        if CONFIG_FILE.exists():
            return _read_text_json(CONFIG_FILE)
    except Exception:
        pass
    return {}


def save_config(cfg: dict) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_site_presets() -> list[str]:
    # priority: site_presets.json > site_presets.txt
    try:
        if SITE_PRESETS_JSON.exists():
            data = _read_text_json(SITE_PRESETS_JSON)
            if isinstance(data, list):
                return [str(x) for x in data if str(x).strip()]
    except Exception:
        pass
    try:
        if SITE_PRESETS_TXT.exists():
            return [ln.strip() for ln in SITE_PRESETS_TXT.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        pass
    return []


# ----------------- GUI -----------------
class ImagePreview(ttk.Frame):
    """Simple image preview widget (PNG/GIF/PPM). Uses Pillow if available for scaling; otherwise Tk."""

    def __init__(self, master: tk.Misc):
        super().__init__(master, padding=6)
        self.canvas = tk.Canvas(self, bg="#1e1e1e", height=260)
        self.canvas.pack(fill="both", expand=True)
        self._img_ref = None  # keep reference
        self._path: Optional[Path] = None
        self.bind("<Configure>", lambda e: self._render())

    def show(self, path: Optional[Path]):
        self._path = path
        self._render()

    def _render(self):
        cv = self.canvas
        cv.delete("all")
        if not self._path or not self._path.exists():
            cv.create_text(10, 10, anchor="nw", fill="#bbb", text="No preview")
            return
        ext = self._path.suffix.lower()
        w = max(100, cv.winfo_width())
        h = max(100, cv.winfo_height())
        try:
            if Image and ImageTk and ext in PNG_EXTS:
                im = Image.open(self._path)
                im.thumbnail((w - 16, h - 16))
                self._img_ref = ImageTk.PhotoImage(im)
            else:
                self._img_ref = tk.PhotoImage(file=str(self._path))
        except Exception:
            cv.create_text(10, 10, anchor="nw", fill="#bbb", text="Preview unsupported")
            return
        if self._img_ref:
            cv.create_image(w // 2, h // 2, image=self._img_ref, anchor="center")


class RunEasyApp(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=10)
        self.pack(fill="both", expand=True)

        # Try to load a pretty theme
        if sv_ttk:
            try:
                sv_ttk.set_theme("light")
            except Exception:
                pass

        master.title("Run-Easy — SOP Pipeline")
        master.minsize(900, 640)

        self.cfg = load_config()
        self.site_presets = load_site_presets()

        # State
        self.path_var = tk.StringVar(value=self.cfg.get("last_path", ""))
        self.site_var = tk.StringVar(value=self.cfg.get("last_site", ""))
        self.baro_var = tk.StringVar(value=str(self.cfg.get("last_baro", "")))
        self.strict_var = tk.BooleanVar(value=True)
        self.version_var = tk.StringVar(value=self._get_pkg_version())
        self.run_dir: Optional[Path] = None
        self.summary: Optional[dict] = None
        self._logq: "queue.Queue[str]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None

        # ----- Top bar -----
        top = ttk.Frame(self)
        top.pack(fill="x", pady=(2, 8))
        ttk.Label(top, text="Run-Easy", font=("Segoe UI", 14, "bold")).pack(side="left")
        ttk.Label(top, textvariable=self.version_var, foreground="#666").pack(side="left", padx=10)
        ttk.Button(top, text="Settings", command=self._settings).pack(side="right")
        ttk.Button(top, text="Help", command=self._help).pack(side="right", padx=(0, 6))

        # ----- Inputs -----
        frm = ttk.LabelFrame(self, text="Inputs")
        frm.pack(fill="x", pady=4)

        r1 = ttk.Frame(frm); r1.pack(fill="x", pady=6)
        ttk.Label(r1, text="Workbook / Folder:").pack(side="left")
        ttk.Entry(r1, textvariable=self.path_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r1, text="Browse…", command=self._browse).pack(side="left")

        r2 = ttk.Frame(frm); r2.pack(fill="x", pady=6)
        ttk.Label(r2, text="Site preset:").pack(side="left")
        if self.site_presets:
            self.site_combo = ttk.Combobox(r2, textvariable=self.site_var, values=self.site_presets, state="readonly")
            self.site_combo.pack(side="left", padx=6)
        else:
            ttk.Entry(r2, textvariable=self.site_var, width=28).pack(side="left", padx=6)
        ttk.Label(r2, text="Baro override (Pa, optional):").pack(side="left", padx=(16, 4))
        ttk.Entry(r2, textvariable=self.baro_var, width=16).pack(side="left")

        r3 = ttk.Frame(frm); r3.pack(fill="x", pady=6)
        ttk.Checkbutton(r3, text="Strict (fail-fast)", variable=self.strict_var).pack(side="left")

        # ----- Actions -----
        act = ttk.Frame(self)
        act.pack(fill="x", pady=8)
        self.btn_run = ttk.Button(act, text="Run", command=self._start)
        self.btn_run.pack(side="left")
        self.btn_open = ttk.Button(act, text="Open Run Folder", command=self._open_run, state="disabled")
        self.btn_open.pack(side="left", padx=6)
        self.btn_zip = ttk.Button(act, text="Open Bundle Zip", command=self._open_zip, state="disabled")
        self.btn_zip.pack(side="left")
        self.prog = ttk.Progressbar(act, mode="indeterminate")
        self.prog.pack(side="right", fill="x", expand=True, padx=6)

        # ----- Middle: Key values + Artifact lists + Preview -----
        mid = ttk.Frame(self); mid.pack(fill="both", expand=True)

        # Key values panel
        kv = ttk.LabelFrame(mid, text="Key values")
        kv.pack(side="left", fill="y", padx=(0, 8), pady=4)
        self.kv_alpha = tk.StringVar(); self.kv_beta = tk.StringVar()
        self.kv_lag = tk.StringVar(); self.kv_span = tk.StringVar(); self.kv_setpts = tk.StringVar()
        for lbl, var in (
            ("Pooled α (alpha):", self.kv_alpha),
            ("Pooled β (beta):", self.kv_beta),
            ("Lag (samples):", self.kv_lag),
            ("Transmitter span (Pa):", self.kv_span),
            ("Set-points (4–20 mA):", self.kv_setpts),
        ):
            row = ttk.Frame(kv); row.pack(fill="x", pady=3, padx=8)
            ttk.Label(row, text=lbl, width=26).pack(side="left")
            ttk.Label(row, textvariable=var).pack(side="left")

        # Artifacts panel
        art = ttk.LabelFrame(mid, text="Artifacts")
        art.pack(side="left", fill="both", expand=True, pady=4)

        columns = ("type", "path")
        self.tree = ttk.Treeview(art, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("type", text="Type")
        self.tree.heading("path", text="Path")
        self.tree.column("type", width=86, stretch=False)
        self.tree.column("path", width=380, stretch=True)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", self._open_selected)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        btns = ttk.Frame(art); btns.pack(fill="x")
        ttk.Button(btns, text="Open selected", command=self._open_selected).pack(side="left")
        ttk.Button(btns, text="Copy path", command=self._copy_path).pack(side="left", padx=6)

        # Preview panel
        prev = ttk.LabelFrame(mid, text="Preview")
        prev.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=4)
        self.preview = ImagePreview(prev)
        self.preview.pack(fill="both", expand=True)

        # Log panel
        lg = ttk.LabelFrame(self, text="Log")
        lg.pack(fill="both", expand=False)
        self.logw = tk.Text(lg, height=10, wrap="word", state="disabled")
        self.logw.pack(fill="both", expand=True)

        # log pump
        self.after(120, self._drain_log)

    # ----- UI helpers -----
    def _get_pkg_version(self) -> str:
        try:
            import importlib.metadata as md  # py3.8+
            v = md.version("kielproc")
            return f"kielproc {v}"
        except Exception:
            return "kielproc (version unknown)"

    def _browse(self) -> None:
        path = filedialog.askopenfilename(title="Select workbook")
        if not path:
            path = filedialog.askdirectory(title="Or select a folder")
        if path:
            self.path_var.set(path)

    def _start(self) -> None:
        p = Path(self.path_var.get().strip())
        if not p.exists():
            messagebox.showerror("Input required", "Select a workbook file or a folder.")
            return
        site = self.site_var.get().strip() or None
        baro = self.baro_var.get().strip() or None
        if baro and _try_float(baro) is None:
            messagebox.showerror("Invalid baro", "Enter barometric override in Pascals (e.g., 101325).")
            return

        # persist last values
        save_config({
            "last_path": str(p),
            "last_site": site or "",
            "last_baro": baro or "",
        })

        # reset UI
        self.btn_run.configure(state="disabled")
        self.btn_open.configure(state="disabled")
        self.btn_zip.configure(state="disabled")
        self._set_keys(alpha=None, beta=None, lag=None, span=None, setpts=None)
        self._fill_tree([])
        self.preview.show(None)
        self._log("Starting…")
        self.prog.start(12)

        def worker():
            try:
                result = call_runeasy_api_or_cli(
                    p, site, baro, self._log, strict=self.strict_var.get()
                )
                self.run_dir = result.run_dir
                self.summary = result.summary
                self._log(f"Run complete: {result.run_dir}")
                self._populate_from_summary(result.summary, result.run_dir)
                self.btn_open.configure(state="normal")
                bundle = next(result.run_dir.glob("*__bundle.zip"), None)
                self.btn_zip.configure(state=("normal" if bundle else "disabled"))
            except Exception as e:
                self._log("ERROR: " + str(e))
                self._log(traceback.format_exc())
                messagebox.showerror("Run-Easy failed", str(e))
            finally:
                self.btn_run.configure(state="normal")
                self.prog.stop()

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _settings(self) -> None:
        msg = (
            "Settings\n\n"
            "• Site presets: place 'site_presets.json' (list of strings) or 'site_presets.txt' next to this app.\n"
            "• Defaults are remembered per user in ~/.kielproc_gui/config.json.\n"
            "• This GUI calls kielproc.run_easy (API), then falls back to the CLI if needed."
        )
        messagebox.showinfo("Settings", msg)

    def _help(self) -> None:
        msg = (
            "Run-Easy usage\n\n"
            "1) Choose a workbook or folder; choose a site preset; optionally set baro override (Pa).\n"
            "2) Click Run.\n"
            "3) When finished, key values show here; open the run folder or any artifact from the list."
        )
        messagebox.showinfo("Help", msg)

    def _open_run(self) -> None:
        if self.run_dir:
            _open_path(self.run_dir)

    def _open_zip(self) -> None:
        if not self.run_dir:
            return
        z = next(self.run_dir.glob("*__bundle.zip"), None)
        if z:
            _open_path(z)

    def _on_select(self, _event=None) -> None:
        sel = self._current_tree_path()
        if sel and sel.suffix.lower() in PNG_EXTS:
            self.preview.show(sel)
        else:
            self.preview.show(None)

    def _open_selected(self, _event=None) -> None:
        p = self._current_tree_path()
        if p and p.exists():
            _open_path(p)

    def _copy_path(self) -> None:
        p = self._current_tree_path()
        if not p:
            return
        self.clipboard_clear()
        self.clipboard_append(str(p))
        self._log(f"Copied path: {p}")

    def _current_tree_path(self) -> Optional[Path]:
        item = self.tree.focus()
        if not item:
            return None
        values = self.tree.item(item, "values")
        if len(values) >= 2:
            return Path(values[1])
        return None

    def _populate_from_summary(self, smry: dict, run_dir: Path) -> None:
        kv = smry.get("key_values", {}) or {}
        alpha = kv.get("alpha")
        beta = kv.get("beta")
        lag = kv.get("lag_samples")
        span = kv.get("transmitter_span")
        setpts = kv.get("transmitter_setpoints")
        self._set_keys(alpha, beta, lag, span, setpts)

        rows: list[tuple[str, str]] = []
        for t in smry.get("plots", []) or []:
            p = (run_dir / t) if not os.path.isabs(t) else Path(t)
            rows.append(("plot", str(p)))
        for t in smry.get("tables", []) or []:
            p = (run_dir / t) if not os.path.isabs(t) else Path(t)
            rows.append(("table", str(p)))
        self._fill_tree(rows)

    def _set_keys(self, alpha, beta, lag, span, setpts) -> None:
        def fmt(v, default="—"):
            return default if v in (None, "", [], {}) else str(v)
        self.kv_alpha.set(fmt(alpha))
        self.kv_beta.set(fmt(beta))
        self.kv_lag.set(fmt(lag))
        if isinstance(span, (list, tuple)) and len(span) == 2:
            self.kv_span.set(f"{span[0]} .. {span[1]} Pa")
        else:
            self.kv_span.set(fmt(span))
        if isinstance(setpts, dict):
            self.kv_setpts.set(f"4 mA: {setpts.get('mA4')}  |  20 mA: {setpts.get('mA20')}")
        else:
            self.kv_setpts.set(fmt(setpts))

    def _fill_tree(self, rows: list[tuple[str, str]]):
        self.tree.delete(*self.tree.get_children())
        for typ, path in rows:
            self.tree.insert("", "end", values=(typ, path))

    # ----- logging pump -----
    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self._logq.put(f"[{stamp}] {msg}")

    def _drain_log(self) -> None:
        try:
            while True:
                line = self._logq.get_nowait()
                self.logw.configure(state="normal")
                self.logw.insert("end", line + "\n")
                self.logw.see("end")
                self.logw.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(120, self._drain_log)


def main() -> None:
    root = tk.Tk()
    try:
        root.iconbitmap(default="")  # no external icon dependency
    except Exception:
        pass
    app = RunEasyApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
