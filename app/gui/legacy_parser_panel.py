from __future__ import annotations

import sys, os, json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# Repo root: .../<repo_root> (contains 'app' and 'kielproc_monorepo')
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Legacy parser package root: .../kielproc_monorepo/tools/legacy_parser
_LEGACY_PKG_ROOT = _REPO_ROOT / "kielproc_monorepo" / "tools" / "legacy_parser"
if str(_LEGACY_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEGACY_PKG_ROOT))

try:
    # The package layout is tools/legacy_parser/legacy_parser/parser.py
    from legacy_parser.parser import parse_legacy_workbook  # type: ignore
except Exception as e:
    parse_legacy_workbook = None  # type: ignore
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


class LegacyParserPanel(ttk.Frame):
    """
    Notebook tab embedding the Legacy XLSX → CSV parser.
    Mirrors tools/legacy_parser/parser_gui.py but as a Frame.
    """

    def __init__(self, master: tk.Misc | None = None) -> None:
        super().__init__(master)

        self.var_infile = tk.StringVar()
        self.var_outdir = tk.StringVar()
        self.var_folder = tk.StringVar()
        self.var_thresh = tk.StringVar(value="1e-6")

        self.columnconfigure(1, weight=1)
        row = 0

        ttk.Label(self, text="Legacy XLSX → CSV Parser", font=("", 12, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", pady=(4, 8)); row += 1

        ttk.Label(self, text="XLSX workbook").grid(row=row, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.var_infile).grid(row=row, column=1, sticky="ew")
        ttk.Button(self, text="Browse…", command=self._pick_xlsx).grid(row=row, column=2, sticky="w"); row += 1

        ttk.Label(self, text="Output base directory").grid(row=row, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(self, textvariable=self.var_outdir).grid(row=row, column=1, sticky="ew", pady=(4, 0))
        ttk.Button(self, text="Choose…", command=self._pick_outdir).grid(row=row, column=2, sticky="w", pady=(4, 0)); row += 1

        ttk.Label(self, text="Folder name (optional)").grid(row=row, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.var_folder).grid(row=row, column=1, sticky="ew"); row += 1

        ttk.Label(self, text="Piccolo flat threshold").grid(row=row, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.var_thresh, width=10).grid(row=row, column=1, sticky="w"); row += 1

        self.btn = ttk.Button(self, text="Parse", command=self._run_parse)
        self.btn.grid(row=row, column=1, sticky="w", pady=6); row += 1

        self.log = ScrolledText(self, height=12, state="disabled")
        self.log.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(6, 0)); row += 1
        self.rowconfigure(row-1, weight=1)

        if _IMPORT_ERR is not None:
            self._log(f"[import] legacy parser unavailable: {_IMPORT_ERR}")

    # --- UI helpers ------------------------------------------------------
    def _pick_xlsx(self):
        p = filedialog.askopenfilename(title="Select legacy workbook", filetypes=[("Excel files", "*.xlsx *.xls *.xlsm")])
        if p:
            self.var_infile.set(p)

    def _pick_outdir(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.var_outdir.set(d)

    def _log(self, msg: str):
        self.log.configure(state="normal")
        try:
            self.log.insert("end", msg + "\n")
            self.log.see("end")
        finally:
            self.log.configure(state="disabled")

    # --- Action ----------------------------------------------------------
    def _run_parse(self):
        if parse_legacy_workbook is None:
            messagebox.showerror("Unavailable", f"Legacy parser import failed: {_IMPORT_ERR}")
            return

        infile = Path(self.var_infile.get())
        if not infile.exists():
            messagebox.showerror("Missing file", f"Workbook not found: {infile}")
            return

        try:
            thresh = float(self.var_thresh.get() or "0.0")
        except ValueError:
            messagebox.showerror("Invalid threshold", f"Not a number: {self.var_thresh.get()}")
            return

        base_out = Path(self.var_outdir.get()) if self.var_outdir.get() else infile.parent
        folder_name = (self.var_folder.get().strip() or f"{infile.stem}_parsed").replace(os.sep, "_")
        outd = base_out / folder_name
        outd.mkdir(parents=True, exist_ok=True)

        self._log(f"[parse] {infile.name} → {outd}")
        try:
            frames, summary = parse_legacy_workbook(infile, return_mode="frames", piccolo_flat_threshold=thresh)
            # Write CSVs and summary JSON
            for i, (port_key, df) in enumerate(frames.items(), start=1):
                csv_path = outd / f"PORT {i}.csv"
                df.to_csv(csv_path, index=False)
                try:
                    summary.setdefault("sheets", [])
                    if i-1 < len(summary["sheets"]):
                        summary["sheets"][i-1]["csv_path"] = str(csv_path)
                except Exception:
                    pass
            (outd / f"{infile.stem}__parse_summary.json").write_text(json.dumps(summary or {}, indent=2))
        except Exception as e:
            self._log(f"[error] {e}")
            messagebox.showerror("Parse failed", str(e))
            return

        self._log("✅ Done.")
        messagebox.showinfo("Parse complete", f"Outputs in: {outd}")
