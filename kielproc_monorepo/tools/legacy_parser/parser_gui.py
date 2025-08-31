import os, sys, json, traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from legacy_parser.parser import parse_legacy_workbook

def main():
    root = tk.Tk()
    root.title("Legacy XLSX → CSV Parser")
    root.geometry("900x600")
    frm = ttk.Frame(root, padding=12); frm.pack(fill="x", expand=False)

    var_xlsx = tk.StringVar()
    var_out  = tk.StringVar()
    var_thresh = tk.StringVar(value="1e-6")

    def pick_xlsx():
        p = filedialog.askopenfilename(title="Select legacy workbook",
                                       filetypes=[("Excel","*.xlsx *.xls")])
        if p: var_xlsx.set(p)
    def pick_out():
        p = filedialog.askdirectory(title="Select output folder")
        if p: var_out.set(p)

    ttk.Label(frm, text="Legacy workbook:").grid(row=0, column=0, sticky="w")
    ttk.Entry(frm, textvariable=var_xlsx, width=70).grid(row=0, column=1, sticky="we")
    ttk.Button(frm, text="Browse…", command=pick_xlsx).grid(row=0, column=2, padx=6)

    ttk.Label(frm, text="Output folder:").grid(row=1, column=0, sticky="w")
    ttk.Entry(frm, textvariable=var_out, width=70).grid(row=1, column=1, sticky="we")
    ttk.Button(frm, text="Browse…", command=pick_out).grid(row=1, column=2, padx=6)

    ttk.Label(frm, text="Piccolo flat threshold (std):").grid(row=2, column=0, sticky="w")
    ttk.Entry(frm, textvariable=var_thresh, width=20).grid(row=2, column=1, sticky="w")

    out_text = tk.Text(root, height=24); out_text.pack(fill="both", expand=True, padx=12, pady=12)
    out_text.configure(state="disabled")
    def log(msg):
        out_text.configure(state="normal"); out_text.insert("end", msg + "\\n"); out_text.configure(state="disabled"); out_text.see("end")

    def run_parse():
        try:
            xlsx = Path(var_xlsx.get())
            if not xlsx.exists():
                raise FileNotFoundError(f"Workbook not found: {xlsx}")
            outd = Path(var_out.get()) if var_out.get() else xlsx.parent/ (xlsx.stem + "_parsed")
            outd.mkdir(parents=True, exist_ok=True)
            thresh = float(var_thresh.get())
            summary = parse_legacy_workbook(xlsx, outd, piccolo_flat_threshold=thresh)
            log("[OK] Parsed: " + xlsx.name)
            log(json.dumps(summary, indent=2))
            messagebox.showinfo("Parse complete", f"Parsed {xlsx.name}\nOutputs in: {outd}")
        except Exception as e:
            log(f"[ERROR] {e}\n{traceback.format_exc()}")
            messagebox.showerror("Error", str(e))

    ttk.Button(frm, text="Parse", command=run_parse).grid(row=3, column=1, sticky="w", pady=8)

    root.mainloop()

if __name__ == "__main__":
    main()
