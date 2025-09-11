import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

from .transmitter_profiles import load_tx_profile, derive_mc_from_gain_bias


class TxCalibrationApp(tk.Tk):
    """Simple GUI for deriving transmitter calibration."""

    def __init__(self) -> None:
        super().__init__()
        self.title("KielProc – Transmitter Helper")
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)
        r = 0

        # Site / season -------------------------------------------------------
        self.site_var = tk.StringVar(value="")
        self.season_var = tk.StringVar(value="summer")
        ttk.Label(frm, text="Site name").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.site_var, width=18).grid(row=r, column=1, sticky="w")
        r += 1
        ttk.Label(frm, text="Season").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.season_var, width=18).grid(row=r, column=1, sticky="w")
        r += 1
        ttk.Label(frm, text="Pressure range (mbar)").grid(row=r, column=0, sticky="w")
        self.range_var = tk.StringVar(value="100")
        ttk.Entry(frm, textvariable=self.range_var, width=18).grid(row=r, column=1, sticky="w")
        r += 1

        # 820 preset inputs ---------------------------------------------------
        self.gain_var = tk.StringVar(value="")
        self.bias_var = tk.StringVar(value="")
        self.bias_unit_var = tk.StringVar(value="tph")
        self.fs_var = tk.StringVar(value="100")
        ttk.Label(frm, text="820 gain (dimensionless)").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.gain_var, width=18).grid(row=r, column=1, sticky="w")
        r += 1
        ttk.Label(frm, text="820 bias").grid(row=r, column=0, sticky="w")
        bias_row = ttk.Frame(frm)
        bias_row.grid(row=r, column=1, sticky="w")
        ttk.Entry(bias_row, textvariable=self.bias_var, width=10).pack(side="left")
        ttk.Combobox(
            bias_row,
            textvariable=self.bias_unit_var,
            values=["tph", "percent"],
            width=8,
            state="readonly",
        ).pack(side="left")
        r += 1
        ttk.Label(frm, text="820 Full-scale (t/h)").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.fs_var, width=18).grid(row=r, column=1, sticky="w")
        r += 1

        # 820 m, c (derived but editable) ------------------------------------
        self.m_var = tk.StringVar(value="")
        self.c_var = tk.StringVar(value="")
        ttk.Label(frm, text="820 slope m (t/h per mbar)").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.m_var, width=18).grid(row=r, column=1, sticky="w")
        r += 1
        ttk.Label(frm, text="820 intercept c (t/h)").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.c_var, width=18).grid(row=r, column=1, sticky="w")
        r += 1

        # Buttons -------------------------------------------------------------
        btns = ttk.Frame(frm)
        btns.grid(row=r, column=0, columnspan=2, sticky="we", pady=6)
        ttk.Button(btns, text="Load Season Preset", command=self._on_load_profile).pack(side="left")
        ttk.Button(btns, text="Derive m,c from gain/bias", command=self._on_derive_mc).pack(side="left", padx=6)
        ttk.Button(btns, text="Run", command=self._on_run).pack(side="right")
        r += 1

    # ------------------------------------------------------------------ events
    def _on_load_profile(self) -> None:
        site = self.site_var.get().strip()
        season = self.season_var.get().strip()
        prof = load_tx_profile(site, season, []) if season else None
        if prof is None:
            messagebox.showerror("Preset not found", f"No preset for site '{site}' and season '{season}'.")
            return
        m, c, rng, meta = prof
        self.m_var.set(f"{m:.6g}")
        self.c_var.set(f"{c:.6g}")
        self.range_var.set(f"{rng:.6g}")
        if meta:
            if meta.get("gain") is not None:
                self.gain_var.set(str(meta["gain"]))
            if meta.get("bias") is not None:
                self.bias_var.set(str(meta["bias"]))
            if meta.get("full_scale_tph") is not None:
                self.fs_var.set(str(meta["full_scale_tph"]))
            if meta.get("bias_unit"):
                self.bias_unit_var.set(meta["bias_unit"])
        messagebox.showinfo("Preset loaded", f"Loaded {season} preset for site '{site or 'default'}'.")

    def _on_derive_mc(self) -> None:
        try:
            g = float(self.gain_var.get())
            b = float(self.bias_var.get())
            fs = float(self.fs_var.get())
            rng = float(self.range_var.get())
            m, c = derive_mc_from_gain_bias(g, b, rng, full_scale_tph=fs, bias_unit=self.bias_unit_var.get())
            self.m_var.set(f"{m:.6g}")
            self.c_var.set(f"{c:.6g}")
        except Exception as e:  # pragma: no cover - GUI feedback
            messagebox.showerror("Derive m,c failed", str(e))

    def _on_run(self) -> None:
        """Placeholder run action; in real usage this would trigger calibration."""
        try:
            m = float(self.m_var.get())
            c = float(self.c_var.get())
        except ValueError as e:  # pragma: no cover - GUI feedback
            messagebox.showerror("Run failed", str(e))
            return
        messagebox.showinfo("Run", f"Using m={m}  c={c}")


def main() -> None:
    app = TxCalibrationApp()
    app.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()
