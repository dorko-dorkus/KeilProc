import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont


def apply_style(root: tk.Tk, font_scale: float = 1.0) -> None:
    """Apply a basic themed style and optionally scale default font sizes."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    default_font = tkfont.nametofont("TkDefaultFont")
    default_font.configure(size=int(default_font.cget("size") * font_scale))
    root.option_add("*Font", default_font)


def make_statusbar(root: tk.Tk):
    """Create a simple status bar at the bottom of *root*.

    Returns a tuple ``(variable, logger)`` where ``variable`` is a ``StringVar``
    bound to the label and ``logger`` is a helper function that sets the
    variable's value.
    """
    var = tk.StringVar(value="")
    bar = ttk.Label(root, textvariable=var, relief="sunken", anchor="w")
    bar.pack(side="bottom", fill="x")

    def log(msg: str) -> None:
        var.set(msg)

    return var, log


def set_grid_weights(widget: tk.Widget, rows: int, cols: int) -> None:
    """Give all rows/cols of *widget* a weight so children expand nicely."""
    for r in range(rows):
        widget.rowconfigure(r, weight=1)
    for c in range(cols):
        widget.columnconfigure(c, weight=1)


def bind_mousewheel(widget: tk.Widget) -> None:
    """Enable basic mouse wheel scrolling on *widget* across platforms."""

    def _on_mousewheel(event):
        delta = event.delta
        if delta == 0 and event.num in (4, 5):
            delta = 120 if event.num == 4 else -120
        widget.yview_scroll(int(-delta / 120), "units")

    widget.bind("<Enter>", lambda _: widget.focus_set())
    widget.bind("<MouseWheel>", _on_mousewheel)
    widget.bind("<Button-4>", _on_mousewheel)
    widget.bind("<Button-5>", _on_mousewheel)


def tooltip(widget: tk.Widget, text: str) -> None:
    """Attach a very small tooltip to *widget*."""
    tip = {"window": None}

    def enter(_):
        if tip["window"] or not text:
            return
        x = widget.winfo_rootx() + 20
        y = widget.winfo_rooty() + 20
        tw = tk.Toplevel(widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = ttk.Label(tw, text=text, relief="solid", borderwidth=1,
                        background="#ffffe0")
        lbl.pack(ipadx=2)
        tip["window"] = tw

    def leave(_):
        tw = tip.pop("window", None)
        if tw:
            tw.destroy()
            tip["window"] = None

    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)


def vcmd_float(root: tk.Tk):
    """Return a ``validatecommand`` tuple allowing only float input."""
    def _validate(P: str) -> bool:
        if P.strip() == "":
            return True
        try:
            float(P)
            return True
        except ValueError:
            return False

    return (root.register(_validate), "%P")


def labeled_row(parent: tk.Widget, label: str, widget: tk.Widget, row: int,
                col: int = 0) -> int:
    """Grid a label and widget on the given *row* and return next row."""
    pad = {"padx": 6, "pady": 4}
    ttk.Label(parent, text=label).grid(row=row, column=col, sticky="e", **pad)
    widget.grid(row=row, column=col + 1, sticky="w", **pad)
    return row + 1
