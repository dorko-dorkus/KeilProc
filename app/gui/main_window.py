"""Minimal launcher that exposes only the Run Easy panel."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class MainWindow(tk.Tk):
    """Unified GUI window with a single Run Easy tab."""

    def __init__(self, parent=None):
        super().__init__()
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(expand=1, fill="both")
        self._insert_run_easy()

    def _insert_run_easy(self) -> None:
        try:
            from .run_easy_panel import RunEasyPanel

            run_easy = RunEasyPanel(self.tabs)
            self.tabs.insert(0, run_easy, text="Run Easy")
            self.tabs.select(run_easy)
        except Exception as e:  # pragma: no cover - exercised via unit tests
            import sys

            print(f"[RunEasy] wiring failed: {e}", file=sys.stderr)


def main() -> None:
    win = MainWindow()
    win.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()
