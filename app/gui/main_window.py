import tkinter as tk
from tkinter import ttk

class MainWindow(tk.Tk):
    def __init__(self, parent=None):  # parent kept for API compatibility
        super().__init__()
        self.setupUi()

    def setupUi(self):
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(expand=1, fill="both")

        # existing tabs assembled elsewhere…
        # Always prepend the Run Easy panel so operators see it first
        self._insert_run_easy()

    def _insert_run_easy(self) -> None:
        """Insert the Run Easy panel as the first tab.

        Any import or wiring issues are treated as non-fatal; the GUI should
        continue to load even if the panel cannot be constructed (e.g. missing
        optional dependencies). In such cases the error is surfaced to the
        console for easier diagnosis.
        """

        try:  # pragma: no cover - exercised via unit tests
            from .run_easy_panel import RunEasyPanel

            run_easy = RunEasyPanel(self.tabs)
            self.tabs.insert(0, run_easy, text="Run Easy")
            self.tabs.select(run_easy)
        except Exception as e:  # pragma: no cover - exercised via unit tests
            # Non-fatal: if wiring fails (e.g., different widget name), surface in console and continue
            import sys

            print(f"[RunEasy] wiring failed: {e}", file=sys.stderr)
