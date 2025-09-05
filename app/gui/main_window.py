"""Compatibility launcher that delegates to the canonical app.
This keeps old entry points working but avoids two competing windows.
"""
from __future__ import annotations

try:
    from kielproc_monorepo.gui.app_gui import App as _App, main as _main  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - fallback for minimal environments
    _App = None  # type: ignore
    _main = None  # type: ignore

if _App is None:  # pragma: no cover - simplified fallback for tests/minimal Tk
    import tkinter as tk
    from tkinter import ttk

    class _App(tk.Tk):  # type: ignore[misc]
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

    def _main() -> None:
        win = _App()
        win.mainloop()


class MainWindow(_App):  # type: ignore[misc]
    """Backwards-compatible alias for the unified GUI window."""
    pass


if __name__ == "__main__":  # pragma: no cover
    _main()
