import types
import sys
import importlib
from pathlib import Path

# Ensure repository root (containing the 'app' package) is importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _stub_tk():
    """Create minimal tkinter stubs sufficient for MainWindow tests."""
    tk = types.ModuleType("tkinter")
    ttk = types.ModuleType("tkinter.ttk")

    class DummyTk:
        def __init__(self, parent=None):
            pass

    class DummyNotebook:
        def __init__(self, parent=None):
            self._tabs = []

        def insert(self, index, widget, text=""):
            self._tabs.insert(index, (widget, text))

        def select(self, widget):
            self.current = widget

        def tabs(self):
            return [w for w, _ in self._tabs]

        def tab(self, widget, option):
            for w, label in self._tabs:
                if w is widget and option == "text":
                    return label

        def pack(self, **kwargs):
            pass

    tk.Tk = DummyTk
    ttk.Notebook = DummyNotebook

    sys.modules["tkinter"] = tk
    sys.modules["tkinter.ttk"] = ttk


def test_run_easy_tab_is_first(monkeypatch):
    _stub_tk()

    # Provide a lightweight stub for RunEasyPanel so import succeeds
    run_easy_mod = types.ModuleType("app.gui.run_easy_panel")

    class DummyPanel:
        def __init__(self, parent=None):
            pass

    run_easy_mod.RunEasyPanel = DummyPanel
    sys.modules["app.gui.run_easy_panel"] = run_easy_mod

    mw_module = importlib.import_module("app.gui.main_window")
    win = mw_module.MainWindow()

    assert len(win.tabs.tabs()) == 1
    tab_widget = win.tabs.tabs()[0]
    assert win.tabs.tab(tab_widget, "text") == "Run Easy"
    assert isinstance(tab_widget, DummyPanel)
