import types
import sys
import importlib
from pathlib import Path

# Ensure repository root (containing the 'app' package) is importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _stub_qt():
    """Create minimal PySide6 stubs sufficient for MainWindow tests."""
    QtWidgets = types.ModuleType("PySide6.QtWidgets")

    class DummyQMainWindow:
        def __init__(self, parent=None):
            pass

        def setCentralWidget(self, w):
            self.central = w

    class DummyTabWidget:
        def __init__(self, parent=None):
            self._tabs = []

        def insertTab(self, index, widget, label):
            self._tabs.insert(index, (widget, label))

        def setCurrentIndex(self, index):
            self.current = index

        def widget(self, index):
            return self._tabs[index][0]

        def tabText(self, index):
            return self._tabs[index][1]

        def count(self):
            return len(self._tabs)

    QtWidgets.QMainWindow = DummyQMainWindow
    QtWidgets.QTabWidget = DummyTabWidget

    base = types.ModuleType("PySide6")
    base.QtWidgets = QtWidgets

    sys.modules.setdefault("PySide6", base)
    sys.modules.setdefault("PySide6.QtWidgets", QtWidgets)


def test_run_easy_tab_is_first(monkeypatch):
    _stub_qt()

    # Provide a lightweight stub for RunEasyPanel so import succeeds
    run_easy_mod = types.ModuleType("app.gui.run_easy_panel")

    class DummyPanel:
        def __init__(self, parent=None):
            pass

    run_easy_mod.RunEasyPanel = DummyPanel
    sys.modules["app.gui.run_easy_panel"] = run_easy_mod

    mw_module = importlib.import_module("app.gui.main_window")
    win = mw_module.MainWindow()

    assert win.tabs.count() == 1
    assert win.tabs.tabText(0) == "Run Easy"
    assert isinstance(win.tabs.widget(0), DummyPanel)
