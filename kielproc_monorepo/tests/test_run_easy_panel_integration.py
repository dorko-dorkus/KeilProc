import types
import sys
import importlib
from pathlib import Path

# Ensure repository root (containing the 'app' package) is importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _stub_tk():
    """Create expansive tkinter stubs sufficient for app_gui imports."""
    tk = types.ModuleType("tkinter")
    ttk = types.ModuleType("tkinter.ttk")

    class DummyWidget:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            def method(*args, **kwargs):
                return None

            return method

    class DummyVar:
        def __init__(self, value=None):
            self._value = value

        def get(self):
            return self._value

        def set(self, v):
            self._value = v

        def trace_add(self, *args, **kwargs):
            pass

    class DummyNotebook(DummyWidget):
        def __init__(self, parent=None):
            super().__init__()
            self._tabs = []

        def insert(self, index, widget, text=""):
            self._tabs.insert(index, (widget, text))

        def add(self, widget, text=""):
            self._tabs.append((widget, text))

        def select(self, widget):
            self.current = widget

        def tabs(self):
            return [w for w, _ in self._tabs]

        def tab(self, widget, option):
            for w, label in self._tabs:
                if w is widget and option == "text":
                    return label

    # Basic widgets and variables
    tk.Tk = DummyWidget
    tk.Canvas = DummyWidget
    tk.Widget = DummyWidget
    tk.StringVar = tk.BooleanVar = tk.DoubleVar = tk.IntVar = DummyVar
    ttk.Notebook = DummyNotebook
    ttk.Frame = ttk.Label = ttk.Entry = ttk.Button = ttk.Checkbutton = ttk.Scrollbar = ttk.Separator = ttk.LabelFrame = ttk.Combobox = ttk.Treeview = DummyWidget
    ttk.Style = DummyWidget

    # Submodules expected during import
    filedialog = types.ModuleType("tkinter.filedialog")
    filedialog.askopenfilename = lambda *a, **k: ""
    filedialog.askdirectory = lambda *a, **k: ""
    messagebox = types.ModuleType("tkinter.messagebox")
    messagebox.showerror = lambda *a, **k: None
    messagebox.showinfo = lambda *a, **k: None
    scrolledtext = types.ModuleType("tkinter.scrolledtext")
    scrolledtext.ScrolledText = DummyWidget

    class DummyFont(DummyWidget):
        def cget(self, key):
            return 10

        def configure(self, **kwargs):
            pass

    font = types.ModuleType("tkinter.font")
    font.Font = DummyFont
    font.nametofont = lambda *a, **k: DummyFont()

    simpledialog = types.ModuleType("tkinter.simpledialog")
    simpledialog.SimpleDialog = DummyWidget

    sys.modules["tkinter"] = tk
    sys.modules["tkinter.ttk"] = ttk
    sys.modules["tkinter.filedialog"] = filedialog
    sys.modules["tkinter.messagebox"] = messagebox
    sys.modules["tkinter.scrolledtext"] = scrolledtext
    sys.modules["tkinter.font"] = font
    sys.modules["tkinter.simpledialog"] = simpledialog


def test_run_easy_tab_is_first(monkeypatch):
    _stub_tk()

    # Provide a lightweight stub for RunEasyPanel so import succeeds
    run_easy_mod = types.ModuleType("app.gui.run_easy_panel")

    class DummyPanel:
        def __init__(self, parent=None):
            pass

    run_easy_mod.RunEasyPanel = DummyPanel
    sys.modules["app.gui.run_easy_panel"] = run_easy_mod

    # Ensure modules depending on tkinter are re-imported after stubbing
    sys.modules.pop("ui_polish", None)
    sys.modules.pop("kielproc_monorepo.gui.app_gui", None)

    mw_module = importlib.import_module("app.gui.main_window")
    win = mw_module.MainWindow()

    notebook = win.__dict__.get("tabs") or win.__dict__.get("nb")
    assert notebook is not None
    tabs = notebook.tabs()
    assert tabs
    tab_widget = tabs[0]
    assert notebook.tab(tab_widget, "text") == "Run Easy"
    assert isinstance(tab_widget, DummyPanel)
