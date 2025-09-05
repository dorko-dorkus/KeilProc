import inspect
import types
import sys
from pathlib import Path

# Ensure repository root is importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_insert_run_easy_tab(monkeypatch):
    mod = __import__("kielproc_monorepo.gui.run_easy_tab", fromlist=["insert_run_easy_tab"])
    insert_run_easy_tab = mod.insert_run_easy_tab

    # Stub RunEasyTab to avoid needing a real Tk environment
    class DummyTab:
        def __init__(self, master=None):
            self.master = master

    monkeypatch.setattr(mod, "RunEasyTab", DummyTab)

    class DummyNotebook:
        def __init__(self):
            self.actions = []

        def insert(self, index, widget, **kw):
            self.actions.append((index, widget, kw.get("text")))

        def select(self, widget):
            self.selected = widget

    nb = DummyNotebook()
    tab = insert_run_easy_tab(nb)

    assert nb.actions[0] == (0, tab, "Run Easy")
    assert isinstance(tab, DummyTab)


def test_app_build_mentions_insert_run_easy_tab():
    app_mod = __import__("kielproc_monorepo.gui.app_gui", fromlist=["App"])
    src = inspect.getsource(app_mod.App._build)
    assert "insert_run_easy_tab" in src
