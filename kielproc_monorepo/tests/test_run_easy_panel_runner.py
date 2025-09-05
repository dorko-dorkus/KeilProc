from pathlib import Path
import queue
from test_run_easy_panel_integration import _stub_tk

def test_runner_respects_output_base(tmp_path, monkeypatch):
    _stub_tk()
    import app.gui.run_easy_panel as rep

    calls = {}

    def fake_run_easy_legacy(src, preset, baro, stamp, *, output_base=None, progress_cb=None):
        calls["output_base"] = output_base
        if progress_cb:
            progress_cb("step")
        return Path("RUN_fake"), {"warnings": [], "errors": []}, [Path("tab.csv")]

    monkeypatch.setattr(rep, "run_easy_legacy", fake_run_easy_legacy)

    q: queue.Queue = queue.Queue()
    preset = rep.SitePreset(name="X", geometry={}, instruments={}, defaults={})
    runner = rep._Runner(tmp_path / "src.xlsx", preset, None, None, tmp_path, q)
    cwd = Path.cwd()
    runner.run()

    assert Path.cwd() == cwd
    assert calls["output_base"] == tmp_path
    assert q.get()[0] == "started"
    assert q.get()[0] == "progress"
    kind, out_dir, *_ = q.get()
    assert kind == "finished"
    assert Path(out_dir) == tmp_path / "RUN_fake"
