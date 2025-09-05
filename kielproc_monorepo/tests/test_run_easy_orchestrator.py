from pathlib import Path

from kielproc.run_easy import Orchestrator, RunInputs, SitePreset


def test_run_all_sequences_steps(tmp_path, monkeypatch):
    preset = SitePreset(name="T", geometry={}, instruments={}, defaults={})
    src = tmp_path / "book.xlsx"
    src.write_text("dummy")
    monkeypatch.chdir(tmp_path)

    orch = Orchestrator(RunInputs(src=src, site=preset))
    called = []

    def rec(name):
        def _inner(*args, **kwargs):
            called.append(name)
        return _inner

    monkeypatch.setattr(orch, "parse", rec("parse"))
    monkeypatch.setattr(orch, "integrate", rec("integrate"))
    monkeypatch.setattr(orch, "map", rec("map"))
    monkeypatch.setattr(orch, "fit", rec("fit"))
    monkeypatch.setattr(orch, "translate", rec("translate"))
    monkeypatch.setattr(orch, "report", rec("report"))

    out = orch.run_all()
    assert out.is_dir()
    assert (out / "run_context.json").exists()
    assert called == ["parse", "integrate", "map", "fit", "translate", "report"]


def test_run_all_uses_output_base(tmp_path, monkeypatch):
    preset = SitePreset(name="T", geometry={}, instruments={}, defaults={})
    src = tmp_path / "book.xlsx"
    src.write_text("dummy")
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)

    run = RunInputs(src=src, site=preset, output_base=tmp_path)
    orch = Orchestrator(run)

    def noop(*args, **kwargs):
        pass

    monkeypatch.setattr(orch, "parse", noop)
    monkeypatch.setattr(orch, "integrate", noop)
    monkeypatch.setattr(orch, "map", noop)
    monkeypatch.setattr(orch, "fit", noop)
    monkeypatch.setattr(orch, "translate", noop)
    monkeypatch.setattr(orch, "report", noop)

    out = orch.run_all()
    assert out.is_dir()
    assert out.parent == tmp_path
