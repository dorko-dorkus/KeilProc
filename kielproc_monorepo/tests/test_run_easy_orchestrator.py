from pathlib import Path

import pandas as pd
import math

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


def test_map_ignores_unknown_geometry_keys(tmp_path):
    geom = {
        "duct_height_m": 1.0,
        "duct_width_m": 1.0,
        "throat_area_m2": math.pi * (0.1 ** 2) / 4.0,
        "static_port_area_m2": 2.0,
        "total_port_area_m2": 1.0,
        # Extra field not recognized by Geometry dataclass
        "duct_diameter_m": 2.5,
    }
    preset = SitePreset(name="T", geometry=geom, instruments={}, defaults={})
    run = RunInputs(src=tmp_path / "dummy.xlsx", site=preset)
    orch = Orchestrator(run)

    base_dir = tmp_path / "run"
    ports_dir = base_dir / "ports_csv"
    ports_dir.mkdir(parents=True)

    df = pd.DataFrame({"VP": [1.0, 2.0], "Temperature": [20.0, 21.0]})
    csv = ports_dir / "p1.csv"
    df.to_csv(csv, index=False)

    orch.map(base_dir)

    mapped_csv = base_dir / "_mapped" / "p1_mapped.csv"
    assert mapped_csv.exists(), "Expected mapped CSV despite extra geometry keys"
