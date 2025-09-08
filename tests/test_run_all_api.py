from pathlib import Path

from kielproc.run_easy import RunConfig, SitePreset, run_all


def test_run_all_executes_stages_in_order(monkeypatch, tmp_path):
    calls = []

    def fake_parse(input_dir, *, file_glob="*"):
        calls.append("parse")
        return ["parsed"]

    def fake_integrate(parsed, *, vp_unit="Pa", temp_unit="C"):
        assert parsed == ["parsed"]
        calls.append("integrate")
        return {"integrated": True}

    def fake_map_ports(integrated):
        assert integrated["integrated"] is True
        calls.append("map")
        return {"mapped": True}

    def fake_fit(mapped, *, baro_pa: float):
        assert mapped["mapped"] is True
        calls.append("fit")
        return {"fit": True, "baro": baro_pa}

    def fake_translate(fitres, *, output_dir: str):
        assert fitres["fit"] is True
        assert output_dir == str(tmp_path / "out")
        calls.append("translate")
        return {"report": True}

    monkeypatch.setattr("kielproc.run_easy.parse", fake_parse)
    monkeypatch.setattr("kielproc.run_easy.integrate", fake_integrate)
    monkeypatch.setattr("kielproc.run_easy.map_ports", fake_map_ports)
    monkeypatch.setattr("kielproc.run_easy.fit", fake_fit)
    monkeypatch.setattr("kielproc.run_easy.translate", fake_translate)

    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()

    cfg = RunConfig(
        input_dir=str(tmp_path / "in"),
        output_dir=str(tmp_path / "out"),
        baro_pa=50_000.0,
        enable_site=True,
        site=SitePreset(name="SiteA"),
    )

    res = run_all(cfg)

    assert calls == ["parse", "integrate", "map", "fit", "translate"]
    assert res["parsed"] == ["parsed"]
    assert res["integrated"] == {"integrated": True}
    assert res["mapped"] == {"mapped": True}
    assert res["fit"] == {"fit": True, "baro": 50_000.0}
    assert res["report"] == {"report": True}
    assert res["baro_pa"] == 50_000.0
    assert res["site_name"] == "SiteA"

