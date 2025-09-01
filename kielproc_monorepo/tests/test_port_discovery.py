import pandas as pd
from kielproc.aggregate import integrate_run, RunConfig

def _write_csv(path, vp=1.0):
    df = pd.DataFrame({
        "VP": [vp],
        "Temperature": [20.0],
        "Static": [101325.0],
    })
    df.to_csv(path, index=False)


def test_integrate_run_port_name_variants(tmp_path):
    names = ["P1.csv", "PORT 2.csv", "Port_3.csv", "Run07_P4.csv"]
    for name in names:
        _write_csv(tmp_path / name)
    cfg = RunConfig(height_m=1.0, width_m=1.0)
    res = integrate_run(tmp_path, cfg)
    assert sorted(res["files"]) == sorted(names)
    assert res["per_port"]["Port"].tolist() == ["P1", "P2", "P3", "P4"]
    assert [(pid, pf.name) for pid, pf in res["pairs"]] == [
        ("P1", "P1.csv"),
        ("P2", "PORT 2.csv"),
        ("P3", "Port_3.csv"),
        ("P4", "Run07_P4.csv"),
    ]


def test_integrate_run_weight_key_variants(tmp_path):
    names = ["Run07_P1.csv", "PORT 2.csv"]
    for name in names:
        _write_csv(tmp_path / name)
    cfg = RunConfig(height_m=1.0, width_m=1.0, weights={"PORT 1": 0.25, "Port_2": 0.75})
    assert cfg.weights == {"P1": 0.25, "P2": 0.75}
    res = integrate_run(tmp_path, cfg)
    assert res["per_port"]["Port"].tolist() == ["P1", "P2"]
    assert [(pid, pf.name) for pid, pf in res["pairs"]] == [
        ("P1", "Run07_P1.csv"),
        ("P2", "PORT 2.csv"),
    ]
