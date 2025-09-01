import pandas as pd
from kielproc.aggregate import integrate_run, RunConfig

def _write_csv(path):
    df = pd.DataFrame({
        "VP": [1.0],
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
    assert res["per_port"]["Port"].tolist() == ["PORT 1", "PORT 2", "PORT 3", "PORT 4"]
