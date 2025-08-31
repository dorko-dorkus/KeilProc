import pandas as pd
from pathlib import Path
from kielproc.legacy_results import ResultsConfig, compute_results

def test_compute_results_basic(tmp_path: Path):
    df = pd.DataFrame({
        "Temperature": [20.0, 21.0, 19.5],
        "VP": [10.0, 11.0, 9.0],
        "Static": [101325.0, 101300.0, 101350.0],
        "Piccolo": [12.0, 12.0, 12.0],
    })
    csv = tmp_path / "sample.csv"
    df.to_csv(csv, index=False)
    cfg = ResultsConfig(static_col="Static", duct_height_m=2.0, duct_width_m=3.0)
    res = compute_results(csv, cfg)
    assert res["n_samples"] == 3
    # Piccolo mean of 12 mA with 6.7 mbar range -> (12-4)/16*6.7
    assert abs(res["piccolo_mbar"] - ((12-4)/16*6.7)) < 1e-6
    assert "mass_kg_s" in res
    assert res["duct_height_m"] == 2.0
    assert res["duct_width_m"] == 3.0
    assert res["area_m2"] == 6.0


def test_compute_results_area_only(tmp_path: Path):
    df = pd.DataFrame({
        "Temperature": [20.0, 21.0, 19.5],
        "VP": [10.0, 11.0, 9.0],
        "Static": [101325.0, 101300.0, 101350.0],
        "Piccolo": [12.0, 12.0, 12.0],
    })
    csv = tmp_path / "sample.csv"
    df.to_csv(csv, index=False)
    cfg = ResultsConfig(static_col="Static", area_m2=6.0)
    res = compute_results(csv, cfg)
    assert res["area_m2"] == 6.0
    assert res["duct_height_m"] is None
    assert res["duct_width_m"] is None


def test_compute_results_dataframe():
    df = pd.DataFrame({
        "Temperature": [20.0, 21.0],
        "VP": [10.0, 11.0],
        "Static": [101325.0, 101300.0],
        "Piccolo": [12.0, 12.0],
    })
    cfg = ResultsConfig(static_col="Static", duct_height_m=2.0, duct_width_m=3.0)
    res = compute_results(df, cfg)
    assert res["n_samples"] == 2
