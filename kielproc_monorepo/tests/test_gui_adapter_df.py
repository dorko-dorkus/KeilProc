import pandas as pd
from pathlib import Path
from kielproc.geometry import Geometry
from kielproc_gui_adapter import (
    map_from_tot_and_static,
    translate_piccolo,
    legacy_results_from_csv,
)
from kielproc.legacy_results import ResultsConfig


def test_map_from_tot_and_static_with_dataframe(tmp_path: Path):
    df = pd.DataFrame({
        "p_t": [10.0, 11.0],
        "p_s": [1.0, 1.2],
    })
    geom = Geometry(duct_height_m=1.0, duct_width_m=1.0, throat_diameter_m=0.1)
    out = tmp_path / "mapped.csv"
    res_path = map_from_tot_and_static(df, "p_t", "p_s", geom, None, out)
    out_df = pd.read_csv(res_path)
    assert {"qt", "dp_vent"}.issubset(out_df.columns)


def test_translate_piccolo_with_dataframe(tmp_path: Path):
    df = pd.DataFrame({"piccolo": [1.0, 2.0]})
    out = tmp_path / "translated.csv"
    res_path = translate_piccolo(df, 2.0, 1.0, "piccolo", "piccolo_translated", out)
    out_df = pd.read_csv(res_path)
    assert out_df["piccolo_translated"].tolist() == [3.0, 5.0]


def test_legacy_results_from_dataframe(tmp_path: Path):
    df = pd.DataFrame({
        "Temperature": [20.0, 21.0],
        "VP": [10.0, 11.0],
        "Static": [101325.0, 101300.0],
        "Piccolo": [12.0, 12.0],
    })
    cfg = ResultsConfig(static_col="Static", duct_height_m=2.0, duct_width_m=3.0)
    out = tmp_path / "results.csv"
    res = legacy_results_from_csv(df, cfg, out)
    assert out.exists() and res["n_samples"] == 2
