import pandas as pd
from pathlib import Path
import math
from kielproc.geometry import Geometry
from kielproc_gui_adapter import (
    map_from_tot_and_static,
    translate_piccolo,
    legacy_results_from_csv,
    compute_setpoints,
    process_legacy_parsed_csv,
    make_actual_vs_linearized_plot,
)
from kielproc.legacy_results import ResultsConfig


def test_map_from_tot_and_static_with_dataframe(tmp_path: Path):
    df = pd.DataFrame({
        "p_t": [10.0, 11.0],
        "p_s": [1.0, 1.2],
    })
    geom = Geometry(
        duct_height_m=1.0,
        duct_width_m=1.0,
        throat_area_m2=math.pi * (0.1 ** 2) / 4.0,
        static_port_area_m2=2.0,
        total_port_area_m2=1.0,
    )
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


def test_compute_setpoints_from_dataframe(tmp_path: Path):
    df = pd.DataFrame({
        "i/p": [0, 25, 50, 75, 100],
        "820": [4, 8, 12, 16, 20],
    })
    json_path = tmp_path / "sp.json"
    csv_path = tmp_path / "sp.csv"
    res = compute_setpoints(df, "i/p", "820", out_json=json_path, out_csv=csv_path)
    assert json_path.exists() and csv_path.exists()
    assert res["optimal_span"]["span"]["mA_low"] == 4.0
    mapping = pd.read_csv(csv_path)
    assert set(mapping.columns) == {"pct", "x", "mA"}


def test_process_legacy_parsed_csv(tmp_path: Path):
    df = pd.DataFrame({"VP": [1.0, 2.0, 3.0]})
    geom = Geometry(
        duct_height_m=1.0,
        duct_width_m=1.0,
        throat_area_m2=math.pi * (0.1 ** 2) / 4.0,
        static_port_area_m2=2.0,
        total_port_area_m2=1.0,
    )
    out = tmp_path / "legacy_qs_qp_dpvent.csv"
    wrote, res_df = process_legacy_parsed_csv(df, geom, None, out)
    assert wrote == out
    assert out.exists()
    assert {"qp", "deltpVent"}.issubset(res_df.columns)


def test_make_actual_vs_linearized_plot():
    df = pd.DataFrame({"qp": [0.0, 1.0, 2.0], "deltpVent": [0.1, 1.1, 2.1]})
    fig = make_actual_vs_linearized_plot(df)
    assert fig is not None
