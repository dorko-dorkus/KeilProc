from pathlib import Path
import json
import pandas as pd

from kielproc.run_easy import RunConfig, SitePreset, run_all


def test_run_all_produces_outputs(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    # Minimal port CSV (P1)
    pd.DataFrame(
        {
            "VP": [10.0, 12.0],
            "Temperature": [25.0, 26.0],
            "Static": [101325.0, 101300.0],
        }
    ).to_csv(in_dir / "run__P1.csv", index=False)

    # Logger CSV for transmitter setpoints
    sp_csv = tmp_path / "setpoints.csv"
    pd.DataFrame({"i/p": [10, 20], "820": [25, 30]}).to_csv(sp_csv, index=False)

    cfg = RunConfig(
        input_dir=str(in_dir),
        output_dir=str(out_dir),
        enable_site=True,
        site=SitePreset(
            name="SiteA",
            geometry={"duct_width_m": 1.0, "duct_height_m": 1.0, "throat_area_m2": 0.25},
        ),
        setpoints_csv=str(sp_csv),
    )

    summary = run_all(cfg)

    # Basic sanity checks
    assert Path(summary["per_port_csv"]).exists()
    assert Path(summary["duct_result_json"]).exists()
    tp = out_dir / "_integrated" / "transmitter_setpoints.csv"
    assert tp.exists()
    assert summary["setpoints"]["rows"] == 2
    assert summary["baro_pa"] == 101_325.0
    assert summary["site_name"] == "SiteA"
    # Venturi result files present and sane
    vr = Path(summary["venturi_result_json"])
    assert vr.exists()
    data = json.loads(vr.read_text())
    assert len(data["curve"]) == 10
    assert data["dp_vent_Pa_at_Qs"] > 0
