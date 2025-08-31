import json
import subprocess
import sys
from pathlib import Path
import pandas as pd


def test_cli_results(tmp_path: Path):
    df = pd.DataFrame({
        "Temperature": [20.0, 21.0, 19.5],
        "VP": [10.0, 11.0, 9.0],
        "Static": [101325.0, 101300.0, 101350.0],
        "Piccolo": [12.0, 12.0, 12.0],
    })
    csv = tmp_path / "sample.csv"
    df.to_csv(csv, index=False)
    cfg = {
        "static_col": "Static",
        "duct_height_m": 2.0,
        "duct_width_m": 3.0,
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    json_out = tmp_path / "out.json"
    csv_out = tmp_path / "out.csv"
    cmd = [
        sys.executable,
        "-m",
        "kielproc.cli",
        "results",
        "--csv",
        str(csv),
        "--config",
        str(cfg_path),
        "--json-out",
        str(json_out),
        "--csv-out",
        str(csv_out),
    ]
    cp = subprocess.run(cmd, capture_output=True, check=True, text=True)
    data = json.loads(cp.stdout)
    assert json_out.exists()
    assert csv_out.exists()
    assert data["n_samples"] == 3
