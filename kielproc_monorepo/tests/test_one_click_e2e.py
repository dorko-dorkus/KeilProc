import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "kielproc_monorepo"))

from kielproc import cli
from kielproc.run_easy import SitePreset
import tools.legacy_parser.legacy_parser.parser as parser_mod


def test_one_click_e2e(tmp_path, monkeypatch, capsys):
    geom = {
        "duct_height_m": 1.0,
        "duct_width_m": 1.0,
        "static_port_area_m2": 1.0,
        "total_port_area_m2": 2.0,
        "throat_width_m": 0.5,
        "throat_height_m": 0.5,
    }
    site = SitePreset(name="TestSite", geometry=geom, instruments={"vp_unit": "Pa", "temp_unit": "C"}, defaults={})
    monkeypatch.setitem(cli.PRESETS, "TestSite", site)

    def fake_parse(xlsx_path, out_dir, return_mode="files", **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_excel(xlsx_path, sheet_name="P1", header=0)
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = ["Time", "Static", "VP", "Temperature", "Piccolo", "pN", "pS", "pE", "pW", "q_mean"]
        csv_path = out_dir / f"{xlsx_path.stem}__P1.csv"
        df.to_csv(csv_path, index=False)
        summary = {"sheets": [{"sheet": "P1", "csv_path": str(csv_path), "mode": "vertical"}]}
        (out_dir / f"{xlsx_path.stem}__parse_summary.json").write_text(json.dumps(summary))
        return summary

    monkeypatch.setattr(parser_mod, "parse_legacy_workbook", fake_parse)

    beta = 0.5
    r = 0.5
    c = (1 - beta**4) * (r**2)
    vp = [1.0, 2.0, 3.0]
    deltp = [c * v for v in vp]
    piccolo = [2 * d + 1 for d in deltp]

    cols = ["Time", "Static Pressure", "Velocity Pressure", "Temperature", "Piccolo", "pN", "pS", "pE", "pW", "q_mean"]
    units = ["s", "Pa", "Pa", "K", "", "", "", "", "", ""]
    data = [
        [0, 101325, vp[0], 293.15, piccolo[0], 1000, 1000, 1000, 1000, 1000],
        [1, 101325, vp[1], 293.15, piccolo[1], 1000, 1000, 1000, 1000, 1000],
        [2, 101325, vp[2], 293.15, piccolo[2], 1000, 1000, 1000, 1000, 1000],
    ]
    df = pd.DataFrame([cols, units] + data)
    wb = tmp_path / "book.xlsx"
    with pd.ExcelWriter(wb) as xls:
        df.to_excel(xls, sheet_name="P1", index=False, header=False)

    args = ["one-click", str(wb), "--site", "TestSite"]
    cli.main(args)
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    art_names = [Path(a).name for a in out["artifacts"]]
    assert "alpha_beta_by_block.csv" in art_names
    fit_csv = Path(out["out_dir"]) / "_fit" / "alpha_beta_by_block.csv"
    tbl = pd.read_csv(fit_csv)
    assert tbl["alpha"].iloc[0] == pytest.approx(2.0, rel=1e-5)
    assert tbl["beta"].iloc[0] == pytest.approx(1.0, rel=1e-5)
