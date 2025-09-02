import pandas as pd
from tools.legacy_parser.legacy_parser.parser import parse_legacy_workbook


def test_legacy_cube_to_dataframe_handles_padding(tmp_path):
    path = tmp_path / "wb.xlsx"
    # Port A with two samples
    rows_a = [
        ["Time", "Static pressure", "Velocity pressure", "Duct air temperature", "Piccolo current"],
        ["s", "Pa", "Pa", "C", "mA"],
        ["2020-01-01 00:00:00", 10, 1, 25, 5],
        ["2020-01-01 00:00:01", 20, 2, 26, 6],
    ]
    # Port B with one sample
    rows_b = [
        ["Time", "Static pressure", "Velocity pressure", "Duct air temperature", "Piccolo current"],
        ["s", "Pa", "Pa", "C", "mA"],
        ["2020-01-01 00:00:00", 30, 3, 27, 7],
    ]
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(rows_a).to_excel(writer, sheet_name="A", header=False, index=False)
        pd.DataFrame(rows_b).to_excel(writer, sheet_name="B", header=False, index=False)

    cube, _ = parse_legacy_workbook(path, return_mode="array")
    assert cube.ports == ["A", "B"]
    assert cube.counts.tolist() == [2, 1]

    df = cube.to_dataframe()

    # DataFrame should contain only 3 rows (sum of counts) and no padded rows
    assert len(df) == 3
    assert ("B", 1) not in df.index

    # Values from the original sheets should be preserved
    assert df.loc[("A", 0), "Static"] == 10
    assert df.loc[("A", 1), "VP"] == 2
    assert df.loc[("B", 0), "Temperature"] == 27
    assert df.loc[("A", 1), "Time_s"] == 1.0
