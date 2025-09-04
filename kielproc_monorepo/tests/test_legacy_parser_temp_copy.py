import hashlib
from pathlib import Path

import pandas as pd

from tools.legacy_parser.legacy_parser.parser import parse_legacy_workbook


def _file_hash(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def test_parser_preserves_original_workbook(tmp_path):
    """The legacy parser should not modify the source workbook."""
    # Create a minimal workbook with one sheet
    path = tmp_path / "wb.xlsx"
    rows = [
        ["Time", "Static pressure", "Velocity pressure", "Duct air temperature", "Piccolo current"],
        ["s", "Pa", "Pa", "C", "mA"],
        ["2020-01-01 00:00:00", 10, 1, 25, 5],
    ]
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="A", header=False, index=False)

    before = _file_hash(path)
    out_dir = tmp_path / "out"
    parse_legacy_workbook(path, out_dir=out_dir, return_mode="files")

    assert path.exists()
    assert _file_hash(path) == before
