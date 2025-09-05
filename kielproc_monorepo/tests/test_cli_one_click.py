import json
from pathlib import Path

from kielproc import cli


def test_cli_one_click(monkeypatch, tmp_path, capsys):
    out_dir = tmp_path / "out"

    def fake_run(src, site, baro_override_Pa=None, run_stamp=None, *, output_base=None):
        assert src == Path(tmp_path / "book.xlsx")
        assert output_base is None
        return out_dir

    monkeypatch.setattr(cli, "run_easy_legacy", fake_run)
    args = ["one-click", str(tmp_path / "book.xlsx"), "--site", "DefaultSite"]
    cli.main(args)
    captured = capsys.readouterr().out.strip()
    data = json.loads(captured)
    assert data["ok"] is True
    assert data["out_dir"] == str(out_dir)
