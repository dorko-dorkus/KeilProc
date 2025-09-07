import json
from pathlib import Path

from kielproc import cli


def test_cli_one_click(monkeypatch, tmp_path, capsys):
    out_dir = tmp_path / "out"

    def fake_run(
        src,
        site,
        baro_override_Pa=None,
        run_stamp=None,
        *,
        output_base=None,
        strict=False,
    ):
        assert src == Path(tmp_path / "book.xlsx")
        assert output_base is None
        assert strict is False
        return out_dir, {"warnings": ["w"], "errors": []}, [str(out_dir / "a.csv")]

    monkeypatch.setattr(cli, "run_easy_legacy", fake_run)
    args = ["one-click", str(tmp_path / "book.xlsx"), "--site", "DefaultSite"]
    cli.main(args)
    captured = capsys.readouterr().out.strip()
    data = json.loads(captured)
    assert data["ok"] is True
    assert data["out_dir"] == str(out_dir)
    assert data["summary"]["warnings"] == ["w"]
    assert data["artifacts"] == [str(out_dir / "a.csv")]


def test_cli_one_click_bundle(monkeypatch, tmp_path, capsys):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "dummy.txt").write_text("x")

    def fake_run(
        src,
        site,
        baro_override_Pa=None,
        run_stamp=None,
        *,
        output_base=None,
        strict=False,
    ):
        assert strict is False
        return out_dir, {"warnings": [], "errors": []}, []

    monkeypatch.setattr(cli, "run_easy_legacy", fake_run)
    args = ["one-click", str(tmp_path / "book.xlsx"), "--bundle"]
    cli.main(args)
    captured = capsys.readouterr().out.strip()
    data = json.loads(captured)
    bundle = Path(data["bundle_zip"])
    assert bundle.exists()


def test_cli_one_click_strict(monkeypatch, tmp_path, capsys):
    out_dir = tmp_path / "out"

    def fake_run(
        src,
        site,
        baro_override_Pa=None,
        run_stamp=None,
        *,
        output_base=None,
        strict=False,
    ):
        assert strict is True
        return out_dir, {"warnings": [], "errors": []}, []

    monkeypatch.setattr(cli, "run_easy_legacy", fake_run)
    args = [
        "one-click",
        str(tmp_path / "book.xlsx"),
        "--site",
        "DefaultSite",
        "--strict",
    ]
    cli.main(args)
    captured = capsys.readouterr().out.strip()
    data = json.loads(captured)
    assert data["ok"] is True

