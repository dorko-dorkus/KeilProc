def test_run_all_returns_triple_and_artifacts(monkeypatch, tmp_path):
    from kielproc import run_all
    import kielproc.run_easy as run_easy

    class StubOrchestrator:
        def __init__(self, run, progress_cb=None):
            self.run = run
            self.summary = {"ok": True}
            self.artifacts = [tmp_path / "a.txt"]
            self.artifacts[0].write_text("")

        def run_all(self):
            out_dir = tmp_path / "RUN_123"
            out_dir.mkdir()
            return out_dir

    monkeypatch.setattr(run_easy, "Orchestrator", StubOrchestrator)

    src = tmp_path / "book.xlsx"
    src.write_text("")
    site = run_easy.SitePreset(name="Dummy", geometry={}, instruments={}, defaults={})
    out, summary, artifacts = run_all(src, site=site, output_base=tmp_path)

    assert out == tmp_path / "RUN_123"
    assert summary == {"ok": True}
    assert artifacts == [str(tmp_path / "a.txt")]
