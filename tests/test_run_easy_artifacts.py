import json
from pathlib import Path

from kielproc.run_easy import Orchestrator, RunInputs, SitePreset


class StubOrchestrator(Orchestrator):
    def parse(self, ports_dir: Path) -> None:
        ports_dir.mkdir(parents=True, exist_ok=True)

    def integrate(self, base_dir: Path) -> None:
        pass


def test_placeholder_artifacts(tmp_path):
    src = tmp_path / "book.xlsx"
    src.write_text("")
    site = SitePreset(
        name="Dummy",
        geometry={
            "duct_height_m": 1.0,
            "duct_width_m": 1.0,
            "static_port_area_m2": 0.01,
            "total_port_area_m2": 0.02,
        },
        instruments={},
        defaults={},
    )
    run = RunInputs(src=src, site=site, output_base=tmp_path)
    orch = StubOrchestrator(run)
    out_dir = orch.run_all()
    manifest = json.loads((out_dir / "summary.json").read_text())
    expected = [
        out_dir / "_mapped/heatmap_velocity.png",
        out_dir / "_fit/alpha_beta_by_block.csv",
        out_dir / "_fit/alpha_beta_by_block.json",
        out_dir / "_fit/alpha_beta_pooled.csv",
        out_dir / "_fit/alpha_beta_pooled.json",
        out_dir / "_translated/translated.csv",
        out_dir / "_report/legacy_results.csv",
        out_dir / "_report/legacy_results.json",
        out_dir / "_report/setpoints.json",
    ]
    for p in expected:
        assert p.exists()
        s = str(p)
        assert s in manifest["tables"] or s in manifest["plots"]
