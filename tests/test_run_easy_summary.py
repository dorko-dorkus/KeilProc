import json
from pathlib import Path

from kielproc.run_easy import Orchestrator, RunInputs, SitePreset


class DummyOrchestrator(Orchestrator):
    def parse(self, ports_dir: Path) -> None:
        pass

    def integrate(self, base_dir: Path) -> None:
        pass

    def map(self, base_dir: Path) -> None:
        pass

    def fit(self, base_dir: Path) -> None:
        pass

    def translate(self, base_dir: Path) -> None:
        pass

    def report(self, base_dir: Path) -> None:
        pass


def test_summary_contains_required_keys(tmp_path):
    src = tmp_path / "book.xlsx"
    src.write_text("")
    site = SitePreset(name="Dummy", geometry={}, instruments={}, defaults={})
    run = RunInputs(src=src, site=site, output_base=tmp_path)
    orch = DummyOrchestrator(run)
    out_dir = orch.run_all()
    manifest = json.loads((out_dir / "summary.json").read_text())
    expected_keys = {
        "alpha": None,
        "beta": None,
        "lag_samples": None,
        "venturi_r": None,
        "venturi_beta": None,
        "transmitter_span": None,
        "transmitter_setpoints": None,
    }
    assert manifest["key_values"] == expected_keys
    assert manifest["qa_gates"] == {"delta_opp_max": 0.01, "w_max": 0.002}
    expected_inputs = {
        "baro_override_Pa": None,
        "site": "Dummy",
        "r": None,
        "beta": None,
        "reference": None,
    }
    assert manifest["inputs"] == expected_inputs
