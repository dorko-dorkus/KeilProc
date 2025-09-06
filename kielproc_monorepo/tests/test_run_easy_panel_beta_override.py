from pathlib import Path
from test_run_easy_panel_integration import _stub_tk
import sys


def test_run_easy_panel_beta_override(tmp_path, monkeypatch):
    _stub_tk()
    sys.modules.pop("app.gui.run_easy_panel", None)
    import app.gui.run_easy_panel as rep

    wb = tmp_path / "src.xlsx"
    wb.write_text("dummy")

    panel = rep.RunEasyPanel()
    panel.path_var.set(str(wb))
    panel.outdir_var.set("")
    panel.stamp_var.set("")
    panel.static_port_area_var.set("1.0")
    panel.total_port_area_var.set("2.0")
    panel.beta_var.set("0.7")
    panel.baro_var.set("")
    panel.height_var.set("")
    panel.width_var.set("")
    panel.throat_var.set("")
    panel.duct_diam_var.set("")
    panel.throat_width_var.set("")
    panel.throat_height_var.set("")

    captured = {}

    def fake_run_easy_legacy(src, preset, baro, stamp, **kwargs):
        captured["beta"] = preset.geometry.get("beta")
        return Path("RUN_fake"), {"warnings": [], "errors": []}, []

    monkeypatch.setattr(rep, "run_easy_legacy", fake_run_easy_legacy)

    panel._process()
    panel._runner.join()

    assert captured["beta"] == 0.7
