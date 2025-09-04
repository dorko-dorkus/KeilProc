from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QComboBox, QDoubleSpinBox, QTextEdit
)

# One‑click pipeline
from keilproc_oneclick.orchestrator import run_easy_legacy
from keilproc_oneclick import presets as kp_presets


def _available_presets():
    # Discover simple, attribute‑style presets from keilproc_oneclick.presets
    out = {}
    for name in dir(kp_presets):
        if name.startswith("_"):
            continue
        obj = getattr(kp_presets, name)
        # very light type check
        if getattr(obj, "name", None) and getattr(obj, "geometry", None):
            out[obj.name] = obj
    if not out:
        # Always provide at least DefaultSite
        out = {kp_presets.DefaultSite.name: kp_presets.DefaultSite}
    return out


class _Runner(QObject):
    started = Signal()
    progress = Signal(str)
    failed = Signal(str)
    finished = Signal(str)  # run directory path

    def __init__(self, src: Path, preset, baro: float | None, stamp: str | None):
        super().__init__()
        self.src = src
        self.preset = preset
        self.baro = baro
        self.stamp = stamp

    def run(self):
        self.started.emit()
        try:
            self.progress.emit("Preflight…")
            out = run_easy_legacy(self.src, self.preset, self.baro, self.stamp)
            self.finished.emit(str(out))
        except Exception as e:
            self.failed.emit(str(e))


class RunEasyPanel(QWidget):
    """Minimal operator UI: preset → input → Process.

    Insert as the first tab of the main QTabWidget to shift others to the right.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._presets = _available_presets()
        self._runner_thread: QThread | None = None
        self._runner: _Runner | None = None
        self._build()

    def _build(self):
        v = QVBoxLayout(self)

        # Preset row
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Site preset:"))
        self.preset = QComboBox()
        for name in sorted(self._presets.keys()):
            self.preset.addItem(name)
        row1.addWidget(self.preset)
        v.addLayout(row1)

        # Input row
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Workbook or folder:"))
        self.path = QLineEdit()
        self.path.setPlaceholderText("/path/to/legacy.xlsx or directory…")
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        row2.addWidget(self.path, 1)
        row2.addWidget(btn_browse)
        v.addLayout(row2)

        # Baro override
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Baro override (Pa, optional):"))
        self.baro = QDoubleSpinBox()
        self.baro.setRange(0.0, 300000.0)
        self.baro.setDecimals(1)
        self.baro.setValue(0.0)  # 0 => not set
        row3.addWidget(self.baro)
        v.addLayout(row3)

        # Process button
        row4 = QHBoxLayout()
        self.btn = QPushButton("Process")
        self.btn.clicked.connect(self._process)
        row4.addWidget(self.btn)
        v.addLayout(row4)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Status and summary will appear here…")
        v.addWidget(self.log, 1)

    # UI helpers
    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select workbook", "", "Excel (*.xlsx *.xls)")
        if path:
            self.path.setText(path)

    def _set_busy(self, busy: bool):
        self.btn.setEnabled(not busy)
        self.path.setEnabled(not busy)
        self.preset.setEnabled(not busy)
        self.baro.setEnabled(not busy)

    # Orchestration
    def _process(self):
        p = self.path.text().strip()
        if not p:
            self.log.append("⚠️ Provide a workbook or directory path.")
            return
        src = Path(p)
        name = self.preset.currentText()
        preset = self._presets[name]
        baro = float(self.baro.value()) or None

        self._runner_thread = QThread(self)
        self._runner = _Runner(src, preset, baro, None)
        self._runner.moveToThread(self._runner_thread)
        self._runner_thread.started.connect(self._runner.run)
        self._runner.started.connect(lambda: (self._set_busy(True), self.log.append("▶︎ Starting…")))
        self._runner.progress.connect(lambda s: self.log.append(s))
        self._runner.failed.connect(self._on_failed)
        self._runner.finished.connect(self._on_finished)
        self._runner.finished.connect(self._runner_thread.quit)
        self._runner.failed.connect(self._runner_thread.quit)
        self._runner_thread.finished.connect(lambda: self._set_busy(False))
        self._runner_thread.start()

    def _on_failed(self, msg: str):
        self.log.append(f"❌ Failed: {msg}")

    def _on_finished(self, out_dir: str):
        self.log.append(f"✅ Done. Output: {out_dir}")
