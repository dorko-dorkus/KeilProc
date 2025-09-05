from PySide6.QtWidgets import QMainWindow, QTabWidget


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()

    def setupUi(self):
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # existing tabs assembled elsewhere…
        # Always prepend the Run Easy panel so operators see it first
        self._insert_run_easy()

    def _insert_run_easy(self) -> None:
        """Insert the Run Easy panel as the first tab.

        Any import or wiring issues are treated as non-fatal; the GUI should
        continue to load even if the panel cannot be constructed (e.g. missing
        optional dependencies).  In such cases the error is surfaced to the
        console for easier diagnosis.
        """

        try:  # pragma: no cover - exercised via unit tests
            from .run_easy_panel import RunEasyPanel

            run_easy = RunEasyPanel(self)
            # Assume a QTabWidget attribute named `tabs`; adjust if your object differs
            self.tabs.insertTab(0, run_easy, "Run Easy")
            self.tabs.setCurrentIndex(0)
        except Exception as e:  # pragma: no cover - exercised via unit tests
            # Non-fatal: if wiring fails (e.g., different widget name), surface in console and continue
            import sys

            print(f"[RunEasy] wiring failed: {e}", file=sys.stderr)
