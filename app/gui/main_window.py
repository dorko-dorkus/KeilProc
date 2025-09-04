from PySide6.QtWidgets import QMainWindow, QTabWidget


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()

    def setupUi(self):
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # existing tabs assembled elsewhere…
        # Insert Run Easy as the first tab so all others shift one index to the right
        try:
            from .run_easy_panel import RunEasyPanel
            run_easy = RunEasyPanel(self)
            # Assume a QTabWidget attribute named `tabs`; adjust if your object differs
            self.tabs.insertTab(0, run_easy, "Run Easy")
            self.tabs.setCurrentIndex(0)
        except Exception as e:
            # Non‑fatal: if wiring fails (e.g., different widget name), surface in console and continue
            import sys
            print(f"[RunEasy] wiring failed: {e}", file=sys.stderr)
