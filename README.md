# KielProc

Comprehensive processing suite for differential pressure data from coal mill ducts using Kiel probes and legacy piccolo tubes.

## Repository structure

- `kielproc_monorepo/` – combined library, GUI and legacy tools.
  - `kielproc/` – backend for physics mapping, lag removal, Deming regression, pooling and reporting.
  - `gui/` – Tkinter application `app_gui.py` and the original reference GUI.
  - `duct_dp_visualizer.py` – legacy Excel/CSV analyzer (GUI and CLI).
  - `tests/` – basic sanity tests built with `pytest`.
  - `Design_and_Validation_Report.pdf`, `Holistic_Legacy_Integration_and_Verification_Plan.pdf` – project documentation.

## Quick start

```bash
python -m pip install -r kielproc_monorepo/requirements.txt -c kielproc_monorepo/constraints.txt
python kielproc_monorepo/gui/app_gui.py  # launch the main GUI
```

Legacy visualizer usage:

```bash
python kielproc_monorepo/duct_dp_visualizer.py            # PySimpleGUI
python kielproc_monorepo/duct_dp_visualizer.py --cli --help  # CLI options
```

## Pre-start checks

- Ports are weighted equally unless `weights.json` provides custom weights.
- `geometry.json` in the run directory defines duct dimensions and is auto-discovered.
- Provide an absolute static column or gauge + barometric; temperature must be in °C (> -273.15 °C) and values above 200 are treated as Kelvin.

## Windows installer

```bash
cd kielproc_monorepo
pyinstaller --noconsole --name KielProc gui/app_gui.py
cd ..
ISCC tools/installer/kielproc.iss
```

The PyInstaller step writes `kielproc_monorepo/dist/KielProc/KielProc.exe`.
The Inno Setup script creates `tools/installer/Output/KielProcInstaller.exe`.


## Features

- Map verification-plane static pressure to throat flow and venturi differential pressure.
- Remove signal lag via cross-correlation and apply Deming regression to translate legacy piccolo data.
- Pool \u03b1, \u03b2 parameters across runs and generate CSV/PNG summary reports.
- Produce flow heat maps and polar cross-section slices for visual analysis.
- Legacy tool supports Excel or CSV logs, per-port slicing, Pearson *r* with Fisher-z CI, Theil–Sen slope with bootstrap CI, sign-flip probabilities, and tidy outputs.

## Testing

The repository uses [nox](https://nox.thea.codes/) to create an isolated
virtual environment with pinned dependencies, run the test suite, and execute
an import-only CLI smoke demo.

```bash
python -m pip install nox
nox
```

## License

This project is released under the [MIT License](LICENSE).

