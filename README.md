# KielProc

Comprehensive processing suite for differential pressure data from coal mill ducts using Kiel probes and legacy piccolo tubes.

## Repository structure

- `kielproc/` – backend for physics mapping, lag removal, Deming regression, pooling and reporting.
- `tests/` – basic sanity tests built with `pytest`.
- `Design_and_Validation_Report.pdf`, `Holistic_Legacy_Integration_and_Verification_Plan.pdf` – project documentation.

## Quick start

```bash
python -m pip install -r requirements.txt -c constraints.txt
python app/gui/main_window.py  # launch the Run Easy GUI
```

## Pre-start checks

- Ports are weighted equally unless `weights.json` provides custom weights.
- `geometry.json` in the run directory defines duct dimensions and is auto-discovered.
- Provide an absolute static column or gauge + barometric; temperature must be in °C (> -273.15 °C) and values above 200 are treated as Kelvin.

## Windows installer

```bash
cd app
pyinstaller --noconsole --name KielProc gui/main_window.py
cd ..
ISCC tools/installer/kielproc.iss
```

The PyInstaller step writes `app/dist/KielProc/KielProc.exe`.
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

