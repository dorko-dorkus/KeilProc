# KielProc

Comprehensive processing suite for differential pressure data from coal mill ducts using Kiel probes and legacy piccolo tubes.

## Installation

```bash
python -m pip install .
```

## Usage

```bash
kielproc one-click <workbook> --site <Preset> --baro-override <Pa>
```

Running `kielproc one-click` executes the full SOP:

```
Parse → Integrate → Map → Fit → Translate → Report
```

Artifacts are written to a `RUN_<STAMP>/` directory:

```
RUN_<STAMP>/
  run_context.json
  _integrated/...
  _fit/...
  _translate/translated.csv
  _report/{legacy_results.csv,json,setpoints.json}
  summary.json
```

## Pre-start checks

- Ports are weighted equally unless `weights.json` provides custom weights.
- `geometry.json` in the run directory defines duct dimensions and is auto-discovered.
- Provide an absolute static column or gauge + barometric; temperature must be in °C (> -273.15 °C) and values above 200 are treated as Kelvin.

## Features

- Map verification-plane static pressure to throat flow and venturi differential pressure.
- Remove signal lag via cross-correlation and apply Deming regression to translate legacy piccolo data.
- Pool \u03b1, \u03b2 parameters across runs and generate CSV/PNG summary reports.

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

