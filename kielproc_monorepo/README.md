
# Kiel / Piccolo Processing Suite (Mono-repo)

This repository combines:
- `kielproc/` — backend library for physics mapping, lag removal, Deming α/β, pooling, and plotting.
- `tests/` — small sanity tests.

## Quick start
```bash
python -m pip install -r requirements.txt -c constraints.txt
```

## GUI features
- Map verification-plane qs→qt→Δp_vent with inputs `r=As/At`, `beta=dt/D1`, `sampling Hz`.
- Translate legacy piccolo via cross-correlation lag removal + Deming regression; pooled α,β.
- Flow heat map (θ vs z) from legacy static arrays; optional geometry scaling (D1,D2,L).
- Polar cross-section slice at a selected plane (scaled radius if geometry provided).

## Outputs
Files are written under the chosen Output directory: CSV tables and PNG plots.

## Notes
- Dynamics are processed in samples; seconds derived from Sampling Hz.
- Throat unknown is fine; geometry is optional for visual scaling only.
