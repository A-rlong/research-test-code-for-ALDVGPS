# AL-DVGPS experiment artifact

This repository contains the prototype implementation and experiment harness for
the manuscript "Lattice-Based Auditable Designated-Verifier Group Proxy Signatures".

## Environment

- Python 3.10 or later
- CPU execution is sufficient for the smoke tests
- Optional GPU support can be installed with the `gpu` extra in `pyproject.toml`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .[dev]
```

On Unix-like systems, replace the activation command with
`source .venv/bin/activate`.

## Tests

```bash
python -m pytest
```

## Reproducing core outputs

If the package has not been installed, set `PYTHONPATH` before running the
wrapper scripts:

```powershell
$env:PYTHONPATH = "src"
```

On Unix-like systems:

```bash
export PYTHONPATH=src
```

Run a single workflow smoke test:

```bash
python scripts/run_flow.py --track GPV-S --verifier-count 1 --mode optimized-hybrid --message "review-smoke" --output artifacts/final/review_smoke_flow.json
```

Run a small benchmark:

```bash
python scripts/run_bench.py --tracks GPV-S --verifier-counts 1 --trials 1 --mode optimized-hybrid --formal-only --workspace-root . --output artifacts/final/review_smoke_bench.csv
```

The full benchmark data used by the manuscript is included under `results/`.
The final public version should be archived on Zenodo and cited by DOI in the
manuscript.
