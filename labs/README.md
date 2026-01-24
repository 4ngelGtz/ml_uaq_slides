# Labs (Python / Jupyter)

## Prerequisites
- Python 3.10+ recommended
- Check that python3 exists:
  - `python3 --version`

## Create a virtual environment
```bash
python3 -m venv .venv
```

## Activate the environment
- macOS/Linux:
  - `source .venv/bin/activate`
- Windows PowerShell:
  - `.venv\Scripts\Activate.ps1`

## Install dependencies
```bash
pip install -U pip
pip install -r requirements.txt
```

## Register the Jupyter kernel
```bash
python -m ipykernel install --user --name ml-uaq --display-name "ML UAQ"
```

## Run Jupyter Lab
```bash
jupyter lab
```

## Open the Week 01 notebooks
- `labs/week01/lab01_numpy_baseline.ipynb`
- `labs/week01/lab02_sklearn_pipeline_cv.ipynb`

## Troubleshooting
- "command not found: jupyter" => activate the venv and re-run the install steps
- Kernel not showing => re-run the ipykernel install, then restart Jupyter Lab
- Permission issues => keep the `--user` flag (already used above)
