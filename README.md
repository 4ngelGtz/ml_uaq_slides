# ML UAQ Slides (LaTeX Beamer)

## Labs (Python / Jupyter)
Quick start (macOS/Linux):
```bash
make venv
make deps
make kernel
make jupyter
```
Then open:
- `labs/week01/lab01_numpy_baseline.ipynb`
- `labs/week01/lab02_sklearn_pipeline_cv.ipynb`

Windows note: activate the venv with `.\.venv\Scripts\Activate.ps1` and run the
same pip/ipykernel steps from `labs/README.md`.
More details in `labs/README.md`.

## Requisitos
- TeX Live (o MacTeX)
- `latexmk` (recomendado)

## Compilar Semana 1
```bash
make week01
```

Nota: en la primera compilacion, TeX Live puede generar cache de fuentes en
`/Users/<usuario>/Library/texlive`. Si ves errores de permisos, asegúrate de
que ese directorio exista y sea escribible por tu usuario.
