# NN Archi

Layout-focused version of the project.

This repository keeps only the neural-network-assisted rectangular layout solver and the sample input data required to run it.

This project is source-available for non-commercial use only. Commercial use requires a separate license from the author.

## Repository layout

- `src/nn_archi/layout_solver.py` - main rectangular layout solver
- `data/sample/spaces.csv` - sample space program input
- `data/sample/relationships.csv` - sample relationship matrix input
- `outputs/layouts/` - generated results

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
nn-archi-layout --rect-length 400
```

Or run the module directly:

```bash
python -m nn_archi.layout_solver --rect-length 400
```

## Outputs

Each run writes these files into `outputs/layouts/` by default:

- `layout_result.csv`
- `corridor_network.json`
- `layout.png`
- `dropped_spaces.csv`

## Notes

- Use `--spaces-csv`, `--relationships-csv`, and `--outdir` to point the solver at different datasets.
- Commercial licensing contact: `zhorliari@gmail.com`
- This is not an OSI open-source license because commercial use is restricted.
