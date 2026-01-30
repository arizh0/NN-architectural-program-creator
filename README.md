# NN Archi

NN Archi is a neural-network-assisted rectangular layout solver for architectural space planning.

It takes a space program table plus a relationship matrix, selects a working subset of spaces, and fits them into a long rectangular planning envelope while balancing adjacency strength, overlap avoidance, bounds, and corridor access. The repository currently focuses on the layout engine only.

This project is source-available for non-commercial use only. Commercial use requires a separate license from the author.

## What It Does

- Loads a space program from CSV with geometry, area, access level, and public-role metadata
- Loads a directed relationship matrix between spaces
- Uses a lightweight message-passing neural network to initialize placement
- Optimizes rectangular room positions inside a fixed planning envelope
- Updates corridor paths with grid-based routing
- Optionally drops the most problematic rooms until a feasible layout is found
- Exports both machine-readable and visual outputs

## Repository Structure

- `src/nn_archi/layout_solver.py` - main solver and CLI
- `data/sample/spaces.csv` - sample space program dataset
- `data/sample/relationships.csv` - sample relationship matrix
- `outputs/layouts/` - generated outputs
- `LICENSE` - non-commercial source license

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Quick Start

Run with the bundled sample dataset:

```bash
nn-archi-layout --rect-length 400
```

Or run the module directly:

```bash
python -m nn_archi.layout_solver --rect-length 400
```

Example with a custom output directory and feasibility pruning:

```bash
python -m nn_archi.layout_solver ^
  --rect-length 400 ^
  --n_spaces 60 ^
  --drop_until_fit ^
  --outdir outputs/layouts/run_01
```

## Inputs

The solver expects two CSV files:

`spaces.csv`

- One row per space
- Must include geometric fields such as `Length_m`, `Depth_m`, and `NetArea_m2`
- Also uses metadata such as `SpaceName`, `AccessLevel`, and `PublicInterfaceRole`

`relationships.csv`

- Square matrix of directed relationship weights
- First column must be `SpaceName`
- Remaining columns must match the space names used by the solver

The repository includes working sample files in `data/sample/`.

## Outputs

Each run writes these files into `outputs/layouts/` by default:

- `layout_result.csv` - final room coordinates, sizes, access metadata, and blocker flags
- `corridor_network.json` - routed corridor polylines
- `layout.png` - rendered rectangular layout preview
- `dropped_spaces.csv` - spaces removed during fit attempts, if any

## Main CLI Options

- `--rect-length` - length of the rectangular planning envelope
- `--spaces-csv` - path to the space program table
- `--relationships-csv` - path to the relationship matrix
- `--outdir` - output folder for generated files
- `--n_spaces` - number of spaces to sample from the dataset
- `--drop_until_fit` - keep pruning spaces until the layout becomes feasible
- `--corridor_w` - corridor width
- `--no_access_legend` - disable the access legend in the output image

## Technical Notes

- Implemented in Python with `numpy`, `pandas`, `matplotlib`, and `torch`
- Uses a compact message-passing neural network for initialization rather than full model training infrastructure
- Designed around rectangular spaces only
- The current sample data comes from a generated research-building program and is included as an example dataset

## License

Non-commercial use is permitted under the license in [LICENSE](LICENSE).

- Commercial licensing contact: `zhorliari@gmail.com`
- This is not an OSI open-source license because commercial use is restricted

## Suggested GitHub Metadata

Description:

`Neural-network-assisted rectangular layout solver for architectural space planning from program and relationship-matrix data.`

Topics:

`architecture`, `space-planning`, `layout-optimization`, `computational-design`, `generative-design`, `rectangular-layout`, `floorplan-generation`, `pytorch`, `graph-neural-network`, `message-passing-neural-network`
