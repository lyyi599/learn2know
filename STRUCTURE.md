# Repository Structure

```
sutskever-30-implementations/
|-- README.md
|-- STRUCTURE.md
|-- .gitignore
|
|-- docs/                            # Project docs and summaries
|-- notebooks/                       # NumPy notebook track (30/30)
|-- pytorch/                         # PyTorch notebook track (planned)
|-- src/                             # Core NumPy implementations
|-- scripts/                         # Demos, training, utilities
|-- assets/
|   |-- images/                      # Figures and plots
|   |-- models/                      # .npz model artifacts
|   `-- results/                     # .json result artifacts
|-- archive/
|   `-- tests/                       # Archived test scripts
`-- .git/
```

Notes:
- Notebooks are under `notebooks/`. Planned PyTorch notebooks live in `pytorch/`.
- Core Python modules live in `src/`, while runnable demos/trainers are in `scripts/`.
- Generated artifacts are in `assets/` to keep the root directory clean.
