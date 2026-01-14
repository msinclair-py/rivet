# Rivet Python Bindings

Python bindings for the Rivet structural alignment library.

## Installation

```bash
pip install rivet-rs
```

## Quick Start

```python
import rivet

# Simple: align PDB files and write full structures
result = rivet.align_pdbs(
    ["protein1.pdb", "protein2.pdb", "protein3.pdb"],
    output_dir="aligned/"
)

# Pairwise alignment
d1 = rivet.Domain.from_pdb("reference.pdb", chain="A")
d2 = rivet.Domain.from_pdb("mobile.pdb", chain="A")
result = rivet.pairwise_align(d1, d2, scan_mode=True)

# Write full structure (all atoms, not just C-alpha)
d2.to_pdb("aligned.pdb", transform=result.transform)
```

## Key Features

- **Full PDB output by default**: `to_pdb()` transforms all atoms (backbone, side chains, waters, ligands)
- **High-level API**: `align_pdbs()` handles loading, alignment, and output in one call
- **Automatic transform composition**: `MultipleAlignmentResult.full_transforms` provides ready-to-use transforms

See the main [README](../README.md) for full documentation.
