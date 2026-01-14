# Rivet

A fast, modern implementation of the STAMP (Structural Alignment of Multiple Proteins) algorithm in Rust with Python bindings.

![CI](https://github.com/msinclair/rivet/actions/workflows/ci.yml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/rivet-rs.svg)](https://pypi.org/project/rivet-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Full PDB Output**: Aligns and transforms complete structures (all atoms), not just C-alpha
- **Simple API**: One function call to align multiple PDB files and write output
- **Fast**: Written in Rust for maximum performance
- **Safe**: Memory-safe by design, zero unsafe code
- **Cross-platform**: Linux, macOS, and Windows

## Installation

```bash
pip install rivet-rs
```

## Quick Start

### Align Multiple Structures (Simplest)

```python
import rivet

# Align PDB files and write full structures to output directory
result = rivet.align_pdbs(
    ["protein1.pdb", "protein2.pdb", "protein3.pdb"],
    output_dir="aligned/",
    chain="A"
)

print(f"Average RMSD: {result.avg_rmsd:.2f} Å")
print(f"Core positions: {result.n_core}")
# Output files written to aligned/aligned_protein1.pdb, etc.
```

### Pairwise Alignment

```python
import rivet

# Load structures
d1 = rivet.Domain.from_pdb("reference.pdb", chain="A")
d2 = rivet.Domain.from_pdb("mobile.pdb", chain="A")

# Align (use scan_mode=True for structures in different coordinate frames)
result = rivet.pairwise_align(d1, d2, scan_mode=True)

print(f"RMSD: {result.rmsd:.2f} Å")
print(f"Score: {result.score:.4f}")
print(f"Aligned: {result.n_aligned} residues")

# Write aligned structure (full PDB with all atoms by default)
d2.to_pdb("mobile_aligned.pdb", transform=result.transform)
```

### Parameters for Remote Homologs

When comparing distantly related structures, use tolerant parameters:

```python
params = rivet.Parameters()
params.e1 = 5.0   # Distance tolerance (default: 2.0)
params.e2 = 10.0  # Conformational tolerance (default: 5.0)

result = rivet.pairwise_align(d1, d2, params, scan_mode=True)
```

## Output Options

By default, `to_pdb()` writes the **full structure** (all atoms: backbone, side chains, waters, ligands):

```python
# Default: full structure with all atoms
d2.to_pdb("output.pdb", transform=result.transform)

# Explicit: C-alpha only (smaller file, faster)
d2.to_pdb("output_ca.pdb", transform=result.transform, full=False)
```

## Multiple Alignment with Manual Control

For more control over the alignment process:

```python
import rivet

# Load domains
domains = [
    rivet.Domain.from_pdb("protein1.pdb", chain="A"),
    rivet.Domain.from_pdb("protein2.pdb", chain="A"),
    rivet.Domain.from_pdb("protein3.pdb", chain="A"),
]

params = rivet.Parameters()
params.e1 = 5.0
params.e2 = 10.0

# Step 1: Pre-align to reference
aligned_domains = [domains[0]]
pre_transforms = [rivet.Transform()]

for i in range(1, len(domains)):
    result = rivet.pairwise_align(domains[0], domains[i], params, scan_mode=True)
    pre_transforms.append(result.transform)

    # Create pre-aligned domain
    coords = result.get_transformed_coordinates(domains[i])
    aligned = rivet.Domain.from_arrays(
        domains[i].id, coords, domains[i].sequence, chain=domains[i].chain
    )
    aligned_domains.append(aligned)

# Step 2: Run multiple alignment
result = rivet.multiple_align(aligned_domains, params, pre_transforms=pre_transforms)

# Step 3: Write output using composed transforms
original_files = ["protein1.pdb", "protein2.pdb", "protein3.pdb"]
for pdb_file, full_transform in zip(original_files, result.full_transforms):
    rivet.transform_pdb_file(pdb_file, f"aligned_{pdb_file}", full_transform)
```

## Database Scanning

```python
query = rivet.Domain.from_pdb("query.pdb", chain="A")
targets = [rivet.Domain.from_pdb(f, chain="A") for f in target_files]

hits = rivet.scan_database(query, targets, score_cutoff=0.3)

for hit in hits:
    print(f"{hit.target_id}: Score={hit.score:.4f}, RMSD={hit.rmsd:.2f} Å")
```

## Rust API

Add to your `Cargo.toml`:

```toml
[dependencies]
stamp-core = "0.1"
```

```rust
use stamp_core::{io::{parse_pdb, transform_pdb}, pairwise::align_pair, types::Parameters};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d1 = parse_pdb("reference.pdb", Some('A'))?;
    let d2 = parse_pdb("mobile.pdb", Some('A'))?;

    let params = Parameters::default();
    let result = align_pair(&d1, &d2, &params)?;

    println!("RMSD: {:.2} Å, Score: {:.4}", result.rmsd, result.score);

    // Transform full PDB (all atoms)
    transform_pdb("mobile.pdb", "mobile_aligned.pdb", &result.transform, None)?;

    Ok(())
}
```

## API Reference

### Functions

| Function | Description |
|----------|-------------|
| `align_pdbs(files, output_dir, ...)` | High-level: align PDB files and write output |
| `pairwise_align(d1, d2, params, scan_mode)` | Align two structures |
| `multiple_align(domains, params, pre_transforms)` | Multiple structure alignment |
| `transform_pdb_file(input, output, transform)` | Transform full PDB file |
| `scan_database(query, targets, ...)` | Scan query against database |
| `compute_rmsd(coords1, coords2)` | Compute RMSD |
| `superpose(fixed, mobile)` | Optimal superposition |

### Classes

| Class | Description |
|-------|-------------|
| `Domain` | Protein domain with coordinates |
| `Parameters` | Alignment parameters |
| `Transform` | 3D rigid body transformation |
| `AlignmentResult` | Pairwise alignment result |
| `MultipleAlignmentResult` | Multiple alignment result with `full_transforms` |

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `e1` | 2.0 | Distance tolerance (Å) - increase for remote homologs |
| `e2` | 5.0 | Conformational tolerance (Å) - increase for flexible regions |
| `n_passes` | 2 | Number of refinement passes |
| `scan_mode` | False | Enable sliding window scan for different coordinate frames |

## Algorithm

STAMP uses the Rossmann-Argos probability measure for structural equivalence:

1. Calculate probability matrix based on inter-residue distances
2. Smith-Waterman dynamic programming to find optimal alignment
3. Extract equivalent residue pairs above threshold
4. Compute optimal superposition (Kabsch algorithm)
5. Iterate until convergence

References:
- Russell & Barton, *Proteins* 14:309-323 (1992)
- Rossmann & Argos, *J. Mol. Biol.* 105:75-95 (1976)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{russell1992multiple,
  title={Multiple protein sequence alignment from tertiary structure comparison},
  author={Russell, Robert B and Barton, Geoffrey J},
  journal={Proteins: Structure, Function, and Bioinformatics},
  volume={14},
  number={2},
  pages={309--323},
  year={1992}
}
```
