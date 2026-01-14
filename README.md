# Rivet

A fast, safe implementation of the STAMP (Structural Alignment of Multiple Proteins) algorithm in Rust with Python bindings.

[![CI](https://github.com/msinclair/rivet/actions/workflows/ci.yml/badge.svg)](https://github.com/msinclair/rivet/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rivet-rs.svg)](https://pypi.org/project/rivet-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Rivet provides structural alignment of protein structures using the STAMP algorithm described in:

- Russell & Barton, *Proteins* 14:309-323 (1992)
- Rossmann & Argos, *J. Mol. Biol.* 105:75-95 (1976)

Key features:
- **Fast**: Written in Rust for maximum performance
- **Safe**: Zero unsafe code, memory-safe by design
- **Easy to use**: Simple Python API via `pip install rivet-rs`
- **Cross-platform**: Supports Linux, macOS, and Windows

## Installation

### Python

```bash
pip install rivet-rs
```

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
stamp-core = "0.1"
```

## Quick Start

### Python

```python
import rivet

# Load protein structures
d1 = rivet.Domain.from_pdb("protein1.pdb", chain="A")
d2 = rivet.Domain.from_pdb("protein2.pdb", chain="A")

# Perform pairwise alignment
result = rivet.pairwise_align(d1, d2)

print(f"RMSD: {result.rmsd:.2f} Å")
print(f"Aligned residues: {result.n_aligned}")
print(f"Score: {result.score:.4f}")

# Get transformation matrix
transform = result.transform
rotated_coords = transform.apply(d2.coordinates)

# Multiple structure alignment
domains = [d1, d2, d3]
multi_result = rivet.multiple_align(domains)
print(f"Core positions: {multi_result.n_core}")
```

### Rust

```rust
use stamp_core::{io::parse_pdb, pairwise::align_pair, types::Parameters};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load structures
    let domain1 = parse_pdb("protein1.pdb", Some('A'))?;
    let domain2 = parse_pdb("protein2.pdb", Some('A'))?;

    // Align with default parameters
    let params = Parameters::default();
    let result = align_pair(&domain1, &domain2, &params)?;

    println!("RMSD: {:.2} Å", result.rmsd);
    println!("Score: {:.4}", result.score);

    Ok(())
}
```

## API Reference

### Python Classes

| Class | Description |
|-------|-------------|
| `Domain` | Protein domain with C-alpha coordinates |
| `Parameters` | Alignment parameters (E1, E2, gap penalties, etc.) |
| `Transform` | 3D rigid body transformation (rotation + translation) |
| `AlignmentResult` | Result of pairwise alignment |
| `MultipleAlignmentResult` | Result of multiple alignment |
| `ScanHit` | Database scan hit |

### Python Functions

| Function | Description |
|----------|-------------|
| `pairwise_align(d1, d2, params=None)` | Align two structures |
| `multiple_align(domains, params=None)` | Align multiple structures |
| `scan_database(query, targets, ...)` | Scan query against database |
| `compute_rmsd(coords1, coords2)` | Compute RMSD |
| `superpose(fixed, mobile)` | Optimal superposition |
| `distance_matrix(coords1, coords2)` | Pairwise distances |
| `centroid(coords)` | Compute centroid |

### Parameters

Key alignment parameters with defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `e1` | 2.0 | Distance tolerance (Å) |
| `e2` | 5.0 | Conformational tolerance (Å) |
| `n_passes` | 2 | Number of fitting passes |
| `max_iter` | 100 | Maximum iterations |
| `use_secondary` | true | Use secondary structure |

## Command Line Interface

The `stamp` CLI is also available:

```bash
# Pairwise alignment
stamp pairwise protein1.pdb protein2.pdb

# Multiple alignment from domain file
stamp treewise domains.dom -o aligned.pdb

# Database scan
stamp scan query.pdb -d database.dom
```

## Algorithm

STAMP uses the Rossmann-Argos probability measure to score structural equivalence between residue pairs:

```
Pij = exp(Dij + Cij)

Where:
- Dij = -distance² / (2×E1²)     [distance component]
- Cij = -Sij / (2×E2²)           [conformational component]
```

The alignment is iteratively refined using:
1. Calculate probability matrix
2. Smith-Waterman dynamic programming
3. Extract equivalent residue pairs
4. Compute optimal superposition (Kabsch algorithm)
5. Check convergence

## Performance

Rivet is designed for high performance:
- Zero-copy NumPy array integration
- Efficient matrix operations via nalgebra
- Optional parallel processing with rayon

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use Rivet in your research, please cite:

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
