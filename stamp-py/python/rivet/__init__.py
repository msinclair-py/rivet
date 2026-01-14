"""Rivet - Structural Alignment of Multiple Proteins (STAMP implementation).

This module provides Python bindings to the STAMP structural alignment
library implemented in Rust. It enables fast and accurate pairwise
and multiple protein structure alignment.

Example:
    >>> import rivet
    >>> d1 = rivet.Domain.from_pdb("1abc.pdb", chain='A')
    >>> d2 = rivet.Domain.from_pdb("2def.pdb", chain='A')
    >>> result = rivet.pairwise_align(d1, d2)
    >>> print(f"RMSD: {result.rmsd:.2f}, Score: {result.score:.4f}")

Classes:
    Domain: Protein domain with C-alpha coordinates
    Parameters: Alignment parameters
    Transform: 3D rigid body transformation
    AlignmentResult: Result of pairwise alignment
    MultipleAlignmentResult: Result of multiple alignment
    ScanHit: Database scan hit

Functions:
    align_pdbs: High-level API to align PDB files and write full output (recommended)
    pairwise_align: Perform pairwise structural alignment
    multiple_align: Perform multiple structure alignment
    scan_database: Scan query against database
    compute_rmsd: Compute RMSD between coordinate sets
    superpose: Compute optimal superposition
    distance_matrix: Compute pairwise distances
    centroid: Compute centroid of coordinates
    transform_pdb_file: Transform entire PDB file (all atoms) with alignment
"""

# Re-export everything from the Rust extension
from rivet.rivet import (
    AlignmentResult,
    Domain,
    MultipleAlignmentResult,
    Parameters,
    ScanHit,
    Transform,
    __version__,
    align_pdbs,
    centroid,
    compute_rmsd,
    distance_matrix,
    multiple_align,
    pairwise_align,
    scan_database,
    superpose,
    transform_pdb_file,
)

__all__ = [
    # Version
    "__version__",
    # Classes
    "Domain",
    "Parameters",
    "Transform",
    "AlignmentResult",
    "MultipleAlignmentResult",
    "ScanHit",
    # Functions
    "align_pdbs",
    "pairwise_align",
    "multiple_align",
    "scan_database",
    "compute_rmsd",
    "superpose",
    "distance_matrix",
    "centroid",
    "transform_pdb_file",
]
