#!/usr/bin/env python3
"""
Example script demonstrating the Rivet (STAMP) Python API.
This script shows common structural alignment workflows.

Install: pip install rivet-rs
"""

import rivet
import numpy as np


def simple_pairwise_example():
    """Simplest pairwise alignment - just align and write output."""
    print("=== Simple Pairwise Alignment ===")

    # Load two structures
    domain1 = rivet.Domain.from_pdb("examples/data/protein1.pdb", chain="A")
    domain2 = rivet.Domain.from_pdb("examples/data/protein2.pdb", chain="A")

    print(f"Domain 1: {domain1.id}, {len(domain1)} residues")
    print(f"Domain 2: {domain2.id}, {len(domain2)} residues")

    # Use tolerant parameters for comparing different structures
    params = rivet.Parameters()
    params.e1 = 5.0   # Distance tolerance
    params.e2 = 10.0  # Conformational tolerance

    # Align (scan_mode=True for structures in different coordinate frames)
    result = rivet.pairwise_align(domain1, domain2, params, scan_mode=True)

    print(f"\nResults:")
    print(f"  RMSD: {result.rmsd:.2f} A")
    print(f"  Score: {result.score:.4f}")
    print(f"  Aligned: {result.n_aligned} residues")

    # Write output - defaults to FULL PDB (all atoms, not just C-alpha)
    domain2.to_pdb("aligned_output.pdb", transform=result.transform)
    print("\nFull structure written to: aligned_output.pdb")

    # If you specifically need C-alpha only (rare), use full=False
    domain2.to_pdb("aligned_ca_only.pdb", transform=result.transform, full=False)
    print("C-alpha only written to: aligned_ca_only.pdb")

    return result


def simple_multiple_alignment_example():
    """Simplest multiple alignment - one function call does everything."""
    print("\n=== Simple Multiple Alignment ===")

    # List of PDB files to align
    pdb_files = [
        "examples/data/protein1.pdb",
        "examples/data/protein2.pdb",
        "examples/data/protein3.pdb",
    ]

    print(f"Aligning {len(pdb_files)} structures...")

    # Use tolerant parameters for comparing different structures
    params = rivet.Parameters()
    params.e1 = 5.0   # Distance tolerance
    params.e2 = 10.0  # Conformational tolerance

    # Single function call: loads, pre-aligns, runs MSA, writes full PDB output
    result = rivet.align_pdbs(
        pdb_files,
        output_dir="aligned_structures",
        chain="A",
        params=params
    )

    print(f"\nResults:")
    print(f"  Average RMSD: {result.avg_rmsd:.2f} A")
    print(f"  Score: {result.score:.4f}")
    print(f"  Core positions: {result.n_core}")

    print("\nAligned structures written to aligned_structures/")

    return result


def detailed_pairwise_example():
    """Pairwise alignment with custom parameters and detailed output."""
    print("\n=== Detailed Pairwise Alignment ===")

    domain1 = rivet.Domain.from_pdb("examples/data/protein1.pdb", chain="A")
    domain2 = rivet.Domain.from_pdb("examples/data/protein2.pdb", chain="A")

    # Custom parameters for remote homologs
    params = rivet.Parameters()
    params.n_passes = 2
    params.e1 = 5.0       # Distance tolerance (larger for remote homologs)
    params.e2 = 10.0      # Conformational tolerance
    params.max_iter = 100
    params.convergence = 0.01

    result = rivet.pairwise_align(domain1, domain2, params, scan_mode=True)

    print(f"\nAlignment Results:")
    print(f"  RMSD: {result.rmsd:.2f} A")
    print(f"  Score (Sc): {result.score:.4f}")
    print(f"  Aligned residues: {result.n_aligned}")
    print(f"  Sequence identity: {result.seq_identity * 100:.1f}%")

    # Access transformation details
    rotation = result.get_rotation()
    translation = result.get_translation()
    print(f"\nTransformation:")
    print(f"  Rotation matrix:\n{rotation}")
    print(f"  Translation: {translation}")

    return result


def detailed_multiple_alignment_example():
    """Multiple alignment with manual control over the process."""
    print("\n=== Detailed Multiple Alignment ===")

    # Load domains
    domains = [
        rivet.Domain.from_pdb("examples/data/protein1.pdb", chain="A"),
        rivet.Domain.from_pdb("examples/data/protein2.pdb", chain="A"),
        rivet.Domain.from_pdb("examples/data/protein3.pdb", chain="A"),
    ]

    params = rivet.Parameters()
    params.e1 = 5.0
    params.e2 = 10.0

    # Step 1: Pre-align all structures to reference (first domain)
    print("Pre-aligning structures to reference...")
    aligned_domains = [domains[0]]
    pre_transforms = [rivet.Transform()]  # Identity for reference

    for i in range(1, len(domains)):
        result = rivet.pairwise_align(domains[0], domains[i], params, scan_mode=True)
        print(f"  {domains[i].id} -> {domains[0].id}: RMSD={result.rmsd:.2f} A")

        pre_transforms.append(result.transform)

        # Create pre-aligned domain
        transformed_coords = result.get_transformed_coordinates(domains[i])
        aligned_domain = rivet.Domain.from_arrays(
            domains[i].id + "_aligned",
            transformed_coords,
            domains[i].sequence,
            chain=domains[i].chain
        )
        aligned_domains.append(aligned_domain)

    # Step 2: Run multiple alignment with pre-alignment transforms
    print("\nRunning multiple alignment...")
    result = rivet.multiple_align(aligned_domains, params, pre_transforms=pre_transforms)

    print(f"\nResults:")
    print(f"  Average RMSD: {result.avg_rmsd:.2f} A")
    print(f"  Score: {result.score:.4f}")
    print(f"  Alignment columns: {result.n_columns}")
    print(f"  Core positions: {result.n_core}")

    # Step 3: Write output using full_transforms (auto-composed)
    print("\nWriting aligned structures...")

    # The full_transforms property gives us the composed transforms
    # (pre-alignment + MSA refinement)
    original_files = [
        "examples/data/protein1.pdb",
        "examples/data/protein2.pdb",
        "examples/data/protein3.pdb",
    ]

    import os
    os.makedirs("detailed_aligned", exist_ok=True)

    for i, (pdb_file, full_transform) in enumerate(zip(original_files, result.full_transforms)):
        output_path = f"detailed_aligned/aligned_{os.path.basename(pdb_file)}"
        rivet.transform_pdb_file(pdb_file, output_path, full_transform)
        print(f"  Written: {output_path}")

    print(f"\nFull transforms:")
    for i, ft in enumerate(result.full_transforms):
        print(f"  Domain {i+1}: translation = {ft.translation}")

    return result


def database_scan_example():
    """Demonstrate database scanning."""
    print("\n=== Database Scan ===")

    query = rivet.Domain.from_pdb("examples/data/query.pdb", chain="A")
    print(f"Query: {query.id}, {len(query)} residues")

    target_files = [
        "examples/data/target1.pdb",
        "examples/data/target2.pdb",
        "examples/data/target3.pdb",
    ]
    targets = [rivet.Domain.from_pdb(f, chain="A") for f in target_files]
    print(f"Database: {len(targets)} structures")

    params = rivet.Parameters()
    params.e1 = 5.0
    params.e2 = 10.0

    hits = rivet.scan_database(
        query,
        targets,
        params=params,
        score_cutoff=0.3,
        max_hits=100
    )

    print(f"\nFound {len(hits)} hits:")
    for hit in hits:
        print(f"  {hit.rank}. {hit.target_id}: "
              f"Score={hit.score:.4f}, RMSD={hit.rmsd:.2f} A, "
              f"Aligned={hit.n_aligned}")

    return hits


def coordinate_utilities():
    """Demonstrate coordinate manipulation utilities."""
    print("\n=== Coordinate Utilities ===")

    # Create domain from numpy arrays
    coords = np.array([
        [0.0, 0.0, 0.0],
        [3.8, 0.0, 0.0],
        [7.6, 0.0, 0.0],
        [11.4, 0.0, 0.0],
    ], dtype=np.float64)
    sequence = "AAAA"

    domain = rivet.Domain.from_arrays("synthetic", coords, sequence)
    print(f"Created domain: {domain.id}, {len(domain)} residues")

    # RMSD calculation
    coords1 = np.random.randn(100, 3)
    coords2 = coords1 + np.random.randn(100, 3) * 0.5
    rmsd = rivet.compute_rmsd(coords1, coords2)
    print(f"\nRMSD between random coords: {rmsd:.2f} A")

    # Optimal superposition
    transform, rmsd = rivet.superpose(coords1, coords2)
    print(f"After superposition: RMSD = {rmsd:.2f} A")

    # Distance matrix
    distances = rivet.distance_matrix(coords[:2], coords[:2])
    print(f"\nDistance matrix:\n{distances}")

    # Centroid
    cx, cy, cz = rivet.centroid(coords)
    print(f"\nCentroid: ({cx:.2f}, {cy:.2f}, {cz:.2f})")


def transform_operations():
    """Demonstrate Transform class operations."""
    print("\n=== Transform Operations ===")

    # Identity transform
    t1 = rivet.Transform()
    print(f"Identity translation: {t1.translation}")

    # Custom transform (90 degree rotation around Z + translation)
    rotation = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    translation = np.array([10.0, 20.0, 30.0])
    t2 = rivet.Transform.from_matrix(rotation, translation)

    # Apply to coordinates
    coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    transformed = t2.apply(coords)
    print(f"\nOriginal: {coords[0]}")
    print(f"Transformed: {transformed[0]}")

    # Inverse and composition
    t2_inv = t2.inverse()
    restored = t2_inv.apply(transformed)
    print(f"Restored: {restored[0]}")

    t_composed = t2.compose(t2_inv)
    print(f"\nComposed rotation (should be identity):\n{t_composed.rotation}")


def main():
    """Run all examples."""
    print("Rivet (STAMP) Python API Examples")
    print("=" * 50)

    # Simple examples (recommended starting point)
    simple_pairwise_example()
    simple_multiple_alignment_example()

    # Detailed examples (for advanced usage)
    detailed_pairwise_example()
    detailed_multiple_alignment_example()

    # Other functionality
    database_scan_example()
    coordinate_utilities()
    transform_operations()

    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    main()
