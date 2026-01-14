#!/bin/bash
# Example script demonstrating Rivet (STAMP) CLI usage
# This script shows common structural alignment workflows

# Set path to the stamp binary (adjust as needed)
STAMP="./target/release/stamp"

# --- Pairwise Alignment ---
# Align two protein structures and output the superposed result

echo "=== Pairwise Alignment ==="
$STAMP pairwise \
    -1 examples/data/protein1.pdb \
    -2 examples/data/protein2.pdb \
    --chain1 A \
    --chain2 A \
    -O output_pairwise.pdb \
    --matrix transform.txt \
    -v

# With secondary structure information (DSSP files)
# $STAMP pairwise \
#     -1 examples/data/protein1.pdb \
#     -2 examples/data/protein2.pdb \
#     --dssp1 examples/data/protein1.dssp \
#     --dssp2 examples/data/protein2.dssp \
#     --use-secondary \
#     -O output_with_ss.pdb

# --- Multiple Alignment ---
# Align 3+ structures using tree-guided approach
# Requires a domain file listing all structures

echo ""
echo "=== Multiple Alignment ==="
# Create a simple domain file
cat > domains.dom << EOF
protein1 examples/data/protein1.pdb A
protein2 examples/data/protein2.pdb A
protein3 examples/data/protein3.pdb A
EOF

$STAMP treewise \
    -f domains.dom \
    -O output_multiple.pdb \
    --refine \
    -v

# --- Database Scan ---
# Scan a query structure against a database of targets

echo ""
echo "=== Database Scan ==="
$STAMP scan \
    --query examples/data/query.pdb \
    --database targets.dom \
    --scancut 2.0 \
    --max-hits 100 \
    -O scan_results.txt \
    -v

# Quick scan mode for faster results
# $STAMP scan \
#     --query examples/data/query.pdb \
#     --database targets.dom \
#     --quick \
#     --max-hits 50

# --- Structure Information ---
# Display information about a structure

echo ""
echo "=== Structure Info ==="
$STAMP info \
    -f examples/data/protein1.pdb \
    --chain A \
    --sequence \
    --secondary

# --- Convert/Transform ---
# Apply a transformation matrix to a PDB file

echo ""
echo "=== Convert/Transform ==="
$STAMP convert \
    -i examples/data/protein2.pdb \
    -o protein2_transformed.pdb \
    -c A \
    --transform transform.txt

# --- Advanced Options ---
# Customize alignment parameters

echo ""
echo "=== Advanced Pairwise (Custom Parameters) ==="
$STAMP pairwise \
    -1 examples/data/protein1.pdb \
    -2 examples/data/protein2.pdb \
    --chain1 A \
    --chain2 A \
    --e1 2.0 \
    --e2 0.5 \
    -n 3 \
    --max-iter 200 \
    --convergence 0.001 \
    -O output_refined.pdb \
    -vv

# Scan mode for structures in different orientations
# $STAMP pairwise \
#     -1 examples/data/protein1.pdb \
#     -2 examples/data/protein2.pdb \
#     --scan-mode \
#     -O output_scanned.pdb

echo ""
echo "Done! Check the output files for results."
