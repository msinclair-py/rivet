//! Core data structures for STAMP.
//!
//! This module defines the fundamental types used throughout the library,
//! including protein domain representations, alignment parameters, and results.

use nalgebra::{Matrix3, Point3, Vector3};
use thiserror::Error;

/// Type alias for 3D coordinates.
pub type Coord3 = Point3<f64>;

/// Type alias for 3D vectors.
pub type Vec3 = Vector3<f64>;

/// Type alias for 3x3 rotation matrices.
pub type RotationMatrix = Matrix3<f64>;

/// Result type for STAMP operations.
pub type StampResult<T> = Result<T, StampError>;

/// Errors that can occur during STAMP operations.
#[derive(Error, Debug)]
pub enum StampError {
    /// Error reading or parsing input files.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Error parsing PDB file format.
    #[error("PDB parsing error: {0}")]
    PdbParse(String),

    /// Error parsing DSSP file format.
    #[error("DSSP parsing error: {0}")]
    DsspParse(String),

    /// Error parsing domain file.
    #[error("Domain file error: {0}")]
    DomainFile(String),

    /// Invalid parameter configuration.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Alignment failed.
    #[error("Alignment failed: {0}")]
    AlignmentFailed(String),

    /// Structure has insufficient residues.
    #[error("Structure too small: {0} residues (minimum {1})")]
    StructureTooSmall(usize, usize),

    /// No common atoms for superposition.
    #[error("No common atoms for superposition")]
    NoCommonAtoms,
}

/// Represents a single amino acid residue.
#[derive(Debug, Clone)]
pub struct Residue {
    /// Residue sequence number.
    pub seq_num: i32,
    /// PDB residue number (with insertion code).
    pub pdb_num: String,
    /// One-letter amino acid code.
    pub aa: char,
    /// C-alpha coordinates.
    pub ca_coord: Coord3,
    /// Secondary structure assignment (H=helix, E=strand, C=coil).
    pub sec_struct: char,
    /// Accessibility (from DSSP).
    pub accessibility: f64,
}

impl Residue {
    /// Creates a new residue with the given properties.
    #[must_use]
    pub fn new(seq_num: i32, pdb_num: String, aa: char, ca_coord: Coord3) -> Self {
        Self {
            seq_num,
            pdb_num,
            aa,
            ca_coord,
            sec_struct: 'C',
            accessibility: 0.0,
        }
    }
}

/// Represents a protein domain for structural alignment.
#[derive(Debug, Clone)]
pub struct Domain {
    /// Domain identifier.
    pub id: String,
    /// Source PDB file path.
    pub pdb_file: String,
    /// Chain identifier.
    pub chain: char,
    /// List of residues.
    pub residues: Vec<Residue>,
    /// Whether DSSP data has been loaded.
    pub has_dssp: bool,
}

impl Domain {
    /// Creates an empty domain with the given identifier.
    #[must_use]
    pub fn new(id: String) -> Self {
        Self {
            id,
            pdb_file: String::new(),
            chain: 'A',
            residues: Vec::new(),
            has_dssp: false,
        }
    }

    /// Returns the number of residues in the domain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.residues.len()
    }

    /// Returns true if the domain has no residues.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.residues.is_empty()
    }

    /// Returns an iterator over C-alpha coordinates.
    pub fn ca_coords(&self) -> impl Iterator<Item = &Coord3> {
        self.residues.iter().map(|r| &r.ca_coord)
    }

    /// Returns the amino acid sequence as a string.
    #[must_use]
    pub fn sequence(&self) -> String {
        self.residues.iter().map(|r| r.aa).collect()
    }

    /// Returns the secondary structure string.
    #[must_use]
    pub fn secondary_structure(&self) -> String {
        self.residues.iter().map(|r| r.sec_struct).collect()
    }
}

/// Represents a 3D rigid body transformation.
#[derive(Debug, Clone)]
pub struct Transform {
    /// 3x3 rotation matrix.
    pub rotation: RotationMatrix,
    /// Translation vector.
    pub translation: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            rotation: RotationMatrix::identity(),
            translation: Vec3::zeros(),
        }
    }
}

impl Transform {
    /// Creates a new identity transformation.
    #[must_use]
    pub fn identity() -> Self {
        Self::default()
    }

    /// Applies the transformation to a point.
    #[must_use]
    pub fn apply(&self, point: &Coord3) -> Coord3 {
        let rotated = self.rotation * point.coords;
        Coord3::from(rotated + self.translation)
    }

    /// Composes this transformation with another (self * other).
    #[must_use]
    pub fn compose(&self, other: &Transform) -> Transform {
        Transform {
            rotation: self.rotation * other.rotation,
            translation: self.rotation * other.translation + self.translation,
        }
    }

    /// Returns the inverse transformation.
    #[must_use]
    pub fn inverse(&self) -> Transform {
        let inv_rotation = self.rotation.transpose();
        Transform {
            rotation: inv_rotation,
            translation: -(inv_rotation * self.translation),
        }
    }
}

/// Parameters controlling the STAMP alignment algorithm.
#[derive(Debug, Clone)]
pub struct Parameters {
    /// Number of initial fit passes.
    pub n_passes: u32,
    /// Initial slide offset.
    pub slide_offset: i32,
    /// Minimum domain separation.
    pub min_domain_sep: usize,
    /// Score cutoff for alignment.
    pub score_cutoff: f64,
    /// Penalty for alignment extension.
    pub pen_ext: f64,
    /// Penalty for gap opening.
    pub pen_gap: f64,
    /// Use secondary structure in scoring.
    pub use_secondary: bool,
    /// First constant in Pij calculation.
    pub e1: f64,
    /// Second constant in Pij calculation.
    pub e2: f64,
    /// Cutoff distance for Pij.
    pub cutoff_dist: f64,
    /// Maximum number of iterations.
    pub max_iter: u32,
    /// Convergence threshold.
    pub convergence: f64,
    /// Verbosity level.
    pub verbosity: u32,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            n_passes: 2,
            slide_offset: 30,
            min_domain_sep: 10,
            score_cutoff: 2.0,
            pen_ext: 0.5,
            pen_gap: 3.0,
            use_secondary: true,
            e1: 2.0,
            e2: 5.0,
            cutoff_dist: 10.0,
            max_iter: 100,
            convergence: 0.01,
            verbosity: 1,
        }
    }
}

/// Result of a pairwise structural alignment.
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    /// Transformation to superpose second structure onto first.
    pub transform: Transform,
    /// Root mean square deviation after superposition.
    pub rmsd: f64,
    /// Number of aligned residue pairs.
    pub n_aligned: usize,
    /// STAMP score (Sc).
    pub score: f64,
    /// Sequence identity of aligned region.
    pub seq_identity: f64,
    /// Alignment path (pairs of indices).
    pub path: Vec<(usize, usize)>,
    /// Per-residue scores.
    pub residue_scores: Vec<f64>,
}

impl AlignmentResult {
    /// Creates a new empty alignment result.
    #[must_use]
    pub fn new() -> Self {
        Self {
            transform: Transform::identity(),
            rmsd: f64::INFINITY,
            n_aligned: 0,
            score: 0.0,
            seq_identity: 0.0,
            path: Vec::new(),
            residue_scores: Vec::new(),
        }
    }
}

impl Default for AlignmentResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Entry in a distance/similarity matrix.
#[derive(Debug, Clone)]
pub struct MatrixEntry {
    /// Index of first structure.
    pub i: usize,
    /// Index of second structure.
    pub j: usize,
    /// Distance or similarity value.
    pub value: f64,
}

/// Represents a node in the guide tree for multiple alignment.
#[derive(Debug, Clone)]
pub enum TreeNode {
    /// Leaf node containing a domain index.
    Leaf(usize),
    /// Internal node with left and right children and branch length.
    Internal {
        /// Left child.
        left: Box<TreeNode>,
        /// Right child.
        right: Box<TreeNode>,
        /// Distance to children.
        height: f64,
    },
}

impl TreeNode {
    /// Creates a new leaf node.
    #[must_use]
    pub fn leaf(index: usize) -> Self {
        TreeNode::Leaf(index)
    }

    /// Creates a new internal node.
    #[must_use]
    pub fn internal(left: TreeNode, right: TreeNode, height: f64) -> Self {
        TreeNode::Internal {
            left: Box::new(left),
            right: Box::new(right),
            height,
        }
    }

    /// Returns all leaf indices in this subtree.
    #[must_use]
    pub fn leaves(&self) -> Vec<usize> {
        match self {
            TreeNode::Leaf(idx) => vec![*idx],
            TreeNode::Internal { left, right, .. } => {
                let mut leaves = left.leaves();
                leaves.extend(right.leaves());
                leaves
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_creation() {
        let domain = Domain::new("test".to_string());
        assert_eq!(domain.id, "test");
        assert!(domain.is_empty());
    }

    #[test]
    fn test_transform_identity() {
        let t = Transform::identity();
        let p = Coord3::new(1.0, 2.0, 3.0);
        let result = t.apply(&p);
        assert!((result.x - p.x).abs() < 1e-10);
        assert!((result.y - p.y).abs() < 1e-10);
        assert!((result.z - p.z).abs() < 1e-10);
    }

    #[test]
    fn test_transform_inverse() {
        let t = Transform {
            rotation: RotationMatrix::identity(),
            translation: Vec3::new(1.0, 2.0, 3.0),
        };
        let inv = t.inverse();
        let composed = t.compose(&inv);
        assert!((composed.translation.x).abs() < 1e-10);
    }

    #[test]
    fn test_parameters_default() {
        let params = Parameters::default();
        assert_eq!(params.n_passes, 2);
        assert!(params.use_secondary);
    }

    #[test]
    fn test_tree_node_leaves() {
        let tree = TreeNode::internal(
            TreeNode::leaf(0),
            TreeNode::internal(TreeNode::leaf(1), TreeNode::leaf(2), 0.5),
            1.0,
        );
        let leaves = tree.leaves();
        assert_eq!(leaves, vec![0, 1, 2]);
    }
}
