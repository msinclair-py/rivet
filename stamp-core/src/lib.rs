//! # stamp-core
//!
//! Core library for STAMP (Structural Alignment of Multiple Proteins).
//!
//! This crate provides the fundamental algorithms and data structures for
//! structural alignment of proteins, including:
//!
//! - Pairwise structural alignment using the Rossmann-Argos scoring scheme
//! - Multiple structure alignment via tree-guided progressive methods
//! - Hierarchical clustering of protein structures
//! - Database scanning capabilities
//!
//! ## Example
//!
//! ```rust,ignore
//! use stamp_core::types::Domain;
//! use stamp_core::pairwise::align_pair;
//!
//! let domain1 = Domain::from_pdb("1abc.pdb")?;
//! let domain2 = Domain::from_pdb("2def.pdb")?;
//!
//! let result = align_pair(&domain1, &domain2, &Default::default())?;
//! println!("RMSD: {:.2} A", result.rmsd);
//! ```
//!
//! ## Module Overview
//!
//! - [`types`]: Core data structures (Domain, Parameters, AlignmentResult)
//! - [`math`]: Linear algebra operations, Kabsch algorithm, transformations
//! - [`io`]: PDB/DSSP file parsing, domain file handling
//! - [`alignment`]: Smith-Waterman dynamic programming, path tracing
//! - [`scoring`]: Rossmann-Argos scoring matrices and functions
//! - [`clustering`]: Hierarchical clustering algorithms
//! - [`pairwise`]: Pairwise structural alignment
//! - [`treewise`]: Tree-guided multiple structure alignment
//! - [`secondary`]: Secondary structure handling and assignment
//! - [`scan`]: Database scanning operations

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod alignment;
pub mod clustering;
pub mod io;
pub mod math;
pub mod pairwise;
pub mod scan;
pub mod scoring;
pub mod secondary;
pub mod treewise;
pub mod types;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::alignment::SmithWaterman;
    pub use crate::pairwise::align_pair;
    pub use crate::scoring::RossmannArgos;
    pub use crate::types::{AlignmentResult, Domain, Parameters, Transform};
}

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Re-export common types at crate root for convenience.
pub use types::{Domain, Parameters, StampError, StampResult};
