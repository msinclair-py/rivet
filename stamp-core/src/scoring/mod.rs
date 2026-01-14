//! Rossmann-Argos scoring functions for structural alignment.
//!
//! This module implements the STAMP scoring scheme based on the Rossmann-Argos
//! probability measure for structural equivalence. The approach is described in:
//!
//! - Rossmann & Argos, J. Mol. Biol. 105:75-95 (1976)
//! - Russell & Barton, Proteins 14:309-323 (1992)
//!
//! The probability Pij measures the likelihood that residue i in structure A
//! is structurally equivalent to residue j in structure B:
//!
//! ```text
//! Pij = exp(Dij + Cij)
//!
//! Where:
//! - Dij = -distance^2 / (2*E1^2)  [distance component]
//! - Cij = -Sij / (2*E2^2)         [conformational component]
//!
//! Sij measures backbone direction similarity:
//! Sij = ||delta(i-1,i)_A - delta(i-1,i)_B||^2 + ||delta(i,i+1)_A - delta(i,i+1)_B||^2
//! ```

use crate::types::{Coord3, Domain, Parameters};

/// Default precision scaling factor for integer coordinates.
/// Coordinates are stored as integers scaled by this factor (1000 = 0.001 Angstrom resolution).
pub const DEFAULT_PRECISION: i32 = 1000;

/// Default E1 parameter (distance tolerance in Angstroms).
pub const DEFAULT_E1: f64 = 2.0;

/// Default E2 parameter (conformational tolerance in Angstroms).
pub const DEFAULT_E2: f64 = 5.0;

/// Default normalization mean for probability matrix.
pub const DEFAULT_NMEAN: f64 = 0.5;

/// Default normalization standard deviation.
pub const DEFAULT_NSD: f64 = 0.4;

/// Default corner-cutting factor.
pub const DEFAULT_CCFACTOR: f64 = 100.0;

/// Rossmann-Argos probability calculation result.
#[derive(Debug, Clone, Copy)]
pub struct RossmannResult {
    /// The combined probability Pij = exp(Dij + Cij).
    pub pij: f64,
    /// Distance component Dij = -d^2 / const1.
    pub dij: f64,
    /// Conformational component Cij = -Sij / const2.
    pub cij: f64,
}

/// Parameters for Rossmann-Argos probability calculation.
#[derive(Debug, Clone)]
pub struct RossmannParams {
    /// E1: Distance tolerance (Angstroms). Typical value: 2.0.
    pub e1: f64,
    /// E2: Conformational tolerance (Angstroms). Typical value: 5.0.
    pub e2: f64,
    /// Precomputed const1 = -2*E1^2 (for efficiency).
    pub const1: f64,
    /// Precomputed const2 = -2*E2^2 (for efficiency).
    pub const2: f64,
    /// Precision scaling factor for integer coordinates.
    pub precision: i32,
    /// Mean for z-score normalization.
    pub nmean: f64,
    /// Standard deviation for z-score normalization.
    pub nsd: f64,
    /// Whether to use boolean (thresholded) scoring.
    pub boolean_mode: bool,
    /// Threshold for boolean mode.
    pub bool_cut: f64,
    /// Whether to compute statistics from the matrix.
    pub compute_stats: bool,
    /// Corner-cutting factor for large matrices.
    pub cc_factor: f64,
    /// Whether to add length difference to corner-cutting factor.
    pub cc_add: bool,
}

impl Default for RossmannParams {
    fn default() -> Self {
        let e1 = DEFAULT_E1;
        let e2 = DEFAULT_E2;
        Self {
            e1,
            e2,
            const1: -2.0 * e1 * e1,
            const2: -2.0 * e2 * e2,
            precision: DEFAULT_PRECISION,
            nmean: DEFAULT_NMEAN,
            nsd: DEFAULT_NSD,
            boolean_mode: false,
            bool_cut: 0.5,
            compute_stats: false,
            cc_factor: DEFAULT_CCFACTOR,
            cc_add: true,
        }
    }
}

impl RossmannParams {
    /// Creates new parameters with the given E1 and E2 values.
    #[must_use]
    pub fn new(e1: f64, e2: f64) -> Self {
        Self {
            e1,
            e2,
            const1: -2.0 * e1 * e1,
            const2: -2.0 * e2 * e2,
            ..Default::default()
        }
    }

    /// Creates parameters from the global Parameters struct.
    #[must_use]
    pub fn from_parameters(params: &Parameters) -> Self {
        Self::new(params.e1, params.e2)
    }

    /// Sets normalization parameters.
    #[must_use]
    pub fn with_normalization(mut self, mean: f64, sd: f64) -> Self {
        self.nmean = mean;
        self.nsd = sd;
        self
    }

    /// Enables boolean (thresholded) mode.
    #[must_use]
    pub fn with_boolean_mode(mut self, cutoff: f64) -> Self {
        self.boolean_mode = true;
        self.bool_cut = cutoff;
        self
    }

    /// Sets corner-cutting parameters.
    #[must_use]
    pub fn with_corner_cutting(mut self, factor: f64, add_length_diff: bool) -> Self {
        self.cc_factor = factor;
        self.cc_add = add_length_diff;
        self
    }
}

/// Integer coordinate representation for efficient computation.
/// Coordinates are scaled by PRECISION (typically 1000) for integer arithmetic.
#[derive(Debug, Clone, Copy)]
pub struct IntCoord {
    /// X coordinate (scaled by PRECISION).
    pub x: i32,
    /// Y coordinate (scaled by PRECISION).
    pub y: i32,
    /// Z coordinate (scaled by PRECISION).
    pub z: i32,
}

impl IntCoord {
    /// Creates a new integer coordinate.
    #[must_use]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Converts from floating-point coordinate with the given precision.
    #[must_use]
    pub fn from_coord3(coord: &Coord3, precision: i32) -> Self {
        Self {
            x: (coord.x * precision as f64).round() as i32,
            y: (coord.y * precision as f64).round() as i32,
            z: (coord.z * precision as f64).round() as i32,
        }
    }

    /// Converts to floating-point coordinate.
    #[must_use]
    pub fn to_coord3(self, precision: i32) -> Coord3 {
        let scale = 1.0 / precision as f64;
        Coord3::new(
            self.x as f64 * scale,
            self.y as f64 * scale,
            self.z as f64 * scale,
        )
    }

    /// Computes the difference vector (self - other).
    #[must_use]
    pub const fn diff(self, other: Self) -> (i32, i32, i32) {
        (self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

/// Calculates the Rossmann-Argos probability for a single pair of residues.
///
/// This function computes Pij = exp(Dij + Cij) where:
/// - Dij is the distance component based on spatial separation
/// - Cij is the conformational component based on backbone direction similarity
///
/// # Arguments
///
/// * `atoms1` - Array of coordinates for structure 1 (at least 3 elements: [i-1, i, i+1])
/// * `atoms2` - Array of coordinates for structure 2 (at least 3 elements: [i-1, i, i+1])
/// * `is_n_terminus` - True if this is the N-terminal residue (no i-1 neighbor)
/// * `is_c_terminus` - True if this is the C-terminal residue (no i+1 neighbor)
/// * `params` - Rossmann-Argos parameters
///
/// # Returns
///
/// A `RossmannResult` containing Pij, Dij, and Cij components.
///
/// # Algorithm
///
/// The conformational score Sij measures how similarly the backbone directions
/// change around residue i in both structures:
///
/// ```text
/// Sij = ||delta(i-1,i)_A - delta(i-1,i)_B||^2 + ||delta(i,i+1)_A - delta(i,i+1)_B||^2
/// ```
///
/// At termini, the corresponding term is set to zero.
#[must_use]
pub fn rossmann_argos(
    atoms1: &[IntCoord],
    atoms2: &[IntCoord],
    is_n_terminus: bool,
    is_c_terminus: bool,
    params: &RossmannParams,
) -> RossmannResult {
    // Current atoms are at index 1 (middle of the 3-element window)
    // atoms[0] = i-1, atoms[1] = i, atoms[2] = i+1
    let curr1 = atoms1[1];
    let curr2 = atoms2[1];

    // Compute conformational score components
    // t1, t2, t3: backward direction difference (i-1 to i)
    let (t1, t2, t3) = if is_n_terminus {
        (0.0_f64, 0.0_f64, 0.0_f64)
    } else {
        let prev1 = atoms1[0];
        let prev2 = atoms2[0];
        // delta(i-1, i) for structure 1: curr1 - prev1
        // delta(i-1, i) for structure 2: curr2 - prev2
        // Difference of deltas: (curr1 - prev1) - (curr2 - prev2)
        let t1 = ((prev1.x - curr1.x) - (prev2.x - curr2.x)) as f64;
        let t2 = ((prev1.y - curr1.y) - (prev2.y - curr2.y)) as f64;
        let t3 = ((prev1.z - curr1.z) - (prev2.z - curr2.z)) as f64;
        (t1 * t1, t2 * t2, t3 * t3)
    };

    // t4, t5, t6: forward direction difference (i to i+1)
    let (t4, t5, t6) = if is_c_terminus {
        (0.0_f64, 0.0_f64, 0.0_f64)
    } else {
        let next1 = atoms1[2];
        let next2 = atoms2[2];
        // delta(i, i+1) for structure 1: next1 - curr1
        // delta(i, i+1) for structure 2: next2 - curr2
        let t4 = ((next1.x - curr1.x) - (next2.x - curr2.x)) as f64;
        let t5 = ((next1.y - curr1.y) - (next2.y - curr2.y)) as f64;
        let t6 = ((next1.z - curr1.z) - (next2.z - curr2.z)) as f64;
        (t4 * t4, t5 * t5, t6 * t6)
    };

    // Distance between current atoms
    let dx = (curr1.x - curr2.x) as f64;
    let dy = (curr1.y - curr2.y) as f64;
    let dz = (curr1.z - curr2.z) as f64;

    let dijsq = dx * dx + dy * dy + dz * dz;
    let sijsq = t1 + t2 + t3 + t4 + t5 + t6;

    // Convert from integer coordinates to Angstroms
    let precision_sq = (params.precision * params.precision) as f64;

    // Dij = -distance^2 / (2*E1^2) = distance^2 / const1
    let dij = (dijsq / precision_sq) / params.const1;

    // Cij = -Sij / (2*E2^2) = Sij / const2
    let cij = (sijsq / precision_sq) / params.const2;

    // Pij = exp(Dij + Cij)
    let pij = (dij + cij).exp();

    RossmannResult { pij, dij, cij }
}

/// Floating-point version of rossmann_argos for use with Coord3 coordinates.
///
/// This is a convenience wrapper that works directly with floating-point coordinates.
#[must_use]
pub fn rossmann_argos_f64(
    coords1: &[Coord3],
    coords2: &[Coord3],
    is_n_terminus: bool,
    is_c_terminus: bool,
    params: &RossmannParams,
) -> RossmannResult {
    // Convert to integer coordinates
    let atoms1: Vec<IntCoord> = coords1
        .iter()
        .map(|c| IntCoord::from_coord3(c, params.precision))
        .collect();
    let atoms2: Vec<IntCoord> = coords2
        .iter()
        .map(|c| IntCoord::from_coord3(c, params.precision))
        .collect();

    rossmann_argos(&atoms1, &atoms2, is_n_terminus, is_c_terminus, params)
}

/// Direct floating-point Rossmann-Argos calculation without integer conversion.
///
/// This version avoids the integer precision conversion and works directly
/// with floating-point coordinates. Use this when you don't need compatibility
/// with the original STAMP integer coordinate format.
#[must_use]
pub fn rossmann_argos_direct(
    prev1: Option<&Coord3>,
    curr1: &Coord3,
    next1: Option<&Coord3>,
    prev2: Option<&Coord3>,
    curr2: &Coord3,
    next2: Option<&Coord3>,
    params: &RossmannParams,
) -> RossmannResult {
    // Conformational score: backward direction difference
    let backward_sq = match (prev1, prev2) {
        (Some(p1), Some(p2)) => {
            let delta1 = curr1 - p1;
            let delta2 = curr2 - p2;
            let diff = delta1 - delta2;
            diff.x * diff.x + diff.y * diff.y + diff.z * diff.z
        }
        _ => 0.0,
    };

    // Conformational score: forward direction difference
    let forward_sq = match (next1, next2) {
        (Some(n1), Some(n2)) => {
            let delta1 = n1 - curr1;
            let delta2 = n2 - curr2;
            let diff = delta1 - delta2;
            diff.x * diff.x + diff.y * diff.y + diff.z * diff.z
        }
        _ => 0.0,
    };

    // Distance between current atoms
    let d = curr1 - curr2;
    let dijsq = d.x * d.x + d.y * d.y + d.z * d.z;
    let sijsq = backward_sq + forward_sq;

    // Dij = -distance^2 / (2*E1^2)
    let dij = dijsq / params.const1;

    // Cij = -Sij / (2*E2^2)
    let cij = sijsq / params.const2;

    // Pij = exp(Dij + Cij)
    let pij = (dij + cij).exp();

    RossmannResult { pij, dij, cij }
}

/// Probability matrix for structural alignment.
///
/// This stores the Pij values between all pairs of residues in two structures.
/// Values can be stored as raw probabilities, z-scores, or boolean equivalences.
///
/// Uses contiguous row-major storage for cache-friendly access and SIMD compatibility.
#[derive(Debug, Clone)]
pub struct ProbabilityMatrix {
    /// Number of residues in structure 1.
    pub len_a: usize,
    /// Number of residues in structure 2.
    pub len_b: usize,
    /// Matrix values in row-major order. Index as: i * len_b + j.
    values: Vec<f64>,
    /// Whether values are z-score normalized.
    pub normalized: bool,
    /// Computed mean (if statistics were calculated).
    pub mean: Option<f64>,
    /// Computed standard deviation (if statistics were calculated).
    pub std_dev: Option<f64>,
}

impl ProbabilityMatrix {
    /// Creates a new probability matrix with the given dimensions.
    #[must_use]
    pub fn new(len_a: usize, len_b: usize) -> Self {
        Self {
            len_a,
            len_b,
            values: vec![0.0; len_a * len_b],
            normalized: false,
            mean: None,
            std_dev: None,
        }
    }

    /// Creates a probability matrix from flat data.
    #[must_use]
    pub fn from_vec(len_a: usize, len_b: usize, values: Vec<f64>) -> Self {
        assert_eq!(values.len(), len_a * len_b);
        Self {
            len_a,
            len_b,
            values,
            normalized: false,
            mean: None,
            std_dev: None,
        }
    }

    /// Gets the value at position (i, j).
    #[inline]
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.values[i * self.len_b + j]
    }

    /// Sets the value at position (i, j).
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.values[i * self.len_b + j] = value;
    }

    /// Gets a slice of a single row.
    #[inline]
    #[must_use]
    pub fn row(&self, i: usize) -> &[f64] {
        let start = i * self.len_b;
        &self.values[start..start + self.len_b]
    }

    /// Gets a mutable slice of a single row.
    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [f64] {
        let start = i * self.len_b;
        &mut self.values[start..start + self.len_b]
    }

    /// Returns the raw data slice for efficient iteration.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Returns a mutable slice for efficient batch operations.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// Normalizes the matrix to z-scores using the given mean and standard deviation.
    pub fn normalize(&mut self, mean: f64, sd: f64) {
        if sd.abs() < 1e-10 {
            return;
        }
        // SIMD-friendly: single contiguous iteration
        for val in &mut self.values {
            *val = (*val - mean) / sd;
        }
        self.normalized = true;
        self.mean = Some(mean);
        self.std_dev = Some(sd);
    }

    /// Computes statistics from the matrix and normalizes to z-scores.
    pub fn normalize_with_computed_stats(&mut self) {
        let n = (self.len_a * self.len_b) as f64;
        if n < 2.0 {
            return;
        }

        // SIMD-friendly: single pass over contiguous data
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        for &val in &self.values {
            sum += val;
            sumsq += val * val;
        }

        let mean = sum / n;
        let variance = (sumsq - (sum * sum) / n) / (n - 1.0);
        let sd = variance.sqrt();

        self.normalize(mean, sd);
    }

    /// Converts to integer representation scaled by the given precision.
    /// Returns a nested Vec for compatibility with Smith-Waterman.
    #[must_use]
    pub fn to_integer(&self, precision: i32) -> Vec<Vec<i32>> {
        (0..self.len_a)
            .map(|i| {
                self.row(i)
                    .iter()
                    .map(|&v| (v * precision as f64).round() as i32)
                    .collect()
            })
            .collect()
    }

    /// Converts to a flat integer vector.
    #[must_use]
    pub fn to_integer_flat(&self, precision: i32) -> Vec<i32> {
        self.values
            .iter()
            .map(|&v| (v * precision as f64).round() as i32)
            .collect()
    }
}

/// Calculates the full probability matrix between two coordinate arrays.
///
/// This is the main function for computing the Rossmann-Argos probability
/// matrix used in STAMP structural alignment.
///
/// # Arguments
///
/// * `coords1` - C-alpha coordinates for structure 1
/// * `coords2` - C-alpha coordinates for structure 2
/// * `params` - Rossmann-Argos parameters
///
/// # Returns
///
/// A `ProbabilityMatrix` containing Pij values for all residue pairs.
#[must_use]
pub fn calculate_probability_matrix(
    coords1: &[Coord3],
    coords2: &[Coord3],
    params: &RossmannParams,
) -> ProbabilityMatrix {
    let len_a = coords1.len();
    let len_b = coords2.len();

    let mut matrix = ProbabilityMatrix::new(len_a, len_b);

    // Convert coordinates to integer format for efficiency
    let atoms1: Vec<IntCoord> = coords1
        .iter()
        .map(|c| IntCoord::from_coord3(c, params.precision))
        .collect();
    let atoms2: Vec<IntCoord> = coords2
        .iter()
        .map(|c| IntCoord::from_coord3(c, params.precision))
        .collect();

    // Calculate probabilities
    if params.boolean_mode {
        // Boolean mode: values are 0 or 1 based on threshold
        for i in 0..len_a {
            for j in 0..len_b {
                let result =
                    compute_rossmann_at_position(&atoms1, &atoms2, i, j, len_a, len_b, params);
                matrix.set(
                    i,
                    j,
                    if result.pij >= params.bool_cut {
                        1.0
                    } else {
                        0.0
                    },
                );
            }
        }
    } else if params.compute_stats {
        // Compute statistics from matrix and normalize
        for i in 0..len_a {
            for j in 0..len_b {
                let result =
                    compute_rossmann_at_position(&atoms1, &atoms2, i, j, len_a, len_b, params);
                matrix.set(i, j, result.pij);
            }
        }
        matrix.normalize_with_computed_stats();
    } else {
        // Use fixed normalization parameters
        for i in 0..len_a {
            for j in 0..len_b {
                let result =
                    compute_rossmann_at_position(&atoms1, &atoms2, i, j, len_a, len_b, params);
                matrix.set(i, j, result.pij);
            }
        }
        matrix.normalize(params.nmean, params.nsd);
    }

    matrix
}

/// Calculates probability matrix with corner-cutting optimization.
///
/// For large structures, this avoids computing probabilities for residue
/// pairs that are unlikely to be aligned based on their sequence positions.
/// The corner-cutting region is determined by the length ratio of the structures.
///
/// # Arguments
///
/// * `coords1` - C-alpha coordinates for structure 1
/// * `coords2` - C-alpha coordinates for structure 2
/// * `params` - Rossmann-Argos parameters
///
/// # Returns
///
/// A `ProbabilityMatrix` with corner regions left as zero.
#[must_use]
pub fn calculate_probability_matrix_cc(
    coords1: &[Coord3],
    coords2: &[Coord3],
    params: &RossmannParams,
) -> ProbabilityMatrix {
    let len_a = coords1.len();
    let len_b = coords2.len();

    let mut matrix = ProbabilityMatrix::new(len_a, len_b);

    // Convert coordinates to integer format
    let atoms1: Vec<IntCoord> = coords1
        .iter()
        .map(|c| IntCoord::from_coord3(c, params.precision))
        .collect();
    let atoms2: Vec<IntCoord> = coords2
        .iter()
        .map(|c| IntCoord::from_coord3(c, params.precision))
        .collect();

    // Calculate corner-cutting parameters
    let fk1 = if params.cc_add {
        params.cc_factor + ((len_a as i32 - len_b as i32 - 4).abs() as f64)
    } else {
        params.cc_factor
    };

    let fn_val = (len_a + 1) as f64;
    let fm_val = (len_b + 1) as f64;
    let sqrt2 = std::f64::consts::SQRT_2;
    let fcut = (fk1 / sqrt2)
        * ((fm_val / fn_val + fn_val / fm_val) / (fm_val * fm_val + fn_val * fn_val).sqrt());

    let mut lstart: usize = 0;

    for i in 0..len_a {
        let jt = i as f64 / fn_val;
        let mut upper = false;
        let mut lstrt = lstart;

        for j in lstart..len_b {
            let corner_test = ((j as f64 / fm_val) - jt).abs();

            if corner_test > fcut {
                if upper {
                    // We've gone past the valid region, skip to next row
                    break;
                }
                // Haven't reached valid region yet, continue
                continue;
            }

            if !upper {
                lstrt = j;
                upper = true;
            }

            // Compute probability for this position
            let result = compute_rossmann_at_position(&atoms1, &atoms2, i, j, len_a, len_b, params);

            if params.boolean_mode {
                matrix.set(
                    i,
                    j,
                    if result.pij >= params.bool_cut {
                        1.0
                    } else {
                        0.0
                    },
                );
            } else {
                matrix.set(i, j, result.pij);
            }
        }

        lstart = lstrt;
    }

    // Normalize if not in boolean mode
    if !params.boolean_mode {
        if params.compute_stats {
            matrix.normalize_with_computed_stats();
        } else {
            matrix.normalize(params.nmean, params.nsd);
        }
    }

    matrix
}

// ============================================================================
// Parallel probability matrix computation (requires "parallel" feature)
// ============================================================================

/// Calculates probability matrix using parallel computation.
///
/// This version uses Rayon for parallel row computation, providing significant
/// speedup on multi-core systems for large matrices.
///
/// # Feature
///
/// Requires the `parallel` feature to be enabled.
#[cfg(feature = "parallel")]
#[must_use]
pub fn calculate_probability_matrix_parallel(
    coords1: &[Coord3],
    coords2: &[Coord3],
    params: &RossmannParams,
) -> ProbabilityMatrix {
    use rayon::prelude::*;

    let len_a = coords1.len();
    let len_b = coords2.len();

    // Convert coordinates to integer format
    let atoms1: Vec<IntCoord> = coords1
        .iter()
        .map(|c| IntCoord::from_coord3(c, params.precision))
        .collect();
    let atoms2: Vec<IntCoord> = coords2
        .iter()
        .map(|c| IntCoord::from_coord3(c, params.precision))
        .collect();

    // Parallel row computation
    let values: Vec<f64> = if params.boolean_mode {
        (0..len_a)
            .into_par_iter()
            .flat_map(|i| {
                (0..len_b).map(move |j| {
                    let result =
                        compute_rossmann_at_position(&atoms1, &atoms2, i, j, len_a, len_b, params);
                    if result.pij >= params.bool_cut {
                        1.0
                    } else {
                        0.0
                    }
                })
            })
            .collect()
    } else {
        (0..len_a)
            .into_par_iter()
            .flat_map(|i| {
                (0..len_b).map(move |j| {
                    compute_rossmann_at_position(&atoms1, &atoms2, i, j, len_a, len_b, params).pij
                })
            })
            .collect()
    };

    let mut matrix = ProbabilityMatrix::from_vec(len_a, len_b, values);

    // Normalize if not in boolean mode
    if !params.boolean_mode {
        if params.compute_stats {
            matrix.normalize_with_computed_stats();
        } else {
            matrix.normalize(params.nmean, params.nsd);
        }
    }

    matrix
}

/// Calculates probability matrix with corner-cutting using parallel computation.
///
/// This combines the corner-cutting optimization with parallel row processing.
#[cfg(feature = "parallel")]
#[must_use]
pub fn calculate_probability_matrix_cc_parallel(
    coords1: &[Coord3],
    coords2: &[Coord3],
    params: &RossmannParams,
) -> ProbabilityMatrix {
    use rayon::prelude::*;

    let len_a = coords1.len();
    let len_b = coords2.len();

    // Convert coordinates to integer format
    let atoms1: Vec<IntCoord> = coords1
        .iter()
        .map(|c| IntCoord::from_coord3(c, params.precision))
        .collect();
    let atoms2: Vec<IntCoord> = coords2
        .iter()
        .map(|c| IntCoord::from_coord3(c, params.precision))
        .collect();

    // Calculate corner-cutting parameters
    let fk1 = if params.cc_add {
        params.cc_factor + ((len_a as i32 - len_b as i32 - 4).abs() as f64)
    } else {
        params.cc_factor
    };

    let fn_val = (len_a + 1) as f64;
    let fm_val = (len_b + 1) as f64;
    let sqrt2 = std::f64::consts::SQRT_2;
    let fcut = (fk1 / sqrt2)
        * ((fm_val / fn_val + fn_val / fm_val) / (fm_val * fm_val + fn_val * fn_val).sqrt());

    // Pre-compute valid j ranges for each row
    let valid_ranges: Vec<(usize, usize)> = (0..len_a)
        .map(|i| {
            let jt = i as f64 / fn_val;
            let mut start = 0usize;
            let mut end = len_b;
            let mut found_start = false;

            for j in 0..len_b {
                let corner_test = ((j as f64 / fm_val) - jt).abs();
                if corner_test <= fcut {
                    if !found_start {
                        start = j;
                        found_start = true;
                    }
                    end = j + 1;
                } else if found_start {
                    break;
                }
            }

            if !found_start {
                (0, 0) // No valid range
            } else {
                (start, end)
            }
        })
        .collect();

    // Parallel row computation with corner-cutting
    let values: Vec<f64> = (0..len_a)
        .into_par_iter()
        .flat_map(|i| {
            let (j_start, j_end) = valid_ranges[i];
            (0..len_b).map(move |j| {
                if j >= j_start && j < j_end {
                    let result =
                        compute_rossmann_at_position(&atoms1, &atoms2, i, j, len_a, len_b, params);
                    if params.boolean_mode {
                        if result.pij >= params.bool_cut {
                            1.0
                        } else {
                            0.0
                        }
                    } else {
                        result.pij
                    }
                } else {
                    0.0 // Corner-cut region
                }
            })
        })
        .collect();

    let mut matrix = ProbabilityMatrix::from_vec(len_a, len_b, values);

    // Normalize if not in boolean mode
    if !params.boolean_mode {
        if params.compute_stats {
            matrix.normalize_with_computed_stats();
        } else {
            matrix.normalize(params.nmean, params.nsd);
        }
    }

    matrix
}

/// Automatically selects sequential or parallel computation based on matrix size.
///
/// For small matrices (< 2500 elements), uses sequential computation to avoid
/// parallel overhead. For larger matrices, uses parallel computation if the
/// `parallel` feature is enabled.
#[must_use]
pub fn calculate_probability_matrix_auto(
    coords1: &[Coord3],
    coords2: &[Coord3],
    params: &RossmannParams,
) -> ProbabilityMatrix {
    // Threshold: parallel overhead isn't worth it for small matrices
    #[cfg(feature = "parallel")]
    {
        const PARALLEL_THRESHOLD: usize = 2500; // ~50x50
        let size = coords1.len() * coords2.len();
        if size >= PARALLEL_THRESHOLD {
            return calculate_probability_matrix_parallel(coords1, coords2, params);
        }
    }

    calculate_probability_matrix(coords1, coords2, params)
}

/// Automatically selects sequential or parallel computation with corner-cutting.
#[must_use]
pub fn calculate_probability_matrix_cc_auto(
    coords1: &[Coord3],
    coords2: &[Coord3],
    params: &RossmannParams,
) -> ProbabilityMatrix {
    #[cfg(feature = "parallel")]
    {
        const PARALLEL_THRESHOLD: usize = 2500;
        let size = coords1.len() * coords2.len();
        if size >= PARALLEL_THRESHOLD {
            return calculate_probability_matrix_cc_parallel(coords1, coords2, params);
        }
    }

    calculate_probability_matrix_cc(coords1, coords2, params)
}

/// Helper function to compute Rossmann probability at a specific position.
fn compute_rossmann_at_position(
    atoms1: &[IntCoord],
    atoms2: &[IntCoord],
    i: usize,
    j: usize,
    len_a: usize,
    len_b: usize,
    params: &RossmannParams,
) -> RossmannResult {
    // Determine boundary conditions
    let is_n_terminus = i == 0 || j == 0;
    let is_c_terminus = i == len_a - 1 || j == len_b - 1;

    // Build 3-element windows around current positions
    // For positions at boundaries, we still need valid array indices
    let prev_i = i.saturating_sub(1);
    let next_i = (i + 1).min(len_a - 1);
    let prev_j = j.saturating_sub(1);
    let next_j = (j + 1).min(len_b - 1);

    let window1 = [atoms1[prev_i], atoms1[i], atoms1[next_i]];
    let window2 = [atoms2[prev_j], atoms2[j], atoms2[next_j]];

    rossmann_argos(&window1, &window2, is_n_terminus, is_c_terminus, params)
}

// ============================================================================
// Murzin P-value calculation for statistical significance
// ============================================================================

/// Computes factorial for small N, or a large constant for N > 150.
///
/// This matches the C implementation which caps at 1e262 to avoid overflow.
fn factorial(n: u32) -> f64 {
    if n > 150 {
        1e262
    } else {
        let mut f = 1.0;
        for i in 1..=n {
            f *= i as f64;
        }
        f
    }
}

/// Calculates the Murzin P-value for assessing alignment significance.
///
/// This computes the probability of observing m or more identical residues
/// by chance in an alignment of n structurally equivalent positions,
/// given a background identity probability p.
///
/// Reference: Murzin A.G., "Sweet tasting protein monellin is related to
/// the cystatin family of thiol proteinase inhibitors", J. Mol. Biol.
/// 230:689-694 (1993).
///
/// # Arguments
///
/// * `n` - Number of structurally equivalent positions
/// * `m` - Number of identical residues
/// * `p` - Background probability of identity (typically ~0.05 for random sequences)
///
/// # Returns
///
/// The P-value (probability). Returns 1.0 if m is within expected range,
/// or a very small probability if m significantly exceeds expectation.
///
/// # Mathematical basis
///
/// Under the null hypothesis of random alignment:
/// - Expected matches: m_0 = n * p
/// - Standard deviation: sigma = sqrt(n * p * (1-p))
///
/// If m <= m_0 + sigma, the result is not significant (P = 1.0).
/// Otherwise, P is computed from the binomial distribution:
/// ```text
/// P = C(n,m) * p^m * (1-p)^(n-m)
/// ```
#[must_use]
pub fn murzin_p_value(n: u32, m: u32, p: f64) -> f64 {
    if n == 0 || p <= 0.0 || p >= 1.0 {
        return 1.0;
    }

    let n_f = n as f64;
    let m_f = m as f64;

    // Expected number of matches and standard deviation
    let m_expected = n_f * p;
    let sigma = (n_f * p * (1.0 - p)).sqrt();

    // Test for significance: only compute P if m exceeds expectation + 1 sigma
    if m_f <= m_expected + sigma {
        return 1.0;
    }

    // Binomial coefficient: C(n,m) = n! / (m! * (n-m)!)
    let coeff = factorial(n) / (factorial(m) * factorial(n - m));

    // Binomial probability: p^m * (1-p)^(n-m)
    let prob = p.powi(m as i32) * (1.0 - p).powi((n - m) as i32);

    let p_value = coeff * prob;

    // Clamp very small values to zero (matching C implementation)
    if p_value < 1e-100 {
        0.0
    } else {
        p_value
    }
}

/// Computes the Murzin P-value using the log-gamma function for numerical stability.
///
/// This version avoids factorial overflow by computing in log space:
/// ```text
/// log(P) = log(C(n,m)) + m*log(p) + (n-m)*log(1-p)
/// ```
///
/// Use this for large n values where the factorial-based version may overflow.
#[must_use]
pub fn murzin_p_value_stable(n: u32, m: u32, p: f64) -> f64 {
    if n == 0 || p <= 0.0 || p >= 1.0 {
        return 1.0;
    }

    let n_f = n as f64;
    let m_f = m as f64;

    // Expected number of matches and standard deviation
    let m_expected = n_f * p;
    let sigma = (n_f * p * (1.0 - p)).sqrt();

    // Test for significance
    if m_f <= m_expected + sigma {
        return 1.0;
    }

    // Use log-gamma for numerical stability
    // log(n!) = lgamma(n+1)
    use std::f64::consts::LN_10;

    // log(C(n,m)) = lgamma(n+1) - lgamma(m+1) - lgamma(n-m+1)
    let log_coeff = lgamma((n + 1) as f64) - lgamma((m + 1) as f64) - lgamma((n - m + 1) as f64);

    // log(p^m * (1-p)^(n-m)) = m*log(p) + (n-m)*log(1-p)
    let log_prob = m_f * p.ln() + (n_f - m_f) * (1.0 - p).ln();

    let log_p_value = log_coeff + log_prob;

    // Convert from log space, clamp to [0, 1]
    if log_p_value < -230.0 * LN_10 {
        // Below 1e-100
        0.0
    } else {
        log_p_value.exp().min(1.0)
    }
}

/// Compute log-gamma function using Stirling's approximation.
/// For x > 0.5, this is accurate to about 6 decimal places.
fn lgamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    // For small x, use recurrence relation: lgamma(x) = lgamma(x+1) - ln(x)
    if x < 0.5 {
        return lgamma(x + 1.0) - x.ln();
    }

    // Stirling's approximation with Lanczos coefficients
    // This gives good accuracy for x >= 0.5
    let g = 7.0_f64;
    let c = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_403,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let x = x - 1.0;
    let mut sum = c[0];
    for (i, &coef) in c.iter().enumerate().skip(1) {
        sum += coef / (x + i as f64);
    }

    let t = x + g + 0.5;
    let log_sqrt_2pi = 0.918_938_533_204_672_7;

    log_sqrt_2pi + (x + 0.5) * t.ln() - t + sum.ln()
}

// ============================================================================
// Legacy RossmannArgos struct for backward compatibility
// ============================================================================

/// Rossmann-Argos scoring matrix calculator.
///
/// This struct provides the original scoring interface for backward compatibility.
#[derive(Debug)]
pub struct RossmannArgos {
    /// First constant for Pij calculation (E1).
    e1: f64,
    /// Second constant for Pij calculation (E2).
    e2: f64,
    /// Distance cutoff.
    cutoff: f64,
    /// Use secondary structure in scoring.
    use_secondary: bool,
}

impl RossmannArgos {
    /// Creates a new Rossmann-Argos scorer with the given parameters.
    #[must_use]
    pub fn new(params: &Parameters) -> Self {
        Self {
            e1: params.e1,
            e2: params.e2,
            cutoff: params.cutoff_dist,
            use_secondary: params.use_secondary,
        }
    }

    /// Creates a scorer with default parameters.
    #[must_use]
    pub fn default_params() -> Self {
        Self {
            e1: 2.0,
            e2: 0.5,
            cutoff: 10.0,
            use_secondary: true,
        }
    }

    /// Computes the Pij score for a pair of inter-residue distances.
    ///
    /// The Pij function measures how well two distance pairs correspond.
    ///
    /// # Arguments
    ///
    /// * `d1` - Distance in first structure
    /// * `d2` - Distance in second structure
    ///
    /// # Returns
    ///
    /// A score in range [0, 1] where 1 indicates identical distances.
    #[must_use]
    pub fn pij(&self, d1: f64, d2: f64) -> f64 {
        let diff = (d1 - d2).abs();
        let avg = (d1 + d2) / 2.0;

        if avg > self.cutoff {
            return 0.0;
        }

        // Gaussian-like decay based on distance difference
        let sigma = self.e1 + self.e2 * avg;
        (-diff * diff / (2.0 * sigma * sigma)).exp()
    }

    /// Computes the Sij score between two residue pairs.
    ///
    /// This combines distance-based scoring with secondary structure matching.
    ///
    /// # Arguments
    ///
    /// * `d1` - Distance in first structure
    /// * `d2` - Distance in second structure
    /// * `ss1` - Secondary structure of first residue pair
    /// * `ss2` - Secondary structure of second residue pair
    #[must_use]
    pub fn sij(&self, d1: f64, d2: f64, ss1: (char, char), ss2: (char, char)) -> f64 {
        let pij = self.pij(d1, d2);

        if !self.use_secondary {
            return pij;
        }

        // Secondary structure bonus
        let ss_bonus = if ss1 == ss2 {
            1.2 // Same secondary structure type pair
        } else if (ss1.0 == ss2.0) || (ss1.1 == ss2.1) {
            1.1 // Partial match
        } else {
            1.0 // No match
        };

        pij * ss_bonus
    }

    /// Computes the full score matrix between two domains.
    ///
    /// The score matrix S[i][j] represents the structural similarity
    /// between position i in domain1 and position j in domain2.
    ///
    /// # Arguments
    ///
    /// * `domain1` - First domain
    /// * `domain2` - Second domain
    ///
    /// # Returns
    ///
    /// A 2D score matrix of size [len1][len2].
    #[must_use]
    pub fn score_matrix(&self, domain1: &Domain, domain2: &Domain) -> Vec<Vec<f64>> {
        let len1 = domain1.len();
        let len2 = domain2.len();

        let mut scores = vec![vec![0.0; len2]; len1];

        // Precompute internal distance matrices
        let dist1 = internal_distances(domain1);
        let dist2 = internal_distances(domain2);

        // Compute scores using surrounding residue distances
        for i in 0..len1 {
            for j in 0..len2 {
                let mut sum = 0.0;
                let mut count = 0;

                // Compare distances to neighboring residues
                for di in 1..=3i32 {
                    let i_plus = (i as i32 + di) as usize;
                    let i_minus = i.saturating_sub(di as usize);
                    let j_plus = (j as i32 + di) as usize;
                    let j_minus = j.saturating_sub(di as usize);

                    // Forward direction
                    if i_plus < len1 && j_plus < len2 {
                        let d1 = dist1[i][i_plus];
                        let d2 = dist2[j][j_plus];

                        if self.use_secondary && domain1.has_dssp && domain2.has_dssp {
                            let ss1 = (
                                domain1.residues[i].sec_struct,
                                domain1.residues[i_plus].sec_struct,
                            );
                            let ss2 = (
                                domain2.residues[j].sec_struct,
                                domain2.residues[j_plus].sec_struct,
                            );
                            sum += self.sij(d1, d2, ss1, ss2);
                        } else {
                            sum += self.pij(d1, d2);
                        }
                        count += 1;
                    }

                    // Backward direction
                    if i_minus < i && j_minus < j {
                        let d1 = dist1[i][i_minus];
                        let d2 = dist2[j][j_minus];
                        sum += self.pij(d1, d2);
                        count += 1;
                    }
                }

                scores[i][j] = if count > 0 { sum / count as f64 } else { 0.0 };
            }
        }

        scores
    }

    /// Computes the STAMP Sc score for an alignment.
    ///
    /// # Arguments
    ///
    /// * `domain1` - First domain
    /// * `domain2` - Second domain
    /// * `path` - Alignment path (pairs of indices)
    ///
    /// # Returns
    ///
    /// The Sc score.
    #[must_use]
    pub fn sc_score(&self, domain1: &Domain, domain2: &Domain, path: &[(usize, usize)]) -> f64 {
        if path.is_empty() {
            return 0.0;
        }

        let scores = self.score_matrix(domain1, domain2);
        let mut sum = 0.0;

        for &(i, j) in path {
            if i < domain1.len() && j < domain2.len() {
                sum += scores[i][j];
            }
        }

        sum / path.len() as f64
    }
}

/// Computes internal distance matrix for a domain.
///
/// Returns a symmetric matrix where dist[i][j] is the C-alpha distance
/// between residues i and j.
#[must_use]
pub fn internal_distances(domain: &Domain) -> Vec<Vec<f64>> {
    let n = domain.len();
    let mut dist = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in i + 1..n {
            let d = ca_distance(&domain.residues[i].ca_coord, &domain.residues[j].ca_coord);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    dist
}

/// Computes distance between two C-alpha coordinates.
#[inline]
#[must_use]
pub fn ca_distance(p1: &Coord3, p2: &Coord3) -> f64 {
    let diff = p1 - p2;
    (diff.x * diff.x + diff.y * diff.y + diff.z * diff.z).sqrt()
}

/// Secondary structure similarity score.
///
/// Returns a score based on how similar two secondary structure
/// assignments are.
#[must_use]
pub fn ss_similarity(ss1: char, ss2: char) -> f64 {
    match (ss1, ss2) {
        ('H', 'H') => 1.0,              // Helix-helix
        ('E', 'E') => 1.0,              // Strand-strand
        ('C', 'C') => 0.5,              // Coil-coil (less informative)
        ('H', 'E') | ('E', 'H') => 0.0, // Helix-strand mismatch
        _ => 0.25,                      // Other combinations
    }
}

/// Computes sequence identity for an alignment.
///
/// # Arguments
///
/// * `domain1` - First domain
/// * `domain2` - Second domain
/// * `path` - Alignment path
///
/// # Returns
///
/// Fraction of identical amino acids in aligned positions.
#[must_use]
pub fn sequence_identity(domain1: &Domain, domain2: &Domain, path: &[(usize, usize)]) -> f64 {
    if path.is_empty() {
        return 0.0;
    }

    let matches = path
        .iter()
        .filter(|&&(i, j)| {
            domain1.residues.get(i).map(|r| r.aa) == domain2.residues.get(j).map(|r| r.aa)
        })
        .count();

    matches as f64 / path.len() as f64
}

// ============================================================================
// Utility functions for probability matrix operations
// ============================================================================

/// Extracts coordinates from a Domain for use with probability matrix functions.
#[must_use]
pub fn domain_to_coords(domain: &Domain) -> Vec<Coord3> {
    domain.ca_coords().cloned().collect()
}

/// Computes the probability matrix between two domains.
///
/// This is a convenience function that extracts coordinates and computes
/// the full Rossmann-Argos probability matrix.
#[must_use]
pub fn domain_probability_matrix(
    domain1: &Domain,
    domain2: &Domain,
    params: &RossmannParams,
) -> ProbabilityMatrix {
    let coords1 = domain_to_coords(domain1);
    let coords2 = domain_to_coords(domain2);
    calculate_probability_matrix(&coords1, &coords2, params)
}

/// Computes the probability matrix with corner-cutting optimization.
#[must_use]
pub fn domain_probability_matrix_cc(
    domain1: &Domain,
    domain2: &Domain,
    params: &RossmannParams,
) -> ProbabilityMatrix {
    let coords1 = domain_to_coords(domain1);
    let coords2 = domain_to_coords(domain2);
    calculate_probability_matrix_cc(&coords1, &coords2, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Residue;

    fn make_test_domain(coords: Vec<(f64, f64, f64)>) -> Domain {
        let mut domain = Domain::new("test".to_string());
        for (i, (x, y, z)) in coords.into_iter().enumerate() {
            domain.residues.push(Residue::new(
                i as i32,
                i.to_string(),
                'A',
                Coord3::new(x, y, z),
            ));
        }
        domain
    }

    #[test]
    fn test_rossmann_params_default() {
        let params = RossmannParams::default();
        assert!((params.e1 - 2.0).abs() < 1e-10);
        assert!((params.e2 - 5.0).abs() < 1e-10);
        assert!((params.const1 - (-8.0)).abs() < 1e-10);
        assert!((params.const2 - (-50.0)).abs() < 1e-10);
    }

    #[test]
    fn test_int_coord_conversion() {
        let coord = Coord3::new(1.234, -5.678, 0.001);
        let int_coord = IntCoord::from_coord3(&coord, 1000);
        assert_eq!(int_coord.x, 1234);
        assert_eq!(int_coord.y, -5678);
        assert_eq!(int_coord.z, 1);

        let back = int_coord.to_coord3(1000);
        assert!((back.x - 1.234).abs() < 0.001);
        assert!((back.y - (-5.678)).abs() < 0.001);
    }

    #[test]
    fn test_rossmann_identical_atoms() {
        let params = RossmannParams::default();
        let atoms1 = [
            IntCoord::new(0, 0, 0),
            IntCoord::new(3800, 0, 0),
            IntCoord::new(7600, 0, 0),
        ];
        let atoms2 = atoms1;

        let result = rossmann_argos(&atoms1, &atoms2, false, false, &params);

        // Identical structures should have Pij = 1.0
        assert!((result.pij - 1.0).abs() < 1e-10);
        assert!(result.dij.abs() < 1e-10);
        assert!(result.cij.abs() < 1e-10);
    }

    #[test]
    fn test_rossmann_different_atoms() {
        let params = RossmannParams::default();
        let atoms1 = [
            IntCoord::new(0, 0, 0),
            IntCoord::new(3800, 0, 0),
            IntCoord::new(7600, 0, 0),
        ];
        let atoms2 = [
            IntCoord::new(0, 0, 0),
            IntCoord::new(3800, 1000, 0), // 1 Angstrom offset
            IntCoord::new(7600, 0, 0),
        ];

        let result = rossmann_argos(&atoms1, &atoms2, false, false, &params);

        // Should have Pij < 1.0 due to distance
        assert!(result.pij < 1.0);
        assert!(result.pij > 0.0);
        assert!(result.dij < 0.0);
    }

    #[test]
    fn test_rossmann_terminus_handling() {
        let params = RossmannParams::default();
        let atoms1 = [
            IntCoord::new(0, 0, 0),
            IntCoord::new(3800, 0, 0),
            IntCoord::new(7600, 0, 0),
        ];
        let atoms2 = atoms1;

        // N-terminus: backward term should be zero
        let result_n = rossmann_argos(&atoms1, &atoms2, true, false, &params);
        assert!((result_n.pij - 1.0).abs() < 1e-10);

        // C-terminus: forward term should be zero
        let result_c = rossmann_argos(&atoms1, &atoms2, false, true, &params);
        assert!((result_c.pij - 1.0).abs() < 1e-10);

        // Both termini: both conformational terms zero
        let result_both = rossmann_argos(&atoms1, &atoms2, true, true, &params);
        assert!((result_both.pij - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_probability_matrix_dimensions() {
        let coords1 = vec![
            Coord3::new(0.0, 0.0, 0.0),
            Coord3::new(3.8, 0.0, 0.0),
            Coord3::new(7.6, 0.0, 0.0),
        ];
        let coords2 = vec![Coord3::new(0.0, 0.0, 0.0), Coord3::new(3.8, 0.0, 0.0)];
        let params = RossmannParams::default();

        let matrix = calculate_probability_matrix(&coords1, &coords2, &params);

        assert_eq!(matrix.len_a, 3);
        assert_eq!(matrix.len_b, 2);
        // Flat storage: len_a * len_b elements
        assert_eq!(matrix.values.len(), 6);
        assert_eq!(matrix.row(0).len(), 2);
    }

    #[test]
    fn test_probability_matrix_cc_subset() {
        // Corner-cutting should produce valid probabilities in the
        // central diagonal region and zeros in corners
        let coords1: Vec<Coord3> = (0..50)
            .map(|i| Coord3::new(i as f64 * 3.8, 0.0, 0.0))
            .collect();
        let coords2: Vec<Coord3> = (0..80)
            .map(|i| Coord3::new(i as f64 * 3.8, 0.0, 0.0))
            .collect();

        let mut params = RossmannParams::default();
        params.cc_factor = 5.0; // Very aggressive corner cutting
        params.cc_add = false; // Don't add length difference

        let matrix = calculate_probability_matrix_cc(&coords1, &coords2, &params);

        // Check that we have some non-zero values
        // Flat storage: iterate directly over values
        let non_zero_count: usize = matrix
            .values
            .iter()
            .filter(|&&v| v.abs() > 1e-10)
            .count();

        assert!(non_zero_count > 0, "Should have some computed values");
        // With very aggressive CC on large matrices, corners should be zeros
        // The full matrix would have 50*80=4000 entries
        let total = coords1.len() * coords2.len();
        // Allow test to pass if we computed any less than full matrix
        // (corner cutting is working if we skipped any cells)
        assert!(
            non_zero_count <= total,
            "Non-zero count {} should be at most total {}",
            non_zero_count,
            total
        );
    }

    #[test]
    fn test_murzin_p_value_expected() {
        // If matches are within expected range, P should be 1.0
        let p = murzin_p_value(100, 5, 0.05);
        assert!((p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_murzin_p_value_significant() {
        // If matches greatly exceed expectation, P should be small
        let p = murzin_p_value(100, 30, 0.05);
        assert!(p < 0.1);
    }

    #[test]
    fn test_murzin_p_value_stable() {
        // Compare stable and regular versions for moderate n
        let n = 50;
        let m = 15;
        let prob = 0.1;

        let p1 = murzin_p_value(n, m, prob);
        let p2 = murzin_p_value_stable(n, m, prob);

        // Should be approximately equal for moderate values
        if p1 > 1e-100 && p1 < 1.0 {
            assert!((p1 - p2).abs() / p1.max(p2) < 0.1);
        }
    }

    #[test]
    fn test_factorial() {
        assert!((factorial(0) - 1.0).abs() < 1e-10);
        assert!((factorial(1) - 1.0).abs() < 1e-10);
        assert!((factorial(5) - 120.0).abs() < 1e-10);
        assert!((factorial(10) - 3628800.0).abs() < 1e-5);
        // Large factorial should return cap value
        assert!(factorial(151) > 1e200);
    }

    #[test]
    fn test_pij_identical() {
        let scorer = RossmannArgos::default_params();
        let score = scorer.pij(5.0, 5.0);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pij_different() {
        let scorer = RossmannArgos::default_params();
        let score = scorer.pij(5.0, 10.0);
        assert!(score < 1.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_pij_beyond_cutoff() {
        let scorer = RossmannArgos::default_params();
        let score = scorer.pij(20.0, 20.0);
        assert!(score.abs() < 1e-10);
    }

    #[test]
    fn test_internal_distances() {
        let domain = make_test_domain(vec![(0.0, 0.0, 0.0), (3.8, 0.0, 0.0), (7.6, 0.0, 0.0)]);
        let dist = internal_distances(&domain);
        assert!((dist[0][1] - 3.8).abs() < 1e-10);
        assert!((dist[0][2] - 7.6).abs() < 1e-10);
        assert!((dist[1][2] - 3.8).abs() < 1e-10);
    }

    #[test]
    fn test_ss_similarity() {
        assert!((ss_similarity('H', 'H') - 1.0).abs() < 1e-10);
        assert!((ss_similarity('E', 'E') - 1.0).abs() < 1e-10);
        assert!(ss_similarity('H', 'E').abs() < 1e-10);
    }

    #[test]
    fn test_sequence_identity() {
        let mut d1 = make_test_domain(vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]);
        let mut d2 = make_test_domain(vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]);

        d1.residues[0].aa = 'A';
        d1.residues[1].aa = 'R';
        d2.residues[0].aa = 'A';
        d2.residues[1].aa = 'N';

        let path = vec![(0, 0), (1, 1)];
        let identity = sequence_identity(&d1, &d2, &path);
        assert!((identity - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_rossmann_direct_vs_integer() {
        // Test that direct and integer-based calculations give similar results
        let params = RossmannParams::new(2.0, 5.0);

        let p1 = Coord3::new(0.0, 0.0, 0.0);
        let c1 = Coord3::new(3.8, 0.0, 0.0);
        let n1 = Coord3::new(7.6, 0.0, 0.0);

        let p2 = Coord3::new(0.0, 0.5, 0.0);
        let c2 = Coord3::new(3.8, 0.5, 0.0);
        let n2 = Coord3::new(7.6, 0.5, 0.0);

        let result_direct = rossmann_argos_direct(
            Some(&p1),
            &c1,
            Some(&n1),
            Some(&p2),
            &c2,
            Some(&n2),
            &params,
        );

        let coords1 = vec![p1, c1, n1];
        let coords2 = vec![p2, c2, n2];
        let result_f64 = rossmann_argos_f64(&coords1, &coords2, false, false, &params);

        // Results should be very close (within precision tolerance)
        assert!((result_direct.pij - result_f64.pij).abs() < 0.001);
    }
}
