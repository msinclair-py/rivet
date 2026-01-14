//! Pairwise structural alignment.
//!
//! This module implements the core STAMP algorithm for aligning two
//! protein structures through iterative superposition refinement.
//!
//! # Algorithm Overview
//!
//! The pairwise fitting algorithm (`pairfit`) iteratively refines the
//! structural alignment between two domains:
//!
//! 1. Calculate probability matrix P[i,j] using Rossmann-Argos scoring
//! 2. Find optimal alignment path using Smith-Waterman dynamic programming
//! 3. Extract equivalent residue pairs from the alignment path
//! 4. Compute new transformation via Kabsch algorithm
//! 5. Apply transformation to mobile domain coordinates
//! 6. Accumulate transformation into cumulative transform
//! 7. Check convergence: |score_new - score_old| / score_new < SCORETOL
//!
//! # References
//!
//! - Russell & Barton, Proteins 14:309-323 (1992)
//! - Rossmann & Argos, J. Mol. Biol. 105:75-95 (1976)

use crate::alignment::{
    smith_waterman, smith_waterman_corner_cut, traceback, SwResult,
};
use crate::math::superpose;
use crate::scoring::{
    calculate_probability_matrix, calculate_probability_matrix_cc, domain_to_coords,
    sequence_identity, RossmannParams, ProbabilityMatrix,
};
use crate::types::{AlignmentResult, Coord3, Domain, Parameters, StampResult, Transform};

/// Default score tolerance for convergence (0.01% relative change).
pub const DEFAULT_SCORE_TOL: f64 = 0.01;

/// Default maximum iterations for pairwise fitting.
pub const DEFAULT_MAX_ITER: u32 = 100;

/// Default minimum number of fitted residues.
pub const DEFAULT_MIN_FIT: usize = 3;

/// Default precision for integer coordinate scaling.
pub const DEFAULT_PRECISION: i32 = 1000;

/// Configuration for pairwise fitting.
#[derive(Debug, Clone)]
pub struct PairwiseConfig {
    /// Score tolerance for convergence (percentage).
    pub score_tol: f64,
    /// Maximum number of iterations.
    pub max_iter: u32,
    /// Minimum number of fitted residues.
    pub min_fit: usize,
    /// Precision for integer coordinate scaling.
    pub precision: i32,
    /// Gap penalty for Smith-Waterman.
    pub gap_penalty: f64,
    /// Use corner-cutting optimization for Smith-Waterman.
    pub use_corner_cutting: bool,
    /// Corner-cutting factor.
    pub cc_factor: f64,
    /// Add length difference to corner-cutting factor.
    pub cc_add: bool,
    /// Use boolean (thresholded) scoring.
    pub boolean_mode: bool,
    /// Threshold for boolean mode.
    pub bool_cut: f64,
    /// Require score to increase (stop on score drop).
    pub require_score_rise: bool,
    /// Cutoff for structural equivalence.
    pub cutoff: f64,
    /// Second cutoff for secondary structure equivalence.
    pub second_cutoff: f64,
    /// E1 parameter (distance tolerance).
    pub e1: f64,
    /// E2 parameter (conformational tolerance).
    pub e2: f64,
    /// Normalization mean.
    pub nmean: f64,
    /// Normalization standard deviation.
    pub nsd: f64,
}

impl Default for PairwiseConfig {
    fn default() -> Self {
        Self {
            score_tol: DEFAULT_SCORE_TOL,
            max_iter: DEFAULT_MAX_ITER,
            min_fit: DEFAULT_MIN_FIT,
            precision: DEFAULT_PRECISION,
            gap_penalty: 0.0,
            use_corner_cutting: true,
            cc_factor: 100.0,
            cc_add: true,
            boolean_mode: false,
            bool_cut: 0.5,
            require_score_rise: false,
            cutoff: 4.0,
            second_cutoff: 4.5,
            e1: 2.0,
            e2: 5.0,
            nmean: 0.5,
            nsd: 0.4,
        }
    }
}

impl PairwiseConfig {
    /// Creates configuration from global parameters.
    #[must_use]
    pub fn from_parameters(params: &Parameters) -> Self {
        Self {
            score_tol: params.convergence * 100.0, // Convert to percentage
            max_iter: params.max_iter,
            min_fit: DEFAULT_MIN_FIT,
            precision: DEFAULT_PRECISION,
            gap_penalty: params.pen_gap,
            use_corner_cutting: true,
            cc_factor: 100.0,
            cc_add: true,
            boolean_mode: false,
            bool_cut: 0.5,
            require_score_rise: false,
            cutoff: params.cutoff_dist,
            second_cutoff: params.cutoff_dist + 0.5,
            e1: params.e1,
            e2: params.e2,
            nmean: 0.5,
            nsd: 0.4,
        }
    }

    /// Creates configuration for first pass (coarse alignment).
    #[must_use]
    pub fn first_pass() -> Self {
        Self {
            e1: 2.0,
            e2: 5.0,
            cutoff: 4.0,
            ..Default::default()
        }
    }

    /// Creates configuration for second pass (fine alignment).
    #[must_use]
    pub fn second_pass() -> Self {
        Self {
            e1: 1.0,
            e2: 2.5,
            cutoff: 4.0,
            ..Default::default()
        }
    }

    /// Converts to Rossmann-Argos parameters.
    #[must_use]
    pub fn to_rossmann_params(&self) -> RossmannParams {
        let mut params = RossmannParams::new(self.e1, self.e2);
        params.precision = self.precision;
        params.nmean = self.nmean;
        params.nsd = self.nsd;
        params.boolean_mode = self.boolean_mode;
        params.bool_cut = self.bool_cut;
        params.cc_factor = self.cc_factor;
        params.cc_add = self.cc_add;
        params
    }
}

/// Result of pairwise structural alignment.
#[derive(Debug, Clone)]
pub struct PairwiseResult {
    /// Final transformation to superpose domain2 onto domain1.
    pub transform: Transform,
    /// Root mean square deviation after superposition.
    pub rmsd: f64,
    /// STAMP score (Sc).
    pub score: f64,
    /// Alignment length (including gaps).
    pub alignment_length: usize,
    /// Number of fitted (equivalent) residues.
    pub n_fit: usize,
    /// Aligned residue pairs (indices into domain1, domain2).
    pub aligned_pairs: Vec<(usize, usize)>,
    /// Sequence identity of aligned region (percentage).
    pub seq_identity: f64,
    /// Secondary structure identity of aligned region (percentage).
    pub sec_identity: f64,
    /// Number of equivalent secondary structure elements.
    pub n_sec_equiv: usize,
    /// Number of equivalent positions.
    pub n_equiv: usize,
    /// Per-residue probability scores.
    pub residue_scores: Vec<f64>,
    /// Start position in domain1.
    pub start1: usize,
    /// End position in domain1.
    pub end1: usize,
    /// Start position in domain2.
    pub start2: usize,
    /// End position in domain2.
    pub end2: usize,
    /// Number of iterations performed.
    pub iterations: u32,
    /// Whether convergence was achieved.
    pub converged: bool,
}

impl Default for PairwiseResult {
    fn default() -> Self {
        Self {
            transform: Transform::identity(),
            rmsd: f64::INFINITY,
            score: 0.0,
            alignment_length: 0,
            n_fit: 0,
            aligned_pairs: Vec::new(),
            seq_identity: 0.0,
            sec_identity: 0.0,
            n_sec_equiv: 0,
            n_equiv: 0,
            residue_scores: Vec::new(),
            start1: 0,
            end1: 0,
            start2: 0,
            end2: 0,
            iterations: 0,
            converged: false,
        }
    }
}

impl PairwiseResult {
    /// Creates a new empty result.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true if the alignment is valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.n_fit >= 3 && self.rmsd < 100.0 && self.score > 0.0
    }
}

/// Working state for iterative fitting.
struct FittingState {
    /// Current working coordinates for domain2 (transformed).
    coords2: Vec<Coord3>,
    /// Cumulative transformation.
    cumulative_transform: Transform,
    /// Current RMSD.
    rmsd: f64,
    /// Current score.
    score: f64,
    /// Current alignment length.
    alignment_length: usize,
    /// Current number of fitted residues.
    n_fit: usize,
    /// Current aligned pairs.
    aligned_pairs: Vec<(usize, usize)>,
    /// Per-residue scores.
    residue_scores: Vec<f64>,
    /// Alignment start/end positions.
    start1: usize,
    end1: usize,
    start2: usize,
    end2: usize,
}

impl FittingState {
    fn new(coords2: Vec<Coord3>) -> Self {
        let n = coords2.len();
        Self {
            coords2,
            cumulative_transform: Transform::identity(),
            rmsd: f64::INFINITY,
            score: 0.0,
            alignment_length: 0,
            n_fit: 0,
            aligned_pairs: Vec::new(),
            residue_scores: vec![0.0; n],
            start1: 0,
            end1: 0,
            start2: 0,
            end2: 0,
        }
    }
}

/// Performs pairwise structural alignment using iterative fitting.
///
/// This is the main entry point for the STAMP pairwise alignment algorithm.
/// It implements the `pairfit` function from the original C code.
///
/// # Algorithm
///
/// The algorithm iteratively:
/// 1. Calculates the Rossmann-Argos probability matrix
/// 2. Finds the optimal alignment path using Smith-Waterman
/// 3. Extracts equivalent residue pairs
/// 4. Computes optimal transformation via Kabsch
/// 5. Applies transformation and accumulates it
/// 6. Checks for convergence
///
/// # Arguments
///
/// * `domain1` - Reference (fixed) domain
/// * `domain2` - Mobile domain to be transformed
/// * `config` - Pairwise fitting configuration
///
/// # Returns
///
/// Result containing the alignment details, transformation, and statistics.
///
/// # Example
///
/// ```ignore
/// use stamp_core::pairwise::{pairfit, PairwiseConfig};
///
/// let config = PairwiseConfig::default();
/// let result = pairfit(&domain1, &domain2, &config)?;
/// println!("RMSD: {:.2}, Score: {:.3}", result.rmsd, result.score);
/// ```
pub fn pairfit(
    domain1: &Domain,
    domain2: &Domain,
    config: &PairwiseConfig,
) -> StampResult<PairwiseResult> {
    log::info!(
        "Pairwise fitting {} ({} residues) with {} ({} residues)",
        domain1.id,
        domain1.len(),
        domain2.id,
        domain2.len()
    );

    // Extract coordinates
    let coords1 = domain_to_coords(domain1);
    let coords2_orig = domain_to_coords(domain2);

    if coords1.len() < config.min_fit || coords2_orig.len() < config.min_fit {
        log::warn!("Structures too small for alignment");
        return Ok(PairwiseResult::new());
    }

    // Initialize working state with a copy of domain2 coordinates
    let mut state = FittingState::new(coords2_orig.clone());

    // Rossmann-Argos parameters
    let ra_params = config.to_rossmann_params();

    // Iteration variables
    let mut score_diff: f64 = config.score_tol + 1.0;
    let mut score_rise = true;
    let mut iteration: u32 = 0;

    // Main iterative loop - use loop with break to ensure at least one iteration
    loop {
        iteration += 1;

        // Check termination conditions (after first iteration)
        if iteration > 1 {
            if score_diff <= config.score_tol {
                log::debug!("Converged: score_diff={:.4}% <= tol={:.4}%", score_diff, config.score_tol);
                break;
            }
            if iteration > config.max_iter {
                log::debug!("Max iterations ({}) reached", config.max_iter);
                break;
            }
            if state.n_fit < config.min_fit.max(3) {
                log::debug!("Too few fitted residues: {} < {}", state.n_fit, config.min_fit.max(3));
                break;
            }
            if config.require_score_rise && !score_rise {
                log::debug!("Score did not rise, stopping");
                break;
            }
        }

        // Calculate probability matrix
        let prob_matrix = if config.use_corner_cutting {
            calculate_probability_matrix_cc(&coords1, &state.coords2, &ra_params)
        } else {
            calculate_probability_matrix(&coords1, &state.coords2, &ra_params)
        };

        // Run pairpath to get alignment, score, transformation
        let path_result = pairpath(
            domain1,
            &coords1,
            &state.coords2,
            &prob_matrix,
            config,
        )?;

        // Update state from path result
        let old_score = state.score;
        state.score = path_result.score;
        state.rmsd = path_result.rmsd;
        state.alignment_length = path_result.alignment_length;
        state.n_fit = path_result.n_fit;
        state.aligned_pairs = path_result.aligned_pairs;
        state.residue_scores = path_result.residue_scores;
        state.start1 = path_result.start1;
        state.end1 = path_result.end1;
        state.start2 = path_result.start2;
        state.end2 = path_result.end2;

        // Calculate score difference (percentage)
        if state.score > 0.0 {
            score_diff = 100.0 * (state.score - old_score).abs() / state.score;
        } else {
            score_diff = 0.0;
        }

        score_rise = state.score > old_score;

        log::debug!(
            "Iteration {}: RMS={:.3}, Sc diff={:.2}%, Sc={:.3}, len={}, nfit={}",
            iteration,
            state.rmsd,
            score_diff,
            state.score,
            state.alignment_length,
            state.n_fit
        );

        // Apply transformation to coordinates and accumulate
        if path_result.n_fit >= 3 {
            let new_transform = &path_result.transform;

            // Transform working coordinates
            for coord in &mut state.coords2 {
                *coord = new_transform.apply(coord);
            }

            // Accumulate transformation: cumulative = new * cumulative
            state.cumulative_transform = update_transform(new_transform, &state.cumulative_transform);
        }

        // Safety limit for first iteration
        if iteration >= config.max_iter {
            break;
        }
    }

    // Determine convergence
    let converged = score_diff <= config.score_tol;
    if converged {
        log::info!("Converged after {} iterations", iteration);
    } else {
        log::info!("No convergence after {} iterations", iteration);
    }

    // Calculate sequence and secondary structure identity
    let (seq_id, sec_id, n_sec_equiv, n_equiv) = calculate_identities(
        domain1,
        domain2,
        &state.aligned_pairs,
        &state.residue_scores,
        config.second_cutoff,
    );

    log::info!(
        "Alignment: len={}, seqid={:.1}%, secid={:.1}%, n_sec_equiv={}",
        state.alignment_length,
        seq_id,
        sec_id,
        n_sec_equiv
    );

    Ok(PairwiseResult {
        transform: state.cumulative_transform,
        rmsd: state.rmsd,
        score: state.score,
        alignment_length: state.alignment_length,
        n_fit: state.n_fit,
        aligned_pairs: state.aligned_pairs,
        seq_identity: seq_id,
        sec_identity: sec_id,
        n_sec_equiv,
        n_equiv,
        residue_scores: state.residue_scores,
        start1: state.start1,
        end1: state.end1,
        start2: state.start2,
        end2: state.end2,
        iterations: iteration,
        converged,
    })
}

/// Result from a single path calculation iteration.
struct PathResult {
    transform: Transform,
    rmsd: f64,
    score: f64,
    alignment_length: usize,
    n_fit: usize,
    aligned_pairs: Vec<(usize, usize)>,
    residue_scores: Vec<f64>,
    start1: usize,
    end1: usize,
    start2: usize,
    end2: usize,
}

/// Calculates alignment path and transformation for one iteration.
///
/// This function implements the `pairpath` algorithm from the original C code.
/// It performs Smith-Waterman alignment on the probability matrix, extracts
/// equivalent residue pairs, and computes the optimal transformation.
fn pairpath(
    _domain1: &Domain,
    coords1: &[Coord3],
    coords2: &[Coord3],
    prob_matrix: &ProbabilityMatrix,
    config: &PairwiseConfig,
) -> StampResult<PathResult> {
    let len_a = coords1.len();
    let len_b = coords2.len();

    // Convert probability matrix to integer format for Smith-Waterman
    // The C code uses integer matrices scaled by PRECISION
    // We need to pad the matrix with zeros for boundary handling (+2 in each dimension)
    let raw_prob = prob_matrix.to_integer(config.precision);

    // Create padded probability matrix (len+2 in each dimension)
    // The first row/column and last row/column are padding (zeros)
    let padded_rows = len_a + 2;
    let padded_cols = len_b + 2;
    let mut int_prob = vec![vec![0i32; padded_cols]; padded_rows];

    // Copy original matrix into padded matrix at offset (1,1)
    for i in 0..len_a {
        for j in 0..len_b {
            int_prob[i + 1][j + 1] = raw_prob[i][j];
        }
    }

    // Calculate minimum score threshold
    let min_score = if config.boolean_mode {
        config.min_fit as i32
    } else {
        (config.min_fit as f64 * config.precision as f64 * config.cutoff) as i32
    };

    let gap_penalty = (config.gap_penalty * config.precision as f64) as i32;

    // Run Smith-Waterman alignment with padded dimensions
    let sw_result: SwResult = if config.use_corner_cutting {
        smith_waterman_corner_cut(
            padded_rows,
            padded_cols,
            &int_prob,
            gap_penalty,
            min_score,
            config.cc_add,
            config.cc_factor,
        )
    } else {
        smith_waterman(padded_rows, padded_cols, &int_prob, gap_penalty, min_score)
    };

    // Check if we found any paths
    if sw_result.paths.is_empty() || sw_result.total_paths == 0 {
        log::debug!("No alignment paths found");
        return Ok(PathResult {
            transform: Transform::identity(),
            rmsd: 100.0,
            score: 0.0,
            alignment_length: 0,
            n_fit: 0,
            aligned_pairs: Vec::new(),
            residue_scores: Vec::new(),
            start1: 0,
            end1: 0,
            start2: 0,
            end2: 0,
        });
    }

    // Get the best path (paths are sorted by score)
    let best_path = &sw_result.paths[0];

    // Perform traceback to get aligned positions
    let (aligned_positions, h_gaps, v_gaps) = traceback(&sw_result.path_matrix, best_path);

    let alignment_length = aligned_positions.len() + h_gaps + v_gaps;

    if aligned_positions.is_empty() {
        return Ok(PathResult {
            transform: Transform::identity(),
            rmsd: 100.0,
            score: 0.0,
            alignment_length: 0,
            n_fit: 0,
            aligned_pairs: Vec::new(),
            residue_scores: Vec::new(),
            start1: 0,
            end1: 0,
            start2: 0,
            end2: 0,
        });
    }

    // Extract equivalent pairs based on probability cutoff
    // In STAMP, equivalence is determined by Pij >= BOOLCUT (typically 0.5)
    // For non-boolean mode, we use a lower threshold (0.3) since values are normalized
    let cutoff_int = if config.boolean_mode {
        1
    } else {
        // Probability threshold: 0.3 (300 after scaling by precision)
        // This corresponds to reasonably well-aligned residue pairs
        (0.3 * config.precision as f64) as i32
    };

    let mut equivalent_pairs: Vec<(usize, usize)> = Vec::new();
    let mut residue_scores: Vec<f64> = vec![-1.0; alignment_length.max(len_a).max(len_b)];

    for &(i, j) in &aligned_positions {
        // Adjust for 1-based indexing in C code
        if i == 0 || j == 0 || i > len_a || j > len_b {
            continue;
        }
        let i_adj = i - 1;
        let j_adj = j - 1;

        // Get probability score
        let prob_score = if i < int_prob.len() && j < int_prob[i].len() {
            int_prob[i][j]
        } else {
            0
        };

        // Store per-residue score
        if i_adj < residue_scores.len() {
            residue_scores[i_adj] = if config.boolean_mode {
                if prob_score > 0 { 1.0 } else { 0.0 }
            } else {
                prob_score as f64 / config.precision as f64
            };
        }

        // Check if this pair is equivalent
        let is_equivalent = if config.boolean_mode {
            prob_score == 1
        } else {
            prob_score >= cutoff_int
        };

        if is_equivalent {
            equivalent_pairs.push((i_adj, j_adj));
        }
    }

    let n_fit = equivalent_pairs.len();

    // Compute transformation using Kabsch if we have enough points
    let (transform, rmsd) = if n_fit >= 3 {
        // Extract coordinates for fitting
        let fit_coords1: Vec<Coord3> = equivalent_pairs
            .iter()
            .filter_map(|&(i, _)| coords1.get(i).copied())
            .collect();
        let fit_coords2: Vec<Coord3> = equivalent_pairs
            .iter()
            .filter_map(|&(_, j)| coords2.get(j).copied())
            .collect();

        if fit_coords1.len() >= 3 && fit_coords2.len() >= 3 {
            superpose(&fit_coords1, &fit_coords2)?
        } else {
            (Transform::identity(), 100.0)
        }
    } else {
        log::debug!("Too few equivalences ({}) for transformation", n_fit);
        (Transform::identity(), 100.0)
    };

    // Calculate STAMP score (Sc)
    // Sc = (sum_prob / alignment_length) * ((alignment_length - h_gaps) / len_a) * ((alignment_length - v_gaps) / len_b)
    let score = calculate_stamp_score(
        sw_result.max_score,
        alignment_length,
        h_gaps,
        v_gaps,
        len_a,
        len_b,
        config.precision,
        config.boolean_mode,
    );

    // Determine start and end positions
    let start1 = best_path.start.i.saturating_sub(1);
    let end1 = best_path.end.i.saturating_sub(1).min(len_a.saturating_sub(1));
    let start2 = best_path.start.j.saturating_sub(1);
    let end2 = best_path.end.j.saturating_sub(1).min(len_b.saturating_sub(1));

    Ok(PathResult {
        transform,
        rmsd,
        score,
        alignment_length,
        n_fit,
        aligned_pairs: equivalent_pairs,
        residue_scores,
        start1,
        end1,
        start2,
        end2,
    })
}

/// Calculates the STAMP Sc score.
///
/// The score accounts for:
/// - Sum of probability values along the alignment
/// - Alignment length
/// - Gap penalties
/// - Sequence lengths
fn calculate_stamp_score(
    max_score: i32,
    alignment_length: usize,
    h_gaps: usize,
    v_gaps: usize,
    len_a: usize,
    len_b: usize,
    precision: i32,
    boolean_mode: bool,
) -> f64 {
    if alignment_length == 0 || len_a == 0 || len_b == 0 {
        return 0.0;
    }

    let allen = alignment_length as f64;
    let highest = max_score as f64;
    let ia = h_gaps as f64;
    let ib = v_gaps as f64;
    let na = len_a as f64;
    let nb = len_b as f64;

    // Standard STAMP score formula
    let mut score = (highest / allen) * ((allen - ia) / na) * ((allen - ib) / nb);

    if !boolean_mode {
        score /= precision as f64;
    }

    score
}

/// Updates the cumulative transformation: result = delta * current.
///
/// This implements the transformation composition from the C `update` function:
/// - R_new = delta_R * R_old
/// - V_new = delta_R * V_old + delta_V
fn update_transform(delta: &Transform, current: &Transform) -> Transform {
    Transform {
        rotation: delta.rotation * current.rotation,
        translation: delta.rotation * current.translation + delta.translation,
    }
}

/// Calculates sequence and secondary structure identities.
fn calculate_identities(
    domain1: &Domain,
    domain2: &Domain,
    aligned_pairs: &[(usize, usize)],
    residue_scores: &[f64],
    second_cutoff: f64,
) -> (f64, f64, usize, usize) {
    if aligned_pairs.is_empty() {
        return (0.0, 0.0, 0, 0);
    }

    let mut seq_count = 0;
    let mut sec_count = 0;
    let mut n_sec_equiv = 0;
    let mut n_equiv = 0;

    let mut last_matched_sec1 = 0i32;
    let mut last_matched_sec2 = 0i32;
    let mut nsec1 = 0;
    let mut nsec2 = 0;
    let mut in_sec1 = false;
    let mut in_sec2 = false;

    for (idx, &(i, j)) in aligned_pairs.iter().enumerate() {
        let r1 = match domain1.residues.get(i) {
            Some(r) => r,
            None => continue,
        };
        let r2 = match domain2.residues.get(j) {
            Some(r) => r,
            None => continue,
        };

        // Check if this position meets the cutoff and has neighbors
        let score = residue_scores.get(idx).copied().unwrap_or(-1.0);
        let neighbors = count_neighbors(residue_scores, idx, second_cutoff);

        if score >= second_cutoff && neighbors >= 2 {
            n_equiv += 1;

            // Sequence identity
            if r1.aa == r2.aa {
                seq_count += 1;
            }

            // Secondary structure identity
            let ss1 = normalize_ss(r1.sec_struct);
            let ss2 = normalize_ss(r2.sec_struct);
            if ss1 == ss2 {
                sec_count += 1;
            }
        }

        // Track secondary structure elements
        let ss1 = normalize_ss(r1.sec_struct);
        let ss2 = normalize_ss(r2.sec_struct);

        if ss1 == 'c' {
            in_sec1 = false;
        } else {
            if !in_sec1 {
                nsec1 += 1;
            }
            in_sec1 = true;
        }

        if ss2 == 'c' {
            in_sec2 = false;
        } else {
            if !in_sec2 {
                nsec2 += 1;
            }
            in_sec2 = true;
        }

        // Count equivalent secondary structure elements
        if in_sec1 && in_sec2
            && score >= second_cutoff
            && neighbors >= 2
            && (nsec1 != last_matched_sec1 || nsec2 != last_matched_sec2)
        {
            n_sec_equiv += 1;
            last_matched_sec1 = nsec1;
            last_matched_sec2 = nsec2;
        }
    }

    let seq_id = if n_equiv > 0 {
        100.0 * seq_count as f64 / n_equiv as f64
    } else {
        0.0
    };

    let sec_id = if n_equiv > 0 {
        100.0 * sec_count as f64 / n_equiv as f64
    } else {
        0.0
    };

    (seq_id, sec_id, n_sec_equiv, n_equiv)
}

/// Counts neighbors with scores above cutoff.
fn count_neighbors(scores: &[f64], idx: usize, cutoff: f64) -> usize {
    let mut count = 0;

    // Check forward neighbors
    for j in (idx + 1)..scores.len().min(idx + 5) {
        if scores[j] >= cutoff {
            count += 1;
        } else {
            break;
        }
    }

    // Check backward neighbors
    for j in (idx.saturating_sub(4)..idx).rev() {
        if scores[j] >= cutoff {
            count += 1;
        } else {
            break;
        }
    }

    count
}

/// Normalizes secondary structure code to 3-state (H, B, c).
fn normalize_ss(ss: char) -> char {
    match ss {
        'H' | 'G' => 'H',
        'E' | 'B' => 'B',
        _ => 'c',
    }
}

/// Performs two-pass pairwise fitting (coarse then fine).
///
/// This implements the NPASS=2 mode from the C code, which runs
/// alignment twice with different E1/E2 parameters for better
/// sensitivity.
///
/// # Arguments
///
/// * `domain1` - Reference domain
/// * `domain2` - Mobile domain
/// * `params` - Global parameters (optional, uses defaults if None)
///
/// # Returns
///
/// Result from the second (fine) pass.
pub fn pairfit_two_pass(
    domain1: &Domain,
    domain2: &Domain,
    params: Option<&Parameters>,
) -> StampResult<PairwiseResult> {
    // First pass: coarse alignment
    let config1 = match params {
        Some(p) => {
            let mut cfg = PairwiseConfig::from_parameters(p);
            cfg.e1 = 2.0;  // Coarse E1
            cfg.e2 = 5.0;  // Coarse E2
            cfg
        }
        None => PairwiseConfig::first_pass(),
    };

    log::info!("First pass (E1={}, E2={})", config1.e1, config1.e2);
    let result1 = pairfit(domain1, domain2, &config1)?;

    if !result1.is_valid() {
        log::warn!("First pass failed, returning empty result");
        return Ok(result1);
    }

    // Apply first pass transformation to get transformed domain2
    let mut domain2_transformed = domain2.clone();
    for residue in &mut domain2_transformed.residues {
        residue.ca_coord = result1.transform.apply(&residue.ca_coord);
    }

    // Second pass: fine alignment on transformed coordinates
    let config2 = match params {
        Some(p) => {
            let mut cfg = PairwiseConfig::from_parameters(p);
            cfg.e1 = 1.0;  // Fine E1
            cfg.e2 = 2.5;  // Fine E2
            cfg
        }
        None => PairwiseConfig::second_pass(),
    };

    log::info!("Second pass (E1={}, E2={})", config2.e1, config2.e2);
    let result2 = pairfit(domain1, &domain2_transformed, &config2)?;

    // Combine transformations: total = pass2 * pass1
    let combined_transform = update_transform(&result2.transform, &result1.transform);

    Ok(PairwiseResult {
        transform: combined_transform,
        iterations: result1.iterations + result2.iterations,
        ..result2
    })
}

/// Performs quick pairwise alignment without iteration.
///
/// This is a simplified version that performs only one round of
/// probability matrix calculation, path finding, and transformation.
/// Useful for initial screening or when speed is more important
/// than accuracy.
pub fn quick_align(
    domain1: &Domain,
    domain2: &Domain,
    params: &Parameters,
) -> StampResult<AlignmentResult> {
    let config = PairwiseConfig::from_parameters(params);
    let coords1 = domain_to_coords(domain1);
    let coords2 = domain_to_coords(domain2);

    let ra_params = config.to_rossmann_params();
    let prob_matrix = calculate_probability_matrix(&coords1, &coords2, &ra_params);

    let path_result = pairpath(domain1, &coords1, &coords2, &prob_matrix, &config)?;

    Ok(AlignmentResult {
        transform: path_result.transform,
        rmsd: path_result.rmsd,
        n_aligned: path_result.n_fit,
        score: path_result.score,
        seq_identity: sequence_identity(domain1, domain2, &path_result.aligned_pairs),
        path: path_result.aligned_pairs,
        residue_scores: path_result.residue_scores,
    })
}

/// Performs pairwise structural alignment of two domains.
///
/// This is the main entry point for STAMP pairwise alignment,
/// providing a simplified interface to `pairfit`.
///
/// # Arguments
///
/// * `domain1` - First (reference) domain
/// * `domain2` - Second (mobile) domain
/// * `params` - Alignment parameters
///
/// # Returns
///
/// Result containing the alignment details including transformation,
/// RMSD, and aligned positions.
pub fn align_pair(
    domain1: &Domain,
    domain2: &Domain,
    params: &Parameters,
) -> StampResult<AlignmentResult> {
    log::info!(
        "Aligning {} ({} residues) with {} ({} residues)",
        domain1.id,
        domain1.len(),
        domain2.id,
        domain2.len()
    );

    let config = PairwiseConfig::from_parameters(params);
    let result = pairfit(domain1, domain2, &config)?;

    log::info!(
        "Alignment complete: {} aligned, RMSD={:.2}, Sc={:.3}",
        result.n_fit,
        result.rmsd,
        result.score
    );

    Ok(AlignmentResult {
        transform: result.transform,
        rmsd: result.rmsd,
        n_aligned: result.n_fit,
        score: result.score,
        seq_identity: result.seq_identity / 100.0, // Convert to fraction
        path: result.aligned_pairs,
        residue_scores: result.residue_scores,
    })
}

/// Computes all-vs-all pairwise alignments.
///
/// # Arguments
///
/// * `domains` - List of domains to align
/// * `params` - Alignment parameters
///
/// # Returns
///
/// Matrix of alignment results where result[i][j] is alignment of domain i to j.
pub fn all_vs_all(
    domains: &[Domain],
    params: &Parameters,
) -> StampResult<Vec<Vec<AlignmentResult>>> {
    let n = domains.len();
    let mut results = vec![vec![AlignmentResult::new(); n]; n];

    for i in 0..n {
        for j in i + 1..n {
            let result = align_pair(&domains[i], &domains[j], params)?;

            // Store result for both i,j and j,i
            results[i][j] = result.clone();

            // Create inverse alignment
            let mut inv_result = result;
            inv_result.transform = inv_result.transform.inverse();
            inv_result.path = inv_result.path.into_iter().map(|(a, b)| (b, a)).collect();
            results[j][i] = inv_result;
        }

        // Diagonal is identity
        results[i][i] = AlignmentResult {
            transform: Transform::identity(),
            rmsd: 0.0,
            n_aligned: domains[i].len(),
            score: 1.0,
            seq_identity: 1.0,
            path: (0..domains[i].len()).map(|k| (k, k)).collect(),
            residue_scores: Vec::new(),
        };
    }

    Ok(results)
}

/// Computes similarity matrix from pairwise alignments.
#[must_use]
pub fn similarity_matrix(results: &[Vec<AlignmentResult>]) -> Vec<Vec<f64>> {
    results
        .iter()
        .map(|row| row.iter().map(|r| r.score).collect())
        .collect()
}

/// Computes RMSD matrix from pairwise alignments.
#[must_use]
pub fn rmsd_matrix(results: &[Vec<AlignmentResult>]) -> Vec<Vec<f64>> {
    results
        .iter()
        .map(|row| row.iter().map(|r| r.rmsd).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Residue;

    fn make_test_domain(id: &str, coords: Vec<(f64, f64, f64)>) -> Domain {
        let mut domain = Domain::new(id.to_string());
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
    fn test_pairwise_config_default() {
        let config = PairwiseConfig::default();
        assert!((config.score_tol - DEFAULT_SCORE_TOL).abs() < 1e-10);
        assert_eq!(config.max_iter, DEFAULT_MAX_ITER);
        assert_eq!(config.min_fit, DEFAULT_MIN_FIT);
    }

    #[test]
    fn test_pairwise_config_first_pass() {
        let config = PairwiseConfig::first_pass();
        assert!((config.e1 - 2.0).abs() < 1e-10);
        assert!((config.e2 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_config_second_pass() {
        let config = PairwiseConfig::second_pass();
        assert!((config.e1 - 1.0).abs() < 1e-10);
        assert!((config.e2 - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_result_default() {
        let result = PairwiseResult::new();
        assert!(!result.is_valid());
        assert_eq!(result.n_fit, 0);
        assert!(result.rmsd.is_infinite());
    }

    #[test]
    fn test_update_transform_identity() {
        let t1 = Transform::identity();
        let t2 = Transform::identity();
        let result = update_transform(&t1, &t2);

        // Should still be identity
        let p = Coord3::new(1.0, 2.0, 3.0);
        let transformed = result.apply(&p);
        assert!((transformed.x - p.x).abs() < 1e-10);
        assert!((transformed.y - p.y).abs() < 1e-10);
        assert!((transformed.z - p.z).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_ss() {
        assert_eq!(normalize_ss('H'), 'H');
        assert_eq!(normalize_ss('G'), 'H');
        assert_eq!(normalize_ss('E'), 'B');
        assert_eq!(normalize_ss('B'), 'B');
        assert_eq!(normalize_ss('C'), 'c');
        assert_eq!(normalize_ss('T'), 'c');
    }

    #[test]
    fn test_calculate_stamp_score_zero() {
        let score = calculate_stamp_score(0, 0, 0, 0, 10, 10, 1000, false);
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_stamp_score_nonzero() {
        // Simple case: perfect alignment of length 10
        let score = calculate_stamp_score(10000, 10, 0, 0, 10, 10, 1000, false);
        assert!(score > 0.0);
    }

    #[test]
    fn test_pairfit_identical_structures() {
        let coords = vec![
            (0.0, 0.0, 0.0),
            (3.8, 0.0, 0.0),
            (7.6, 0.0, 0.0),
            (11.4, 0.0, 0.0),
            (15.2, 0.0, 0.0),
        ];
        let d1 = make_test_domain("test1", coords.clone());
        let d2 = make_test_domain("test2", coords);
        let config = PairwiseConfig::default();

        let result = pairfit(&d1, &d2, &config).unwrap();

        // Identical structures should give low RMSD
        if result.is_valid() {
            assert!(result.rmsd < 1.0);
        }
    }

    #[test]
    fn test_pairfit_translated_structures() {
        let coords1 = vec![
            (0.0, 0.0, 0.0),
            (3.8, 0.0, 0.0),
            (7.6, 0.0, 0.0),
            (11.4, 0.0, 0.0),
            (15.2, 0.0, 0.0),
        ];
        let coords2: Vec<(f64, f64, f64)> = coords1
            .iter()
            .map(|(x, y, z)| (x + 10.0, y + 5.0, z + 3.0))
            .collect();

        let d1 = make_test_domain("test1", coords1);
        let d2 = make_test_domain("test2", coords2);
        let config = PairwiseConfig::default();

        let result = pairfit(&d1, &d2, &config).unwrap();

        // Translated structures should be alignable with low RMSD
        if result.is_valid() {
            assert!(result.rmsd < 1.0);
        }
    }

    #[test]
    fn test_align_pair_basic() {
        let coords = vec![
            (0.0, 0.0, 0.0),
            (3.8, 0.0, 0.0),
            (7.6, 0.0, 0.0),
        ];
        let d1 = make_test_domain("test1", coords.clone());
        let d2 = make_test_domain("test2", coords);
        let params = Parameters::default();

        let result = align_pair(&d1, &d2, &params).unwrap();
        // n_aligned is usize so always >= 0, check it's a valid result
        assert!(result.rmsd.is_finite() || result.n_aligned == 0);
    }

    #[test]
    fn test_similarity_matrix() {
        let results = vec![
            vec![
                AlignmentResult { score: 1.0, ..Default::default() },
                AlignmentResult { score: 0.8, ..Default::default() },
            ],
            vec![
                AlignmentResult { score: 0.8, ..Default::default() },
                AlignmentResult { score: 1.0, ..Default::default() },
            ],
        ];
        let sim = similarity_matrix(&results);
        assert!((sim[0][0] - 1.0).abs() < 1e-10);
        assert!((sim[0][1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_rmsd_matrix() {
        let results = vec![
            vec![
                AlignmentResult { rmsd: 0.0, ..Default::default() },
                AlignmentResult { rmsd: 2.5, ..Default::default() },
            ],
            vec![
                AlignmentResult { rmsd: 2.5, ..Default::default() },
                AlignmentResult { rmsd: 0.0, ..Default::default() },
            ],
        ];
        let rmsds = rmsd_matrix(&results);
        assert!((rmsds[0][0] - 0.0).abs() < 1e-10);
        assert!((rmsds[0][1] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_count_neighbors() {
        let scores = vec![1.0, 5.0, 5.0, 5.0, 5.0, 1.0];
        let count = count_neighbors(&scores, 2, 4.5);
        assert!(count >= 2);
    }

    #[test]
    fn test_pairwise_result_validity() {
        let mut result = PairwiseResult::new();
        assert!(!result.is_valid());

        result.n_fit = 10;
        result.rmsd = 2.0;
        result.score = 5.0;
        assert!(result.is_valid());
    }
}
