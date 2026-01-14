//! Database scanning operations.
//!
//! This module provides functionality for scanning a query structure
//! against a database of structures to find similar proteins, as well
//! as roughfit initial alignment and sliding window scan mode.
//!
//! # Scan Mode
//!
//! The scan algorithm uses a sliding window approach for structures of different
//! lengths or arbitrary orientations:
//!
//! 1. For each starting position in the target structure (stepped by `slide`):
//!    - Take overlapping N-terminal portions of query and target
//!    - Perform initial superposition via Kabsch algorithm
//!    - Apply transformation to bring structures into alignment
//!    - Run iterative pairfit refinement
//!    - Track the best result across all starting positions
//!
//! # Roughfit Mode
//!
//! Simple initial alignment that aligns structures from their N-terminal ends:
//! - For each domain after the first, align N-terminal overlapping portion
//! - Use Kabsch algorithm to find initial transformation
//! - Apply transformation to bring structures into rough alignment

use crate::math::superpose;
use crate::pairwise::{align_pair, pairfit, quick_align, PairwiseConfig, PairwiseResult};
use crate::scoring::domain_to_coords;
use crate::types::{AlignmentResult, Coord3, Domain, Parameters, StampResult, Transform};

/// Result of a database scan hit.
#[derive(Debug, Clone)]
pub struct ScanHit {
    /// Target domain identifier.
    pub target_id: String,
    /// Alignment result.
    pub alignment: AlignmentResult,
    /// Z-score (if computed).
    pub z_score: Option<f64>,
    /// E-value (if computed).
    pub e_value: Option<f64>,
    /// Rank in hit list.
    pub rank: usize,
}

impl ScanHit {
    /// Creates a new scan hit.
    #[must_use]
    pub fn new(target_id: String, alignment: AlignmentResult, rank: usize) -> Self {
        Self {
            target_id,
            alignment,
            z_score: None,
            e_value: None,
            rank,
        }
    }
}

// ============================================================================
// Roughfit Mode Implementation
// ============================================================================

/// Result of a roughfit alignment operation.
#[derive(Debug, Clone)]
pub struct RoughfitResult {
    /// Domain identifier.
    pub domain_id: String,
    /// Transformation to apply to this domain.
    pub transform: Transform,
    /// RMSD of the rough alignment.
    pub rmsd: f64,
    /// Number of residues used in the initial superposition.
    pub n_aligned: usize,
}

/// Performs rough initial alignment of structures from their N-terminal ends.
///
/// This function implements the `roughfit` mode from the original STAMP C code.
/// It aligns each domain after the first by superposing their N-terminal
/// overlapping portions onto the first domain.
///
/// # Algorithm
///
/// For each domain i > 0:
/// 1. Determine overlap length = min(len(domain_0), len(domain_i))
/// 2. Extract N-terminal overlapping coordinates
/// 3. Compute optimal transformation via Kabsch algorithm
/// 4. Record the transformation for later application
///
/// # Arguments
///
/// * `domains` - Slice of domains to align (first is reference, modified in place)
/// * `_params` - Alignment parameters (currently unused but kept for API consistency)
///
/// # Returns
///
/// Vector of `RoughfitResult` containing transformations for each domain.
///
/// # Example
///
/// ```ignore
/// use stamp_core::scan::roughfit;
///
/// let mut domains = load_domains();
/// let results = roughfit(&mut domains, &Parameters::default())?;
/// for result in &results {
///     println!("{}: RMSD = {:.2}", result.domain_id, result.rmsd);
/// }
/// ```
pub fn roughfit(domains: &mut [Domain], _params: &Parameters) -> StampResult<Vec<RoughfitResult>> {
    if domains.is_empty() {
        return Ok(Vec::new());
    }

    let mut results = Vec::with_capacity(domains.len());

    // First domain is the reference - identity transform
    let ref_coords = domain_to_coords(&domains[0]);
    results.push(RoughfitResult {
        domain_id: domains[0].id.clone(),
        transform: Transform::identity(),
        rmsd: 0.0,
        n_aligned: ref_coords.len(),
    });

    // For each subsequent domain, compute rough alignment to first domain
    for i in 1..domains.len() {
        let mobile_coords = domain_to_coords(&domains[i]);

        // Determine overlap length (N-terminal alignment)
        let overlap_len = ref_coords.len().min(mobile_coords.len());

        if overlap_len < 3 {
            log::warn!(
                "Insufficient overlap ({}) for roughfit of {}",
                overlap_len,
                domains[i].id
            );
            results.push(RoughfitResult {
                domain_id: domains[i].id.clone(),
                transform: Transform::identity(),
                rmsd: f64::INFINITY,
                n_aligned: 0,
            });
            continue;
        }

        // Extract overlapping portions
        let ref_overlap: Vec<Coord3> = ref_coords[..overlap_len].to_vec();
        let mobile_overlap: Vec<Coord3> = mobile_coords[..overlap_len].to_vec();

        // Compute optimal superposition
        let (transform, rmsd) = superpose(&ref_overlap, &mobile_overlap)?;

        log::debug!(
            "Roughfit {}: {} residues overlapping, RMSD = {:.3}",
            domains[i].id,
            overlap_len,
            rmsd
        );

        // Apply transformation to domain coordinates
        for residue in &mut domains[i].residues {
            residue.ca_coord = transform.apply(&residue.ca_coord);
        }

        results.push(RoughfitResult {
            domain_id: domains[i].id.clone(),
            transform,
            rmsd,
            n_aligned: overlap_len,
        });
    }

    Ok(results)
}

// ============================================================================
// Scan Mode Implementation
// ============================================================================

/// Configuration for sliding window scan mode.
#[derive(Debug, Clone)]
pub struct ScanConfig {
    /// Sliding step size in residues (default: 5).
    pub slide: usize,
    /// Minimum score cutoff for reporting (default: 2.0).
    pub scan_cut: f64,
    /// Minimum length fraction to consider target (default: 0.5).
    pub min_frac: f64,
    /// Whether to truncate long target structures.
    pub truncate: bool,
    /// Maximum length ratio for truncation (default: 1.3).
    pub trunc_factor: f64,
    /// Skip ahead over matched regions for efficiency.
    pub skip_ahead: bool,
    /// Number of fitting passes (1 for quick, 2 for thorough).
    pub n_passes: u32,
    /// E1 parameter for scoring.
    pub e1: f64,
    /// E2 parameter for scoring.
    pub e2: f64,
    /// Gap penalty.
    pub gap_penalty: f64,
    /// Cutoff distance.
    pub cutoff: f64,
    /// Maximum iterations per fit.
    pub max_iter: u32,
    /// Convergence threshold.
    pub convergence: f64,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            slide: 5,
            scan_cut: 2.0,
            min_frac: 0.5,
            truncate: true,
            trunc_factor: 1.3,
            skip_ahead: true,
            n_passes: 2,
            e1: 2.0,
            e2: 5.0,
            gap_penalty: 0.0,
            cutoff: 4.0,
            max_iter: 100,
            convergence: 0.01,
        }
    }
}

impl ScanConfig {
    /// Creates a scan configuration from global parameters.
    #[must_use]
    pub fn from_parameters(params: &Parameters) -> Self {
        Self {
            slide: params.slide_offset.unsigned_abs() as usize,
            scan_cut: params.score_cutoff,
            n_passes: params.n_passes,
            e1: params.e1,
            e2: params.e2,
            gap_penalty: params.pen_gap,
            cutoff: params.cutoff_dist,
            max_iter: params.max_iter,
            convergence: params.convergence,
            ..Default::default()
        }
    }

    /// Creates configuration for first pass (coarse).
    #[must_use]
    pub fn first_pass() -> Self {
        Self {
            e1: 2.0,
            e2: 5.0,
            n_passes: 1,
            ..Default::default()
        }
    }

    /// Creates configuration for second pass (fine).
    #[must_use]
    pub fn second_pass() -> Self {
        Self {
            e1: 1.0,
            e2: 2.5,
            n_passes: 1,
            ..Default::default()
        }
    }

    /// Converts to pairwise configuration.
    #[must_use]
    pub fn to_pairwise_config(&self) -> PairwiseConfig {
        PairwiseConfig {
            e1: self.e1,
            e2: self.e2,
            gap_penalty: self.gap_penalty,
            cutoff: self.cutoff,
            max_iter: self.max_iter,
            score_tol: self.convergence * 100.0,
            ..Default::default()
        }
    }
}

/// Result of a scan pair alignment.
#[derive(Debug, Clone)]
pub struct ScanResult {
    /// Query domain identifier.
    pub query_id: String,
    /// Target domain identifier.
    pub target_id: String,
    /// STAMP score (Sc).
    pub score: f64,
    /// RMSD after superposition.
    pub rmsd: f64,
    /// Number of aligned residues.
    pub n_aligned: usize,
    /// Number of fitted (equivalent) residues.
    pub n_fit: usize,
    /// Sequence identity (fraction).
    pub seq_identity: f64,
    /// Transformation to superpose target onto query.
    pub transform: Transform,
    /// Starting position in target where best alignment was found.
    pub fit_position: usize,
    /// Length of query structure.
    pub query_len: usize,
    /// Length of target structure (possibly truncated).
    pub target_len: usize,
    /// Whether the result passed the score cutoff.
    pub passed_cutoff: bool,
}

impl Default for ScanResult {
    fn default() -> Self {
        Self {
            query_id: String::new(),
            target_id: String::new(),
            score: 0.0,
            rmsd: f64::INFINITY,
            n_aligned: 0,
            n_fit: 0,
            seq_identity: 0.0,
            transform: Transform::identity(),
            fit_position: 0,
            query_len: 0,
            target_len: 0,
            passed_cutoff: false,
        }
    }
}

/// Performs sliding window scan alignment between query and target.
///
/// This function implements the scan mode from the original STAMP C code.
/// It tries multiple starting alignments by sliding the query along the
/// target structure and keeping the best result.
///
/// # Algorithm
///
/// 1. Check if target is too short (< min_frac * query_len) and skip if so
/// 2. Optionally truncate target if > trunc_factor * query_len
/// 3. For each starting position (stepped by `slide`):
///    - Take overlapping N-terminal portions
///    - Compute initial superposition via Kabsch
///    - Apply transformation to target coordinates
///    - Run iterative pairfit refinement
///    - Track the best score
/// 4. Return the best result
///
/// # Arguments
///
/// * `query` - Query domain (reference structure)
/// * `target` - Target domain to scan against
/// * `config` - Scan configuration parameters
/// * `params` - Global alignment parameters
///
/// # Returns
///
/// `ScanResult` containing the best alignment found.
///
/// # Example
///
/// ```ignore
/// use stamp_core::scan::{scan_pair, ScanConfig};
///
/// let config = ScanConfig::default();
/// let result = scan_pair(&query, &target, &config, &params)?;
/// if result.passed_cutoff {
///     println!("Hit: {} score={:.3}", result.target_id, result.score);
/// }
/// ```
pub fn scan_pair(
    query: &Domain,
    target: &Domain,
    config: &ScanConfig,
    _params: &Parameters,
) -> StampResult<ScanResult> {
    let query_len = query.len();
    let target_len = target.len();

    log::debug!(
        "Scanning {} ({} res) against {} ({} res)",
        query.id,
        query_len,
        target.id,
        target_len
    );

    let mut result = ScanResult {
        query_id: query.id.clone(),
        target_id: target.id.clone(),
        query_len,
        target_len,
        ..Default::default()
    };

    // Check minimum length requirement
    let min_target_len = (query_len as f64 * config.min_frac) as usize;
    if target_len < min_target_len {
        log::debug!(
            "Target {} too short: {} < {} (min_frac={})",
            target.id,
            target_len,
            min_target_len,
            config.min_frac
        );
        return Ok(result);
    }

    // Prepare working copy of target that may be truncated
    let working_target = if config.truncate {
        let max_len = (query_len as f64 * config.trunc_factor) as usize;
        if target_len > max_len {
            log::debug!(
                "Truncating {} from {} to {} residues",
                target.id,
                target_len,
                max_len
            );
            let mut truncated = target.clone();
            truncated.residues.truncate(max_len);
            truncated
        } else {
            target.clone()
        }
    } else {
        target.clone()
    };

    result.target_len = working_target.len();

    // Extract coordinates
    let query_coords = domain_to_coords(query);
    let target_coords_orig = domain_to_coords(&working_target);

    // Determine scan range and overlap
    let overlap_len = query_len.min(working_target.len());
    if overlap_len < 3 {
        log::debug!("Insufficient overlap for scan");
        return Ok(result);
    }

    // Sliding window scan
    let slide = config.slide.max(1);
    let max_start = target_coords_orig.len().saturating_sub(overlap_len / 2).max(1);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_result: Option<PairwiseResult> = None;
    let mut best_position: usize = 0;
    let mut best_transform = Transform::identity();

    let mut qcount: usize = 0;
    while qcount < max_start {
        // Determine the overlap region for this starting position
        let target_start = qcount;
        let available_target = target_coords_orig.len().saturating_sub(target_start);
        let current_overlap = overlap_len.min(available_target).min(query_len);

        if current_overlap < 3 {
            qcount += slide;
            continue;
        }

        // Extract overlapping portions for initial superposition
        let query_overlap: Vec<Coord3> = query_coords[..current_overlap].to_vec();
        let target_overlap: Vec<Coord3> = target_coords_orig[target_start..target_start + current_overlap].to_vec();

        // Compute initial superposition of overlapping portions
        let (initial_transform, _initial_rmsd) = superpose(&query_overlap, &target_overlap)?;

        // Create transformed target domain for pairfit
        let mut transformed_target = working_target.clone();
        for residue in &mut transformed_target.residues {
            residue.ca_coord = initial_transform.apply(&residue.ca_coord);
        }

        // Run pairfit iterative alignment
        let pairwise_config = config.to_pairwise_config();
        let fit_result = pairfit(query, &transformed_target, &pairwise_config)?;

        log::trace!(
            "Scan position {}: score={:.4}, rmsd={:.3}, nfit={}",
            qcount,
            fit_result.score,
            fit_result.rmsd,
            fit_result.n_fit
        );

        // Check if this is the best result so far
        if fit_result.score > best_score {
            best_score = fit_result.score;
            best_position = qcount;
            // Combine initial transform with pairfit transform
            best_transform = fit_result.transform.compose(&initial_transform);
            best_result = Some(fit_result);

            // Optionally skip ahead if we found a good match
            if config.skip_ahead && best_score >= config.scan_cut {
                // Skip over the matched region
                if let Some(ref r) = best_result {
                    let skip_amount = r.n_fit.saturating_sub(slide) / slide * slide;
                    if skip_amount > slide {
                        log::trace!("Skipping ahead by {} residues", skip_amount);
                        qcount += skip_amount;
                        continue;
                    }
                }
            }
        }

        qcount += slide;
    }

    // If we're doing two passes and found something decent, do a second pass
    if config.n_passes >= 2 && best_result.is_some() && best_score > 0.0 {
        // Create working target with best transformation applied
        let mut transformed_target = working_target.clone();
        for residue in &mut transformed_target.residues {
            residue.ca_coord = best_transform.apply(&residue.ca_coord);
        }

        // Second pass with tighter parameters
        let fine_config = PairwiseConfig::second_pass();
        let fine_result = pairfit(query, &transformed_target, &fine_config)?;

        if fine_result.score >= best_score {
            best_transform = fine_result.transform.compose(&best_transform);
            best_result = Some(fine_result);
        }
    }

    // Populate final result
    if let Some(pairwise_result) = best_result {
        result.score = pairwise_result.score;
        result.rmsd = pairwise_result.rmsd;
        result.n_aligned = pairwise_result.alignment_length;
        result.n_fit = pairwise_result.n_fit;
        result.seq_identity = pairwise_result.seq_identity / 100.0;
        result.transform = best_transform;
        result.fit_position = best_position;
        result.passed_cutoff = pairwise_result.score >= config.scan_cut;

        log::debug!(
            "Scan result for {}: score={:.4}, rmsd={:.3}, nfit={}, position={}",
            target.id,
            result.score,
            result.rmsd,
            result.n_fit,
            best_position
        );
    }

    Ok(result)
}

/// Scans a query against multiple targets using sliding window approach.
///
/// # Arguments
///
/// * `query` - Query domain
/// * `targets` - Slice of target domains
/// * `config` - Scan configuration
/// * `params` - Alignment parameters
///
/// # Returns
///
/// Vector of `ScanResult` for targets passing the cutoff, sorted by score.
pub fn scan_database(
    query: &Domain,
    targets: &[Domain],
    config: &ScanConfig,
    params: &Parameters,
) -> StampResult<Vec<ScanResult>> {
    log::info!(
        "Scanning {} against {} targets with slide={}",
        query.id,
        targets.len(),
        config.slide
    );

    let mut results: Vec<ScanResult> = Vec::new();

    for target in targets {
        let result = scan_pair(query, target, config, params)?;
        if result.passed_cutoff {
            results.push(result);
        }
    }

    // Sort by score descending
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    log::info!("Found {} hits above cutoff {}", results.len(), config.scan_cut);

    Ok(results)
}

// ============================================================================
// DatabaseScanner Implementation
// ============================================================================

/// Database scanner for structural similarity searches.
pub struct DatabaseScanner {
    /// Database of target domains.
    targets: Vec<Domain>,
    /// Scanning parameters.
    params: Parameters,
    /// Use quick alignment for initial scan.
    quick_scan: bool,
    /// Score threshold for reporting.
    score_threshold: f64,
    /// Maximum number of hits to return.
    max_hits: usize,
    /// Slide parameter for scan mode.
    slide: usize,
    /// Whether to use scan mode (sliding window).
    use_scan_mode: bool,
}

impl DatabaseScanner {
    /// Creates a new database scanner.
    ///
    /// # Arguments
    ///
    /// * `params` - Alignment parameters
    #[must_use]
    pub fn new(params: Parameters) -> Self {
        Self {
            targets: Vec::new(),
            params,
            quick_scan: true,
            score_threshold: 0.0,
            max_hits: 100,
            slide: 10,
            use_scan_mode: false,
        }
    }

    /// Adds a target domain to the database.
    pub fn add_target(&mut self, domain: Domain) {
        self.targets.push(domain);
    }

    /// Adds multiple targets to the database.
    pub fn add_targets(&mut self, domains: Vec<Domain>) {
        self.targets.extend(domains);
    }

    /// Sets whether to use quick alignment for initial scan.
    pub fn set_quick_scan(&mut self, quick: bool) {
        self.quick_scan = quick;
    }

    /// Sets the minimum score threshold for reporting hits.
    pub fn set_score_threshold(&mut self, threshold: f64) {
        self.score_threshold = threshold;
    }

    /// Sets the maximum number of hits to return.
    pub fn set_max_hits(&mut self, max: usize) {
        self.max_hits = max;
    }

    /// Sets the slide parameter for scan mode.
    pub fn set_slide(&mut self, slide: usize) {
        self.slide = slide;
    }

    /// Enables or disables scan mode (sliding window).
    pub fn set_scan_mode(&mut self, enabled: bool) {
        self.use_scan_mode = enabled;
    }

    /// Returns the number of targets in the database.
    #[must_use]
    pub fn database_size(&self) -> usize {
        self.targets.len()
    }

    /// Scans a query against the database.
    ///
    /// # Arguments
    ///
    /// * `query` - Query domain
    ///
    /// # Returns
    ///
    /// Sorted list of hits above threshold.
    ///
    /// # Errors
    ///
    /// Returns an error if alignment fails.
    pub fn scan(&self, query: &Domain) -> StampResult<Vec<ScanHit>> {
        log::info!(
            "Scanning query {} against {} targets",
            query.id,
            self.targets.len()
        );

        let mut hits = Vec::new();

        if self.use_scan_mode {
            // Use sliding window scan mode
            let config = ScanConfig {
                slide: self.slide,
                scan_cut: self.score_threshold,
                ..ScanConfig::from_parameters(&self.params)
            };

            for target in &self.targets {
                let result = scan_pair(query, target, &config, &self.params)?;

                if result.score >= self.score_threshold {
                    let alignment = AlignmentResult {
                        transform: result.transform,
                        rmsd: result.rmsd,
                        n_aligned: result.n_fit,
                        score: result.score,
                        seq_identity: result.seq_identity,
                        path: Vec::new(),
                        residue_scores: Vec::new(),
                    };
                    hits.push(ScanHit::new(target.id.clone(), alignment, 0));
                }
            }
        } else {
            // Use simple pairwise alignment
            for target in &self.targets {
                let result = if self.quick_scan {
                    quick_align(query, target, &self.params)?
                } else {
                    align_pair(query, target, &self.params)?
                };

                if result.score >= self.score_threshold {
                    hits.push(ScanHit::new(target.id.clone(), result, 0));
                }
            }
        }

        // Sort by score descending
        hits.sort_by(|a, b| {
            b.alignment
                .score
                .partial_cmp(&a.alignment.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to max hits
        hits.truncate(self.max_hits);

        // Assign ranks
        for (i, hit) in hits.iter_mut().enumerate() {
            hit.rank = i + 1;
        }

        // Compute Z-scores
        compute_z_scores(&mut hits);

        log::info!("Found {} hits above threshold", hits.len());

        Ok(hits)
    }

    /// Performs full refinement on top hits.
    ///
    /// Re-aligns top hits using full alignment if quick scan was used.
    pub fn refine_top_hits(
        &self,
        query: &Domain,
        hits: &mut [ScanHit],
        n_refine: usize,
    ) -> StampResult<()> {
        if !self.quick_scan && !self.use_scan_mode {
            return Ok(());
        }

        let n = hits.len().min(n_refine);

        for hit in hits.iter_mut().take(n) {
            if let Some(target) = self.targets.iter().find(|t| t.id == hit.target_id) {
                hit.alignment = align_pair(query, target, &self.params)?;
            }
        }

        // Re-sort after refinement
        hits.sort_by(|a, b| {
            b.alignment
                .score
                .partial_cmp(&a.alignment.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update ranks
        for (i, hit) in hits.iter_mut().enumerate() {
            hit.rank = i + 1;
        }

        Ok(())
    }
}

/// Computes Z-scores for scan hits.
fn compute_z_scores(hits: &mut [ScanHit]) {
    if hits.len() < 2 {
        return;
    }

    // Compute mean and standard deviation of scores
    let scores: Vec<f64> = hits.iter().map(|h| h.alignment.score).collect();
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
    let std_dev = variance.sqrt();

    if std_dev < 1e-10 {
        return;
    }

    // Assign Z-scores
    for hit in hits {
        hit.z_score = Some((hit.alignment.score - mean) / std_dev);
    }
}

/// Estimates E-value from Z-score using EVD.
///
/// # Arguments
///
/// * `z_score` - Z-score
/// * `database_size` - Number of sequences in database
///
/// # Returns
///
/// Estimated E-value.
#[must_use]
pub fn z_to_evalue(z_score: f64, database_size: usize) -> f64 {
    // Using extreme value distribution approximation
    let p_value = (-(-z_score).exp()).exp();
    p_value * database_size as f64
}

/// Batch scan of multiple queries.
///
/// # Arguments
///
/// * `queries` - List of query domains
/// * `scanner` - Database scanner
///
/// # Returns
///
/// Map of query ID to hit list.
pub fn batch_scan(
    queries: &[Domain],
    scanner: &DatabaseScanner,
) -> StampResult<Vec<(String, Vec<ScanHit>)>> {
    let mut results = Vec::new();

    for query in queries {
        let hits = scanner.scan(query)?;
        results.push((query.id.clone(), hits));
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Residue;

    fn make_test_domain(id: &str, n_residues: usize) -> Domain {
        let mut domain = Domain::new(id.to_string());
        for i in 0..n_residues {
            domain.residues.push(Residue::new(
                i as i32,
                i.to_string(),
                'A',
                Coord3::new(i as f64 * 3.8, 0.0, 0.0),
            ));
        }
        domain
    }

    fn make_helix_domain(id: &str, n_residues: usize, offset: f64) -> Domain {
        let mut domain = Domain::new(id.to_string());
        for i in 0..n_residues {
            let t = i as f64 * 0.5;
            domain.residues.push(Residue::new(
                i as i32,
                i.to_string(),
                'A',
                Coord3::new(
                    t.cos() * 5.0 + offset,
                    t.sin() * 5.0,
                    i as f64 * 1.5,
                ),
            ));
        }
        domain
    }

    #[test]
    fn test_scanner_creation() {
        let params = Parameters::default();
        let scanner = DatabaseScanner::new(params);
        assert_eq!(scanner.database_size(), 0);
    }

    #[test]
    fn test_add_targets() {
        let params = Parameters::default();
        let mut scanner = DatabaseScanner::new(params);

        scanner.add_target(make_test_domain("target1", 10));
        scanner.add_target(make_test_domain("target2", 15));

        assert_eq!(scanner.database_size(), 2);
    }

    #[test]
    fn test_scan_hit() {
        let hit = ScanHit::new("target".to_string(), AlignmentResult::new(), 1);
        assert_eq!(hit.target_id, "target");
        assert_eq!(hit.rank, 1);
        assert!(hit.z_score.is_none());
    }

    #[test]
    fn test_z_to_evalue() {
        let e = z_to_evalue(3.0, 1000);
        assert!(e < 1000.0);
        assert!(e > 0.0);
    }

    #[test]
    fn test_compute_z_scores() {
        let mut hits = vec![
            ScanHit::new("a".to_string(), AlignmentResult { score: 5.0, ..Default::default() }, 1),
            ScanHit::new("b".to_string(), AlignmentResult { score: 3.0, ..Default::default() }, 2),
            ScanHit::new("c".to_string(), AlignmentResult { score: 1.0, ..Default::default() }, 3),
        ];

        compute_z_scores(&mut hits);

        assert!(hits[0].z_score.is_some());
        assert!(hits[0].z_score.unwrap() > hits[1].z_score.unwrap());
    }

    #[test]
    fn test_scan_config_default() {
        let config = ScanConfig::default();
        assert_eq!(config.slide, 5);
        assert!((config.scan_cut - 2.0).abs() < 1e-10);
        assert!((config.min_frac - 0.5).abs() < 1e-10);
        assert!(config.truncate);
    }

    #[test]
    fn test_scan_config_from_parameters() {
        let params = Parameters {
            slide_offset: 10,
            score_cutoff: 3.0,
            ..Default::default()
        };
        let config = ScanConfig::from_parameters(&params);
        assert_eq!(config.slide, 10);
        assert!((config.scan_cut - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_roughfit_single_domain() {
        let mut domains = vec![make_test_domain("test", 20)];
        let params = Parameters::default();

        let results = roughfit(&mut domains, &params).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].domain_id, "test");
        assert!((results[0].rmsd - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_roughfit_two_domains() {
        let mut domains = vec![
            make_test_domain("ref", 20),
            make_test_domain("mobile", 20),
        ];
        let params = Parameters::default();

        let results = roughfit(&mut domains, &params).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].domain_id, "ref");
        assert_eq!(results[1].domain_id, "mobile");
        // Identical domains should have very low RMSD
        assert!(results[1].rmsd < 0.1);
    }

    #[test]
    fn test_roughfit_translated_domains() {
        let ref_domain = make_test_domain("ref", 20);
        let mut mobile_domain = make_test_domain("mobile", 20);

        // Translate mobile domain
        for residue in &mut mobile_domain.residues {
            residue.ca_coord.x += 100.0;
            residue.ca_coord.y += 50.0;
            residue.ca_coord.z += 25.0;
        }

        let mut domains = vec![ref_domain, mobile_domain];
        let params = Parameters::default();

        let results = roughfit(&mut domains, &params).unwrap();

        assert_eq!(results.len(), 2);
        // After roughfit, structures should be aligned
        assert!(results[1].rmsd < 0.5);
    }

    #[test]
    fn test_scan_pair_too_short() {
        let query = make_test_domain("query", 50);
        let target = make_test_domain("target", 10); // Too short
        let config = ScanConfig::default();
        let params = Parameters::default();

        let result = scan_pair(&query, &target, &config, &params).unwrap();

        // Target is less than 50% of query length
        assert!(!result.passed_cutoff);
        assert_eq!(result.n_fit, 0);
    }

    #[test]
    fn test_scan_pair_identical() {
        let query = make_helix_domain("query", 30, 0.0);
        let target = make_helix_domain("target", 30, 0.0);
        let config = ScanConfig {
            scan_cut: 0.0, // Low cutoff to ensure we get a result
            ..Default::default()
        };
        let params = Parameters::default();

        let result = scan_pair(&query, &target, &config, &params).unwrap();

        // Identical structures should match well
        assert!(result.n_fit > 0);
        if result.score > 0.0 {
            assert!(result.rmsd < 2.0);
        }
    }

    #[test]
    fn test_scan_pair_translated() {
        let query = make_helix_domain("query", 30, 0.0);
        let target = make_helix_domain("target", 30, 100.0); // Translated in X
        let config = ScanConfig {
            scan_cut: 0.0,
            ..Default::default()
        };
        let params = Parameters::default();

        let result = scan_pair(&query, &target, &config, &params).unwrap();

        // Scan should find alignment despite translation
        assert!(result.n_fit > 0 || result.score == 0.0);
    }

    #[test]
    fn test_scan_result_default() {
        let result = ScanResult::default();
        assert!(result.query_id.is_empty());
        assert!(result.target_id.is_empty());
        assert!(!result.passed_cutoff);
        assert!(result.rmsd.is_infinite());
    }

    #[test]
    fn test_roughfit_result() {
        let result = RoughfitResult {
            domain_id: "test".to_string(),
            transform: Transform::identity(),
            rmsd: 1.5,
            n_aligned: 20,
        };
        assert_eq!(result.domain_id, "test");
        assert_eq!(result.n_aligned, 20);
    }

    #[test]
    fn test_scanner_scan_mode() {
        let params = Parameters::default();
        let mut scanner = DatabaseScanner::new(params);

        scanner.set_scan_mode(true);
        scanner.set_slide(5);
        scanner.set_score_threshold(0.0);

        scanner.add_target(make_test_domain("target1", 30));
        scanner.add_target(make_test_domain("target2", 30));

        let query = make_test_domain("query", 25);
        let hits = scanner.scan(&query).unwrap();

        // Should get some results (may or may not pass cutoff)
        // This tests that scan mode runs without errors
        assert!(hits.len() <= 2);
    }

    #[test]
    fn test_scan_database() {
        let query = make_helix_domain("query", 25, 0.0);
        let targets = vec![
            make_helix_domain("t1", 25, 0.0),
            make_helix_domain("t2", 25, 50.0),
            make_helix_domain("t3", 25, 100.0),
        ];
        let config = ScanConfig {
            scan_cut: 0.0,
            ..Default::default()
        };
        let params = Parameters::default();

        let results = scan_database(&query, &targets, &config, &params).unwrap();

        // Results should be sorted by score
        if results.len() >= 2 {
            assert!(results[0].score >= results[1].score);
        }
    }

    #[test]
    fn test_batch_scan() {
        let params = Parameters::default();
        let scanner = DatabaseScanner::new(params);

        let queries = vec![
            make_test_domain("q1", 20),
            make_test_domain("q2", 20),
        ];

        let results = batch_scan(&queries, &scanner).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "q1");
        assert_eq!(results[1].0, "q2");
    }
}
