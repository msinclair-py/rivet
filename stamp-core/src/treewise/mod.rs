//! Tree-guided multiple structure alignment.
//!
//! This module implements progressive multiple alignment following
//! a guide tree derived from pairwise similarities.

use crate::clustering::{HierarchicalClustering, LinkageMethod};
use crate::pairwise::{all_vs_all, similarity_matrix};
use crate::types::{AlignmentResult, Domain, Parameters, StampResult, Transform, TreeNode};

/// Result of multiple structure alignment.
#[derive(Debug, Clone)]
pub struct MultipleAlignmentResult {
    /// Transformations to apply to each domain.
    pub transforms: Vec<Transform>,
    /// Overall alignment quality score.
    pub score: f64,
    /// Average pairwise RMSD.
    pub avg_rmsd: f64,
    /// Core positions (columns with all structures aligned).
    pub core_positions: Vec<usize>,
    /// Alignment columns: each column is vec of Option<residue_index> per domain.
    pub columns: Vec<Vec<Option<usize>>>,
}

impl MultipleAlignmentResult {
    /// Creates a new empty result.
    #[must_use]
    pub fn new(n_domains: usize) -> Self {
        Self {
            transforms: vec![Transform::identity(); n_domains],
            score: 0.0,
            avg_rmsd: 0.0,
            core_positions: Vec::new(),
            columns: Vec::new(),
        }
    }

    /// Returns the number of alignment columns.
    #[must_use]
    pub fn n_columns(&self) -> usize {
        self.columns.len()
    }

    /// Returns the number of core positions.
    #[must_use]
    pub fn n_core(&self) -> usize {
        self.core_positions.len()
    }
}

/// Performs tree-guided multiple structure alignment.
///
/// # Arguments
///
/// * `domains` - List of domains to align
/// * `params` - Alignment parameters
///
/// # Returns
///
/// Multiple alignment result with transformations and alignment columns.
///
/// # Errors
///
/// Returns an error if pairwise alignments fail.
pub fn multiple_align(
    domains: &[Domain],
    params: &Parameters,
) -> StampResult<MultipleAlignmentResult> {
    let n = domains.len();

    if n == 0 {
        return Ok(MultipleAlignmentResult::new(0));
    }

    if n == 1 {
        let mut result = MultipleAlignmentResult::new(1);
        result.columns = (0..domains[0].len()).map(|i| vec![Some(i)]).collect();
        result.core_positions = (0..domains[0].len()).collect();
        return Ok(result);
    }

    log::info!("Performing multiple alignment of {} structures", n);

    // Step 1: All-vs-all pairwise alignment
    log::debug!("Computing pairwise alignments");
    let pairwise_results = all_vs_all(domains, params)?;
    let similarities = similarity_matrix(&pairwise_results);

    // Step 2: Build guide tree using UPGMA
    log::debug!("Building guide tree");
    let tree = build_guide_tree(&similarities);

    // Step 3: Progressive alignment following tree
    log::debug!("Performing progressive alignment");
    let result = progressive_align(domains, &pairwise_results, &tree, params)?;

    log::info!(
        "Multiple alignment complete: {} columns, {} core positions",
        result.n_columns(),
        result.n_core()
    );

    Ok(result)
}

/// Builds a guide tree from similarity matrix using UPGMA.
fn build_guide_tree(similarities: &[Vec<f64>]) -> TreeNode {
    let n = similarities.len();

    // Convert similarities to distances
    let max_sim = similarities
        .iter()
        .flat_map(|row| row.iter())
        .cloned()
        .fold(0.0f64, f64::max);

    let mut clustering = HierarchicalClustering::new(n, LinkageMethod::Average);

    for i in 0..n {
        for j in i + 1..n {
            let distance = max_sim - similarities[i][j];
            clustering.set_distance(i, j, distance);
        }
    }

    clustering.cluster()
}

/// Performs progressive alignment following guide tree.
fn progressive_align(
    domains: &[Domain],
    pairwise: &[Vec<AlignmentResult>],
    tree: &TreeNode,
    params: &Parameters,
) -> StampResult<MultipleAlignmentResult> {
    let n = domains.len();
    let mut result = MultipleAlignmentResult::new(n);

    // Recursively align following tree structure
    let (columns, _) = align_tree_node(tree, domains, pairwise, params)?;
    result.columns = columns;

    // Find core positions (all domains present)
    result.core_positions = result
        .columns
        .iter()
        .enumerate()
        .filter_map(|(i, col)| {
            if col.iter().all(|x| x.is_some()) {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    // Compute average RMSD using core positions
    if !result.core_positions.is_empty() {
        let mut sum_rmsd = 0.0;
        let mut count = 0;
        for i in 0..n {
            for j in i + 1..n {
                sum_rmsd += pairwise[i][j].rmsd;
                count += 1;
            }
        }
        if count > 0 {
            result.avg_rmsd = sum_rmsd / count as f64;
        }
    }

    // Compute overall score
    result.score = result.n_core() as f64 / result.n_columns().max(1) as f64;

    Ok(result)
}

/// Recursively aligns tree nodes.
#[allow(clippy::type_complexity)]
fn align_tree_node(
    node: &TreeNode,
    domains: &[Domain],
    pairwise: &[Vec<AlignmentResult>],
    _params: &Parameters,
) -> StampResult<(Vec<Vec<Option<usize>>>, Vec<usize>)> {
    match node {
        TreeNode::Leaf(idx) => {
            // Single domain: each residue is its own column
            let columns: Vec<Vec<Option<usize>>> = (0..domains[*idx].len())
                .map(|i| {
                    let mut col = vec![None; domains.len()];
                    col[*idx] = Some(i);
                    col
                })
                .collect();
            Ok((columns, vec![*idx]))
        }
        TreeNode::Internal { left, right, .. } => {
            // Recursively align subtrees
            let (left_columns, left_members) = align_tree_node(left, domains, pairwise, _params)?;
            let (right_columns, right_members) = align_tree_node(right, domains, pairwise, _params)?;

            // Merge alignments using best pairwise alignment between groups
            let merged = merge_alignments(
                &left_columns,
                &right_columns,
                &left_members,
                &right_members,
                pairwise,
            );

            let mut all_members = left_members;
            all_members.extend(right_members);

            Ok((merged, all_members))
        }
    }
}

/// Merges two alignment blocks based on pairwise alignments.
fn merge_alignments(
    left: &[Vec<Option<usize>>],
    right: &[Vec<Option<usize>>],
    left_members: &[usize],
    right_members: &[usize],
    pairwise: &[Vec<AlignmentResult>],
) -> Vec<Vec<Option<usize>>> {
    // Find best pairwise alignment between groups
    let mut best_path = Vec::new();
    let mut best_score = f64::NEG_INFINITY;

    for &i in left_members {
        for &j in right_members {
            if pairwise[i][j].score > best_score {
                best_score = pairwise[i][j].score;
                best_path = pairwise[i][j].path.clone();
            }
        }
    }

    if best_path.is_empty() {
        // No alignment found, concatenate
        let mut result = left.to_vec();
        result.extend(right.iter().cloned());
        return result;
    }

    // Build merged alignment using path
    let mut result = Vec::new();
    let mut left_idx = 0;
    let mut right_idx = 0;

    for &(li, ri) in &best_path {
        // Add unaligned left columns
        while left_idx < li && left_idx < left.len() {
            result.push(left[left_idx].clone());
            left_idx += 1;
        }

        // Add unaligned right columns
        while right_idx < ri && right_idx < right.len() {
            result.push(right[right_idx].clone());
            right_idx += 1;
        }

        // Merge aligned columns
        if left_idx < left.len() && right_idx < right.len() {
            let mut merged = left[left_idx].clone();
            for (k, val) in right[right_idx].iter().enumerate() {
                if val.is_some() {
                    merged[k] = *val;
                }
            }
            result.push(merged);
            left_idx += 1;
            right_idx += 1;
        }
    }

    // Add remaining columns
    while left_idx < left.len() {
        result.push(left[left_idx].clone());
        left_idx += 1;
    }
    while right_idx < right.len() {
        result.push(right[right_idx].clone());
        right_idx += 1;
    }

    result
}

/// Refines a multiple alignment by iterative superposition.
pub fn refine_alignment(
    domains: &[Domain],
    result: &mut MultipleAlignmentResult,
    _params: &Parameters,
) -> StampResult<()> {
    if domains.is_empty() || result.core_positions.is_empty() {
        return Ok(());
    }

    // Use first domain as reference
    let reference = &domains[0];

    for (i, domain) in domains.iter().enumerate().skip(1) {
        // Extract core coordinates
        let mut ref_coords = Vec::new();
        let mut mob_coords = Vec::new();

        for &col in &result.core_positions {
            if let (Some(ref_idx), Some(mob_idx)) = (result.columns[col][0], result.columns[col][i])
            {
                if let (Some(ref_res), Some(mob_res)) = (
                    reference.residues.get(ref_idx),
                    domain.residues.get(mob_idx),
                ) {
                    ref_coords.push(ref_res.ca_coord);
                    mob_coords.push(mob_res.ca_coord);
                }
            }
        }

        if ref_coords.len() >= 3 {
            let (transform, _) = crate::math::superpose(&ref_coords, &mob_coords)?;
            result.transforms[i] = transform;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Coord3, Residue};

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
    fn test_multiple_align_single() {
        let domain = make_test_domain("test", vec![(0.0, 0.0, 0.0), (3.8, 0.0, 0.0)]);
        let params = Parameters::default();

        let result = multiple_align(&[domain], &params).unwrap();
        assert_eq!(result.n_columns(), 2);
        assert_eq!(result.n_core(), 2);
    }

    #[test]
    fn test_build_guide_tree() {
        let sim = vec![
            vec![1.0, 0.8, 0.5],
            vec![0.8, 1.0, 0.6],
            vec![0.5, 0.6, 1.0],
        ];
        let tree = build_guide_tree(&sim);
        let leaves = tree.leaves();
        assert_eq!(leaves.len(), 3);
    }

    #[test]
    fn test_multiple_alignment_result() {
        let result = MultipleAlignmentResult::new(3);
        assert_eq!(result.transforms.len(), 3);
        assert_eq!(result.n_columns(), 0);
    }
}
