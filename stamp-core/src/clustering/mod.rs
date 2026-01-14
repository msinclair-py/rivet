//! Hierarchical clustering algorithms.
//!
//! This module implements clustering methods for grouping protein structures
//! based on pairwise similarity scores.

use crate::types::{MatrixEntry, TreeNode};

/// Linkage method for hierarchical clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkageMethod {
    /// Single linkage (minimum distance).
    Single,
    /// Complete linkage (maximum distance).
    Complete,
    /// Average linkage (UPGMA).
    Average,
    /// Ward's method (minimum variance).
    Ward,
}

/// Hierarchical clustering using distance matrix.
#[derive(Debug)]
pub struct HierarchicalClustering {
    /// Number of items.
    n: usize,
    /// Distance matrix (lower triangular).
    distances: Vec<Vec<f64>>,
    /// Linkage method.
    linkage: LinkageMethod,
    /// Cluster sizes (for UPGMA).
    sizes: Vec<usize>,
}

impl HierarchicalClustering {
    /// Creates a new clustering instance.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of items to cluster
    /// * `linkage` - Linkage method to use
    #[must_use]
    pub fn new(n: usize, linkage: LinkageMethod) -> Self {
        let distances = vec![vec![0.0; n]; n];
        let sizes = vec![1; n];

        Self {
            n,
            distances,
            linkage,
            sizes,
        }
    }

    /// Sets the distance between two items.
    pub fn set_distance(&mut self, i: usize, j: usize, distance: f64) {
        self.distances[i][j] = distance;
        self.distances[j][i] = distance;
    }

    /// Loads distances from a list of matrix entries.
    pub fn load_distances(&mut self, entries: &[MatrixEntry]) {
        for entry in entries {
            self.set_distance(entry.i, entry.j, entry.value);
        }
    }

    /// Performs hierarchical clustering and returns the tree.
    #[must_use]
    pub fn cluster(&mut self) -> TreeNode {
        // Initialize each item as a leaf node
        let mut nodes: Vec<Option<TreeNode>> = (0..self.n).map(|i| Some(TreeNode::leaf(i))).collect();
        let mut active: Vec<bool> = vec![true; self.n];

        // Merge until one cluster remains
        for _ in 0..self.n - 1 {
            // Find minimum distance pair
            let (min_i, min_j, min_dist) = self.find_min_pair(&active);

            // Get the nodes to merge
            let left = nodes[min_i].take().expect("Node should exist");
            let right = nodes[min_j].take().expect("Node should exist");

            // Create new internal node
            let new_node = TreeNode::internal(left, right, min_dist);

            // Update distances using linkage method
            self.update_distances(min_i, min_j, &active);

            // Mark j as inactive, keep merged cluster at i
            active[min_j] = false;
            nodes[min_i] = Some(new_node);
        }

        // Find and return the root
        nodes.into_iter().flatten().next().expect("Root should exist")
    }

    /// Finds the pair with minimum distance among active clusters.
    fn find_min_pair(&self, active: &[bool]) -> (usize, usize, f64) {
        let mut min_dist = f64::INFINITY;
        let mut min_i = 0;
        let mut min_j = 0;

        for i in 0..self.n {
            if !active[i] {
                continue;
            }
            for j in i + 1..self.n {
                if !active[j] {
                    continue;
                }
                if self.distances[i][j] < min_dist {
                    min_dist = self.distances[i][j];
                    min_i = i;
                    min_j = j;
                }
            }
        }

        (min_i, min_j, min_dist)
    }

    /// Updates distances after merging clusters i and j.
    fn update_distances(&mut self, i: usize, j: usize, active: &[bool]) {
        let size_i = self.sizes[i];
        let size_j = self.sizes[j];
        let new_size = size_i + size_j;

        for k in 0..self.n {
            if !active[k] || k == i || k == j {
                continue;
            }

            let d_ik = self.distances[i][k];
            let d_jk = self.distances[j][k];

            let new_dist = match self.linkage {
                LinkageMethod::Single => d_ik.min(d_jk),
                LinkageMethod::Complete => d_ik.max(d_jk),
                LinkageMethod::Average => {
                    (size_i as f64 * d_ik + size_j as f64 * d_jk) / new_size as f64
                }
                LinkageMethod::Ward => {
                    let size_k = self.sizes[k] as f64;
                    let total = size_i as f64 + size_j as f64 + size_k;
                    let d_ij = self.distances[i][j];
                    ((size_i as f64 + size_k) * d_ik
                        + (size_j as f64 + size_k) * d_jk
                        - size_k * d_ij)
                        / total
                }
            };

            self.distances[i][k] = new_dist;
            self.distances[k][i] = new_dist;
        }

        self.sizes[i] = new_size;
    }

    /// Returns clusters at a given distance threshold.
    #[must_use]
    pub fn clusters_at_threshold(&mut self, threshold: f64) -> Vec<Vec<usize>> {
        let tree = self.cluster();
        cut_tree(&tree, threshold)
    }
}

/// Cuts a tree at a given height threshold.
///
/// Returns a list of clusters (leaf indices) at the specified height.
#[must_use]
pub fn cut_tree(tree: &TreeNode, threshold: f64) -> Vec<Vec<usize>> {
    match tree {
        TreeNode::Leaf(idx) => vec![vec![*idx]],
        TreeNode::Internal { left, right, height } => {
            if *height <= threshold {
                // Keep this cluster together
                vec![tree.leaves()]
            } else {
                // Split into sub-clusters
                let mut clusters = cut_tree(left, threshold);
                clusters.extend(cut_tree(right, threshold));
                clusters
            }
        }
    }
}

/// Converts a similarity matrix to a distance matrix.
///
/// # Arguments
///
/// * `similarities` - Matrix of similarity scores
/// * `max_sim` - Maximum similarity value for conversion
///
/// # Returns
///
/// Distance matrix where distance = max_sim - similarity.
#[must_use]
pub fn similarity_to_distance(similarities: &[Vec<f64>], max_sim: f64) -> Vec<Vec<f64>> {
    similarities
        .iter()
        .map(|row| row.iter().map(|&s| max_sim - s).collect())
        .collect()
}

/// Computes the cophenetic correlation coefficient.
///
/// Measures how well the tree represents the original distances.
#[must_use]
pub fn cophenetic_correlation(original: &[Vec<f64>], tree: &TreeNode) -> f64 {
    let n = original.len();
    if n < 2 {
        return 1.0;
    }

    // Compute cophenetic distances
    let cophenetic = cophenetic_distances(tree, n);

    // Compute correlation
    let mut sum_xy = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut count = 0;

    for i in 0..n {
        for j in i + 1..n {
            let x = original[i][j];
            let y = cophenetic[i][j];
            sum_xy += x * y;
            sum_x += x;
            sum_y += y;
            sum_x2 += x * x;
            sum_y2 += y * y;
            count += 1;
        }
    }

    let n_f = count as f64;
    let numerator = n_f * sum_xy - sum_x * sum_y;
    let denominator = ((n_f * sum_x2 - sum_x * sum_x) * (n_f * sum_y2 - sum_y * sum_y)).sqrt();

    if denominator.abs() < 1e-10 {
        1.0
    } else {
        numerator / denominator
    }
}

/// Computes cophenetic distance matrix from a tree.
fn cophenetic_distances(tree: &TreeNode, n: usize) -> Vec<Vec<f64>> {
    let mut distances = vec![vec![0.0; n]; n];
    compute_cophenetic_recursive(tree, &mut distances, 0.0);
    distances
}

/// Recursively computes cophenetic distances.
fn compute_cophenetic_recursive(node: &TreeNode, distances: &mut [Vec<f64>], _current_height: f64) {
    if let TreeNode::Internal { left, right, height } = node {
        let left_leaves = left.leaves();
        let right_leaves = right.leaves();

        // Set distance between all pairs crossing this node
        for &i in &left_leaves {
            for &j in &right_leaves {
                distances[i][j] = *height;
                distances[j][i] = *height;
            }
        }

        // Recurse
        compute_cophenetic_recursive(left, distances, *height);
        compute_cophenetic_recursive(right, distances, *height);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_linkage() {
        let mut hc = HierarchicalClustering::new(3, LinkageMethod::Single);
        hc.set_distance(0, 1, 1.0);
        hc.set_distance(0, 2, 3.0);
        hc.set_distance(1, 2, 2.0);

        let tree = hc.cluster();
        let leaves = tree.leaves();
        assert_eq!(leaves.len(), 3);
    }

    #[test]
    fn test_complete_linkage() {
        let mut hc = HierarchicalClustering::new(3, LinkageMethod::Complete);
        hc.set_distance(0, 1, 1.0);
        hc.set_distance(0, 2, 3.0);
        hc.set_distance(1, 2, 2.0);

        let tree = hc.cluster();
        let leaves = tree.leaves();
        assert_eq!(leaves.len(), 3);
    }

    #[test]
    fn test_cut_tree() {
        let tree = TreeNode::internal(
            TreeNode::internal(TreeNode::leaf(0), TreeNode::leaf(1), 1.0),
            TreeNode::leaf(2),
            2.0,
        );

        let clusters = cut_tree(&tree, 1.5);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn test_similarity_to_distance() {
        let sim = vec![vec![1.0, 0.8], vec![0.8, 1.0]];
        let dist = similarity_to_distance(&sim, 1.0);
        assert!((dist[0][1] - 0.2).abs() < 1e-10);
    }
}
