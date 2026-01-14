//! Hierarchical clustering algorithms.
//!
//! This module implements clustering methods for grouping protein structures
//! based on pairwise similarity scores. Includes both UPGMA and Neighbor-Joining
//! algorithms for guide tree construction.

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
        let mut nodes: Vec<Option<TreeNode>> =
            (0..self.n).map(|i| Some(TreeNode::leaf(i))).collect();
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
        nodes
            .into_iter()
            .flatten()
            .next()
            .expect("Root should exist")
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
                    ((size_i as f64 + size_k) * d_ik + (size_j as f64 + size_k) * d_jk
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

// =============================================================================
// Neighbor-Joining Algorithm
// =============================================================================

/// Neighbor-Joining tree building algorithm.
///
/// Unlike UPGMA, Neighbor-Joining does not assume a molecular clock
/// (equal rates of evolution), making it more appropriate for
/// phylogenetically diverse sets of structures.
///
/// # Algorithm
///
/// 1. Calculate Q-matrix from distance matrix
/// 2. Find pair (i,j) with minimum Q(i,j)
/// 3. Join i and j to create new node
/// 4. Calculate distances from new node to all others
/// 5. Repeat until only 2 nodes remain
///
/// # References
///
/// Saitou N, Nei M (1987). "The neighbor-joining method: a new method
/// for reconstructing phylogenetic trees". Molecular Biology and Evolution.
#[derive(Debug)]
pub struct NeighborJoining {
    /// Number of items.
    n: usize,
    /// Distance matrix.
    distances: Vec<Vec<f64>>,
    /// Sum of distances for each node (for Q-matrix calculation).
    row_sums: Vec<f64>,
    /// Nodes created during tree building.
    nodes: Vec<Option<TreeNode>>,
    /// Whether each index is still active.
    active: Vec<bool>,
    /// Number of currently active nodes.
    n_active: usize,
}

impl NeighborJoining {
    /// Creates a new Neighbor-Joining instance.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of items to cluster
    #[must_use]
    pub fn new(n: usize) -> Self {
        let distances = vec![vec![0.0; n]; n];
        let row_sums = vec![0.0; n];
        let nodes: Vec<Option<TreeNode>> = (0..n).map(|i| Some(TreeNode::leaf(i))).collect();
        let active = vec![true; n];

        Self {
            n,
            distances,
            row_sums,
            nodes,
            active,
            n_active: n,
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

    /// Computes row sums for Q-matrix calculation.
    fn compute_row_sums(&mut self) {
        for i in 0..self.n {
            if self.active[i] {
                self.row_sums[i] = (0..self.n)
                    .filter(|&j| self.active[j])
                    .map(|j| self.distances[i][j])
                    .sum();
            }
        }
    }

    /// Computes Q-matrix value for pair (i, j).
    ///
    /// Q(i,j) = (n-2) * d(i,j) - r_i - r_j
    /// where r_i is the sum of distances from i to all other active nodes.
    #[inline]
    fn q_value(&self, i: usize, j: usize) -> f64 {
        if self.n_active <= 2 {
            // When only 2 nodes remain, just use the distance
            return self.distances[i][j];
        }
        (self.n_active as f64 - 2.0) * self.distances[i][j] - self.row_sums[i] - self.row_sums[j]
    }

    /// Finds the pair with minimum Q-value among active nodes.
    fn find_min_q_pair(&self) -> (usize, usize, f64) {
        let mut min_q = f64::INFINITY;
        let mut min_i = 0;
        let mut min_j = 0;

        for i in 0..self.n {
            if !self.active[i] {
                continue;
            }
            for j in i + 1..self.n {
                if !self.active[j] {
                    continue;
                }
                let q = self.q_value(i, j);
                if q < min_q {
                    min_q = q;
                    min_i = i;
                    min_j = j;
                }
            }
        }

        (min_i, min_j, min_q)
    }

    /// Calculates branch lengths for joining nodes i and j.
    ///
    /// Returns (length_i, length_j) - the branch lengths from the new node
    /// to i and j respectively.
    fn calculate_branch_lengths(&self, i: usize, j: usize) -> (f64, f64) {
        let d_ij = self.distances[i][j];

        if self.n_active <= 2 {
            // For final join, split distance equally
            return (d_ij / 2.0, d_ij / 2.0);
        }

        // limb_i = d(i,j)/2 + (r_i - r_j) / (2*(n-2))
        let n_minus_2 = (self.n_active - 2) as f64;
        let limb_i = d_ij / 2.0 + (self.row_sums[i] - self.row_sums[j]) / (2.0 * n_minus_2);
        let limb_j = d_ij - limb_i;

        // Ensure non-negative branch lengths
        (limb_i.max(0.0), limb_j.max(0.0))
    }

    /// Updates distances after joining nodes i and j into new node u.
    ///
    /// For all other nodes k: d(u,k) = (d(i,k) + d(j,k) - d(i,j)) / 2
    fn update_distances(&mut self, i: usize, j: usize) {
        let d_ij = self.distances[i][j];

        for k in 0..self.n {
            if !self.active[k] || k == i || k == j {
                continue;
            }

            // New distance from the joined node to k
            let new_dist = (self.distances[i][k] + self.distances[j][k] - d_ij) / 2.0;

            // Store at position i (which will represent the new node)
            self.distances[i][k] = new_dist.max(0.0);
            self.distances[k][i] = new_dist.max(0.0);
        }

        // Reset distance to self
        self.distances[i][i] = 0.0;
    }

    /// Performs Neighbor-Joining clustering and returns the tree.
    #[must_use]
    pub fn cluster(&mut self) -> TreeNode {
        if self.n == 0 {
            panic!("Cannot cluster empty set");
        }
        if self.n == 1 {
            return self.nodes[0].take().expect("Node should exist");
        }

        // Main NJ loop
        while self.n_active > 2 {
            // Compute row sums for Q-matrix
            self.compute_row_sums();

            // Find minimum Q pair
            let (min_i, min_j, _) = self.find_min_q_pair();

            // Calculate branch lengths
            let (limb_i, limb_j) = self.calculate_branch_lengths(min_i, min_j);

            // Get the nodes to merge
            let left = self.nodes[min_i].take().expect("Node should exist");
            let right = self.nodes[min_j].take().expect("Node should exist");

            // Create new internal node with proper branch lengths
            // We use the sum of branch lengths as the height
            let new_node = TreeNode::Internal {
                left: Box::new(left),
                right: Box::new(right),
                height: limb_i + limb_j,
            };

            // Update distances
            self.update_distances(min_i, min_j);

            // Mark j as inactive, keep merged cluster at i
            self.active[min_j] = false;
            self.nodes[min_i] = Some(new_node);
            self.n_active -= 1;
        }

        // Final join of the last two nodes
        let remaining: Vec<usize> = (0..self.n).filter(|&i| self.active[i]).collect();
        assert_eq!(remaining.len(), 2);

        let i = remaining[0];
        let j = remaining[1];
        let (limb_i, limb_j) = self.calculate_branch_lengths(i, j);

        let left = self.nodes[i].take().expect("Node should exist");
        let right = self.nodes[j].take().expect("Node should exist");

        TreeNode::Internal {
            left: Box::new(left),
            right: Box::new(right),
            height: limb_i + limb_j,
        }
    }
}

/// Tree building method selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeMethod {
    /// UPGMA (Unweighted Pair Group Method with Arithmetic Mean).
    /// Assumes equal evolutionary rates (molecular clock).
    Upgma,
    /// Neighbor-Joining.
    /// Does not assume equal evolutionary rates.
    NeighborJoin,
}

impl Default for TreeMethod {
    fn default() -> Self {
        TreeMethod::Upgma
    }
}

/// Builds a guide tree from a distance matrix using the specified method.
///
/// # Arguments
///
/// * `distances` - Distance matrix (n x n)
/// * `method` - Tree building method (UPGMA or Neighbor-Joining)
///
/// # Returns
///
/// Root node of the constructed tree.
#[must_use]
pub fn build_tree(distances: &[Vec<f64>], method: TreeMethod) -> TreeNode {
    let n = distances.len();

    match method {
        TreeMethod::Upgma => {
            let mut hc = HierarchicalClustering::new(n, LinkageMethod::Average);
            for i in 0..n {
                for j in i + 1..n {
                    hc.set_distance(i, j, distances[i][j]);
                }
            }
            hc.cluster()
        }
        TreeMethod::NeighborJoin => {
            let mut nj = NeighborJoining::new(n);
            for i in 0..n {
                for j in i + 1..n {
                    nj.set_distance(i, j, distances[i][j]);
                }
            }
            nj.cluster()
        }
    }
}

/// Builds a guide tree from matrix entries using the specified method.
///
/// # Arguments
///
/// * `n` - Number of items
/// * `entries` - List of distance matrix entries
/// * `method` - Tree building method
#[must_use]
pub fn build_tree_from_entries(n: usize, entries: &[MatrixEntry], method: TreeMethod) -> TreeNode {
    match method {
        TreeMethod::Upgma => {
            let mut hc = HierarchicalClustering::new(n, LinkageMethod::Average);
            hc.load_distances(entries);
            hc.cluster()
        }
        TreeMethod::NeighborJoin => {
            let mut nj = NeighborJoining::new(n);
            nj.load_distances(entries);
            nj.cluster()
        }
    }
}

/// Cuts a tree at a given height threshold.
///
/// Returns a list of clusters (leaf indices) at the specified height.
#[must_use]
pub fn cut_tree(tree: &TreeNode, threshold: f64) -> Vec<Vec<usize>> {
    match tree {
        TreeNode::Leaf(idx) => vec![vec![*idx]],
        TreeNode::Internal {
            left,
            right,
            height,
        } => {
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
    if let TreeNode::Internal {
        left,
        right,
        height,
    } = node
    {
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

    // Neighbor-Joining tests
    #[test]
    fn test_neighbor_joining_basic() {
        // Simple 4-taxon case
        let mut nj = NeighborJoining::new(4);

        // Set up a distance matrix
        // Based on Wikipedia NJ example
        nj.set_distance(0, 1, 7.0);
        nj.set_distance(0, 2, 11.0);
        nj.set_distance(0, 3, 14.0);
        nj.set_distance(1, 2, 6.0);
        nj.set_distance(1, 3, 9.0);
        nj.set_distance(2, 3, 7.0);

        let tree = nj.cluster();
        let leaves = tree.leaves();
        assert_eq!(leaves.len(), 4);
    }

    #[test]
    fn test_neighbor_joining_three_taxa() {
        let mut nj = NeighborJoining::new(3);
        nj.set_distance(0, 1, 2.0);
        nj.set_distance(0, 2, 4.0);
        nj.set_distance(1, 2, 4.0);

        let tree = nj.cluster();
        let leaves = tree.leaves();
        assert_eq!(leaves.len(), 3);
    }

    #[test]
    fn test_neighbor_joining_two_taxa() {
        let mut nj = NeighborJoining::new(2);
        nj.set_distance(0, 1, 5.0);

        let tree = nj.cluster();
        let leaves = tree.leaves();
        assert_eq!(leaves.len(), 2);

        // Height should be the total distance
        if let TreeNode::Internal { height, .. } = tree {
            assert!((height - 5.0).abs() < 1e-10);
        } else {
            panic!("Expected internal node");
        }
    }

    #[test]
    fn test_build_tree_upgma() {
        let distances = vec![
            vec![0.0, 1.0, 3.0],
            vec![1.0, 0.0, 2.0],
            vec![3.0, 2.0, 0.0],
        ];
        let tree = build_tree(&distances, TreeMethod::Upgma);
        assert_eq!(tree.leaves().len(), 3);
    }

    #[test]
    fn test_build_tree_nj() {
        let distances = vec![
            vec![0.0, 1.0, 3.0],
            vec![1.0, 0.0, 2.0],
            vec![3.0, 2.0, 0.0],
        ];
        let tree = build_tree(&distances, TreeMethod::NeighborJoin);
        assert_eq!(tree.leaves().len(), 3);
    }

    #[test]
    fn test_tree_method_default() {
        assert_eq!(TreeMethod::default(), TreeMethod::Upgma);
    }

    #[test]
    fn test_nj_ultrametric_distances() {
        // For ultrametric distances (molecular clock), NJ and UPGMA should
        // produce similar trees
        let mut nj = NeighborJoining::new(4);
        // Ultrametric: d(a,b) <= max(d(a,c), d(b,c)) for all a,b,c
        nj.set_distance(0, 1, 2.0);
        nj.set_distance(0, 2, 4.0);
        nj.set_distance(0, 3, 4.0);
        nj.set_distance(1, 2, 4.0);
        nj.set_distance(1, 3, 4.0);
        nj.set_distance(2, 3, 2.0);

        let tree = nj.cluster();
        let leaves = tree.leaves();
        assert_eq!(leaves.len(), 4);
    }

    #[test]
    fn test_build_tree_from_entries() {
        let entries = vec![
            MatrixEntry {
                i: 0,
                j: 1,
                value: 1.0,
            },
            MatrixEntry {
                i: 0,
                j: 2,
                value: 3.0,
            },
            MatrixEntry {
                i: 1,
                j: 2,
                value: 2.0,
            },
        ];

        let tree_upgma = build_tree_from_entries(3, &entries, TreeMethod::Upgma);
        let tree_nj = build_tree_from_entries(3, &entries, TreeMethod::NeighborJoin);

        assert_eq!(tree_upgma.leaves().len(), 3);
        assert_eq!(tree_nj.leaves().len(), 3);
    }
}
