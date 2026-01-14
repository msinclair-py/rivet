//! Smith-Waterman dynamic programming and path tracing.
//!
//! This module implements the local alignment algorithm used by STAMP
//! for finding optimal structural correspondences between proteins.
//!
//! # Features
//!
//! - Standard Smith-Waterman local alignment
//! - Corner-cutting optimization for efficiency (reduces matrix cells computed)
//! - Multiple alignment path tracking (not just optimal)
//! - Bitfield path directions for traceback (DIAG=1, HORIZ=2, VERT=4)
//! - Alignment reconstruction from path matrix
//!
//! # References
//!
//! Based on the STAMP implementation by Russell and Barton:
//! - `sw7ccs.c`: Corner-cutting Smith-Waterman
//! - `swstruc.c`: Standard Smith-Waterman for structures
//! - `aliseq.c`: Alignment reconstruction

use crate::types::{Parameters, StampResult};

/// Direction bitmasks for traceback in dynamic programming matrix.
/// These can be OR'd together when multiple directions yield the same score.
pub mod direction {
    /// Diagonal move (match/mismatch) - bit 0.
    pub const DIAG: u8 = 0b001;
    /// Horizontal move (gap in sequence 1) - bit 1.
    pub const HORIZ: u8 = 0b010;
    /// Vertical move (gap in sequence 2) - bit 2.
    pub const VERT: u8 = 0b100;
    /// No direction (start of alignment or zero score).
    pub const NONE: u8 = 0b000;
}

/// Direction for traceback in dynamic programming matrix (enum version).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// No direction (start of alignment).
    None,
    /// Diagonal move (match/mismatch).
    Diagonal,
    /// Horizontal move (gap in sequence 1).
    Horizontal,
    /// Vertical move (gap in sequence 2).
    Vertical,
}

impl Direction {
    /// Converts the direction to its bitmask representation.
    #[must_use]
    pub const fn to_bitmask(self) -> u8 {
        match self {
            Direction::None => direction::NONE,
            Direction::Diagonal => direction::DIAG,
            Direction::Horizontal => direction::HORIZ,
            Direction::Vertical => direction::VERT,
        }
    }

    /// Creates a direction from a bitmask (returns primary direction).
    #[must_use]
    pub const fn from_bitmask(mask: u8) -> Self {
        if mask & direction::DIAG != 0 {
            Direction::Diagonal
        } else if mask & direction::HORIZ != 0 {
            Direction::Horizontal
        } else if mask & direction::VERT != 0 {
            Direction::Vertical
        } else {
            Direction::None
        }
    }
}

/// Coordinate in the alignment matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AlignmentCoord {
    /// Row index (position in first sequence).
    pub i: usize,
    /// Column index (position in second sequence).
    pub j: usize,
}

impl AlignmentCoord {
    /// Creates a new alignment coordinate.
    #[must_use]
    pub const fn new(i: usize, j: usize) -> Self {
        Self { i, j }
    }
}

/// An alignment path with start, end coordinates and score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AlignmentPath {
    /// Start coordinate of the alignment.
    pub start: AlignmentCoord,
    /// End coordinate of the alignment.
    pub end: AlignmentCoord,
    /// Maximum score achieved on this path.
    pub score: i32,
}

impl AlignmentPath {
    /// Creates a new alignment path.
    #[must_use]
    pub const fn new(start: AlignmentCoord, end: AlignmentCoord, score: i32) -> Self {
        Self { start, end, score }
    }

    /// Returns true if the path is valid (has distinct start and end).
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.start.i != self.end.i && self.start.j != self.end.j
    }
}

/// List of alignment paths starting from a particular position.
#[derive(Debug, Clone, Default)]
pub struct PathList {
    /// Collection of paths.
    pub paths: Vec<AlignmentPath>,
}

impl PathList {
    /// Creates an empty path list.
    #[must_use]
    pub fn new() -> Self {
        Self { paths: Vec::new() }
    }

    /// Adds a path if it doesn't duplicate an existing one.
    /// If a path with the same start.j exists, keeps the one with higher score.
    /// Returns true if the path was added.
    pub fn add_if_better(&mut self, path: AlignmentPath) -> bool {
        for existing in &mut self.paths {
            if existing.start.j == path.start.j {
                if path.score > existing.score {
                    *existing = path;
                }
                return false;
            }
        }
        self.paths.push(path);
        true
    }

    /// Returns the number of paths.
    #[must_use]
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    /// Returns true if there are no paths.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }
}

/// Cell in the dynamic programming matrix.
#[derive(Debug, Clone)]
pub struct DpCell {
    /// Score at this cell.
    pub score: f64,
    /// Direction for traceback.
    pub direction: Direction,
}

impl Default for DpCell {
    fn default() -> Self {
        Self {
            score: 0.0,
            direction: Direction::None,
        }
    }
}

/// Internal path state used during DP computation.
#[derive(Debug, Clone, Copy, Default)]
struct PathState {
    /// Current column score (for DP).
    col: i32,
    /// Maximum score seen on this path.
    score: i32,
    /// Start coordinate of the path.
    start: AlignmentCoord,
    /// End coordinate where maximum was achieved.
    end: AlignmentCoord,
}

/// Result of Smith-Waterman alignment.
#[derive(Debug, Clone)]
pub struct SwResult {
    /// Maximum alignment score found.
    pub max_score: i32,
    /// Path matrix storing direction bitmasks.
    pub path_matrix: Vec<Vec<u8>>,
    /// All discovered alignment paths (sorted by score).
    pub paths: Vec<AlignmentPath>,
    /// Total number of paths found.
    pub total_paths: usize,
}

impl SwResult {
    /// Creates a new empty result.
    #[must_use]
    pub fn new(len_a: usize, len_b: usize) -> Self {
        Self {
            max_score: -1,
            path_matrix: vec![vec![0u8; len_a]; len_b],
            paths: Vec::new(),
            total_paths: 0,
        }
    }
}

/// Smith-Waterman local alignment algorithm for structural data.
///
/// This implementation follows the original STAMP algorithm with support for:
/// - Multiple path tracking
/// - Bitfield direction storage for traceback
/// - Corner-cutting optimization
#[derive(Debug)]
pub struct SmithWaterman {
    /// Dynamic programming matrix (for simple API).
    matrix: Vec<Vec<DpCell>>,
    /// Length of first sequence.
    len1: usize,
    /// Length of second sequence.
    len2: usize,
    /// Gap opening penalty.
    gap_open: f64,
    /// Gap extension penalty.
    gap_extend: f64,
}

impl SmithWaterman {
    /// Creates a new Smith-Waterman aligner.
    ///
    /// # Arguments
    ///
    /// * `len1` - Length of first structure
    /// * `len2` - Length of second structure
    /// * `params` - Alignment parameters
    #[must_use]
    pub fn new(len1: usize, len2: usize, params: &Parameters) -> Self {
        let matrix = vec![vec![DpCell::default(); len2 + 1]; len1 + 1];

        Self {
            matrix,
            len1,
            len2,
            gap_open: params.pen_gap,
            gap_extend: params.pen_ext,
        }
    }

    /// Fills the dynamic programming matrix with the given score matrix.
    ///
    /// # Arguments
    ///
    /// * `scores` - Matrix of pairwise similarity scores \[i\]\[j\]
    pub fn fill(&mut self, scores: &[Vec<f64>]) {
        for i in 1..=self.len1 {
            for j in 1..=self.len2 {
                let score_ij = scores[i - 1][j - 1];

                // Diagonal: match/mismatch
                let diag = self.matrix[i - 1][j - 1].score + score_ij;

                // Horizontal: gap in sequence 1
                let horiz = if self.matrix[i][j - 1].direction == Direction::Horizontal {
                    self.matrix[i][j - 1].score - self.gap_extend
                } else {
                    self.matrix[i][j - 1].score - self.gap_open
                };

                // Vertical: gap in sequence 2
                let vert = if self.matrix[i - 1][j].direction == Direction::Vertical {
                    self.matrix[i - 1][j].score - self.gap_extend
                } else {
                    self.matrix[i - 1][j].score - self.gap_open
                };

                // Take maximum (local alignment allows starting fresh with 0)
                let (score, direction) = if diag >= horiz && diag >= vert && diag >= 0.0 {
                    (diag, Direction::Diagonal)
                } else if horiz >= vert && horiz >= 0.0 {
                    (horiz, Direction::Horizontal)
                } else if vert >= 0.0 {
                    (vert, Direction::Vertical)
                } else {
                    (0.0, Direction::None)
                };

                self.matrix[i][j] = DpCell { score, direction };
            }
        }
    }

    /// Traces back from the maximum score to find the optimal path.
    ///
    /// # Returns
    ///
    /// Vector of (i, j) pairs representing aligned positions.
    #[must_use]
    pub fn traceback(&self) -> Vec<(usize, usize)> {
        // Find maximum score position
        let mut max_score = f64::NEG_INFINITY;
        let mut max_i = 0;
        let mut max_j = 0;

        for i in 1..=self.len1 {
            for j in 1..=self.len2 {
                if self.matrix[i][j].score > max_score {
                    max_score = self.matrix[i][j].score;
                    max_i = i;
                    max_j = j;
                }
            }
        }

        // Trace back
        let mut path = Vec::new();
        let mut i = max_i;
        let mut j = max_j;

        while i > 0 && j > 0 && self.matrix[i][j].direction != Direction::None {
            match self.matrix[i][j].direction {
                Direction::Diagonal => {
                    path.push((i - 1, j - 1));
                    i -= 1;
                    j -= 1;
                }
                Direction::Horizontal => {
                    j -= 1;
                }
                Direction::Vertical => {
                    i -= 1;
                }
                Direction::None => break,
            }
        }

        path.reverse();
        path
    }

    /// Returns the maximum alignment score.
    #[must_use]
    pub fn max_score(&self) -> f64 {
        let mut max = f64::NEG_INFINITY;
        for i in 1..=self.len1 {
            for j in 1..=self.len2 {
                if self.matrix[i][j].score > max {
                    max = self.matrix[i][j].score;
                }
            }
        }
        max
    }

    /// Returns the score at a specific position.
    #[must_use]
    pub fn score_at(&self, i: usize, j: usize) -> f64 {
        self.matrix[i][j].score
    }
}

/// Performs standard Smith-Waterman local alignment with multiple path tracking.
///
/// This function implements the `swstruc` algorithm from the original STAMP code,
/// storing path directions in a bitfield matrix and tracking all significant paths.
///
/// # Arguments
///
/// * `len_a` - Length of first sequence (+ padding used by original)
/// * `len_b` - Length of second sequence (+ padding used by original)
/// * `prob` - Probability/similarity matrix \[i\]\[j\] (integer scores)
/// * `penalty` - Gap penalty (typically 0 for structural alignment)
/// * `min_score` - Minimum score for a path to be recorded
///
/// # Returns
///
/// Result containing the maximum score, path matrix, and all discovered paths.
///
/// # Example
///
/// ```ignore
/// use stamp_core::alignment::smith_waterman;
///
/// let prob = vec![vec![5, 0, 0], vec![0, 5, 0], vec![0, 0, 5]];
/// let result = smith_waterman(4, 4, &prob, 0, 1);
/// assert!(result.max_score > 0);
/// ```
#[must_use]
pub fn smith_waterman(
    len_a: usize,
    len_b: usize,
    prob: &[Vec<i32>],
    penalty: i32,
    min_score: i32,
) -> SwResult {
    let mut result = SwResult::new(len_a, len_b);
    let mut path_lists: Vec<PathList> = (0..len_a).map(|_| PathList::new()).collect();

    // Allocate working arrays (old and new row states)
    let mut old: Vec<PathState> = (0..len_a)
        .map(|i| PathState {
            col: 0,
            score: 0,
            start: AlignmentCoord::new(i, 1),
            end: AlignmentCoord::new(i, 1),
        })
        .collect();

    let mut new: Vec<PathState> = old.clone();

    // Initialize path matrix borders
    for i in 0..len_a.saturating_sub(1) {
        result.path_matrix[0][i] = direction::NONE;
    }
    for j in 0..len_b.saturating_sub(1) {
        result.path_matrix[j][0] = direction::NONE;
    }

    let mut match_score: i32 = -1;

    // Main DP loop
    for j in 1..len_b.saturating_sub(1) {
        for i in 1..len_a.saturating_sub(1) {
            let im1 = i - 1;

            // Calculate scores from three directions
            let diag = old[im1].col + prob[i][j];
            let horiz = old[i].col - penalty;
            let vert = new[im1].col - penalty;

            // Find maximum of (diag, horiz, vert, 0)
            let rtemp = max4(diag, horiz, vert, 0);

            // Store direction bitmask (multiple directions possible)
            let mut dir_mask = direction::NONE;
            if rtemp == diag {
                dir_mask |= direction::DIAG;
            }
            if rtemp == horiz {
                dir_mask |= direction::HORIZ;
            }
            if rtemp == vert {
                dir_mask |= direction::VERT;
            }
            result.path_matrix[j][i] = dir_mask;

            if rtemp > 0 {
                // Update path state based on which direction gave max
                if diag == rtemp {
                    if old[im1].col == 0 {
                        // Starting a new path
                        new[i].start = AlignmentCoord::new(i, j);
                        new[i].score = rtemp;
                        new[i].end = AlignmentCoord::new(i, j);
                    } else {
                        // Extending from diagonal
                        new[i].start = old[im1].start;
                        if rtemp >= old[im1].score {
                            new[i].score = rtemp;
                            new[i].end = AlignmentCoord::new(i, j);
                        } else {
                            new[i].score = old[im1].score;
                            new[i].end = old[im1].end;
                        }
                    }
                } else if horiz == rtemp {
                    // Extending horizontally
                    new[i].start = old[i].start;
                    if horiz >= old[i].score {
                        new[i].score = horiz;
                        new[i].end = AlignmentCoord::new(i, j);
                    } else {
                        new[i].score = old[i].score;
                        new[i].end = old[i].end;
                    }
                } else if vert == rtemp {
                    // Extending vertically
                    new[i].start = new[im1].start;
                    if vert > new[im1].score {
                        new[i].score = vert;
                        new[i].end = AlignmentCoord::new(i, j);
                    } else {
                        new[i].score = new[im1].score;
                        new[i].end = new[im1].end;
                    }
                }
            }

            // Check if we should record a path (at matrix edges)
            if (i == len_a.saturating_sub(2)) || (j == len_b.saturating_sub(2)) {
                let path = AlignmentPath::new(new[i].start, new[i].end, new[i].score);
                if new[i].score >= min_score
                    && new[i].start.i > 0
                    && path.is_valid()
                {
                    if path_lists[new[i].start.i].add_if_better(path) {
                        result.total_paths += 1;
                    }
                }
            } else if rtemp == 0 && old[im1].score > 0 {
                // Path ended (score dropped to 0)
                let path = AlignmentPath::new(old[im1].start, old[im1].end, old[im1].score);
                if old[im1].score >= min_score
                    && old[im1].start.i > 0
                    && path.is_valid()
                {
                    if path_lists[old[im1].start.i].add_if_better(path) {
                        result.total_paths += 1;
                    }
                }
                new[i].score = 0;
            }

            if rtemp > match_score {
                match_score = rtemp;
            }
            new[i].col = rtemp;
        }

        // Swap old and new
        std::mem::swap(&mut old, &mut new);

        // Reset start to end for next row
        for k in 0..len_a.saturating_sub(1) {
            new[k].start = new[k].end;
        }
    }

    result.max_score = match_score;

    // Collect all paths and sort by score descending
    result.paths = collect_and_sort_paths(&path_lists);

    result
}

/// Performs Smith-Waterman alignment with corner-cutting optimization.
///
/// This function implements the `sw7ccs` algorithm from the original STAMP code.
/// The corner-cutting optimization skips matrix cells that are far from the diagonal,
/// based on the ratio of sequence lengths. This significantly speeds up alignment
/// of sequences with similar lengths.
///
/// # Arguments
///
/// * `len_a` - Length of first sequence (+ padding)
/// * `len_b` - Length of second sequence (+ padding)
/// * `prob` - Probability/similarity matrix \[i\]\[j\]
/// * `penalty` - Gap penalty
/// * `min_score` - Minimum score for a path to be recorded
/// * `auto_corner` - If true, automatically adjust fk based on length difference
/// * `fk` - Corner-cutting factor (typically 2.0-4.0)
///
/// # Corner-Cutting Formula
///
/// The algorithm skips cells where `|i/m - j/n| > fcut`, where:
/// - `fcut = (fk / sqrt(2)) * ((m/n + n/m) / sqrt(m^2 + n^2))`
/// - If `auto_corner` is true: `fk1 = fk + |len_a - len_b - 4|`
///
/// # Returns
///
/// Result containing the maximum score, path matrix, and all discovered paths.
#[must_use]
pub fn smith_waterman_corner_cut(
    len_a: usize,
    len_b: usize,
    prob: &[Vec<i32>],
    penalty: i32,
    min_score: i32,
    auto_corner: bool,
    fk: f64,
) -> SwResult {
    let mut result = SwResult::new(len_a, len_b);
    let mut path_lists: Vec<PathList> = (0..len_a).map(|_| PathList::new()).collect();

    // Allocate working arrays
    let mut old: Vec<PathState> = (0..len_a)
        .map(|i| PathState {
            col: 0,
            score: 0,
            start: AlignmentCoord::new(i, 1),
            end: AlignmentCoord::new(i, 1),
        })
        .collect();

    let mut new: Vec<PathState> = old.clone();

    // Initialize path matrix borders
    for i in 0..len_a.saturating_sub(1) {
        result.path_matrix[0][i] = direction::NONE;
    }
    for j in 0..len_b.saturating_sub(1) {
        result.path_matrix[j][0] = direction::NONE;
    }

    // Calculate corner-cutting parameters
    let fk1 = if auto_corner {
        fk + ((len_a as i32 - len_b as i32 - 4).abs() as f64)
    } else {
        fk
    };

    let fn_val = len_a as f64;
    let fm_val = len_b as f64;
    let fcut = (fk1 / 2.0_f64.sqrt())
        * ((fm_val / fn_val + fn_val / fm_val) / (fm_val * fm_val + fn_val * fn_val).sqrt());

    let mut match_score: i32 = -1;
    let mut lstart = 1usize;

    // Main DP loop with corner cutting
    for j in 1..len_b.saturating_sub(1) {
        let jt = j as f64 / fn_val;
        let mut upper = false;
        let mut lstrt = 0usize;

        for i in lstart..len_a.saturating_sub(1) {
            // Corner cutting check
            let corner_dist = ((i as f64 / fm_val) - jt).abs();
            if corner_dist > fcut {
                if upper {
                    // We've passed the valid region, break to next j
                    break;
                }
                // Haven't reached valid region yet, continue
                continue;
            }

            // We're in the valid region
            if !upper {
                lstrt = i;
                upper = true;
            }

            let im1 = i - 1;

            // Calculate scores from three directions
            let diag = old[im1].col + prob[i][j];
            let horiz = old[i].col - penalty;
            let vert = new[im1].col - penalty;

            let rtemp = max4(diag, horiz, vert, 0);

            // Store direction bitmask
            let mut dir_mask = direction::NONE;
            if rtemp == diag {
                dir_mask |= direction::DIAG;
            }
            if rtemp == horiz {
                dir_mask |= direction::HORIZ;
            }
            if rtemp == vert {
                dir_mask |= direction::VERT;
            }
            result.path_matrix[j][i] = dir_mask;

            if rtemp > 0 {
                if diag == rtemp {
                    if old[im1].col == 0 {
                        new[i].start = AlignmentCoord::new(i, j);
                        new[i].score = rtemp;
                        new[i].end = AlignmentCoord::new(i, j);
                    } else {
                        new[i].start = old[im1].start;
                        if rtemp >= old[im1].score {
                            new[i].score = rtemp;
                            new[i].end = AlignmentCoord::new(i, j);
                        } else {
                            new[i].score = old[im1].score;
                            new[i].end = old[im1].end;
                        }
                    }
                } else if horiz == rtemp {
                    new[i].start = old[i].start;
                    if horiz >= old[i].score {
                        new[i].score = horiz;
                        new[i].end = AlignmentCoord::new(i, j);
                    } else {
                        new[i].score = old[i].score;
                        new[i].end = old[i].end;
                    }
                } else if vert == rtemp {
                    new[i].start = new[im1].start;
                    if vert > new[im1].score {
                        new[i].score = vert;
                        new[i].end = AlignmentCoord::new(i, j);
                    } else {
                        new[i].score = new[im1].score;
                        new[i].end = new[im1].end;
                    }
                }
            }

            // Check if we should record a path
            if (i == len_a.saturating_sub(2)) || (j == len_b.saturating_sub(2)) {
                let path = AlignmentPath::new(new[i].start, new[i].end, new[i].score);
                if new[i].score >= min_score
                    && new[i].start.i > 0
                    && path.is_valid()
                {
                    if path_lists[new[i].start.i].add_if_better(path) {
                        result.total_paths += 1;
                    }
                }
            } else if rtemp == 0 && old[im1].score > 0 {
                let path = AlignmentPath::new(old[im1].start, old[im1].end, old[im1].score);
                if old[im1].score >= min_score
                    && old[im1].start.i > 0
                    && path.is_valid()
                {
                    if path_lists[old[im1].start.i].add_if_better(path) {
                        result.total_paths += 1;
                    }
                }
                new[i].score = 0;
            }

            if rtemp > match_score {
                match_score = rtemp;
            }
            new[i].col = rtemp;
        }

        lstart = lstrt.max(1);

        // Swap old and new
        std::mem::swap(&mut old, &mut new);

        // Reset start to end for next row
        for k in 0..len_a.saturating_sub(1) {
            new[k].start = new[k].end;
        }
    }

    result.max_score = match_score;
    result.paths = collect_and_sort_paths(&path_lists);

    result
}

/// Performs traceback through the path matrix to reconstruct an alignment.
///
/// Given a path's start and end coordinates and the path matrix, this function
/// traces back from the end to the start, following the direction bits.
///
/// # Arguments
///
/// * `path_matrix` - Matrix of direction bitmasks \[j\]\[i\]
/// * `path` - The alignment path to trace
///
/// # Returns
///
/// Tuple of:
/// - Vector of (i, j) aligned position pairs (only diagonal moves)
/// - Number of horizontal gaps
/// - Number of vertical gaps
///
/// # Example
///
/// ```ignore
/// use stamp_core::alignment::{smith_waterman, traceback};
///
/// let result = smith_waterman(4, 4, &prob, 0, 1);
/// if let Some(best_path) = result.paths.first() {
///     let (aligned, hgaps, vgaps) = traceback(&result.path_matrix, best_path);
///     println!("Aligned {} positions", aligned.len());
/// }
/// ```
#[must_use]
pub fn traceback(path_matrix: &[Vec<u8>], path: &AlignmentPath) -> (Vec<(usize, usize)>, usize, usize) {
    let mut aligned_pairs = Vec::new();
    let mut hgap_count = 0usize;
    let mut vgap_count = 0usize;

    let mut i = path.end.i;
    let mut j = path.end.j;

    // Trace back from end to start
    while i >= path.start.i && j >= path.start.j && (i > 0 || j > 0) {
        let dir = path_matrix[j][i];

        if dir & direction::DIAG != 0 {
            // Diagonal move - this is an aligned pair
            aligned_pairs.push((i, j));
            if i == 0 || j == 0 {
                break;
            }
            i -= 1;
            j -= 1;
        } else if dir & direction::HORIZ != 0 {
            // Horizontal move - gap in sequence A
            hgap_count += 1;
            if j == 0 {
                break;
            }
            j -= 1;
        } else if dir & direction::VERT != 0 {
            // Vertical move - gap in sequence B
            vgap_count += 1;
            if i == 0 {
                break;
            }
            i -= 1;
        } else {
            // No valid direction, end traceback
            break;
        }

        // Stop if we've reached the start
        if i < path.start.i || j < path.start.j {
            break;
        }
    }

    // Reverse to get forward order
    aligned_pairs.reverse();

    (aligned_pairs, hgap_count, vgap_count)
}

/// Extracts aligned residue pairs from an alignment path.
///
/// This is a convenience wrapper around `traceback` that returns only the
/// aligned position pairs, suitable for use with structure superposition.
///
/// # Arguments
///
/// * `path_matrix` - Matrix of direction bitmasks
/// * `path` - The alignment path to extract pairs from
///
/// # Returns
///
/// Vector of (i, j) pairs where i is the position in sequence A and j is
/// the position in sequence B. These pairs represent structurally aligned
/// residues.
#[must_use]
pub fn extract_aligned_pairs(path_matrix: &[Vec<u8>], path: &AlignmentPath) -> Vec<(usize, usize)> {
    let (pairs, _, _) = traceback(path_matrix, path);
    pairs
}

/// Performs traceback to reconstruct alignment strings.
///
/// This function produces alignment strings similar to the original `aliseq` function,
/// marking aligned positions and gaps.
///
/// # Arguments
///
/// * `seq_a` - First sequence (as bytes/chars)
/// * `seq_b` - Second sequence (as bytes/chars)
/// * `path_matrix` - Matrix of direction bitmasks
/// * `path` - The alignment path
///
/// # Returns
///
/// Result containing:
/// - Aligned sequence A (with gap characters)
/// - Aligned sequence B (with gap characters)
/// - Alignment statistics
pub fn traceback_to_strings(
    seq_a: &[u8],
    seq_b: &[u8],
    path_matrix: &[Vec<u8>],
    path: &AlignmentPath,
) -> StampResult<AlignmentStrings> {
    let mut aligned_a = Vec::new();
    let mut aligned_b = Vec::new();
    let mut hgap_count = 0usize;
    let mut vgap_count = 0usize;

    let mut i = path.end.i;
    let mut j = path.end.j;

    while i >= path.start.i && j >= path.start.j && (i > 0 || j > 0) {
        let dir = path_matrix[j][i];

        if dir & direction::DIAG != 0 {
            // Diagonal - both sequences have a residue
            if i > 0 && j > 0 && i <= seq_a.len() && j <= seq_b.len() {
                aligned_a.push(seq_a[i - 1]);
                aligned_b.push(seq_b[j - 1]);
            }
            if i == 0 || j == 0 {
                break;
            }
            i -= 1;
            j -= 1;
        } else if dir & direction::HORIZ != 0 {
            // Horizontal - gap in sequence A
            aligned_a.push(b'-');
            if j > 0 && j <= seq_b.len() {
                aligned_b.push(seq_b[j - 1]);
            }
            hgap_count += 1;
            if j == 0 {
                break;
            }
            j -= 1;
        } else if dir & direction::VERT != 0 {
            // Vertical - gap in sequence B
            if i > 0 && i <= seq_a.len() {
                aligned_a.push(seq_a[i - 1]);
            }
            aligned_b.push(b'-');
            vgap_count += 1;
            if i == 0 {
                break;
            }
            i -= 1;
        } else {
            break;
        }

        if i < path.start.i || j < path.start.j {
            break;
        }
    }

    // Reverse to get forward order
    aligned_a.reverse();
    aligned_b.reverse();

    let length = aligned_a.len().max(aligned_b.len());

    Ok(AlignmentStrings {
        seq_a: aligned_a,
        seq_b: aligned_b,
        hgap_count,
        vgap_count,
        length,
    })
}

/// Result of alignment string reconstruction.
#[derive(Debug, Clone)]
pub struct AlignmentStrings {
    /// Aligned sequence A (with gaps as '-').
    pub seq_a: Vec<u8>,
    /// Aligned sequence B (with gaps as '-').
    pub seq_b: Vec<u8>,
    /// Number of horizontal gaps (gaps in seq A).
    pub hgap_count: usize,
    /// Number of vertical gaps (gaps in seq B).
    pub vgap_count: usize,
    /// Total alignment length.
    pub length: usize,
}

/// Global alignment using Needleman-Wunsch algorithm.
#[derive(Debug)]
pub struct NeedlemanWunsch {
    /// Dynamic programming matrix.
    matrix: Vec<Vec<DpCell>>,
    /// Length of first sequence.
    len1: usize,
    /// Length of second sequence.
    len2: usize,
    /// Gap opening penalty.
    gap_open: f64,
    /// Gap extension penalty.
    gap_extend: f64,
}

impl NeedlemanWunsch {
    /// Creates a new Needleman-Wunsch aligner.
    #[must_use]
    pub fn new(len1: usize, len2: usize, params: &Parameters) -> Self {
        let mut matrix = vec![vec![DpCell::default(); len2 + 1]; len1 + 1];

        // Initialize first row and column with gap penalties
        for i in 1..=len1 {
            matrix[i][0] = DpCell {
                score: -(params.pen_gap + (i - 1) as f64 * params.pen_ext),
                direction: Direction::Vertical,
            };
        }
        for j in 1..=len2 {
            matrix[0][j] = DpCell {
                score: -(params.pen_gap + (j - 1) as f64 * params.pen_ext),
                direction: Direction::Horizontal,
            };
        }

        Self {
            matrix,
            len1,
            len2,
            gap_open: params.pen_gap,
            gap_extend: params.pen_ext,
        }
    }

    /// Fills the dynamic programming matrix.
    pub fn fill(&mut self, scores: &[Vec<f64>]) {
        for i in 1..=self.len1 {
            for j in 1..=self.len2 {
                let score_ij = scores[i - 1][j - 1];

                let diag = self.matrix[i - 1][j - 1].score + score_ij;

                let horiz = if self.matrix[i][j - 1].direction == Direction::Horizontal {
                    self.matrix[i][j - 1].score - self.gap_extend
                } else {
                    self.matrix[i][j - 1].score - self.gap_open
                };

                let vert = if self.matrix[i - 1][j].direction == Direction::Vertical {
                    self.matrix[i - 1][j].score - self.gap_extend
                } else {
                    self.matrix[i - 1][j].score - self.gap_open
                };

                let (score, direction) = if diag >= horiz && diag >= vert {
                    (diag, Direction::Diagonal)
                } else if horiz >= vert {
                    (horiz, Direction::Horizontal)
                } else {
                    (vert, Direction::Vertical)
                };

                self.matrix[i][j] = DpCell { score, direction };
            }
        }
    }

    /// Traces back from the end to find the optimal global alignment.
    #[must_use]
    pub fn traceback(&self) -> Vec<(usize, usize)> {
        let mut path = Vec::new();
        let mut i = self.len1;
        let mut j = self.len2;

        while i > 0 && j > 0 {
            match self.matrix[i][j].direction {
                Direction::Diagonal => {
                    path.push((i - 1, j - 1));
                    i -= 1;
                    j -= 1;
                }
                Direction::Horizontal => {
                    j -= 1;
                }
                Direction::Vertical => {
                    i -= 1;
                }
                Direction::None => break,
            }
        }

        path.reverse();
        path
    }

    /// Returns the final alignment score.
    #[must_use]
    pub fn final_score(&self) -> f64 {
        self.matrix[self.len1][self.len2].score
    }
}

/// Computes alignment path from equivalence lists.
///
/// # Arguments
///
/// * `equiv1` - Equivalence indices for structure 1
/// * `equiv2` - Equivalence indices for structure 2
///
/// # Returns
///
/// Vector of aligned position pairs.
pub fn path_from_equivalences(equiv1: &[i32], equiv2: &[i32]) -> StampResult<Vec<(usize, usize)>> {
    let mut path = Vec::new();

    for (i, &e1) in equiv1.iter().enumerate() {
        if e1 >= 0 {
            if let Some(&e2) = equiv2.get(e1 as usize) {
                if e2 == i as i32 {
                    path.push((i, e1 as usize));
                }
            }
        }
    }

    Ok(path)
}

// Helper function: maximum of 4 values
#[inline]
fn max4(a: i32, b: i32, c: i32, d: i32) -> i32 {
    a.max(b).max(c).max(d)
}

// Helper function: collect and sort all paths by score descending
fn collect_and_sort_paths(path_lists: &[PathList]) -> Vec<AlignmentPath> {
    let mut all_paths: Vec<AlignmentPath> = path_lists
        .iter()
        .flat_map(|pl| pl.paths.iter().copied())
        .collect();

    all_paths.sort_by(|a, b| b.score.cmp(&a.score));
    all_paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smith_waterman_simple() {
        let params = Parameters::default();
        let mut sw = SmithWaterman::new(3, 3, &params);

        // Simple identity scoring
        let scores = vec![
            vec![5.0, -1.0, -1.0],
            vec![-1.0, 5.0, -1.0],
            vec![-1.0, -1.0, 5.0],
        ];

        sw.fill(&scores);
        let path = sw.traceback();

        assert_eq!(path.len(), 3);
        assert_eq!(path, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn test_needleman_wunsch_simple() {
        let params = Parameters::default();
        let mut nw = NeedlemanWunsch::new(3, 3, &params);

        let scores = vec![
            vec![5.0, -1.0, -1.0],
            vec![-1.0, 5.0, -1.0],
            vec![-1.0, -1.0, 5.0],
        ];

        nw.fill(&scores);
        let path = nw.traceback();

        assert_eq!(path.len(), 3);
    }

    #[test]
    fn test_path_from_equivalences() {
        let equiv1 = vec![0, 1, 2];
        let equiv2 = vec![0, 1, 2];
        let path = path_from_equivalences(&equiv1, &equiv2).unwrap();
        assert_eq!(path, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn test_direction_bitmask_conversion() {
        assert_eq!(Direction::Diagonal.to_bitmask(), direction::DIAG);
        assert_eq!(Direction::Horizontal.to_bitmask(), direction::HORIZ);
        assert_eq!(Direction::Vertical.to_bitmask(), direction::VERT);
        assert_eq!(Direction::None.to_bitmask(), direction::NONE);

        assert_eq!(Direction::from_bitmask(direction::DIAG), Direction::Diagonal);
        assert_eq!(Direction::from_bitmask(direction::HORIZ), Direction::Horizontal);
        assert_eq!(Direction::from_bitmask(direction::VERT), Direction::Vertical);
        assert_eq!(Direction::from_bitmask(direction::NONE), Direction::None);
    }

    #[test]
    fn test_smith_waterman_with_path_tracking() {
        // Create a simple scoring matrix (identity-like)
        // Using len+2 to match original C code padding convention
        let len_a = 5; // 3 residues + 2 padding
        let len_b = 5;

        let mut prob = vec![vec![0i32; len_b]; len_a];
        // Set diagonal matches
        prob[1][1] = 10;
        prob[2][2] = 10;
        prob[3][3] = 10;

        let result = smith_waterman(len_a, len_b, &prob, 0, 1);

        assert!(result.max_score > 0);
    }

    #[test]
    fn test_smith_waterman_corner_cut() {
        let len_a = 5;
        let len_b = 5;

        let mut prob = vec![vec![0i32; len_b]; len_a];
        prob[1][1] = 10;
        prob[2][2] = 10;
        prob[3][3] = 10;

        let result = smith_waterman_corner_cut(len_a, len_b, &prob, 0, 1, true, 2.0);

        assert!(result.max_score > 0);
    }

    #[test]
    fn test_alignment_path_validity() {
        let valid_path = AlignmentPath::new(
            AlignmentCoord::new(1, 1),
            AlignmentCoord::new(5, 5),
            100,
        );
        assert!(valid_path.is_valid());

        let invalid_path = AlignmentPath::new(
            AlignmentCoord::new(1, 1),
            AlignmentCoord::new(1, 5),
            100,
        );
        assert!(!invalid_path.is_valid());
    }

    #[test]
    fn test_path_list_add_if_better() {
        let mut pl = PathList::new();

        let path1 = AlignmentPath::new(
            AlignmentCoord::new(1, 1),
            AlignmentCoord::new(5, 5),
            100,
        );

        let path2 = AlignmentPath::new(
            AlignmentCoord::new(1, 1),
            AlignmentCoord::new(6, 6),
            150,
        );

        assert!(pl.add_if_better(path1));
        assert_eq!(pl.len(), 1);

        // Same start.j, but higher score - should update
        assert!(!pl.add_if_better(path2));
        assert_eq!(pl.len(), 1);
        assert_eq!(pl.paths[0].score, 150);
    }

    #[test]
    fn test_traceback_simple() {
        // Create a simple path matrix for a diagonal alignment
        let len_a = 5;
        let len_b = 5;
        let mut path_matrix = vec![vec![direction::NONE; len_a]; len_b];

        // Set diagonal path
        path_matrix[1][1] = direction::DIAG;
        path_matrix[2][2] = direction::DIAG;
        path_matrix[3][3] = direction::DIAG;

        let path = AlignmentPath::new(
            AlignmentCoord::new(1, 1),
            AlignmentCoord::new(3, 3),
            30,
        );

        let (aligned, hgaps, vgaps) = traceback(&path_matrix, &path);

        assert!(!aligned.is_empty());
        assert_eq!(hgaps, 0);
        assert_eq!(vgaps, 0);
    }

    #[test]
    fn test_max4() {
        assert_eq!(max4(1, 2, 3, 4), 4);
        assert_eq!(max4(4, 3, 2, 1), 4);
        assert_eq!(max4(-1, -2, -3, 0), 0);
        assert_eq!(max4(5, 5, 3, 2), 5);
    }

    #[test]
    fn test_alignment_coord_default() {
        let coord = AlignmentCoord::default();
        assert_eq!(coord.i, 0);
        assert_eq!(coord.j, 0);
    }

    #[test]
    fn test_alignment_path_default() {
        let path = AlignmentPath::default();
        assert_eq!(path.start.i, 0);
        assert_eq!(path.end.i, 0);
        assert_eq!(path.score, 0);
    }

    #[test]
    fn test_sw_result_initialization() {
        let result = SwResult::new(10, 8);
        assert_eq!(result.max_score, -1);
        assert_eq!(result.path_matrix.len(), 8);
        assert_eq!(result.path_matrix[0].len(), 10);
        assert!(result.paths.is_empty());
        assert_eq!(result.total_paths, 0);
    }

    #[test]
    fn test_extract_aligned_pairs() {
        let len_a = 5;
        let len_b = 5;
        let mut path_matrix = vec![vec![direction::NONE; len_a]; len_b];

        path_matrix[1][1] = direction::DIAG;
        path_matrix[2][2] = direction::DIAG;
        path_matrix[3][3] = direction::DIAG;

        let path = AlignmentPath::new(
            AlignmentCoord::new(1, 1),
            AlignmentCoord::new(3, 3),
            30,
        );

        let pairs = extract_aligned_pairs(&path_matrix, &path);
        assert!(!pairs.is_empty());
    }
}
