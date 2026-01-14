//! Linear algebra operations and geometric transformations.
//!
//! This module provides mathematical utilities for structural alignment,
//! including the Kabsch algorithm for optimal superposition, centroid
//! calculations, and RMSD computation.
//!
//! # SIMD Optimization
//!
//! When the `simd` feature is enabled, certain functions use SIMD intrinsics
//! for improved performance on supported platforms.

use crate::matrix::CoordsSoA;
use crate::types::{Coord3, RotationMatrix, StampResult, Transform, Vec3};
use nalgebra::{Matrix3, SVD};

/// Computes the centroid of a set of points.
///
/// # Arguments
///
/// * `points` - Iterator over 3D coordinates
///
/// # Returns
///
/// The geometric center of the points.
///
/// # Panics
///
/// Panics if the iterator is empty.
#[must_use]
pub fn centroid<'a>(points: impl Iterator<Item = &'a Coord3>) -> Coord3 {
    let mut sum = Vec3::zeros();
    let mut count = 0usize;

    for p in points {
        sum += p.coords;
        count += 1;
    }

    assert!(count > 0, "Cannot compute centroid of empty point set");
    Coord3::from(sum / count as f64)
}

/// Computes the root mean square deviation between two point sets.
///
/// # Arguments
///
/// * `points1` - First set of coordinates
/// * `points2` - Second set of coordinates (must have same length)
///
/// # Returns
///
/// The RMSD value in the same units as the input coordinates.
#[must_use]
pub fn rmsd<'a>(
    points1: impl Iterator<Item = &'a Coord3>,
    points2: impl Iterator<Item = &'a Coord3>,
) -> f64 {
    let mut sum_sq = 0.0;
    let mut count = 0usize;

    for (p1, p2) in points1.zip(points2) {
        let diff = p1.coords - p2.coords;
        sum_sq += diff.dot(&diff);
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }

    (sum_sq / count as f64).sqrt()
}

/// Computes the optimal rotation matrix using the Kabsch algorithm.
///
/// Given two sets of corresponding points, finds the rotation matrix R
/// that minimizes the RMSD when applied to the second set.
///
/// Both point sets should be centered at the origin before calling this function.
///
/// # Arguments
///
/// * `fixed` - Reference point set (centered)
/// * `mobile` - Point set to be rotated (centered)
///
/// # Returns
///
/// The optimal 3x3 rotation matrix.
///
/// # Algorithm
///
/// Uses singular value decomposition (SVD) of the covariance matrix:
/// 1. Compute H = sum(mobile_i * fixed_i^T)
/// 2. SVD: H = U * S * V^T
/// 3. R = V * U^T
/// 4. If det(R) < 0, flip sign of last column of V
#[must_use]
pub fn kabsch<'a>(
    fixed: impl Iterator<Item = &'a Coord3>,
    mobile: impl Iterator<Item = &'a Coord3>,
) -> RotationMatrix {
    // Build covariance matrix H
    let mut h = Matrix3::<f64>::zeros();

    for (f, m) in fixed.zip(mobile) {
        h += m.coords * f.coords.transpose();
    }

    // SVD decomposition
    let svd = SVD::new(h, true, true);
    let u = svd.u.expect("SVD failed to compute U");
    let v_t = svd.v_t.expect("SVD failed to compute V^T");

    // Compute rotation matrix
    let mut rotation = v_t.transpose() * u.transpose();

    // Handle reflection case (det < 0)
    if rotation.determinant() < 0.0 {
        let mut v = v_t.transpose();
        v.column_mut(2).neg_mut();
        rotation = v * u.transpose();
    }

    rotation
}

/// Computes the optimal superposition of two point sets.
///
/// Returns the transformation (rotation + translation) that minimizes
/// the RMSD between the two point sets.
///
/// # Arguments
///
/// * `fixed` - Reference point set
/// * `mobile` - Point set to be transformed
///
/// # Returns
///
/// Result containing the optimal transformation and the resulting RMSD.
pub fn superpose(fixed: &[Coord3], mobile: &[Coord3]) -> StampResult<(Transform, f64)> {
    if fixed.is_empty() || mobile.is_empty() {
        return Ok((Transform::identity(), 0.0));
    }

    // Compute centroids
    let centroid_fixed = centroid(fixed.iter());
    let centroid_mobile = centroid(mobile.iter());

    // Center the point sets
    let centered_fixed: Vec<Coord3> = fixed
        .iter()
        .map(|p| Coord3::from(p.coords - centroid_fixed.coords))
        .collect();
    let centered_mobile: Vec<Coord3> = mobile
        .iter()
        .map(|p| Coord3::from(p.coords - centroid_mobile.coords))
        .collect();

    // Compute optimal rotation
    let rotation = kabsch(centered_fixed.iter(), centered_mobile.iter());

    // Compute translation
    let translation = centroid_fixed.coords - rotation * centroid_mobile.coords;

    let transform = Transform {
        rotation,
        translation,
    };

    // Compute RMSD after transformation
    let transformed: Vec<Coord3> = mobile.iter().map(|p| transform.apply(p)).collect();
    let final_rmsd = rmsd(fixed.iter(), transformed.iter());

    Ok((transform, final_rmsd))
}

/// Computes the squared distance between two points.
#[inline]
#[must_use]
pub fn distance_squared(p1: &Coord3, p2: &Coord3) -> f64 {
    let diff = p1.coords - p2.coords;
    diff.dot(&diff)
}

/// Computes the Euclidean distance between two points.
#[inline]
#[must_use]
pub fn distance(p1: &Coord3, p2: &Coord3) -> f64 {
    distance_squared(p1, p2).sqrt()
}

/// Computes distance matrix between two sets of points.
///
/// Returns a matrix where element [i][j] is the distance between
/// points1[i] and points2[j].
#[must_use]
pub fn distance_matrix(points1: &[Coord3], points2: &[Coord3]) -> Vec<Vec<f64>> {
    points1
        .iter()
        .map(|p1| points2.iter().map(|p2| distance(p1, p2)).collect())
        .collect()
}

/// Applies a transformation to a set of points in place.
pub fn transform_points(points: &mut [Coord3], transform: &Transform) {
    for point in points {
        *point = transform.apply(point);
    }
}

/// Creates a rotation matrix from Euler angles (in radians).
///
/// Uses the ZYX convention (yaw-pitch-roll).
#[must_use]
pub fn rotation_from_euler(roll: f64, pitch: f64, yaw: f64) -> RotationMatrix {
    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();

    RotationMatrix::new(
        cy * cp,
        cy * sp * sr - sy * cr,
        cy * sp * cr + sy * sr,
        sy * cp,
        sy * sp * sr + cy * cr,
        sy * sp * cr - cy * sr,
        -sp,
        cp * sr,
        cp * cr,
    )
}

// ============================================================================
// Structure-of-Arrays (SoA) optimized functions
// ============================================================================

/// Computes RMSD using Structure-of-Arrays layout for better vectorization.
///
/// This function processes coordinates in SoA format which allows for
/// more efficient SIMD operations.
#[must_use]
pub fn rmsd_soa(points1: &CoordsSoA, points2: &CoordsSoA) -> f64 {
    assert_eq!(
        points1.len(),
        points2.len(),
        "Point sets must have equal length"
    );
    let n = points1.len();
    if n == 0 {
        return 0.0;
    }

    let mut sum_sq = 0.0;

    // Process x, y, z components separately for vectorization
    for i in 0..n {
        let dx = points1.x[i] - points2.x[i];
        let dy = points1.y[i] - points2.y[i];
        let dz = points1.z[i] - points2.z[i];
        sum_sq += dx * dx + dy * dy + dz * dz;
    }

    (sum_sq / n as f64).sqrt()
}

/// Computes the covariance matrix for Kabsch algorithm using SoA layout.
///
/// Returns a 3x3 matrix H = sum(mobile_i * fixed_i^T).
#[must_use]
pub fn covariance_matrix_soa(fixed: &CoordsSoA, mobile: &CoordsSoA) -> Matrix3<f64> {
    assert_eq!(fixed.len(), mobile.len());
    let n = fixed.len();

    // Accumulate 9 matrix elements
    let mut h = [[0.0f64; 3]; 3];

    for i in 0..n {
        // mobile row, fixed column -> outer product contribution
        h[0][0] += mobile.x[i] * fixed.x[i];
        h[0][1] += mobile.x[i] * fixed.y[i];
        h[0][2] += mobile.x[i] * fixed.z[i];
        h[1][0] += mobile.y[i] * fixed.x[i];
        h[1][1] += mobile.y[i] * fixed.y[i];
        h[1][2] += mobile.y[i] * fixed.z[i];
        h[2][0] += mobile.z[i] * fixed.x[i];
        h[2][1] += mobile.z[i] * fixed.y[i];
        h[2][2] += mobile.z[i] * fixed.z[i];
    }

    Matrix3::new(
        h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2],
    )
}

/// Computes optimal rotation using Kabsch algorithm with SoA coordinates.
///
/// Both point sets should be centered at the origin.
#[must_use]
pub fn kabsch_soa(fixed: &CoordsSoA, mobile: &CoordsSoA) -> RotationMatrix {
    let h = covariance_matrix_soa(fixed, mobile);

    // SVD decomposition
    let svd = SVD::new(h, true, true);
    let u = svd.u.expect("SVD failed to compute U");
    let v_t = svd.v_t.expect("SVD failed to compute V^T");

    // Compute rotation matrix
    let mut rotation = v_t.transpose() * u.transpose();

    // Handle reflection case (det < 0)
    if rotation.determinant() < 0.0 {
        let mut v = v_t.transpose();
        v.column_mut(2).neg_mut();
        rotation = v * u.transpose();
    }

    rotation
}

/// Computes optimal superposition using SoA coordinates.
///
/// Returns the transformation and resulting RMSD.
pub fn superpose_soa(fixed: &CoordsSoA, mobile: &CoordsSoA) -> StampResult<(Transform, f64)> {
    if fixed.is_empty() || mobile.is_empty() {
        return Ok((Transform::identity(), 0.0));
    }

    // Compute centroids
    let (cx_fixed, cy_fixed, cz_fixed) = fixed.centroid();
    let (cx_mobile, cy_mobile, cz_mobile) = mobile.centroid();

    // Center the point sets
    let mut centered_fixed = fixed.clone();
    let mut centered_mobile = mobile.clone();
    centered_fixed.center();
    centered_mobile.center();

    // Compute optimal rotation
    let rotation = kabsch_soa(&centered_fixed, &centered_mobile);

    // Compute translation
    let centroid_fixed = Vec3::new(cx_fixed, cy_fixed, cz_fixed);
    let centroid_mobile = Vec3::new(cx_mobile, cy_mobile, cz_mobile);
    let translation = centroid_fixed - rotation * centroid_mobile;

    let transform = Transform {
        rotation,
        translation,
    };

    // Compute final RMSD
    let mut transformed_mobile = mobile.clone();
    transform_soa(&mut transformed_mobile, &transform);
    let final_rmsd = rmsd_soa(fixed, &transformed_mobile);

    Ok((transform, final_rmsd))
}

/// Applies a transformation to SoA coordinates in place.
pub fn transform_soa(coords: &mut CoordsSoA, transform: &Transform) {
    let r = &transform.rotation;
    let t = &transform.translation;

    for i in 0..coords.len() {
        let x = coords.x[i];
        let y = coords.y[i];
        let z = coords.z[i];

        coords.x[i] = r[(0, 0)] * x + r[(0, 1)] * y + r[(0, 2)] * z + t.x;
        coords.y[i] = r[(1, 0)] * x + r[(1, 1)] * y + r[(1, 2)] * z + t.y;
        coords.z[i] = r[(2, 0)] * x + r[(2, 1)] * y + r[(2, 2)] * z + t.z;
    }
}

/// Computes squared distances between corresponding points in SoA format.
///
/// Returns a vector of squared distances.
#[must_use]
pub fn squared_distances_soa(points1: &CoordsSoA, points2: &CoordsSoA) -> Vec<f64> {
    assert_eq!(points1.len(), points2.len());
    let n = points1.len();

    let mut distances = Vec::with_capacity(n);
    for i in 0..n {
        let dx = points1.x[i] - points2.x[i];
        let dy = points1.y[i] - points2.y[i];
        let dz = points1.z[i] - points2.z[i];
        distances.push(dx * dx + dy * dy + dz * dz);
    }
    distances
}

/// Computes the sum of squared distances (useful for RMSD and other metrics).
#[must_use]
pub fn sum_squared_distances_soa(points1: &CoordsSoA, points2: &CoordsSoA) -> f64 {
    assert_eq!(points1.len(), points2.len());

    let mut sum = 0.0;
    for i in 0..points1.len() {
        let dx = points1.x[i] - points2.x[i];
        let dy = points1.y[i] - points2.y[i];
        let dz = points1.z[i] - points2.z[i];
        sum += dx * dx + dy * dy + dz * dz;
    }
    sum
}

// ============================================================================
// Batch operations for improved cache efficiency
// ============================================================================

/// Computes a distance matrix between two sets of points.
///
/// Uses a cache-friendly access pattern for better performance.
#[must_use]
pub fn distance_matrix_flat(points1: &[Coord3], points2: &[Coord3]) -> Vec<f64> {
    let n1 = points1.len();
    let n2 = points2.len();
    let mut distances = vec![0.0; n1 * n2];

    for (i, p1) in points1.iter().enumerate() {
        for (j, p2) in points2.iter().enumerate() {
            distances[i * n2 + j] = distance(p1, p2);
        }
    }
    distances
}

/// Computes a squared distance matrix (avoids sqrt for efficiency).
#[must_use]
pub fn squared_distance_matrix_flat(points1: &[Coord3], points2: &[Coord3]) -> Vec<f64> {
    let n1 = points1.len();
    let n2 = points2.len();
    let mut distances = vec![0.0; n1 * n2];

    for (i, p1) in points1.iter().enumerate() {
        for (j, p2) in points2.iter().enumerate() {
            distances[i * n2 + j] = distance_squared(p1, p2);
        }
    }
    distances
}

// ============================================================================
// TM-score calculation
// ============================================================================

/// Computes the TM-score normalization factor d0.
///
/// d0 is a length-dependent scale that makes TM-score length-independent.
/// For proteins with L < 22, a minimum value is used to avoid negative d0.
///
/// Formula: d0 = 1.24 * ∛(L - 15) - 1.8
#[inline]
#[must_use]
pub fn tm_d0(length: usize) -> f64 {
    // For very short proteins, use minimum d0 to avoid issues
    if length < 22 {
        return 0.5; // Minimum practical d0
    }
    1.24 * ((length as f64 - 15.0).cbrt()) - 1.8
}

/// Computes TM-score between two aligned structures.
///
/// TM-score is a length-independent metric for protein structural similarity
/// that ranges from 0 to 1:
/// - TM-score > 0.5: Generally indicates same fold
/// - TM-score > 0.17: Above random similarity threshold
/// - TM-score = 1.0: Perfect structural match
///
/// # Arguments
///
/// * `aligned_distances` - Squared distances between aligned residue pairs
///   after optimal superposition (NOT square rooted)
/// * `target_length` - Length of the target/reference protein for normalization
///
/// # Returns
///
/// TM-score value in range [0, 1]
///
/// # Formula
///
/// TM-score = (1/L_target) * Σ(1 / (1 + (d_i / d_0)²))
///
/// Where d_0 = 1.24 * ∛(L_target - 15) - 1.8
#[must_use]
pub fn tm_score(aligned_distances_sq: &[f64], target_length: usize) -> f64 {
    if aligned_distances_sq.is_empty() || target_length == 0 {
        return 0.0;
    }

    let d0 = tm_d0(target_length);
    let d0_sq = d0 * d0;

    let sum: f64 = aligned_distances_sq
        .iter()
        .map(|&d_sq| 1.0 / (1.0 + d_sq / d0_sq))
        .sum();

    sum / target_length as f64
}

/// Computes TM-score from aligned coordinate pairs.
///
/// This is a convenience function that computes distances and TM-score together.
///
/// # Arguments
///
/// * `coords1` - Coordinates of aligned residues in structure 1
/// * `coords2` - Coordinates of aligned residues in structure 2 (after superposition)
/// * `target_length` - Length of the target protein for normalization
///   (typically the length of the shorter protein)
#[must_use]
pub fn tm_score_from_coords(coords1: &[Coord3], coords2: &[Coord3], target_length: usize) -> f64 {
    assert_eq!(
        coords1.len(),
        coords2.len(),
        "Aligned coordinate sets must have equal length"
    );

    let distances_sq: Vec<f64> = coords1
        .iter()
        .zip(coords2.iter())
        .map(|(c1, c2)| distance_squared(c1, c2))
        .collect();

    tm_score(&distances_sq, target_length)
}

/// Computes TM-score using SoA coordinate layout.
///
/// More efficient version using Structure-of-Arrays layout.
#[must_use]
pub fn tm_score_soa(coords1: &CoordsSoA, coords2: &CoordsSoA, target_length: usize) -> f64 {
    assert_eq!(
        coords1.len(),
        coords2.len(),
        "Aligned coordinate sets must have equal length"
    );

    if coords1.is_empty() || target_length == 0 {
        return 0.0;
    }

    let d0 = tm_d0(target_length);
    let d0_sq = d0 * d0;

    let mut sum = 0.0;
    for i in 0..coords1.len() {
        let dx = coords1.x[i] - coords2.x[i];
        let dy = coords1.y[i] - coords2.y[i];
        let dz = coords1.z[i] - coords2.z[i];
        let d_sq = dx * dx + dy * dy + dz * dz;
        sum += 1.0 / (1.0 + d_sq / d0_sq);
    }

    sum / target_length as f64
}

/// Computes both TM-score and RMSD simultaneously from aligned coordinates.
///
/// This is more efficient than computing them separately when both are needed.
///
/// # Returns
///
/// Tuple of (tm_score, rmsd)
#[must_use]
pub fn tm_score_and_rmsd(
    coords1: &[Coord3],
    coords2: &[Coord3],
    target_length: usize,
) -> (f64, f64) {
    assert_eq!(
        coords1.len(),
        coords2.len(),
        "Aligned coordinate sets must have equal length"
    );

    if coords1.is_empty() || target_length == 0 {
        return (0.0, 0.0);
    }

    let d0 = tm_d0(target_length);
    let d0_sq = d0 * d0;

    let mut tm_sum = 0.0;
    let mut rmsd_sum = 0.0;

    for (c1, c2) in coords1.iter().zip(coords2.iter()) {
        let d_sq = distance_squared(c1, c2);
        tm_sum += 1.0 / (1.0 + d_sq / d0_sq);
        rmsd_sum += d_sq;
    }

    let n = coords1.len();
    let tm = tm_sum / target_length as f64;
    let rmsd_val = (rmsd_sum / n as f64).sqrt();

    (tm, rmsd_val)
}

/// Result of TM-score alignment with all metrics.
#[derive(Debug, Clone)]
pub struct TmAlignResult {
    /// TM-score normalized by the first structure's length.
    pub tm_score_1: f64,
    /// TM-score normalized by the second structure's length.
    pub tm_score_2: f64,
    /// RMSD of aligned residues.
    pub rmsd: f64,
    /// Number of aligned residues.
    pub n_aligned: usize,
    /// Length of first structure.
    pub length_1: usize,
    /// Length of second structure.
    pub length_2: usize,
}

impl TmAlignResult {
    /// Returns the average TM-score (mean of both normalizations).
    #[must_use]
    pub fn tm_score_avg(&self) -> f64 {
        (self.tm_score_1 + self.tm_score_2) / 2.0
    }

    /// Returns the maximum TM-score (more conservative comparison).
    #[must_use]
    pub fn tm_score_max(&self) -> f64 {
        self.tm_score_1.max(self.tm_score_2)
    }

    /// Returns the minimum TM-score (more stringent comparison).
    #[must_use]
    pub fn tm_score_min(&self) -> f64 {
        self.tm_score_1.min(self.tm_score_2)
    }
}

/// Computes full TM-alignment metrics with both normalizations.
///
/// # Arguments
///
/// * `coords1` - Aligned coordinates from structure 1
/// * `coords2` - Aligned coordinates from structure 2 (after superposition)
/// * `length_1` - Total length of structure 1
/// * `length_2` - Total length of structure 2
#[must_use]
pub fn tm_align_full(
    coords1: &[Coord3],
    coords2: &[Coord3],
    length_1: usize,
    length_2: usize,
) -> TmAlignResult {
    assert_eq!(coords1.len(), coords2.len());

    let n_aligned = coords1.len();
    if n_aligned == 0 {
        return TmAlignResult {
            tm_score_1: 0.0,
            tm_score_2: 0.0,
            rmsd: 0.0,
            n_aligned: 0,
            length_1,
            length_2,
        };
    }

    let d0_1 = tm_d0(length_1);
    let d0_sq_1 = d0_1 * d0_1;
    let d0_2 = tm_d0(length_2);
    let d0_sq_2 = d0_2 * d0_2;

    let mut tm_sum_1 = 0.0;
    let mut tm_sum_2 = 0.0;
    let mut rmsd_sum = 0.0;

    for (c1, c2) in coords1.iter().zip(coords2.iter()) {
        let d_sq = distance_squared(c1, c2);
        tm_sum_1 += 1.0 / (1.0 + d_sq / d0_sq_1);
        tm_sum_2 += 1.0 / (1.0 + d_sq / d0_sq_2);
        rmsd_sum += d_sq;
    }

    TmAlignResult {
        tm_score_1: tm_sum_1 / length_1 as f64,
        tm_score_2: tm_sum_2 / length_2 as f64,
        rmsd: (rmsd_sum / n_aligned as f64).sqrt(),
        n_aligned,
        length_1,
        length_2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_centroid() {
        let points = vec![
            Coord3::new(0.0, 0.0, 0.0),
            Coord3::new(2.0, 0.0, 0.0),
            Coord3::new(0.0, 2.0, 0.0),
        ];
        let c = centroid(points.iter());
        assert!((c.x - 2.0 / 3.0).abs() < 1e-10);
        assert!((c.y - 2.0 / 3.0).abs() < 1e-10);
        assert!(c.z.abs() < 1e-10);
    }

    #[test]
    fn test_rmsd_identical() {
        let points = vec![Coord3::new(1.0, 2.0, 3.0), Coord3::new(4.0, 5.0, 6.0)];
        let r = rmsd(points.iter(), points.iter());
        assert!(r.abs() < 1e-10);
    }

    #[test]
    fn test_kabsch_identity() {
        let points = vec![
            Coord3::new(1.0, 0.0, 0.0),
            Coord3::new(0.0, 1.0, 0.0),
            Coord3::new(0.0, 0.0, 1.0),
        ];
        let rotation = kabsch(points.iter(), points.iter());
        let identity = RotationMatrix::identity();
        for i in 0..3 {
            for j in 0..3 {
                assert!((rotation[(i, j)] - identity[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_superpose() {
        let fixed = vec![
            Coord3::new(0.0, 0.0, 0.0),
            Coord3::new(1.0, 0.0, 0.0),
            Coord3::new(0.0, 1.0, 0.0),
        ];
        let mobile = vec![
            Coord3::new(5.0, 5.0, 0.0),
            Coord3::new(6.0, 5.0, 0.0),
            Coord3::new(5.0, 6.0, 0.0),
        ];

        let (transform, final_rmsd) = superpose(&fixed, &mobile).unwrap();
        assert!(final_rmsd < 1e-10);

        // Verify transformed points match
        for (f, m) in fixed.iter().zip(mobile.iter()) {
            let transformed = transform.apply(m);
            assert!((transformed.x - f.x).abs() < 1e-10);
            assert!((transformed.y - f.y).abs() < 1e-10);
            assert!((transformed.z - f.z).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rotation_from_euler() {
        // 90 degree rotation around Z axis
        let rot = rotation_from_euler(0.0, 0.0, PI / 2.0);
        let p = Vec3::new(1.0, 0.0, 0.0);
        let result = rot * p;
        assert!((result.x).abs() < 1e-10);
        assert!((result.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance() {
        let p1 = Coord3::new(0.0, 0.0, 0.0);
        let p2 = Coord3::new(3.0, 4.0, 0.0);
        assert!((distance(&p1, &p2) - 5.0).abs() < 1e-10);
    }

    // TM-score tests
    #[test]
    fn test_tm_d0() {
        // Test d0 calculation for various lengths
        // For L=100: d0 = 1.24 * (85)^(1/3) - 1.8 ≈ 3.65
        let d0_100 = tm_d0(100);
        assert!((d0_100 - 3.65).abs() < 0.1);

        // For L=200: d0 = 1.24 * (185)^(1/3) - 1.8 ≈ 5.27
        let d0_200 = tm_d0(200);
        assert!((d0_200 - 5.27).abs() < 0.1);

        // For short proteins, should return minimum d0
        let d0_short = tm_d0(15);
        assert!((d0_short - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tm_score_perfect() {
        // Perfect alignment (zero distances) should give TM-score = n_aligned/L_target
        let distances_sq = vec![0.0; 100];
        let tm = tm_score(&distances_sq, 100);
        assert!((tm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tm_score_partial() {
        // Partial alignment: 50 residues aligned out of 100
        let distances_sq = vec![0.0; 50];
        let tm = tm_score(&distances_sq, 100);
        assert!((tm - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tm_score_with_distances() {
        // With non-zero distances, TM-score should be less than n_aligned/L_target
        let d0 = tm_d0(100);
        // If all distances equal d0, each term is 1/(1+1) = 0.5
        let distances_sq = vec![d0 * d0; 100];
        let tm = tm_score(&distances_sq, 100);
        assert!((tm - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tm_score_from_coords_identical() {
        let coords: Vec<Coord3> = (0..50).map(|i| Coord3::new(i as f64, 0.0, 0.0)).collect();
        let tm = tm_score_from_coords(&coords, &coords, 50);
        assert!((tm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tm_align_full() {
        let coords1: Vec<Coord3> = (0..50).map(|i| Coord3::new(i as f64, 0.0, 0.0)).collect();
        let coords2: Vec<Coord3> = (0..50)
            .map(|i| Coord3::new(i as f64 + 0.5, 0.0, 0.0))
            .collect();

        let result = tm_align_full(&coords1, &coords2, 100, 80);

        // Check that both TM-scores are computed
        assert!(result.tm_score_1 > 0.0 && result.tm_score_1 < 1.0);
        assert!(result.tm_score_2 > 0.0 && result.tm_score_2 < 1.0);
        // Shorter normalization length gives higher TM-score
        assert!(result.tm_score_2 > result.tm_score_1);
        assert_eq!(result.n_aligned, 50);
        assert!((result.rmsd - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tm_score_and_rmsd() {
        let coords1: Vec<Coord3> = (0..50).map(|i| Coord3::new(i as f64, 0.0, 0.0)).collect();
        let (tm, rmsd_val) = tm_score_and_rmsd(&coords1, &coords1, 50);
        assert!((tm - 1.0).abs() < 1e-10);
        assert!(rmsd_val.abs() < 1e-10);
    }
}
