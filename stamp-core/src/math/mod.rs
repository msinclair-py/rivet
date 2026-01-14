//! Linear algebra operations and geometric transformations.
//!
//! This module provides mathematical utilities for structural alignment,
//! including the Kabsch algorithm for optimal superposition, centroid
//! calculations, and RMSD computation.

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
}
