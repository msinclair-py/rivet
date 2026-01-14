//! Efficient matrix types with contiguous memory layout.
//!
//! This module provides matrix types optimized for cache-friendly access patterns
//! and SIMD operations. All matrices use row-major, contiguous storage instead of
//! nested `Vec<Vec<T>>` for better performance.

use std::ops::{Index, IndexMut};

/// A 2D matrix with contiguous row-major storage.
///
/// This is more efficient than `Vec<Vec<T>>` because:
/// - Single allocation instead of `rows + 1` allocations
/// - Better cache locality for row-wise iteration
/// - Enables SIMD operations on contiguous data
/// - No pointer indirection for element access
#[derive(Debug, Clone)]
pub struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Clone + Default> Matrix<T> {
    /// Creates a new matrix filled with default values.
    #[must_use]
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![T::default(); rows * cols],
            rows,
            cols,
        }
    }

    /// Creates a matrix filled with a specific value.
    #[must_use]
    pub fn filled(rows: usize, cols: usize, value: T) -> Self {
        Self {
            data: vec![value; rows * cols],
            rows,
            cols,
        }
    }
}

impl<T> Matrix<T> {
    /// Creates a matrix from existing data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != rows * cols`.
    #[must_use]
    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length {} doesn't match dimensions {}x{}",
            data.len(),
            rows,
            cols
        );
        Self { data, rows, cols }
    }

    /// Returns the number of rows.
    #[inline]
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    #[inline]
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the total number of elements.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the matrix is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Gets a reference to the element at (row, col).
    #[inline]
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            Some(&self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// Gets a mutable reference to the element at (row, col).
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.rows && col < self.cols {
            Some(&mut self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// Sets the value at (row, col).
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.rows && col < self.cols);
        self.data[row * self.cols + col] = value;
    }

    /// Returns a slice of a single row.
    #[inline]
    #[must_use]
    pub fn row(&self, row: usize) -> &[T] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Returns a mutable slice of a single row.
    #[inline]
    pub fn row_mut(&mut self, row: usize) -> &mut [T] {
        let start = row * self.cols;
        &mut self.data[start..start + self.cols]
    }

    /// Returns the raw data slice for SIMD operations.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns the raw data slice mutably.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consumes the matrix and returns the underlying data.
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Iterator over all elements in row-major order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Mutable iterator over all elements.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }

    /// Iterator over rows.
    pub fn rows_iter(&self) -> impl Iterator<Item = &[T]> {
        self.data.chunks(self.cols)
    }

    /// Iterator over (row, col, value) tuples.
    pub fn enumerate(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        self.data.iter().enumerate().map(move |(idx, val)| {
            let row = idx / self.cols;
            let col = idx % self.cols;
            (row, col, val)
        })
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &T {
        &self.data[row * self.cols + col]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        &mut self.data[row * self.cols + col]
    }
}

/// Symmetric matrix storing only the upper triangle.
///
/// This saves approximately 50% memory for distance/similarity matrices.
/// Elements on the diagonal are NOT stored (assumed to be 0 or identity).
#[derive(Debug, Clone)]
pub struct SymmetricMatrix<T> {
    data: Vec<T>,
    n: usize,
}

impl<T: Clone + Default> SymmetricMatrix<T> {
    /// Creates a new symmetric matrix of size n×n.
    #[must_use]
    pub fn new(n: usize) -> Self {
        // Store upper triangle: n*(n-1)/2 elements
        let size = n * (n - 1) / 2;
        Self {
            data: vec![T::default(); size],
            n,
        }
    }
}

impl<T> SymmetricMatrix<T> {
    /// Returns the dimension of the matrix.
    #[inline]
    #[must_use]
    pub fn size(&self) -> usize {
        self.n
    }

    /// Computes the linear index for upper-triangular storage.
    #[inline]
    fn linear_index(&self, i: usize, j: usize) -> usize {
        let (min, max) = if i < j { (i, j) } else { (j, i) };
        // Index into upper triangle: sum of (n-1) + (n-2) + ... + (n-min) + (max - min - 1)
        min * self.n - min * (min + 1) / 2 + max - min - 1
    }

    /// Gets the value at (i, j).
    ///
    /// # Panics
    ///
    /// Panics if i == j (diagonal not stored) or indices out of bounds.
    #[inline]
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> &T {
        assert!(i != j, "Diagonal elements not stored in SymmetricMatrix");
        assert!(i < self.n && j < self.n);
        &self.data[self.linear_index(i, j)]
    }

    /// Sets the value at (i, j). Due to symmetry, also sets (j, i).
    ///
    /// # Panics
    ///
    /// Panics if i == j or indices out of bounds.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: T) {
        assert!(i != j, "Cannot set diagonal in SymmetricMatrix");
        assert!(i < self.n && j < self.n);
        let idx = self.linear_index(i, j);
        self.data[idx] = value;
    }

    /// Returns the raw data slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
}

impl<T: Clone + Default> SymmetricMatrix<T> {
    /// Gets the value at (i, j), returning the default for diagonal.
    #[must_use]
    pub fn get_or_default(&self, i: usize, j: usize) -> T {
        if i == j {
            T::default()
        } else {
            self.get(i, j).clone()
        }
    }
}

/// Structure-of-Arrays layout for 3D coordinates.
///
/// This layout is optimal for SIMD operations because x, y, z values
/// are stored contiguously, allowing vectorized processing.
#[derive(Debug, Clone, Default)]
pub struct CoordsSoA {
    /// X coordinates.
    pub x: Vec<f64>,
    /// Y coordinates.
    pub y: Vec<f64>,
    /// Z coordinates.
    pub z: Vec<f64>,
}

impl CoordsSoA {
    /// Creates empty coordinate arrays.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates coordinate arrays with the given capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            x: Vec::with_capacity(capacity),
            y: Vec::with_capacity(capacity),
            z: Vec::with_capacity(capacity),
        }
    }

    /// Creates SoA layout from Array-of-Structures (Coord3 slice).
    #[must_use]
    pub fn from_coords(coords: &[crate::types::Coord3]) -> Self {
        let n = coords.len();
        let mut soa = Self::with_capacity(n);
        for c in coords {
            soa.x.push(c.x);
            soa.y.push(c.y);
            soa.z.push(c.z);
        }
        soa
    }

    /// Returns the number of coordinates.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Returns true if empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Pushes a new coordinate.
    pub fn push(&mut self, x: f64, y: f64, z: f64) {
        self.x.push(x);
        self.y.push(y);
        self.z.push(z);
    }

    /// Gets the coordinate at index i as a tuple.
    #[inline]
    #[must_use]
    pub fn get(&self, i: usize) -> (f64, f64, f64) {
        (self.x[i], self.y[i], self.z[i])
    }

    /// Converts back to Array-of-Structures layout.
    #[must_use]
    pub fn to_coords(&self) -> Vec<crate::types::Coord3> {
        (0..self.len())
            .map(|i| crate::types::Coord3::new(self.x[i], self.y[i], self.z[i]))
            .collect()
    }

    /// Computes the centroid.
    #[must_use]
    pub fn centroid(&self) -> (f64, f64, f64) {
        if self.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let n = self.len() as f64;
        let cx: f64 = self.x.iter().sum::<f64>() / n;
        let cy: f64 = self.y.iter().sum::<f64>() / n;
        let cz: f64 = self.z.iter().sum::<f64>() / n;
        (cx, cy, cz)
    }

    /// Centers the coordinates by subtracting the centroid.
    pub fn center(&mut self) {
        let (cx, cy, cz) = self.centroid();
        for x in &mut self.x {
            *x -= cx;
        }
        for y in &mut self.y {
            *y -= cy;
        }
        for z in &mut self.z {
            *z -= cz;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_basic() {
        let mut m: Matrix<f64> = Matrix::new(3, 4);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        assert_eq!(m.len(), 12);

        m.set(1, 2, 5.0);
        assert!((m[(1, 2)] - 5.0).abs() < 1e-10);
        assert!((m.get(1, 2).unwrap() - &5.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_row_access() {
        let mut m: Matrix<i32> = Matrix::new(3, 4);
        for i in 0..4 {
            m.set(1, i, i as i32);
        }

        let row = m.row(1);
        assert_eq!(row, &[0, 1, 2, 3]);
    }

    #[test]
    fn test_matrix_iteration() {
        let m: Matrix<i32> = Matrix::filled(2, 3, 1);
        let sum: i32 = m.iter().sum();
        assert_eq!(sum, 6);
    }

    #[test]
    fn test_symmetric_matrix() {
        let mut sm: SymmetricMatrix<f64> = SymmetricMatrix::new(5);
        sm.set(0, 3, 1.5);
        assert!((sm.get(0, 3) - &1.5).abs() < 1e-10);
        assert!((sm.get(3, 0) - &1.5).abs() < 1e-10); // Symmetric access
    }

    #[test]
    fn test_symmetric_matrix_size() {
        let sm: SymmetricMatrix<f64> = SymmetricMatrix::new(100);
        // Should store 100*99/2 = 4950 elements (not 10000)
        assert_eq!(sm.as_slice().len(), 4950);
    }

    #[test]
    fn test_coords_soa() {
        let coords = vec![
            crate::types::Coord3::new(1.0, 2.0, 3.0),
            crate::types::Coord3::new(4.0, 5.0, 6.0),
        ];
        let soa = CoordsSoA::from_coords(&coords);

        assert_eq!(soa.len(), 2);
        assert_eq!(soa.x, vec![1.0, 4.0]);
        assert_eq!(soa.y, vec![2.0, 5.0]);
        assert_eq!(soa.z, vec![3.0, 6.0]);
    }

    #[test]
    fn test_coords_soa_centroid() {
        let mut soa = CoordsSoA::new();
        soa.push(0.0, 0.0, 0.0);
        soa.push(2.0, 4.0, 6.0);

        let (cx, cy, cz) = soa.centroid();
        assert!((cx - 1.0).abs() < 1e-10);
        assert!((cy - 2.0).abs() < 1e-10);
        assert!((cz - 3.0).abs() < 1e-10);
    }
}
