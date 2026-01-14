//! Python bindings for STAMP structural alignment library.
//!
//! This module provides Python access to STAMP functionality via PyO3.
//!
//! # Usage
//!
//! ```python
//! import stamp
//!
//! # Load structures
//! domain_a = stamp.Domain.from_pdb("1abc.pdb", chain="A")
//! domain_b = stamp.Domain.from_pdb("2xyz.pdb", chain="B")
//!
//! # Pairwise alignment
//! result = stamp.pairwise_align(domain_a, domain_b)
//! print(f"RMSD: {result.rmsd}, Score: {result.score}")
//!
//! # Multiple alignment
//! domains = [stamp.Domain.from_pdb(f) for f in files]
//! msa = stamp.multiple_align(domains)
//! ```

use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use stamp_core::io::{parse_dssp, parse_pdb, write_pdb};
use stamp_core::treewise::multiple_align as core_multiple_align;
use stamp_core::types::{
    Coord3, Domain as CoreDomain, Parameters as CoreParameters, Residue, Transform as CoreTransform,
};

// ============================================================================
// Domain class
// ============================================================================

/// Python wrapper for Domain.
///
/// Represents a protein domain with C-alpha coordinates and optional
/// secondary structure information.
#[pyclass(name = "Domain")]
#[derive(Clone)]
pub struct PyDomain {
    inner: CoreDomain,
}

#[pymethods]
impl PyDomain {
    /// Creates a new empty domain with the given identifier.
    ///
    /// Args:
    ///     id: Domain identifier string.
    ///     chain: Optional chain identifier (default 'A').
    #[new]
    #[pyo3(signature = (id, chain=None))]
    fn new(id: String, chain: Option<char>) -> Self {
        let mut inner = CoreDomain::new(id);
        if let Some(c) = chain {
            inner.chain = c;
        }
        Self { inner }
    }

    /// Creates a domain from a PDB file.
    ///
    /// Args:
    ///     path: Path to PDB file.
    ///     chain: Optional chain identifier. If None, uses first chain found.
    ///
    /// Returns:
    ///     Domain object with C-alpha coordinates.
    ///
    /// Raises:
    ///     ValueError: If the file cannot be parsed.
    ///     IOError: If the file cannot be read.
    #[staticmethod]
    #[pyo3(signature = (path, chain=None))]
    fn from_pdb(path: &str, chain: Option<char>) -> PyResult<Self> {
        let inner = parse_pdb(path, chain)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse PDB: {}", e)))?;
        Ok(Self { inner })
    }

    /// Creates a domain from a DSSP file.
    ///
    /// Loads both coordinates and secondary structure from DSSP.
    ///
    /// Args:
    ///     path: Path to DSSP file.
    ///     chain: Optional chain identifier.
    ///
    /// Returns:
    ///     Domain object with coordinates and secondary structure.
    #[staticmethod]
    #[pyo3(signature = (pdb_path, dssp_path, chain=None))]
    fn from_dssp(pdb_path: &str, dssp_path: &str, chain: Option<char>) -> PyResult<Self> {
        let mut inner = parse_pdb(pdb_path, chain)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse PDB: {}", e)))?;
        parse_dssp(dssp_path, &mut inner)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse DSSP: {}", e)))?;
        Ok(Self { inner })
    }

    /// Loads secondary structure from a DSSP file.
    ///
    /// Args:
    ///     path: Path to DSSP file.
    fn load_dssp(&mut self, path: &str) -> PyResult<()> {
        parse_dssp(path, &mut self.inner)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse DSSP: {}", e)))
    }

    /// Writes the domain to a PDB file.
    ///
    /// Args:
    ///     path: Output file path.
    ///     transform: Optional transformation to apply before writing.
    #[pyo3(signature = (path, transform=None))]
    fn to_pdb(&self, path: &str, transform: Option<&PyTransform>) -> PyResult<()> {
        let trans = transform.map(|t| t.inner.clone());
        write_pdb(path, &self.inner, trans.as_ref())
            .map_err(|e| PyIOError::new_err(format!("Failed to write PDB: {}", e)))
    }

    /// Returns the domain identifier.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// Sets the domain identifier.
    #[setter]
    fn set_id(&mut self, id: String) {
        self.inner.id = id;
    }

    /// Returns the chain identifier.
    #[getter]
    fn chain(&self) -> char {
        self.inner.chain
    }

    /// Sets the chain identifier.
    #[setter]
    fn set_chain(&mut self, chain: char) {
        self.inner.chain = chain;
    }

    /// Returns the number of residues.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Returns True if domain has no residues.
    fn __bool__(&self) -> bool {
        !self.inner.is_empty()
    }

    /// Returns the amino acid sequence as a string.
    #[getter]
    fn sequence(&self) -> String {
        self.inner.sequence()
    }

    /// Returns the secondary structure string.
    #[getter]
    fn secondary_structure(&self) -> String {
        self.inner.secondary_structure()
    }

    /// Returns True if DSSP data has been loaded.
    #[getter]
    fn has_dssp(&self) -> bool {
        self.inner.has_dssp
    }

    /// Returns C-alpha coordinates as numpy array (N x 3).
    #[getter]
    fn coordinates<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.len();
        let mut coords = Vec::with_capacity(n * 3);

        for residue in &self.inner.residues {
            coords.push(residue.ca_coord.x);
            coords.push(residue.ca_coord.y);
            coords.push(residue.ca_coord.z);
        }

        let arr = ndarray::Array2::from_shape_vec((n, 3), coords).unwrap();
        arr.to_pyarray(py)
    }

    /// Sets C-alpha coordinates from numpy array (N x 3).
    #[setter]
    fn set_coordinates(&mut self, coords: PyReadonlyArray2<f64>) -> PyResult<()> {
        let arr = coords.as_array();

        if arr.shape()[1] != 3 {
            return Err(PyValueError::new_err("Coordinates must have shape (N, 3)"));
        }

        if arr.shape()[0] != self.inner.len() {
            return Err(PyValueError::new_err(format!(
                "Expected {} coordinates, got {}",
                self.inner.len(),
                arr.shape()[0]
            )));
        }

        for (i, row) in arr.rows().into_iter().enumerate() {
            self.inner.residues[i].ca_coord = Coord3::new(row[0], row[1], row[2]);
        }

        Ok(())
    }

    /// Creates a domain from coordinates and sequence.
    ///
    /// Args:
    ///     id: Domain identifier.
    ///     coords: Nx3 numpy array of C-alpha coordinates.
    ///     sequence: Amino acid sequence string.
    ///     chain: Optional chain identifier (default 'A').
    ///
    /// Returns:
    ///     New Domain object.
    #[staticmethod]
    #[pyo3(signature = (id, coords, sequence, chain=None))]
    fn from_arrays(
        id: String,
        coords: PyReadonlyArray2<f64>,
        sequence: &str,
        chain: Option<char>,
    ) -> PyResult<Self> {
        let arr = coords.as_array();

        if arr.shape()[1] != 3 {
            return Err(PyValueError::new_err("Coordinates must have shape (N, 3)"));
        }

        let seq_chars: Vec<char> = sequence.chars().collect();
        if arr.shape()[0] != seq_chars.len() {
            return Err(PyValueError::new_err(
                "Number of coordinates must match sequence length",
            ));
        }

        let mut inner = CoreDomain::new(id);
        inner.chain = chain.unwrap_or('A');

        for (i, row) in arr.rows().into_iter().enumerate() {
            let residue = Residue::new(
                i as i32,
                (i + 1).to_string(),
                seq_chars[i],
                Coord3::new(row[0], row[1], row[2]),
            );
            inner.residues.push(residue);
        }

        Ok(Self { inner })
    }

    /// Returns a copy of this domain.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Returns secondary structure content as (helix, strand, coil) tuple.
    fn ss_content(&self) -> (f64, f64, f64) {
        stamp_core::secondary::ss_content(&self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "Domain(id='{}', chain='{}', n_residues={})",
            self.inner.id,
            self.inner.chain,
            self.inner.len()
        )
    }

    fn __str__(&self) -> String {
        format!(
            "<Domain {} chain {} with {} residues>",
            self.inner.id,
            self.inner.chain,
            self.inner.len()
        )
    }
}

// ============================================================================
// Parameters class
// ============================================================================

/// Python wrapper for alignment parameters.
///
/// All parameters have sensible defaults matching the original STAMP behavior.
#[pyclass(name = "Parameters")]
#[derive(Clone)]
pub struct PyParameters {
    inner: CoreParameters,
}

#[pymethods]
impl PyParameters {
    /// Creates parameters with default values.
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreParameters::default(),
        }
    }

    /// Number of refinement passes.
    #[getter]
    fn n_passes(&self) -> u32 {
        self.inner.n_passes
    }

    #[setter]
    fn set_n_passes(&mut self, value: u32) {
        self.inner.n_passes = value;
    }

    /// E1 parameter for Pij calculation.
    #[getter]
    fn e1(&self) -> f64 {
        self.inner.e1
    }

    #[setter]
    fn set_e1(&mut self, value: f64) {
        self.inner.e1 = value;
    }

    /// E2 parameter for Pij calculation.
    #[getter]
    fn e2(&self) -> f64 {
        self.inner.e2
    }

    #[setter]
    fn set_e2(&mut self, value: f64) {
        self.inner.e2 = value;
    }

    /// Gap opening penalty.
    #[getter]
    fn gap_open(&self) -> f64 {
        self.inner.pen_gap
    }

    #[setter]
    fn set_gap_open(&mut self, value: f64) {
        self.inner.pen_gap = value;
    }

    /// Gap extension penalty.
    #[getter]
    fn gap_extend(&self) -> f64 {
        self.inner.pen_ext
    }

    #[setter]
    fn set_gap_extend(&mut self, value: f64) {
        self.inner.pen_ext = value;
    }

    /// Whether to use secondary structure in scoring.
    #[getter]
    fn use_secondary(&self) -> bool {
        self.inner.use_secondary
    }

    #[setter]
    fn set_use_secondary(&mut self, value: bool) {
        self.inner.use_secondary = value;
    }

    /// Maximum iterations for refinement.
    #[getter]
    fn max_iter(&self) -> u32 {
        self.inner.max_iter
    }

    #[setter]
    fn set_max_iter(&mut self, value: u32) {
        self.inner.max_iter = value;
    }

    /// Distance cutoff for scoring.
    #[getter]
    fn cutoff_dist(&self) -> f64 {
        self.inner.cutoff_dist
    }

    #[setter]
    fn set_cutoff_dist(&mut self, value: f64) {
        self.inner.cutoff_dist = value;
    }

    /// Score cutoff for alignment.
    #[getter]
    fn score_cutoff(&self) -> f64 {
        self.inner.score_cutoff
    }

    #[setter]
    fn set_score_cutoff(&mut self, value: f64) {
        self.inner.score_cutoff = value;
    }

    /// Convergence threshold.
    #[getter]
    fn convergence(&self) -> f64 {
        self.inner.convergence
    }

    #[setter]
    fn set_convergence(&mut self, value: f64) {
        self.inner.convergence = value;
    }

    /// Verbosity level.
    #[getter]
    fn verbosity(&self) -> u32 {
        self.inner.verbosity
    }

    #[setter]
    fn set_verbosity(&mut self, value: u32) {
        self.inner.verbosity = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "Parameters(n_passes={}, e1={}, e2={}, gap_open={}, use_secondary={})",
            self.inner.n_passes,
            self.inner.e1,
            self.inner.e2,
            self.inner.pen_gap,
            self.inner.use_secondary
        )
    }
}

// ============================================================================
// Transform class
// ============================================================================

/// Python wrapper for 3D rigid body transformation.
///
/// Represents a rotation followed by translation.
#[pyclass(name = "Transform")]
#[derive(Clone)]
pub struct PyTransform {
    inner: CoreTransform,
}

#[pymethods]
impl PyTransform {
    /// Creates an identity transformation.
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreTransform::identity(),
        }
    }

    /// Creates a transformation from rotation matrix and translation vector.
    ///
    /// Args:
    ///     rotation: 3x3 rotation matrix as numpy array.
    ///     translation: 3-element translation vector as numpy array.
    #[staticmethod]
    fn from_matrix(
        rotation: PyReadonlyArray2<f64>,
        translation: PyReadonlyArray2<f64>,
    ) -> PyResult<Self> {
        let rot = rotation.as_array();
        let trans = translation.as_array();

        if rot.shape() != [3, 3] {
            return Err(PyValueError::new_err("Rotation must be 3x3 matrix"));
        }

        let rotation_matrix = stamp_core::types::RotationMatrix::new(
            rot[[0, 0]],
            rot[[0, 1]],
            rot[[0, 2]],
            rot[[1, 0]],
            rot[[1, 1]],
            rot[[1, 2]],
            rot[[2, 0]],
            rot[[2, 1]],
            rot[[2, 2]],
        );

        // Handle both (3,) and (3,1) or (1,3) shapes
        let (tx, ty, tz) = if trans.shape()[0] >= 3 {
            (trans[[0, 0]], trans[[1, 0]], trans[[2, 0]])
        } else if trans.shape().len() == 1 || trans.shape()[1] >= 3 {
            (trans[[0, 0]], trans[[0, 1]], trans[[0, 2]])
        } else {
            return Err(PyValueError::new_err(
                "Translation must be 3-element vector",
            ));
        };

        let translation_vec = stamp_core::types::Vec3::new(tx, ty, tz);

        Ok(Self {
            inner: CoreTransform {
                rotation: rotation_matrix,
                translation: translation_vec,
            },
        })
    }

    /// Returns the rotation matrix as 3x3 numpy array.
    #[getter]
    fn rotation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let mut flat = Vec::with_capacity(9);
        for i in 0..3 {
            for j in 0..3 {
                flat.push(self.inner.rotation[(i, j)]);
            }
        }
        let arr = ndarray::Array2::from_shape_vec((3, 3), flat).unwrap();
        arr.to_pyarray(py)
    }

    /// Returns the translation vector as numpy array.
    #[getter]
    fn translation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let arr = ndarray::Array1::from_vec(vec![
            self.inner.translation.x,
            self.inner.translation.y,
            self.inner.translation.z,
        ]);
        arr.to_pyarray(py)
    }

    /// Returns the full 4x4 transformation matrix.
    fn transformation_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let mut matrix = vec![0.0; 16];

        // Rotation part
        for i in 0..3 {
            for j in 0..3 {
                matrix[i * 4 + j] = self.inner.rotation[(i, j)];
            }
        }

        // Translation part
        matrix[3] = self.inner.translation.x;
        matrix[7] = self.inner.translation.y;
        matrix[11] = self.inner.translation.z;

        // Bottom row
        matrix[15] = 1.0;

        let arr = ndarray::Array2::from_shape_vec((4, 4), matrix).unwrap();
        arr.to_pyarray(py)
    }

    /// Applies this transformation to coordinates.
    ///
    /// Args:
    ///     coords: Nx3 array of coordinates.
    ///
    /// Returns:
    ///     Transformed Nx3 array.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        coords: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = coords.as_array();

        if arr.shape()[1] != 3 {
            return Err(PyValueError::new_err("Coordinates must have shape (N, 3)"));
        }

        let n = arr.shape()[0];
        let mut result = Vec::with_capacity(n * 3);

        for row in arr.rows() {
            let point = Coord3::new(row[0], row[1], row[2]);
            let transformed = self.inner.apply(&point);
            result.push(transformed.x);
            result.push(transformed.y);
            result.push(transformed.z);
        }

        let out_arr = ndarray::Array2::from_shape_vec((n, 3), result).unwrap();
        Ok(out_arr.to_pyarray(py))
    }

    /// Returns the inverse transformation.
    fn inverse(&self) -> Self {
        Self {
            inner: self.inner.inverse(),
        }
    }

    /// Composes this transformation with another (self * other).
    fn compose(&self, other: &PyTransform) -> Self {
        Self {
            inner: self.inner.compose(&other.inner),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Transform(translation=[{:.3}, {:.3}, {:.3}])",
            self.inner.translation.x, self.inner.translation.y, self.inner.translation.z
        )
    }
}

// ============================================================================
// AlignmentResult class
// ============================================================================

/// Python wrapper for pairwise alignment result.
///
/// Contains the transformation, statistics, and alignment path.
#[pyclass(name = "AlignmentResult")]
pub struct PyAlignmentResult {
    /// Root mean square deviation.
    #[pyo3(get)]
    rmsd: f64,
    /// Number of aligned positions.
    #[pyo3(get)]
    n_aligned: usize,
    /// STAMP score (Sc).
    #[pyo3(get)]
    score: f64,
    /// Sequence identity of aligned region.
    #[pyo3(get)]
    seq_identity: f64,
    /// Transformation to superpose second structure onto first.
    transform: CoreTransform,
    /// Alignment path (pairs of indices).
    path: Vec<(usize, usize)>,
}

#[pymethods]
impl PyAlignmentResult {
    /// Returns the transformation as a Transform object.
    #[getter]
    fn transform(&self) -> PyTransform {
        PyTransform {
            inner: self.transform.clone(),
        }
    }

    /// Returns the rotation matrix as numpy array (3x3).
    fn get_rotation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let mut flat = Vec::with_capacity(9);
        for i in 0..3 {
            for j in 0..3 {
                flat.push(self.transform.rotation[(i, j)]);
            }
        }
        let arr = ndarray::Array2::from_shape_vec((3, 3), flat).unwrap();
        arr.to_pyarray(py)
    }

    /// Returns the translation vector as numpy array (3,).
    fn get_translation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let arr = ndarray::Array1::from_vec(vec![
            self.transform.translation.x,
            self.transform.translation.y,
            self.transform.translation.z,
        ]);
        arr.to_pyarray(py)
    }

    /// Returns the 4x4 transformation matrix.
    fn transformation_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let mut matrix = vec![0.0; 16];

        for i in 0..3 {
            for j in 0..3 {
                matrix[i * 4 + j] = self.transform.rotation[(i, j)];
            }
        }

        matrix[3] = self.transform.translation.x;
        matrix[7] = self.transform.translation.y;
        matrix[11] = self.transform.translation.z;
        matrix[15] = 1.0;

        let arr = ndarray::Array2::from_shape_vec((4, 4), matrix).unwrap();
        arr.to_pyarray(py)
    }

    /// Returns the alignment path as list of (i, j) tuples.
    #[getter]
    fn aligned_pairs(&self) -> Vec<(usize, usize)> {
        self.path.clone()
    }

    /// Applies transformation to coordinates.
    ///
    /// Args:
    ///     coords: Nx3 array of coordinates.
    ///
    /// Returns:
    ///     Transformed Nx3 array.
    fn transform_coordinates<'py>(
        &self,
        py: Python<'py>,
        coords: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = coords.as_array();

        if arr.shape()[1] != 3 {
            return Err(PyValueError::new_err("Coordinates must have shape (N, 3)"));
        }

        let n = arr.shape()[0];
        let mut result = Vec::with_capacity(n * 3);

        for row in arr.rows() {
            let point = Coord3::new(row[0], row[1], row[2]);
            let transformed = self.transform.apply(&point);
            result.push(transformed.x);
            result.push(transformed.y);
            result.push(transformed.z);
        }

        let out_arr = ndarray::Array2::from_shape_vec((n, 3), result).unwrap();
        Ok(out_arr.to_pyarray(py))
    }

    /// Returns the transformed coordinates of the mobile domain.
    fn get_transformed_coordinates<'py>(
        &self,
        py: Python<'py>,
        domain: &PyDomain,
    ) -> Bound<'py, PyArray2<f64>> {
        let n = domain.inner.len();
        let mut coords = Vec::with_capacity(n * 3);

        for residue in &domain.inner.residues {
            let transformed = self.transform.apply(&residue.ca_coord);
            coords.push(transformed.x);
            coords.push(transformed.y);
            coords.push(transformed.z);
        }

        let arr = ndarray::Array2::from_shape_vec((n, 3), coords).unwrap();
        arr.to_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "AlignmentResult(rmsd={:.3}, n_aligned={}, score={:.4}, seq_identity={:.3})",
            self.rmsd, self.n_aligned, self.score, self.seq_identity
        )
    }
}

// ============================================================================
// MultipleAlignmentResult class
// ============================================================================

/// Python wrapper for multiple alignment result.
#[pyclass(name = "MultipleAlignmentResult")]
pub struct PyMultipleAlignmentResult {
    /// Average pairwise RMSD.
    #[pyo3(get)]
    avg_rmsd: f64,
    /// Overall alignment score.
    #[pyo3(get)]
    score: f64,
    /// Number of alignment columns.
    #[pyo3(get)]
    n_columns: usize,
    /// Number of core positions (all structures present).
    #[pyo3(get)]
    n_core: usize,
    /// Transformations for each domain.
    transforms: Vec<CoreTransform>,
    /// Core position indices.
    core_positions: Vec<usize>,
    /// Alignment columns.
    columns: Vec<Vec<Option<usize>>>,
}

#[pymethods]
impl PyMultipleAlignmentResult {
    /// Returns the transformation for a specific domain.
    fn get_transform(&self, index: usize) -> PyResult<PyTransform> {
        self.transforms
            .get(index)
            .map(|t| PyTransform { inner: t.clone() })
            .ok_or_else(|| PyValueError::new_err(format!("Invalid domain index: {}", index)))
    }

    /// Returns all transformations as a list.
    #[getter]
    fn transforms(&self) -> Vec<PyTransform> {
        self.transforms
            .iter()
            .map(|t| PyTransform { inner: t.clone() })
            .collect()
    }

    /// Returns core position indices.
    #[getter]
    fn core_positions(&self) -> Vec<usize> {
        self.core_positions.clone()
    }

    /// Returns alignment columns as list of lists.
    ///
    /// Each column is a list where element i is the residue index for
    /// domain i, or None if that domain has a gap at this position.
    #[getter]
    fn columns(&self) -> Vec<Vec<Option<usize>>> {
        self.columns.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "MultipleAlignmentResult(n_structures={}, n_columns={}, n_core={}, avg_rmsd={:.3})",
            self.transforms.len(),
            self.n_columns,
            self.n_core,
            self.avg_rmsd
        )
    }
}

// ============================================================================
// ScanHit class
// ============================================================================

/// Result of a database scan hit.
#[pyclass(name = "ScanHit")]
pub struct PyScanHit {
    /// Target domain identifier.
    #[pyo3(get)]
    target_id: String,
    /// Rank in hit list.
    #[pyo3(get)]
    rank: usize,
    /// STAMP score.
    #[pyo3(get)]
    score: f64,
    /// RMSD after superposition.
    #[pyo3(get)]
    rmsd: f64,
    /// Number of aligned positions.
    #[pyo3(get)]
    n_aligned: usize,
    /// Sequence identity.
    #[pyo3(get)]
    seq_identity: f64,
    /// Z-score (if computed).
    #[pyo3(get)]
    z_score: Option<f64>,
    /// E-value (if computed).
    #[pyo3(get)]
    e_value: Option<f64>,
}

#[pymethods]
impl PyScanHit {
    fn __repr__(&self) -> String {
        format!(
            "ScanHit(target='{}', rank={}, score={:.4}, rmsd={:.3})",
            self.target_id, self.rank, self.score, self.rmsd
        )
    }
}

// ============================================================================
// Module functions
// ============================================================================

/// Performs pairwise structural alignment.
///
/// Args:
///     domain1: First (reference) domain.
///     domain2: Second (mobile) domain.
///     params: Optional alignment parameters.
///
/// Returns:
///     AlignmentResult containing transformation and statistics.
///
/// Raises:
///     ValueError: If alignment fails.
#[pyfunction]
#[pyo3(name = "pairwise_align", signature = (domain1, domain2, params=None, scan_mode=None, slide=None))]
fn pairwise_align(
    domain1: &PyDomain,
    domain2: &PyDomain,
    params: Option<&PyParameters>,
    scan_mode: Option<bool>,
    slide: Option<usize>,
) -> PyResult<PyAlignmentResult> {
    let core_params = params.map(|p| p.inner.clone()).unwrap_or_default();

    // Use scan mode for structures in different orientations
    let result = if scan_mode.unwrap_or(false) {
        use stamp_core::scan::{scan_pair, ScanConfig};

        let scan_config = ScanConfig {
            slide: slide.unwrap_or(5),
            n_passes: core_params.n_passes,
            e1: core_params.e1,
            e2: core_params.e2,
            gap_penalty: core_params.pen_gap,
            cutoff: core_params.cutoff_dist,
            max_iter: core_params.max_iter,
            convergence: core_params.convergence,
            ..ScanConfig::default()
        };

        let scan_result = scan_pair(&domain1.inner, &domain2.inner, &scan_config, &core_params)
            .map_err(|e| PyValueError::new_err(format!("Scan alignment failed: {}", e)))?;

        // Convert ScanResult to AlignmentResult format
        stamp_core::types::AlignmentResult {
            transform: scan_result.transform,
            rmsd: scan_result.rmsd,
            n_aligned: scan_result.n_aligned,
            score: scan_result.score,
            seq_identity: scan_result.seq_identity / 100.0,
            path: Vec::new(),
            residue_scores: Vec::new(),
        }
    } else {
        stamp_core::pairwise::align_pair(&domain1.inner, &domain2.inner, &core_params)
            .map_err(|e| PyValueError::new_err(format!("Alignment failed: {}", e)))?
    };

    Ok(PyAlignmentResult {
        rmsd: result.rmsd,
        n_aligned: result.n_aligned,
        score: result.score,
        seq_identity: result.seq_identity,
        transform: result.transform,
        path: result.path,
    })
}

/// Performs multiple structure alignment.
///
/// Args:
///     domains: List of Domain objects to align.
///     params: Optional alignment parameters.
///
/// Returns:
///     MultipleAlignmentResult with transformations and alignment.
///
/// Raises:
///     ValueError: If alignment fails or fewer than 2 domains provided.
#[pyfunction]
#[pyo3(name = "multiple_align", signature = (domains, params=None))]
fn multiple_align(
    domains: Vec<PyRef<PyDomain>>,
    params: Option<&PyParameters>,
) -> PyResult<PyMultipleAlignmentResult> {
    if domains.len() < 2 {
        return Err(PyValueError::new_err(
            "Need at least 2 domains for multiple alignment",
        ));
    }

    let core_domains: Vec<CoreDomain> = domains.iter().map(|d| d.inner.clone()).collect();
    let core_params = params.map(|p| p.inner.clone()).unwrap_or_default();

    let result = core_multiple_align(&core_domains, &core_params)
        .map_err(|e| PyValueError::new_err(format!("Multiple alignment failed: {}", e)))?;

    Ok(PyMultipleAlignmentResult {
        avg_rmsd: result.avg_rmsd,
        score: result.score,
        n_columns: result.n_columns(),
        n_core: result.n_core(),
        transforms: result.transforms,
        core_positions: result.core_positions,
        columns: result.columns,
    })
}

/// Scans a query structure against a database.
///
/// Args:
///     query: Query domain.
///     targets: List of target domains.
///     params: Optional alignment parameters.
///     score_cutoff: Minimum score for reporting hits (default 0.0).
///     max_hits: Maximum number of hits to return (default 100).
///     quick: Use quick alignment mode (default False).
///
/// Returns:
///     List of ScanHit objects sorted by score.
#[pyfunction]
#[pyo3(name = "scan_database", signature = (query, targets, params=None, score_cutoff=0.0, max_hits=100, quick=false))]
fn scan_database(
    query: &PyDomain,
    targets: Vec<PyRef<PyDomain>>,
    params: Option<&PyParameters>,
    score_cutoff: f64,
    max_hits: usize,
    quick: bool,
) -> PyResult<Vec<PyScanHit>> {
    let core_params = params.map(|p| p.inner.clone()).unwrap_or_default();

    let mut scanner = stamp_core::scan::DatabaseScanner::new(core_params);
    scanner.set_quick_scan(quick);
    scanner.set_score_threshold(score_cutoff);
    scanner.set_max_hits(max_hits);

    for target in targets {
        scanner.add_target(target.inner.clone());
    }

    let hits = scanner
        .scan(&query.inner)
        .map_err(|e| PyValueError::new_err(format!("Scan failed: {}", e)))?;

    Ok(hits
        .into_iter()
        .map(|h| PyScanHit {
            target_id: h.target_id,
            rank: h.rank,
            score: h.alignment.score,
            rmsd: h.alignment.rmsd,
            n_aligned: h.alignment.n_aligned,
            seq_identity: h.alignment.seq_identity,
            z_score: h.z_score,
            e_value: h.e_value,
        })
        .collect())
}

/// Computes RMSD between two coordinate sets.
///
/// Args:
///     coords1: First coordinate array (N x 3).
///     coords2: Second coordinate array (N x 3).
///
/// Returns:
///     RMSD value in same units as input coordinates.
#[pyfunction]
fn compute_rmsd(coords1: PyReadonlyArray2<f64>, coords2: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let arr1 = coords1.as_array();
    let arr2 = coords2.as_array();

    if arr1.shape() != arr2.shape() {
        return Err(PyValueError::new_err(
            "Coordinate arrays must have same shape",
        ));
    }

    if arr1.shape()[1] != 3 {
        return Err(PyValueError::new_err("Coordinates must have shape (N, 3)"));
    }

    let points1: Vec<Coord3> = arr1
        .rows()
        .into_iter()
        .map(|r| Coord3::new(r[0], r[1], r[2]))
        .collect();

    let points2: Vec<Coord3> = arr2
        .rows()
        .into_iter()
        .map(|r| Coord3::new(r[0], r[1], r[2]))
        .collect();

    Ok(stamp_core::math::rmsd(points1.iter(), points2.iter()))
}

/// Computes optimal superposition of two coordinate sets.
///
/// Args:
///     fixed: Reference coordinates (N x 3).
///     mobile: Coordinates to transform (N x 3).
///
/// Returns:
///     Tuple of (Transform, rmsd).
#[pyfunction]
fn superpose(
    fixed: PyReadonlyArray2<f64>,
    mobile: PyReadonlyArray2<f64>,
) -> PyResult<(PyTransform, f64)> {
    let arr1 = fixed.as_array();
    let arr2 = mobile.as_array();

    if arr1.shape() != arr2.shape() {
        return Err(PyValueError::new_err(
            "Coordinate arrays must have same shape",
        ));
    }

    let points1: Vec<Coord3> = arr1
        .rows()
        .into_iter()
        .map(|r| Coord3::new(r[0], r[1], r[2]))
        .collect();

    let points2: Vec<Coord3> = arr2
        .rows()
        .into_iter()
        .map(|r| Coord3::new(r[0], r[1], r[2]))
        .collect();

    let (transform, rmsd) = stamp_core::math::superpose(&points1, &points2)
        .map_err(|e| PyValueError::new_err(format!("Superposition failed: {}", e)))?;

    Ok((PyTransform { inner: transform }, rmsd))
}

/// Computes distance matrix between two coordinate sets.
///
/// Args:
///     coords1: First coordinate array (M x 3).
///     coords2: Second coordinate array (N x 3).
///
/// Returns:
///     MxN distance matrix as numpy array.
#[pyfunction]
fn distance_matrix<'py>(
    py: Python<'py>,
    coords1: PyReadonlyArray2<f64>,
    coords2: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr1 = coords1.as_array();
    let arr2 = coords2.as_array();

    if arr1.shape()[1] != 3 || arr2.shape()[1] != 3 {
        return Err(PyValueError::new_err("Coordinates must have shape (N, 3)"));
    }

    let m = arr1.shape()[0];
    let n = arr2.shape()[0];

    let points1: Vec<Coord3> = arr1
        .rows()
        .into_iter()
        .map(|r| Coord3::new(r[0], r[1], r[2]))
        .collect();

    let points2: Vec<Coord3> = arr2
        .rows()
        .into_iter()
        .map(|r| Coord3::new(r[0], r[1], r[2]))
        .collect();

    let dist = stamp_core::math::distance_matrix(&points1, &points2);
    let flat: Vec<f64> = dist.into_iter().flatten().collect();

    let arr = ndarray::Array2::from_shape_vec((m, n), flat).unwrap();
    Ok(arr.to_pyarray(py))
}

/// Computes centroid of a coordinate set.
///
/// Args:
///     coords: Coordinate array (N x 3).
///
/// Returns:
///     Centroid as (x, y, z) tuple.
#[pyfunction]
fn centroid(coords: PyReadonlyArray2<f64>) -> PyResult<(f64, f64, f64)> {
    let arr = coords.as_array();

    if arr.shape()[1] != 3 {
        return Err(PyValueError::new_err("Coordinates must have shape (N, 3)"));
    }

    let points: Vec<Coord3> = arr
        .rows()
        .into_iter()
        .map(|r| Coord3::new(r[0], r[1], r[2]))
        .collect();

    let c = stamp_core::math::centroid(points.iter());
    Ok((c.x, c.y, c.z))
}

// ============================================================================
// Module definition
// ============================================================================

/// Rivet - Structural Alignment of Multiple Proteins (STAMP implementation)
///
/// This module provides Python bindings to the STAMP structural alignment
/// library, enabling pairwise and multiple structure alignment.
///
/// Example:
///     >>> import rivet
///     >>> d1 = rivet.Domain.from_pdb("1abc.pdb", chain='A')
///     >>> d2 = rivet.Domain.from_pdb("2def.pdb", chain='A')
///     >>> result = rivet.pairwise_align(d1, d2)
///     >>> print(f"RMSD: {result.rmsd:.2f}")
#[pymodule]
fn rivet(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyDomain>()?;
    m.add_class::<PyParameters>()?;
    m.add_class::<PyTransform>()?;
    m.add_class::<PyAlignmentResult>()?;
    m.add_class::<PyMultipleAlignmentResult>()?;
    m.add_class::<PyScanHit>()?;

    // Alignment functions
    m.add_function(wrap_pyfunction!(pairwise_align, m)?)?;
    m.add_function(wrap_pyfunction!(multiple_align, m)?)?;
    m.add_function(wrap_pyfunction!(scan_database, m)?)?;

    // Utility functions
    m.add_function(wrap_pyfunction!(compute_rmsd, m)?)?;
    m.add_function(wrap_pyfunction!(superpose, m)?)?;
    m.add_function(wrap_pyfunction!(distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(centroid, m)?)?;

    // Version
    m.add("__version__", stamp_core::VERSION)?;

    Ok(())
}
