"""Tests for the rivet Python module."""

import numpy as np
import pytest

import rivet


class TestDomain:
    """Tests for Domain class."""

    def test_create_empty_domain(self):
        """Test creating an empty domain."""
        domain = rivet.Domain("test")
        assert domain.id == "test"
        assert domain.chain == "A"
        assert len(domain) == 0

    def test_create_domain_with_chain(self):
        """Test creating a domain with specific chain."""
        domain = rivet.Domain("test", chain="B")
        assert domain.chain == "B"

    def test_from_arrays(self):
        """Test creating domain from arrays."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
        ])
        sequence = "AGA"

        domain = rivet.Domain.from_arrays("test", coords, sequence)
        assert len(domain) == 3
        assert domain.sequence == "AGA"

        retrieved_coords = domain.coordinates
        np.testing.assert_allclose(retrieved_coords, coords)

    def test_sequence_and_ss(self):
        """Test sequence and secondary structure access."""
        coords = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
        domain = rivet.Domain.from_arrays("test", coords, "AL")

        assert domain.sequence == "AL"
        assert len(domain.secondary_structure) == 2

    def test_copy(self):
        """Test domain copy."""
        coords = np.array([[0.0, 0.0, 0.0]])
        domain = rivet.Domain.from_arrays("orig", coords, "A")
        copied = domain.copy()

        assert copied.id == domain.id
        assert len(copied) == len(domain)

    def test_repr(self):
        """Test string representation."""
        domain = rivet.Domain("test")
        rep = repr(domain)
        assert "test" in rep
        assert "Domain" in rep


class TestParameters:
    """Tests for Parameters class."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = rivet.Parameters()
        assert params.n_passes == 2
        assert params.e1 == 2.0
        assert params.e2 == 5.0
        assert params.use_secondary is True

    def test_modify_parameters(self):
        """Test modifying parameters."""
        params = rivet.Parameters()
        params.n_passes = 3
        params.e1 = 1.5
        params.use_secondary = False

        assert params.n_passes == 3
        assert params.e1 == 1.5
        assert params.use_secondary is False

    def test_repr(self):
        """Test string representation."""
        params = rivet.Parameters()
        rep = repr(params)
        assert "Parameters" in rep


class TestTransform:
    """Tests for Transform class."""

    def test_identity_transform(self):
        """Test identity transformation."""
        t = rivet.Transform()
        coords = np.array([[1.0, 2.0, 3.0]])

        transformed = t.apply(coords)
        np.testing.assert_allclose(transformed, coords)

    def test_transformation_matrix(self):
        """Test 4x4 transformation matrix."""
        t = rivet.Transform()
        matrix = t.transformation_matrix()

        assert matrix.shape == (4, 4)
        np.testing.assert_allclose(matrix, np.eye(4))

    def test_inverse(self):
        """Test inverse transformation."""
        t = rivet.Transform()
        inv = t.inverse()
        composed = t.compose(inv)

        matrix = composed.transformation_matrix()
        np.testing.assert_allclose(matrix, np.eye(4), atol=1e-10)

    def test_repr(self):
        """Test string representation."""
        t = rivet.Transform()
        rep = repr(t)
        assert "Transform" in rep


class TestAlignment:
    """Tests for alignment functions."""

    @pytest.fixture
    def pdb_domains(self):
        """Load real PDB structures for alignment testing.

        Uses lysozyme structures (1lyz and 2lyz) which are homologous
        proteins that should align well with STAMP.
        """
        import os
        test_pdbs = os.path.join(os.path.dirname(__file__), "..", "..", "test_pdbs")
        d1 = rivet.Domain.from_pdb(os.path.join(test_pdbs, "1lyz.pdb"), chain="A")
        d2 = rivet.Domain.from_pdb(os.path.join(test_pdbs, "2lyz.pdb"), chain="A")
        return d1, d2

    @pytest.fixture
    def simple_domains(self):
        """Create simple test domains for basic functionality tests."""
        n_residues = 20
        coords1 = []
        coords2 = []

        for i in range(n_residues):
            x = i * 3.8
            y = 2.0 * np.sin(np.radians(100 * i))
            z = 2.0 * np.cos(np.radians(100 * i))
            coords1.append([x, y, z])
            coords2.append([x + 0.5, y + 0.3, z + 0.2])

        coords1 = np.array(coords1)
        coords2 = np.array(coords2)
        sequence = "A" * n_residues

        d1 = rivet.Domain.from_arrays("domain1", coords1, sequence)
        d2 = rivet.Domain.from_arrays("domain2", coords2, sequence)

        return d1, d2

    def test_pairwise_real_structures(self, pdb_domains):
        """Test alignment of real protein structures (lysozyme)."""
        d1, d2 = pdb_domains
        result = rivet.pairwise_align(d1, d2, scan_mode=True)

        # Lysozyme structures should align well
        assert result.rmsd < 5.0
        assert result.n_aligned > 50  # Should align most of the ~130 residues
        assert result.score >= 0

    def test_pairwise_with_params(self, pdb_domains):
        """Test alignment with custom parameters."""
        d1, d2 = pdb_domains
        params = rivet.Parameters()
        params.n_passes = 1

        result = rivet.pairwise_align(d1, d2, params=params, scan_mode=True)
        assert result.n_aligned > 0

    def test_transformation_matrix(self, simple_domains):
        """Test getting transformation matrix."""
        d1, d2 = simple_domains
        result = rivet.pairwise_align(d1, d2)

        matrix = result.transformation_matrix()
        assert matrix.shape == (4, 4)

    def test_aligned_pairs(self, pdb_domains):
        """Test getting aligned pairs."""
        d1, d2 = pdb_domains
        result = rivet.pairwise_align(d1, d2, scan_mode=True)

        # Verify the result object is valid
        pairs = result.aligned_pairs
        assert isinstance(pairs, list)


class TestMultipleAlignment:
    """Tests for multiple alignment."""

    @pytest.fixture
    def multiple_domains(self):
        """Create multiple test domains with alpha-helix geometry."""
        n_residues = 15
        coords = []
        for i in range(n_residues):
            x = i * 1.5
            y = 2.3 * np.cos(np.radians(100 * i))
            z = 2.3 * np.sin(np.radians(100 * i))
            coords.append([x, y, z])
        coords = np.array(coords)
        sequence = "A" * n_residues

        domains = []
        for i in range(3):
            d = rivet.Domain.from_arrays(f"domain{i}", coords.copy(), sequence)
            domains.append(d)

        return domains

    def test_multiple_align(self, multiple_domains):
        """Test multiple alignment."""
        result = rivet.multiple_align(multiple_domains)

        assert result.n_columns > 0
        assert result.n_core >= 0
        assert len(result.transforms) == len(multiple_domains)

    def test_multiple_align_min_domains(self):
        """Test that multiple alignment requires at least 2 domains."""
        coords = np.array([[0.0, 0.0, 0.0]])
        d = rivet.Domain.from_arrays("single", coords, "A")

        with pytest.raises(ValueError):
            rivet.multiple_align([d])


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_rmsd_identical(self):
        """Test RMSD of identical coordinates."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        rmsd = rivet.compute_rmsd(coords, coords)
        assert abs(rmsd) < 1e-10

    def test_compute_rmsd_different(self):
        """Test RMSD of different coordinates."""
        coords1 = np.array([[0.0, 0.0, 0.0]])
        coords2 = np.array([[1.0, 0.0, 0.0]])

        rmsd = rivet.compute_rmsd(coords1, coords2)
        assert abs(rmsd - 1.0) < 1e-10

    def test_superpose(self):
        """Test superposition."""
        fixed = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        # Translated copy
        mobile = fixed + np.array([5.0, 5.0, 0.0])

        transform, rmsd = rivet.superpose(fixed, mobile)

        assert rmsd < 1e-10
        assert isinstance(transform, rivet.Transform)

    def test_distance_matrix(self):
        """Test distance matrix computation."""
        coords1 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        coords2 = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ])

        dist = rivet.distance_matrix(coords1, coords2)

        assert dist.shape == (2, 2)
        assert abs(dist[0, 0]) < 1e-10  # origin to origin
        assert abs(dist[0, 1] - 5.0) < 1e-10  # origin to (3,4,0)

    def test_centroid(self):
        """Test centroid computation."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ])

        cx, cy, cz = rivet.centroid(coords)

        assert abs(cx - 2.0 / 3.0) < 1e-10
        assert abs(cy - 2.0 / 3.0) < 1e-10
        assert abs(cz) < 1e-10


class TestVersion:
    """Tests for version information."""

    def test_version_exists(self):
        """Test that version is defined."""
        assert hasattr(rivet, "__version__")
        assert isinstance(rivet.__version__, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
