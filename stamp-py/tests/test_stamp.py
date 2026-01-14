"""Tests for the stamp Python module."""

import numpy as np
import pytest

import stamp


class TestDomain:
    """Tests for Domain class."""

    def test_create_empty_domain(self):
        """Test creating an empty domain."""
        domain = stamp.Domain("test")
        assert domain.id == "test"
        assert domain.chain == "A"
        assert len(domain) == 0

    def test_create_domain_with_chain(self):
        """Test creating a domain with specific chain."""
        domain = stamp.Domain("test", chain="B")
        assert domain.chain == "B"

    def test_from_arrays(self):
        """Test creating domain from arrays."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
        ])
        sequence = "AGA"

        domain = stamp.Domain.from_arrays("test", coords, sequence)
        assert len(domain) == 3
        assert domain.sequence == "AGA"

        retrieved_coords = domain.coordinates
        np.testing.assert_allclose(retrieved_coords, coords)

    def test_sequence_and_ss(self):
        """Test sequence and secondary structure access."""
        coords = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
        domain = stamp.Domain.from_arrays("test", coords, "AL")

        assert domain.sequence == "AL"
        assert len(domain.secondary_structure) == 2

    def test_copy(self):
        """Test domain copy."""
        coords = np.array([[0.0, 0.0, 0.0]])
        domain = stamp.Domain.from_arrays("orig", coords, "A")
        copied = domain.copy()

        assert copied.id == domain.id
        assert len(copied) == len(domain)

    def test_repr(self):
        """Test string representation."""
        domain = stamp.Domain("test")
        rep = repr(domain)
        assert "test" in rep
        assert "Domain" in rep


class TestParameters:
    """Tests for Parameters class."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = stamp.Parameters()
        assert params.n_passes == 2
        assert params.e1 == 2.0
        assert params.e2 == 0.5
        assert params.use_secondary is True

    def test_modify_parameters(self):
        """Test modifying parameters."""
        params = stamp.Parameters()
        params.n_passes = 3
        params.e1 = 1.5
        params.use_secondary = False

        assert params.n_passes == 3
        assert params.e1 == 1.5
        assert params.use_secondary is False

    def test_repr(self):
        """Test string representation."""
        params = stamp.Parameters()
        rep = repr(params)
        assert "Parameters" in rep


class TestTransform:
    """Tests for Transform class."""

    def test_identity_transform(self):
        """Test identity transformation."""
        t = stamp.Transform()
        coords = np.array([[1.0, 2.0, 3.0]])

        transformed = t.apply(coords)
        np.testing.assert_allclose(transformed, coords)

    def test_transformation_matrix(self):
        """Test 4x4 transformation matrix."""
        t = stamp.Transform()
        matrix = t.transformation_matrix()

        assert matrix.shape == (4, 4)
        np.testing.assert_allclose(matrix, np.eye(4))

    def test_inverse(self):
        """Test inverse transformation."""
        t = stamp.Transform()
        inv = t.inverse()
        composed = t.compose(inv)

        matrix = composed.transformation_matrix()
        np.testing.assert_allclose(matrix, np.eye(4), atol=1e-10)

    def test_repr(self):
        """Test string representation."""
        t = stamp.Transform()
        rep = repr(t)
        assert "Transform" in rep


class TestAlignment:
    """Tests for alignment functions."""

    @pytest.fixture
    def simple_domains(self):
        """Create simple test domains."""
        coords1 = np.array([
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
            [11.4, 0.0, 0.0],
            [15.2, 0.0, 0.0],
        ])
        coords2 = coords1.copy()

        d1 = stamp.Domain.from_arrays("domain1", coords1, "AAAAA")
        d2 = stamp.Domain.from_arrays("domain2", coords2, "AAAAA")

        return d1, d2

    def test_pairwise_identical(self, simple_domains):
        """Test alignment of identical structures."""
        d1, d2 = simple_domains
        result = stamp.pairwise_align(d1, d2)

        assert result.rmsd < 0.1
        assert result.n_aligned > 0
        assert result.score >= 0

    def test_pairwise_with_params(self, simple_domains):
        """Test alignment with custom parameters."""
        d1, d2 = simple_domains
        params = stamp.Parameters()
        params.n_passes = 1

        result = stamp.pairwise_align(d1, d2, params=params)
        assert result.n_aligned > 0

    def test_transformation_matrix(self, simple_domains):
        """Test getting transformation matrix."""
        d1, d2 = simple_domains
        result = stamp.pairwise_align(d1, d2)

        matrix = result.transformation_matrix()
        assert matrix.shape == (4, 4)

    def test_aligned_pairs(self, simple_domains):
        """Test getting aligned pairs."""
        d1, d2 = simple_domains
        result = stamp.pairwise_align(d1, d2)

        pairs = result.aligned_pairs
        assert len(pairs) > 0
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)


class TestMultipleAlignment:
    """Tests for multiple alignment."""

    @pytest.fixture
    def multiple_domains(self):
        """Create multiple test domains."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
        ])

        domains = []
        for i in range(3):
            d = stamp.Domain.from_arrays(f"domain{i}", coords.copy(), "AAA")
            domains.append(d)

        return domains

    def test_multiple_align(self, multiple_domains):
        """Test multiple alignment."""
        result = stamp.multiple_align(multiple_domains)

        assert result.n_columns > 0
        assert result.n_core >= 0
        assert len(result.transforms) == len(multiple_domains)

    def test_multiple_align_min_domains(self):
        """Test that multiple alignment requires at least 2 domains."""
        coords = np.array([[0.0, 0.0, 0.0]])
        d = stamp.Domain.from_arrays("single", coords, "A")

        with pytest.raises(ValueError):
            stamp.multiple_align([d])


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_rmsd_identical(self):
        """Test RMSD of identical coordinates."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        rmsd = stamp.compute_rmsd(coords, coords)
        assert abs(rmsd) < 1e-10

    def test_compute_rmsd_different(self):
        """Test RMSD of different coordinates."""
        coords1 = np.array([[0.0, 0.0, 0.0]])
        coords2 = np.array([[1.0, 0.0, 0.0]])

        rmsd = stamp.compute_rmsd(coords1, coords2)
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

        transform, rmsd = stamp.superpose(fixed, mobile)

        assert rmsd < 1e-10
        assert isinstance(transform, stamp.Transform)

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

        dist = stamp.distance_matrix(coords1, coords2)

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

        cx, cy, cz = stamp.centroid(coords)

        assert abs(cx - 2.0 / 3.0) < 1e-10
        assert abs(cy - 2.0 / 3.0) < 1e-10
        assert abs(cz) < 1e-10


class TestVersion:
    """Tests for version information."""

    def test_version_exists(self):
        """Test that version is defined."""
        assert hasattr(stamp, "__version__")
        assert isinstance(stamp.__version__, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
