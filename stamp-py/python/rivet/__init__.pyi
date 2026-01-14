"""Type stubs for rivet module."""

from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

__version__: str

class Domain:
    """Protein domain with C-alpha coordinates."""

    id: str
    chain: str
    sequence: str
    secondary_structure: str
    has_dssp: bool
    coordinates: npt.NDArray[np.float64]

    def __init__(self, id: str, chain: Optional[str] = None) -> None: ...
    def __len__(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @staticmethod
    def from_pdb(path: str, chain: Optional[str] = None) -> Domain: ...
    @staticmethod
    def from_dssp(
        pdb_path: str, dssp_path: str, chain: Optional[str] = None
    ) -> Domain: ...
    @staticmethod
    def from_arrays(
        id: str,
        coords: npt.NDArray[np.float64],
        sequence: str,
        chain: Optional[str] = None,
    ) -> Domain: ...
    def load_dssp(self, path: str) -> None: ...
    def to_pdb(self, path: str, transform: Optional[Transform] = None) -> None: ...
    def copy(self) -> Domain: ...
    def ss_content(self) -> Tuple[float, float, float]: ...

class Parameters:
    """Alignment parameters."""

    n_passes: int
    e1: float
    e2: float
    gap_open: float
    gap_extend: float
    use_secondary: bool
    max_iter: int
    cutoff_dist: float
    score_cutoff: float
    convergence: float
    verbosity: int

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

class Transform:
    """3D rigid body transformation."""

    rotation: npt.NDArray[np.float64]
    translation: npt.NDArray[np.float64]

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def from_matrix(
        rotation: npt.NDArray[np.float64], translation: npt.NDArray[np.float64]
    ) -> Transform: ...
    def transformation_matrix(self) -> npt.NDArray[np.float64]: ...
    def apply(self, coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def inverse(self) -> Transform: ...
    def compose(self, other: Transform) -> Transform: ...

class AlignmentResult:
    """Result of pairwise alignment."""

    rmsd: float
    n_aligned: int
    score: float
    seq_identity: float
    transform: Transform
    aligned_pairs: List[Tuple[int, int]]

    def __repr__(self) -> str: ...
    def get_rotation(self) -> npt.NDArray[np.float64]: ...
    def get_translation(self) -> npt.NDArray[np.float64]: ...
    def transformation_matrix(self) -> npt.NDArray[np.float64]: ...
    def transform_coordinates(
        self, coords: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def get_transformed_coordinates(
        self, domain: Domain
    ) -> npt.NDArray[np.float64]: ...

class MultipleAlignmentResult:
    """Result of multiple alignment."""

    avg_rmsd: float
    score: float
    n_columns: int
    n_core: int
    transforms: List[Transform]
    core_positions: List[int]
    columns: List[List[Optional[int]]]

    def __repr__(self) -> str: ...
    def get_transform(self, index: int) -> Transform: ...

class ScanHit:
    """Database scan hit."""

    target_id: str
    rank: int
    score: float
    rmsd: float
    n_aligned: int
    seq_identity: float
    z_score: Optional[float]
    e_value: Optional[float]

    def __repr__(self) -> str: ...

def pairwise_align(
    domain1: Domain, domain2: Domain, params: Optional[Parameters] = None
) -> AlignmentResult:
    """Perform pairwise structural alignment."""
    ...

def multiple_align(
    domains: List[Domain], params: Optional[Parameters] = None
) -> MultipleAlignmentResult:
    """Perform multiple structure alignment."""
    ...

def scan_database(
    query: Domain,
    targets: List[Domain],
    params: Optional[Parameters] = None,
    score_cutoff: float = 0.0,
    max_hits: int = 100,
    quick: bool = False,
) -> List[ScanHit]:
    """Scan query against database of structures."""
    ...

def compute_rmsd(
    coords1: npt.NDArray[np.float64], coords2: npt.NDArray[np.float64]
) -> float:
    """Compute RMSD between two coordinate sets."""
    ...

def superpose(
    fixed: npt.NDArray[np.float64], mobile: npt.NDArray[np.float64]
) -> Tuple[Transform, float]:
    """Compute optimal superposition."""
    ...

def distance_matrix(
    coords1: npt.NDArray[np.float64], coords2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute pairwise distance matrix."""
    ...

def centroid(coords: npt.NDArray[np.float64]) -> Tuple[float, float, float]:
    """Compute centroid of coordinates."""
    ...
