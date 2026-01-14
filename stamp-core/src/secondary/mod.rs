//! Secondary structure handling and assignment.
//!
//! This module provides utilities for working with protein secondary structure,
//! including DSSP-style assignment and secondary structure comparison.

use crate::types::{Coord3, Domain};

/// Secondary structure element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecondaryStructure {
    /// Alpha helix (H).
    Helix,
    /// Beta strand (E).
    Strand,
    /// Coil/loop (C).
    Coil,
    /// 3-10 helix (G).
    Helix310,
    /// Pi helix (I).
    HelixPi,
    /// Beta bridge (B).
    Bridge,
    /// Turn (T).
    Turn,
    /// Bend (S).
    Bend,
}

impl SecondaryStructure {
    /// Creates from DSSP character code.
    #[must_use]
    pub fn from_dssp(c: char) -> Self {
        match c {
            'H' => Self::Helix,
            'E' => Self::Strand,
            'G' => Self::Helix310,
            'I' => Self::HelixPi,
            'B' => Self::Bridge,
            'T' => Self::Turn,
            'S' => Self::Bend,
            _ => Self::Coil,
        }
    }

    /// Converts to DSSP character code.
    #[must_use]
    pub fn to_dssp(self) -> char {
        match self {
            Self::Helix => 'H',
            Self::Strand => 'E',
            Self::Helix310 => 'G',
            Self::HelixPi => 'I',
            Self::Bridge => 'B',
            Self::Turn => 'T',
            Self::Bend => 'S',
            Self::Coil => 'C',
        }
    }

    /// Converts to simplified three-state classification.
    #[must_use]
    pub fn to_three_state(self) -> char {
        match self {
            Self::Helix | Self::Helix310 | Self::HelixPi => 'H',
            Self::Strand | Self::Bridge => 'E',
            _ => 'C',
        }
    }

    /// Returns true if this is a helical structure.
    #[must_use]
    pub fn is_helix(self) -> bool {
        matches!(self, Self::Helix | Self::Helix310 | Self::HelixPi)
    }

    /// Returns true if this is an extended structure.
    #[must_use]
    pub fn is_extended(self) -> bool {
        matches!(self, Self::Strand | Self::Bridge)
    }
}

/// Secondary structure element (SSE) - a contiguous region of same type.
#[derive(Debug, Clone)]
pub struct SecondaryElement {
    /// Type of secondary structure.
    pub ss_type: SecondaryStructure,
    /// Starting residue index.
    pub start: usize,
    /// Ending residue index (inclusive).
    pub end: usize,
    /// Length of element.
    pub length: usize,
}

impl SecondaryElement {
    /// Creates a new secondary structure element.
    #[must_use]
    pub fn new(ss_type: SecondaryStructure, start: usize, end: usize) -> Self {
        Self {
            ss_type,
            start,
            end,
            length: end - start + 1,
        }
    }
}

/// Extracts secondary structure elements from a domain.
///
/// # Arguments
///
/// * `domain` - Domain with secondary structure assignments
///
/// # Returns
///
/// List of secondary structure elements.
#[must_use]
pub fn extract_elements(domain: &Domain) -> Vec<SecondaryElement> {
    if domain.residues.is_empty() {
        return Vec::new();
    }

    let mut elements = Vec::new();
    let mut current_ss = domain.residues[0].sec_struct;
    let mut start = 0;

    for (i, residue) in domain.residues.iter().enumerate().skip(1) {
        let ss = residue.sec_struct;
        if ss != current_ss {
            // End current element
            elements.push(SecondaryElement::new(
                SecondaryStructure::from_dssp(current_ss),
                start,
                i - 1,
            ));
            current_ss = ss;
            start = i;
        }
    }

    // Add final element
    elements.push(SecondaryElement::new(
        SecondaryStructure::from_dssp(current_ss),
        start,
        domain.residues.len() - 1,
    ));

    elements
}

/// Assigns secondary structure using simplified geometric criteria.
///
/// This is a basic assignment method based on backbone geometry.
/// For accurate assignments, use DSSP.
///
/// # Arguments
///
/// * `domain` - Domain to assign secondary structure to
pub fn assign_secondary_structure(domain: &mut Domain) {
    let n = domain.len();
    if n < 4 {
        return;
    }

    // Compute backbone angles and assign based on geometry
    for i in 1..n - 2 {
        let ca_prev = &domain.residues[i - 1].ca_coord;
        let ca_curr = &domain.residues[i].ca_coord;
        let ca_next = &domain.residues[i + 1].ca_coord;
        let ca_next2 = &domain.residues[i + 2].ca_coord;

        // Compute virtual bond angle
        let v1 = ca_curr.coords - ca_prev.coords;
        let v2 = ca_next.coords - ca_curr.coords;
        let v3 = ca_next2.coords - ca_next.coords;

        let angle1 = angle_between(&v1, &v2);
        let angle2 = angle_between(&v2, &v3);

        // Compute local curvature via dihedral-like measure
        let dihedral = dihedral_angle(ca_prev, ca_curr, ca_next, ca_next2);

        // Simple classification based on geometry
        domain.residues[i].sec_struct = if is_helical(angle1, angle2, dihedral) {
            'H'
        } else if is_extended(angle1, angle2, dihedral) {
            'E'
        } else {
            'C'
        };
    }
}

/// Computes angle between two vectors in radians.
fn angle_between(v1: &nalgebra::Vector3<f64>, v2: &nalgebra::Vector3<f64>) -> f64 {
    let cos_angle = v1.dot(v2) / (v1.norm() * v2.norm());
    cos_angle.clamp(-1.0, 1.0).acos()
}

/// Computes dihedral angle between four points.
fn dihedral_angle(p1: &Coord3, p2: &Coord3, p3: &Coord3, p4: &Coord3) -> f64 {
    let b1 = p2.coords - p1.coords;
    let b2 = p3.coords - p2.coords;
    let b3 = p4.coords - p3.coords;

    let n1 = b1.cross(&b2);
    let n2 = b2.cross(&b3);

    let m1 = n1.cross(&b2.normalize());

    let x = n1.dot(&n2);
    let y = m1.dot(&n2);

    (-y).atan2(x)
}

/// Determines if geometry is consistent with helix.
fn is_helical(angle1: f64, angle2: f64, dihedral: f64) -> bool {
    use std::f64::consts::PI;

    let helix_angle = 91.0 * PI / 180.0; // ~91 degrees for alpha helix
    let helix_dihedral = 50.0 * PI / 180.0; // ~50 degrees

    let angle_diff1 = (angle1 - helix_angle).abs();
    let angle_diff2 = (angle2 - helix_angle).abs();
    let dihedral_diff = (dihedral.abs() - helix_dihedral).abs();

    angle_diff1 < 0.3 && angle_diff2 < 0.3 && dihedral_diff < 0.5
}

/// Determines if geometry is consistent with extended strand.
fn is_extended(angle1: f64, angle2: f64, dihedral: f64) -> bool {
    use std::f64::consts::PI;

    let strand_angle = 120.0 * PI / 180.0; // ~120 degrees for strand

    let angle_diff1 = (angle1 - strand_angle).abs();
    let angle_diff2 = (angle2 - strand_angle).abs();

    angle_diff1 < 0.3 && angle_diff2 < 0.3 && dihedral.abs() > 2.5
}

/// Computes secondary structure similarity between two domains.
///
/// # Arguments
///
/// * `domain1` - First domain
/// * `domain2` - Second domain
/// * `path` - Alignment path
///
/// # Returns
///
/// Fraction of aligned positions with matching secondary structure.
#[must_use]
pub fn ss_similarity(domain1: &Domain, domain2: &Domain, path: &[(usize, usize)]) -> f64 {
    if path.is_empty() {
        return 0.0;
    }

    let matches = path
        .iter()
        .filter(|&&(i, j)| {
            let ss1 = domain1
                .residues
                .get(i)
                .map(|r| SecondaryStructure::from_dssp(r.sec_struct).to_three_state());
            let ss2 = domain2
                .residues
                .get(j)
                .map(|r| SecondaryStructure::from_dssp(r.sec_struct).to_three_state());
            ss1 == ss2
        })
        .count();

    matches as f64 / path.len() as f64
}

/// Computes secondary structure content of a domain.
///
/// # Returns
///
/// Tuple of (helix_fraction, strand_fraction, coil_fraction).
#[must_use]
pub fn ss_content(domain: &Domain) -> (f64, f64, f64) {
    if domain.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut helix = 0;
    let mut strand = 0;
    let mut coil = 0;

    for residue in &domain.residues {
        match SecondaryStructure::from_dssp(residue.sec_struct).to_three_state() {
            'H' => helix += 1,
            'E' => strand += 1,
            _ => coil += 1,
        }
    }

    let n = domain.len() as f64;
    (helix as f64 / n, strand as f64 / n, coil as f64 / n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Residue;

    #[test]
    fn test_secondary_structure_conversion() {
        let ss = SecondaryStructure::from_dssp('H');
        assert_eq!(ss, SecondaryStructure::Helix);
        assert_eq!(ss.to_dssp(), 'H');
        assert_eq!(ss.to_three_state(), 'H');
        assert!(ss.is_helix());
    }

    #[test]
    fn test_three_state() {
        assert_eq!(SecondaryStructure::Helix310.to_three_state(), 'H');
        assert_eq!(SecondaryStructure::Bridge.to_three_state(), 'E');
        assert_eq!(SecondaryStructure::Turn.to_three_state(), 'C');
    }

    #[test]
    fn test_secondary_element() {
        let elem = SecondaryElement::new(SecondaryStructure::Helix, 5, 15);
        assert_eq!(elem.length, 11);
        assert_eq!(elem.start, 5);
        assert_eq!(elem.end, 15);
    }

    #[test]
    fn test_extract_elements_empty() {
        let domain = Domain::new("empty".to_string());
        let elements = extract_elements(&domain);
        assert!(elements.is_empty());
    }

    #[test]
    fn test_ss_content() {
        let mut domain = Domain::new("test".to_string());
        domain.residues.push(Residue::new(0, "1".to_string(), 'A', Coord3::new(0.0, 0.0, 0.0)));
        domain.residues.push(Residue::new(1, "2".to_string(), 'A', Coord3::new(1.0, 0.0, 0.0)));
        domain.residues[0].sec_struct = 'H';
        domain.residues[1].sec_struct = 'E';

        let (h, e, c) = ss_content(&domain);
        assert!((h - 0.5).abs() < 1e-10);
        assert!((e - 0.5).abs() < 1e-10);
        assert!(c.abs() < 1e-10);
    }
}
