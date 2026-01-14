//! File I/O operations for PDB, DSSP, and domain files.
//!
//! This module handles parsing of structural data from various file formats
//! used in protein structure analysis. It is based on the original C implementation
//! from STAMP (igetca.c, getdomain.c, igetcadssp.c).
//!
//! # Supported Formats
//!
//! - **PDB**: Protein Data Bank format for atomic coordinates
//! - **DSSP**: Dictionary of Secondary Structure of Proteins format
//! - **Domain files**: STAMP-specific domain specification format
//!
//! # Features
//!
//! - Support for both CA (alpha carbon) and P (phosphorus) atom types
//! - Brookhaven numbering with insertion codes
//! - Chain selection and residue range filtering
//! - Alternate conformation handling
//! - Transformation matrices for pre-aligned structures

use crate::types::{
    Coord3, Domain, Residue, RotationMatrix, StampError, StampResult, Transform, Vec3,
};
use std::fs::File;
use std::io::{BufRead, BufReader, Write as IoWrite};
use std::path::Path;

/// Brookhaven-style residue number with chain ID and insertion code.
///
/// This represents the standard PDB residue numbering scheme which includes:
/// - Residue sequence number
/// - Chain identifier
/// - Insertion code for handling insertions in the sequence
///
/// # Example
///
/// ```
/// use stamp_core::io::BrookhavenNumber;
///
/// // Standard residue: chain A, residue 42, no insertion
/// let resid = BrookhavenNumber::new(42, 'A', ' ');
///
/// // Residue with insertion code: chain A, residue 42A
/// let resid_ins = BrookhavenNumber::new(42, 'A', 'A');
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrookhavenNumber {
    /// Residue sequence number.
    pub n: i32,
    /// Chain identifier.
    pub cid: char,
    /// Insertion code (space if none).
    pub ins: char,
}

impl BrookhavenNumber {
    /// Creates a new Brookhaven number.
    #[must_use]
    pub fn new(n: i32, cid: char, ins: char) -> Self {
        Self { n, cid, ins }
    }

    /// Creates a wildcard that matches any residue in the given chain.
    #[must_use]
    pub fn chain_wildcard(cid: char) -> Self {
        Self {
            n: 0,
            cid,
            ins: '?',
        }
    }

    /// Creates a wildcard that matches any residue.
    #[must_use]
    pub fn any() -> Self {
        Self {
            n: 0,
            cid: '?',
            ins: '?',
        }
    }

    /// Checks if this number matches a specific position.
    #[must_use]
    pub fn matches(&self, other: &BrookhavenNumber) -> bool {
        (self.cid == '?' || self.cid == other.cid)
            && (self.ins == '?' || self.ins == other.ins)
            && (self.n == 0 || self.n == other.n)
    }

    /// Parses a Brookhaven number from a PDB-style string.
    ///
    /// Format: "chain residue_number insertion_code" or compact forms.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }

        let mut chars = s.chars();
        let cid = chars.next()?;
        let cid = if cid == '_' { ' ' } else { cid };

        // Skip whitespace
        let rest: String = chars.collect();
        let parts: Vec<&str> = rest.split_whitespace().collect();

        if parts.is_empty() {
            return Some(Self::chain_wildcard(cid));
        }

        let n: i32 = parts[0].parse().ok()?;
        let ins = if parts.len() > 1 {
            let ins_char = parts[1].chars().next().unwrap_or('_');
            if ins_char == '_' {
                ' '
            } else {
                ins_char
            }
        } else {
            ' '
        };

        Some(Self::new(n, cid, ins))
    }

    /// Formats the number for display, using underscores for spaces.
    #[must_use]
    pub fn to_string_display(&self) -> String {
        let cid = if self.cid == ' ' { '_' } else { self.cid };
        let ins = if self.ins == ' ' { '_' } else { self.ins };
        format!("{} {} {}", cid, self.n, ins)
    }
}

impl Default for BrookhavenNumber {
    fn default() -> Self {
        Self::any()
    }
}

impl std::fmt::Display for BrookhavenNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.ins == ' ' || self.ins == '_' {
            write!(f, "{}{}", self.cid, self.n)
        } else {
            write!(f, "{}{}{}", self.cid, self.n, self.ins)
        }
    }
}

/// Type of atom to extract from PDB files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AtomType {
    /// C-alpha atoms for protein structures.
    #[default]
    Ca,
    /// Phosphorus atoms for nucleic acid structures.
    P,
}

impl AtomType {
    /// Returns the PDB atom name for this type.
    #[must_use]
    pub fn atom_name(&self) -> &'static str {
        match self {
            AtomType::Ca => " CA ",
            AtomType::P => " P  ",
        }
    }
}

/// Type of domain selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainSelectionType {
    /// Select all residues from the file.
    All,
    /// Select a specific chain.
    Chain,
    /// Select a specific range of residues.
    Range,
}

/// Specification for a single segment within a domain.
#[derive(Debug, Clone)]
pub struct DomainSegment {
    /// Selection type for this segment.
    pub selection_type: DomainSelectionType,
    /// Start of the range (for Range type).
    pub start: BrookhavenNumber,
    /// End of the range (for Range type).
    pub end: BrookhavenNumber,
    /// Whether to reverse the coordinates (N to C becomes C to N).
    pub reverse: bool,
}

impl DomainSegment {
    /// Creates a segment that selects all residues.
    #[must_use]
    pub fn all() -> Self {
        Self {
            selection_type: DomainSelectionType::All,
            start: BrookhavenNumber::any(),
            end: BrookhavenNumber::any(),
            reverse: false,
        }
    }

    /// Creates a segment that selects a specific chain.
    #[must_use]
    pub fn chain(chain_id: char) -> Self {
        Self {
            selection_type: DomainSelectionType::Chain,
            start: BrookhavenNumber::chain_wildcard(chain_id),
            end: BrookhavenNumber::chain_wildcard(chain_id),
            reverse: false,
        }
    }

    /// Creates a segment that selects a range of residues.
    #[must_use]
    pub fn range(start: BrookhavenNumber, end: BrookhavenNumber) -> Self {
        Self {
            selection_type: DomainSelectionType::Range,
            start,
            end,
            reverse: false,
        }
    }

    /// Sets the reverse flag.
    #[must_use]
    pub fn with_reverse(mut self, reverse: bool) -> Self {
        self.reverse = reverse;
        self
    }
}

impl Default for DomainSegment {
    fn default() -> Self {
        Self::all()
    }
}

/// Specification for loading a domain from a PDB file.
///
/// This supports the full STAMP domain file format including:
/// - Multiple segments per domain
/// - Chain and range selection
/// - Initial transformation matrices
/// - Reverse coordinate ordering
///
/// # Domain File Format
///
/// Simple format:
/// ```text
/// domain_id pdb_file chain start end
/// 1abc     1abc.pdb  A     1     100
/// ```
///
/// Extended format with braces:
/// ```text
/// pdb_file domain_id { CHAIN A }
/// pdb_file domain_id { A 1 _ TO A 100 _ }
/// pdb_file domain_id { ALL }
/// pdb_file domain_id { REVERSE CHAIN A }
/// pdb_file domain_id { A 1 _ TO A 50 _ A 60 _ TO A 100 _ }
/// ```
///
/// With transformation matrix:
/// ```text
/// pdb_file domain_id { CHAIN A }
/// R11 R12 R13 V1
/// R21 R22 R23 V2
/// R31 R32 R33 V3
/// ```
#[derive(Debug, Clone)]
pub struct DomainSpec {
    /// Domain identifier.
    pub id: String,
    /// Path to PDB or DSSP file.
    pub filename: String,
    /// Segments defining which parts of the file to include.
    pub segments: Vec<DomainSegment>,
    /// Optional initial transformation.
    pub transform: Option<Transform>,
    /// Atom type to extract.
    pub atom_type: AtomType,
}

impl DomainSpec {
    /// Creates a new domain specification with default settings.
    #[must_use]
    pub fn new(id: String, filename: String) -> Self {
        Self {
            id,
            filename,
            segments: vec![DomainSegment::all()],
            transform: None,
            atom_type: AtomType::Ca,
        }
    }

    /// Creates a specification for a specific chain.
    #[must_use]
    pub fn with_chain(id: String, filename: String, chain: char) -> Self {
        Self {
            id,
            filename,
            segments: vec![DomainSegment::chain(chain)],
            transform: None,
            atom_type: AtomType::Ca,
        }
    }

    /// Creates a specification for a residue range.
    #[must_use]
    pub fn with_range(id: String, filename: String, chain: char, start: i32, end: i32) -> Self {
        let start = BrookhavenNumber::new(start, chain, ' ');
        let end = BrookhavenNumber::new(end, chain, ' ');
        Self {
            id,
            filename,
            segments: vec![DomainSegment::range(start, end)],
            transform: None,
            atom_type: AtomType::Ca,
        }
    }

    /// Sets the atom type.
    #[must_use]
    pub fn with_atom_type(mut self, atom_type: AtomType) -> Self {
        self.atom_type = atom_type;
        self
    }

    /// Sets the initial transformation.
    #[must_use]
    pub fn with_transform(mut self, transform: Transform) -> Self {
        self.transform = Some(transform);
        self
    }

    /// Loads the domain according to this specification.
    ///
    /// # Errors
    ///
    /// Returns an error if the PDB file cannot be read or parsed.
    pub fn load(&self) -> StampResult<Domain> {
        let mut domain = Domain::new(self.id.clone());
        domain.pdb_file = self.filename.clone();

        for segment in &self.segments {
            let residues = parse_pdb_coords(
                &self.filename,
                self.atom_type,
                segment.selection_type,
                &segment.start,
                &segment.end,
            )?;

            let mut residues = residues;
            if segment.reverse {
                residues.reverse();
            }

            // Renumber residues sequentially
            let offset = domain.residues.len() as i32;
            for (i, residue) in residues.iter_mut().enumerate() {
                residue.seq_num = offset + i as i32;
            }

            domain.residues.extend(residues);
        }

        // Set chain from first segment if available
        if let Some(segment) = self.segments.first() {
            if segment.start.cid != '?' {
                domain.chain = segment.start.cid;
            }
        }

        // Apply initial transformation if present
        if let Some(ref transform) = self.transform {
            for residue in &mut domain.residues {
                residue.ca_coord = transform.apply(&residue.ca_coord);
            }
        }

        if domain.is_empty() {
            return Err(StampError::PdbParse(format!(
                "No atoms found for domain {} in {}",
                self.id, self.filename
            )));
        }

        log::debug!(
            "Loaded domain {} with {} residues from {}",
            self.id,
            domain.len(),
            self.filename
        );

        Ok(domain)
    }
}

/// Options controlling PDB parsing behavior.
#[derive(Debug, Clone)]
pub struct PdbParseOptions {
    /// Atom type to extract.
    pub atom_type: AtomType,
    /// Whether to accept only the first alternate conformation.
    pub first_alt_only: bool,
    /// Allowed alternate conformation indicators.
    pub allowed_alt: Vec<char>,
    /// Whether to include HETATM records.
    pub include_hetatm: bool,
}

impl Default for PdbParseOptions {
    fn default() -> Self {
        Self {
            atom_type: AtomType::Ca,
            first_alt_only: true,
            allowed_alt: vec![' ', 'A', '1', 'L', 'O'],
            include_hetatm: false,
        }
    }
}

/// Parses a PDB file and extracts C-alpha or phosphorus coordinates.
///
/// This is the main PDB parsing function that handles:
/// - Standard ATOM records
/// - Optional HETATM records
/// - Alternate conformations (selects first by default)
/// - Brookhaven numbering with insertion codes
/// - MODEL/ENDMDL for NMR structures (reads first model only)
///
/// # Arguments
///
/// * `path` - Path to the PDB file
/// * `chain` - Optional chain identifier (None for first chain found)
///
/// # Returns
///
/// A `Domain` containing the extracted residues.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
///
/// # Example
///
/// ```no_run
/// use stamp_core::io::parse_pdb;
///
/// // Parse chain A from a PDB file
/// let domain = parse_pdb("1abc.pdb", Some('A'))?;
/// println!("Found {} residues", domain.len());
/// # Ok::<(), stamp_core::types::StampError>(())
/// ```
pub fn parse_pdb<P: AsRef<Path>>(path: P, chain: Option<char>) -> StampResult<Domain> {
    parse_pdb_with_options(path, chain, &PdbParseOptions::default())
}

/// Parses a PDB file with custom options.
///
/// # Arguments
///
/// * `path` - Path to the PDB file
/// * `chain` - Optional chain identifier
/// * `options` - Parsing options
///
/// # Returns
///
/// A `Domain` containing the extracted residues.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn parse_pdb_with_options<P: AsRef<Path>>(
    path: P,
    chain: Option<char>,
    options: &PdbParseOptions,
) -> StampResult<Domain> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut domain = Domain::new(
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string(),
    );
    domain.pdb_file = path.to_string_lossy().to_string();

    let target_chain = chain.unwrap_or('?');
    let mut found_chain = if target_chain == '?' {
        None
    } else {
        Some(target_chain)
    };
    let mut seq_num = 0i32;

    // Track last residue to avoid duplicates from alternate conformations
    let mut last_resid: Option<(char, i32, char)> = None;

    let atom_name = options.atom_type.atom_name();

    for line in reader.lines() {
        let line = line?;

        // Handle MODEL/ENDMDL for NMR structures - only read first model
        if line.starts_with("ENDMDL") || line.starts_with("END   ") {
            if !domain.is_empty() {
                break;
            }
            continue;
        }

        // Check record type
        let is_atom = line.starts_with("ATOM");
        let is_hetatm = line.starts_with("HETATM");

        if !(is_atom || options.include_hetatm && is_hetatm) {
            continue;
        }

        if line.len() < 54 {
            continue;
        }

        // Parse atom name (columns 13-16, 0-indexed: 12-16)
        let line_atom_name = line.get(12..16).unwrap_or("");
        if line_atom_name != atom_name {
            continue;
        }

        // Parse chain ID (column 22, 0-indexed: 21)
        let line_chain = line.chars().nth(21).unwrap_or(' ');

        // If no specific chain requested, use first chain found
        if found_chain.is_none() {
            found_chain = Some(line_chain);
        }

        // Skip if not the target chain
        if let Some(fc) = found_chain {
            if line_chain != fc {
                continue;
            }
        }

        // Parse alternate conformation indicator (column 17, 0-indexed: 16)
        let alt = line.chars().nth(16).unwrap_or(' ');
        if options.first_alt_only && !options.allowed_alt.contains(&alt) {
            continue;
        }

        // Parse residue number (columns 23-26, 0-indexed: 22-26)
        let res_num_str = line.get(22..26).unwrap_or("0").trim();
        let res_num: i32 = res_num_str.parse().unwrap_or(0);

        // Parse insertion code (column 27, 0-indexed: 26)
        let ins_code = line.chars().nth(26).unwrap_or(' ');

        // Skip duplicate residues from alternate conformations
        let current_resid = (line_chain, res_num, ins_code);
        if let Some(last) = last_resid {
            if last == current_resid {
                continue;
            }
        }
        last_resid = Some(current_resid);

        // Parse residue name (columns 18-20, 0-indexed: 17-20)
        let res_name = line.get(17..20).unwrap_or("UNK").trim();
        let aa = three_to_one(res_name);

        // Format PDB number string
        let pdb_num = if ins_code == ' ' {
            res_num_str.to_string()
        } else {
            format!("{}{}", res_num_str, ins_code)
        };

        // Parse coordinates (columns 31-38, 39-46, 47-54, 0-indexed: 30-38, 38-46, 46-54)
        let x: f64 = line
            .get(30..38)
            .unwrap_or("0.0")
            .trim()
            .parse()
            .map_err(|_| StampError::PdbParse(format!("Invalid X coordinate in line: {}", line)))?;
        let y: f64 = line
            .get(38..46)
            .unwrap_or("0.0")
            .trim()
            .parse()
            .map_err(|_| StampError::PdbParse(format!("Invalid Y coordinate in line: {}", line)))?;
        let z: f64 = line
            .get(46..54)
            .unwrap_or("0.0")
            .trim()
            .parse()
            .map_err(|_| StampError::PdbParse(format!("Invalid Z coordinate in line: {}", line)))?;

        let residue = Residue::new(seq_num, pdb_num, aa, Coord3::new(x, y, z));
        domain.residues.push(residue);
        seq_num += 1;
    }

    domain.chain = found_chain.unwrap_or('_');

    if domain.is_empty() {
        return Err(StampError::PdbParse(format!(
            "No {} atoms found in chain '{}' of {}",
            match options.atom_type {
                AtomType::Ca => "C-alpha",
                AtomType::P => "phosphorus",
            },
            domain.chain,
            path.display()
        )));
    }

    log::debug!(
        "Parsed {} residues from {} chain {}",
        domain.len(),
        path.display(),
        domain.chain
    );

    Ok(domain)
}

/// Parses specific coordinates from a PDB file with selection criteria.
///
/// This function implements the selection logic from the C igetca function,
/// supporting ALL, CHAIN, and RANGE selection types.
///
/// # Arguments
///
/// * `path` - Path to the PDB file
/// * `atom_type` - Type of atom to extract (CA or P)
/// * `selection_type` - Type of selection (All, Chain, or Range)
/// * `start` - Start of selection range
/// * `end` - End of selection range
///
/// # Returns
///
/// Vector of residues matching the selection criteria.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn parse_pdb_coords<P: AsRef<Path>>(
    path: P,
    atom_type: AtomType,
    selection_type: DomainSelectionType,
    start: &BrookhavenNumber,
    end: &BrookhavenNumber,
) -> StampResult<Vec<Residue>> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut residues = Vec::new();
    let mut begun = false;
    let mut seq_num = 0i32;

    let allowed_alt = [' ', 'A', '1', 'L', 'O'];
    let mut last_resid: Option<(char, i32, char)> = None;

    let atom_name = atom_type.atom_name();

    for line in reader.lines() {
        let line = line?;

        // Handle MODEL/ENDMDL and END
        if line.starts_with("ENDMDL") || line.starts_with("END   ") {
            if begun {
                break;
            }
            continue;
        }

        if !line.starts_with("ATOM") {
            continue;
        }

        if line.len() < 54 {
            continue;
        }

        // Parse atom name
        let line_atom_name = line.get(12..16).unwrap_or("");
        if line_atom_name != atom_name {
            continue;
        }

        // Parse chain ID
        let cid = line.chars().nth(21).unwrap_or(' ');

        // Parse residue number
        let res_num_str = line.get(22..26).unwrap_or("0").trim();
        let number: i32 = res_num_str.parse().unwrap_or(0);

        // Parse insertion code
        let ins = line.chars().nth(26).unwrap_or(' ');

        // Parse alternate conformation
        let alt = line.chars().nth(16).unwrap_or(' ');

        // Create current brookhaven number
        let current = BrookhavenNumber::new(number, cid, ins);

        // Check if we should start reading
        if !begun {
            match selection_type {
                DomainSelectionType::All => begun = true,
                DomainSelectionType::Chain => {
                    if start.cid == cid {
                        begun = true;
                    }
                }
                DomainSelectionType::Range => {
                    if start.matches(&current) {
                        begun = true;
                    }
                }
            }
        }

        // Check if we should stop (for Chain type, when chain changes)
        if begun && selection_type == DomainSelectionType::Chain && start.cid != cid {
            break;
        }

        if !begun {
            continue;
        }

        // Check alternate conformation
        if !allowed_alt.contains(&alt) {
            continue;
        }

        // Skip duplicate residues
        let current_resid = (cid, number, ins);
        if let Some(last) = last_resid {
            if last == current_resid {
                continue;
            }
        }
        last_resid = Some(current_resid);

        // Parse residue name and amino acid
        let res_name = line.get(17..20).unwrap_or("UNK").trim();
        let aa = three_to_one(res_name);

        // Format PDB number string
        let pdb_num = if ins == ' ' {
            res_num_str.to_string()
        } else {
            format!("{}{}", res_num_str, ins)
        };

        // Parse coordinates
        let x: f64 = line
            .get(30..38)
            .unwrap_or("0.0")
            .trim()
            .parse()
            .unwrap_or(0.0);
        let y: f64 = line
            .get(38..46)
            .unwrap_or("0.0")
            .trim()
            .parse()
            .unwrap_or(0.0);
        let z: f64 = line
            .get(46..54)
            .unwrap_or("0.0")
            .trim()
            .parse()
            .unwrap_or(0.0);

        let residue = Residue::new(seq_num, pdb_num, aa, Coord3::new(x, y, z));
        residues.push(residue);
        seq_num += 1;

        // Check if we've reached the end (for Range type)
        if selection_type == DomainSelectionType::Range && end.matches(&current) {
            break;
        }
    }

    if !begun {
        return Err(StampError::PdbParse(format!(
            "Start of sequence not found in PDB file {}",
            path.display()
        )));
    }

    Ok(residues)
}

/// Parses a DSSP file and adds secondary structure information to a domain.
///
/// DSSP (Dictionary of Secondary Structure of Proteins) files contain
/// secondary structure assignments calculated from 3D coordinates.
///
/// # DSSP Format
///
/// The relevant columns are:
/// - Columns 1-5: Sequential residue number
/// - Column 11: Chain identifier
/// - Column 14: One-letter amino acid code (! for chain break)
/// - Column 17: Secondary structure code
/// - Columns 35-38: Accessibility
/// - Columns 116-122: X coordinate (optional)
/// - Columns 123-129: Y coordinate (optional)
/// - Columns 130-136: Z coordinate (optional)
///
/// # Secondary Structure Codes
///
/// - H: Alpha helix
/// - G: 3-10 helix
/// - I: Pi helix
/// - E: Extended strand
/// - B: Residue in isolated beta-bridge
/// - T: Hydrogen bonded turn
/// - S: Bend
/// - (space): Loop or irregular
///
/// # Arguments
///
/// * `path` - Path to the DSSP file
/// * `domain` - Domain to update with secondary structure
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
///
/// # Example
///
/// ```no_run
/// use stamp_core::io::{parse_pdb, parse_dssp};
///
/// let mut domain = parse_pdb("1abc.pdb", Some('A'))?;
/// parse_dssp("1abc.dssp", &mut domain)?;
/// println!("Secondary structure: {}", domain.secondary_structure());
/// # Ok::<(), stamp_core::types::StampError>(())
/// ```
pub fn parse_dssp<P: AsRef<Path>>(path: P, domain: &mut Domain) -> StampResult<()> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut in_residue_section = false;
    let mut dssp_data: Vec<(String, char, char, f64)> = Vec::new();

    for line in reader.lines() {
        let line = line?;

        // DSSP residue section starts after the header line containing "#  RESIDUE"
        if line.contains("  #  RESIDUE") {
            in_residue_section = true;
            continue;
        }

        if !in_residue_section {
            continue;
        }

        if line.len() < 17 {
            continue;
        }

        // Check for chain break (indicated by '!' in column 14)
        let aa_char = line.chars().nth(13).unwrap_or(' ');
        if aa_char == '!' {
            continue;
        }

        // Parse chain (column 12, 0-indexed: 11)
        let chain = line.chars().nth(11).unwrap_or(' ');
        if chain != domain.chain && chain != ' ' && domain.chain != '_' {
            continue;
        }

        // Parse residue number (columns 6-10, 0-indexed: 5-10)
        let res_num_str = line.get(5..10).unwrap_or("").trim();

        // Parse insertion code (column 11, 0-indexed: 10) - some DSSP versions
        let ins_code = line.chars().nth(10).unwrap_or(' ');
        let pdb_num = if ins_code == ' ' {
            res_num_str.to_string()
        } else {
            format!("{}{}", res_num_str, ins_code)
        };

        // Parse secondary structure (column 17, 0-indexed: 16)
        let sec_struct = line.chars().nth(16).unwrap_or(' ');
        let sec_struct = match sec_struct {
            'H' | 'G' | 'I' => 'H', // All helix types -> H
            'E' | 'B' => 'E',       // Strand types -> E
            _ => 'C',               // Everything else -> C (coil)
        };

        // Parse accessibility (columns 35-38, 0-indexed: 34-38)
        let accessibility: f64 = line
            .get(34..38)
            .unwrap_or("0")
            .trim()
            .parse()
            .unwrap_or(0.0);

        dssp_data.push((pdb_num, sec_struct, aa_char, accessibility));
    }

    // Match DSSP data to domain residues by PDB number
    for (pdb_num, sec_struct, _aa, accessibility) in &dssp_data {
        if let Some(residue) = domain.residues.iter_mut().find(|r| &r.pdb_num == pdb_num) {
            residue.sec_struct = *sec_struct;
            residue.accessibility = *accessibility;
        }
    }

    domain.has_dssp = true;

    log::debug!(
        "Added DSSP data from {} ({} residues matched)",
        path.display(),
        dssp_data.len()
    );

    Ok(())
}

/// Parses coordinates directly from a DSSP file.
///
/// Some DSSP files contain CA coordinates in addition to secondary structure.
/// This function extracts both coordinates and secondary structure.
///
/// # Arguments
///
/// * `path` - Path to the DSSP file
/// * `selection_type` - Type of selection
/// * `start` - Start of selection range
/// * `end` - End of selection range
///
/// # Returns
///
/// Vector of residues with coordinates and secondary structure.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn parse_dssp_coords<P: AsRef<Path>>(
    path: P,
    selection_type: DomainSelectionType,
    start: &BrookhavenNumber,
    end: &BrookhavenNumber,
) -> StampResult<Vec<Residue>> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut residues = Vec::new();
    let mut in_residue_section = false;
    let mut begun = false;
    let mut seq_num = 0i32;

    for line in reader.lines() {
        let line = line?;

        // Find start of residue section
        if line.contains("  #  RESIDUE") {
            in_residue_section = true;
            continue;
        }

        if !in_residue_section {
            continue;
        }

        // Need at least 136 characters for coordinates
        if line.len() < 136 {
            continue;
        }

        // Check for chain break
        let aa_char = line.chars().nth(13).unwrap_or(' ');
        if aa_char == '!' {
            continue;
        }

        // Parse chain
        let cid = line.chars().nth(11).unwrap_or(' ');

        // Parse residue number
        let res_num_str = line.get(5..10).unwrap_or("").trim();
        let number: i32 = res_num_str.parse().unwrap_or(0);

        // Parse insertion code
        let ins = line.chars().nth(10).unwrap_or(' ');

        let current = BrookhavenNumber::new(number, cid, ins);

        // Check if we should start
        if !begun {
            match selection_type {
                DomainSelectionType::All => begun = true,
                DomainSelectionType::Chain => {
                    if start.cid == cid {
                        begun = true;
                    }
                }
                DomainSelectionType::Range => {
                    if start.matches(&current) {
                        begun = true;
                    }
                }
            }
        }

        // Check if we should stop
        if begun && selection_type == DomainSelectionType::Chain && start.cid != cid {
            break;
        }

        if !begun {
            continue;
        }

        // Parse secondary structure
        let sec_struct = line.chars().nth(16).unwrap_or(' ');
        let sec_struct = match sec_struct {
            'H' | 'G' | 'I' => 'H',
            'E' | 'B' => 'E',
            _ => 'C',
        };

        // Parse accessibility
        let accessibility: f64 = line
            .get(34..38)
            .unwrap_or("0")
            .trim()
            .parse()
            .unwrap_or(0.0);

        // Parse coordinates (columns 116-122, 123-129, 130-136)
        let x: f64 = line
            .get(115..122)
            .unwrap_or("0.0")
            .trim()
            .parse()
            .unwrap_or(0.0);
        let y: f64 = line
            .get(122..129)
            .unwrap_or("0.0")
            .trim()
            .parse()
            .unwrap_or(0.0);
        let z: f64 = line
            .get(129..136)
            .unwrap_or("0.0")
            .trim()
            .parse()
            .unwrap_or(0.0);

        let pdb_num = if ins == ' ' {
            res_num_str.to_string()
        } else {
            format!("{}{}", res_num_str, ins)
        };

        let mut residue = Residue::new(seq_num, pdb_num, aa_char, Coord3::new(x, y, z));
        residue.sec_struct = sec_struct;
        residue.accessibility = accessibility;

        residues.push(residue);
        seq_num += 1;

        // Check if we've reached the end
        if selection_type == DomainSelectionType::Range && end.matches(&current) {
            break;
        }
    }

    if !begun {
        return Err(StampError::DsspParse(format!(
            "Start of sequence not found in DSSP file {}",
            path.display()
        )));
    }

    Ok(residues)
}

/// Parses a STAMP domain specification file.
///
/// Domain files specify which parts of PDB files to load and how to combine them.
/// The format supports several variations:
///
/// # Simple Format
///
/// ```text
/// # Comments start with # or %
/// domain_id pdb_file [chain] [start] [end]
/// 1abc     1abc.pdb  A       1       100
/// ```
///
/// # Extended Format (with braces)
///
/// ```text
/// pdb_file domain_id { selection_spec }
///
/// Selection specs:
///   ALL                    - All residues
///   CHAIN A                - Chain A only
///   A 1 _ TO A 100 _       - Range from A1 to A100
///   REVERSE CHAIN A        - Chain A, reversed
///   A 1 _ TO A 50 _  B 1 _ TO B 50 _  - Multiple segments
/// ```
///
/// # With Transformation Matrix
///
/// ```text
/// pdb_file domain_id { CHAIN A }
/// R11 R12 R13 V1
/// R21 R22 R23 V2
/// R31 R32 R33 V3
/// ```
///
/// # Arguments
///
/// * `path` - Path to the domain file
///
/// # Returns
///
/// Vector of domain specifications and a flag indicating if transformations were found.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn parse_domain_file<P: AsRef<Path>>(path: P) -> StampResult<(Vec<DomainSpec>, bool)> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut specs = Vec::new();
    let mut got_transform = false;

    let mut lines: Vec<String> = Vec::new();
    for line in reader.lines() {
        lines.push(line?);
    }

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') || line.starts_with('%') {
            i += 1;
            continue;
        }

        // Check for extended format (contains '{')
        if line.contains('{') {
            let (spec, has_trans, consumed) = parse_domain_extended(&lines[i..], path)?;
            specs.push(spec);
            if has_trans {
                got_transform = true;
            }
            i += consumed;
        } else {
            // Simple format
            let spec = parse_domain_simple(line, path)?;
            specs.push(spec);
            i += 1;
        }
    }

    // Check for duplicate IDs
    for i in 0..specs.len() {
        for j in (i + 1)..specs.len() {
            if specs[i].id == specs[j].id {
                return Err(StampError::DomainFile(format!(
                    "Duplicate domain identifier: {}",
                    specs[i].id
                )));
            }
        }
    }

    log::info!(
        "Parsed {} domain specifications from {}",
        specs.len(),
        path.display()
    );

    Ok((specs, got_transform))
}

/// Parses a simple domain specification line.
fn parse_domain_simple(line: &str, path: &Path) -> StampResult<DomainSpec> {
    let parts: Vec<&str> = line.split_whitespace().collect();

    if parts.len() < 2 {
        return Err(StampError::DomainFile(format!(
            "Invalid domain specification in {}: {}",
            path.display(),
            line
        )));
    }

    let id = parts[0].to_string();
    let filename = parts[1].to_string();

    let mut spec = DomainSpec::new(id, filename);

    if parts.len() >= 3 {
        // Has chain specification
        let chain = parts[2].chars().next().unwrap_or('_');
        let chain = if chain == '_' { ' ' } else { chain };

        if parts.len() >= 5 {
            // Has range specification
            let start: i32 = parts[3].parse().map_err(|_| {
                StampError::DomainFile(format!("Invalid start residue: {}", parts[3]))
            })?;
            let end: i32 = parts[4].parse().map_err(|_| {
                StampError::DomainFile(format!("Invalid end residue: {}", parts[4]))
            })?;

            spec.segments = vec![DomainSegment::range(
                BrookhavenNumber::new(start, chain, ' '),
                BrookhavenNumber::new(end, chain, ' '),
            )];
        } else {
            spec.segments = vec![DomainSegment::chain(chain)];
        }
    }

    Ok(spec)
}

/// Parses an extended domain specification with braces.
fn parse_domain_extended(lines: &[String], path: &Path) -> StampResult<(DomainSpec, bool, usize)> {
    // Collect all lines until we find the closing brace
    let mut combined = String::new();
    let mut consumed = 0;

    for line in lines {
        let trimmed = line.trim();
        if !trimmed.starts_with('#') && !trimmed.starts_with('%') {
            combined.push_str(trimmed);
            combined.push(' ');
        }
        consumed += 1;
        if trimmed.contains('}') {
            break;
        }
    }

    // Find the opening brace
    let brace_start = combined.find('{').ok_or_else(|| {
        StampError::DomainFile(format!(
            "Missing '{{' in domain specification: {}",
            combined
        ))
    })?;

    let brace_end = combined.find('}').ok_or_else(|| {
        StampError::DomainFile(format!(
            "Missing '}}' in domain specification: {}",
            combined
        ))
    })?;

    // Parse filename and ID (before the brace)
    let header: Vec<&str> = combined[..brace_start].split_whitespace().collect();
    if header.len() < 2 {
        return Err(StampError::DomainFile(format!(
            "Missing filename or ID in domain specification in {}",
            path.display()
        )));
    }

    let filename = header[0].to_string();
    let id = header[1].to_string();

    // Parse descriptor (between braces)
    let descriptor = combined[brace_start + 1..brace_end].trim().to_uppercase();

    // Parse segments from descriptor
    let segments = parse_domain_descriptor(&descriptor)?;

    let mut spec = DomainSpec::new(id, filename);
    spec.segments = segments;

    // Check for transformation matrix (after the closing brace)
    let mut has_transform = false;
    let remaining = combined[brace_end + 1..].trim();

    if !remaining.is_empty() {
        // Try to parse transformation matrix
        let values: Vec<f64> = remaining
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        if values.len() >= 12 {
            let rotation = RotationMatrix::new(
                values[0], values[1], values[2], values[4], values[5], values[6], values[8],
                values[9], values[10],
            );
            let translation = Vec3::new(values[3], values[7], values[11]);
            spec.transform = Some(Transform {
                rotation,
                translation,
            });
            has_transform = true;
        }
    }

    // If no transformation found inline, check following lines
    if !has_transform && consumed < lines.len() {
        let mut matrix_lines = Vec::new();
        for line in &lines[consumed..] {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('%') {
                continue;
            }
            // Check if this could be a transformation line (starts with a number)
            if trimmed
                .chars()
                .next()
                .map_or(false, |c| c.is_ascii_digit() || c == '-' || c == '.')
            {
                matrix_lines.push(trimmed.to_string());
                consumed += 1;
                if matrix_lines.len() >= 3 {
                    break;
                }
            } else {
                break;
            }
        }

        if matrix_lines.len() >= 3 {
            let all_values: String = matrix_lines.join(" ");
            let values: Vec<f64> = all_values
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            if values.len() >= 12 {
                let rotation = RotationMatrix::new(
                    values[0], values[1], values[2], values[4], values[5], values[6], values[8],
                    values[9], values[10],
                );
                let translation = Vec3::new(values[3], values[7], values[11]);
                spec.transform = Some(Transform {
                    rotation,
                    translation,
                });
                has_transform = true;
            }
        }
    }

    Ok((spec, has_transform, consumed))
}

/// Parses the descriptor part of an extended domain specification.
fn parse_domain_descriptor(descriptor: &str) -> StampResult<Vec<DomainSegment>> {
    let mut segments = Vec::new();
    let tokens: Vec<&str> = descriptor.split_whitespace().collect();

    if tokens.is_empty() {
        return Err(StampError::DomainFile(
            "Empty domain descriptor".to_string(),
        ));
    }

    let mut i = 0;
    while i < tokens.len() {
        let mut reverse = false;

        // Check for REVERSE keyword
        if tokens[i] == "REVERSE" {
            reverse = true;
            i += 1;
            if i >= tokens.len() {
                return Err(StampError::DomainFile(
                    "REVERSE keyword not followed by selection".to_string(),
                ));
            }
        }

        if tokens[i] == "ALL" {
            segments.push(DomainSegment::all().with_reverse(reverse));
            i += 1;
        } else if tokens[i] == "CHAIN" {
            i += 1;
            if i >= tokens.len() {
                return Err(StampError::DomainFile(
                    "CHAIN keyword not followed by chain ID".to_string(),
                ));
            }
            let chain = tokens[i].chars().next().unwrap_or('_');
            let chain = if chain == '_' { ' ' } else { chain };
            segments.push(DomainSegment::chain(chain).with_reverse(reverse));
            i += 1;
        } else {
            // Range specification: cid n ins TO cid n ins
            if i + 6 >= tokens.len() {
                return Err(StampError::DomainFile(format!(
                    "Invalid range specification starting at: {}",
                    tokens[i..].join(" ")
                )));
            }

            // Parse start
            let start_cid = tokens[i].chars().next().unwrap_or('_');
            let start_cid = if start_cid == '_' { ' ' } else { start_cid };
            i += 1;

            let start_n: i32 = tokens[i].parse().map_err(|_| {
                StampError::DomainFile(format!("Invalid residue number: {}", tokens[i]))
            })?;
            i += 1;

            let start_ins = tokens[i].chars().next().unwrap_or('_');
            let start_ins = if start_ins == '_' { ' ' } else { start_ins };
            i += 1;

            // Skip "TO"
            if tokens[i] != "TO" {
                return Err(StampError::DomainFile(format!(
                    "Expected 'TO' in range specification, found: {}",
                    tokens[i]
                )));
            }
            i += 1;

            // Parse end
            let end_cid = tokens[i].chars().next().unwrap_or('_');
            let end_cid = if end_cid == '_' { ' ' } else { end_cid };
            i += 1;

            let end_n: i32 = tokens[i].parse().map_err(|_| {
                StampError::DomainFile(format!("Invalid residue number: {}", tokens[i]))
            })?;
            i += 1;

            let end_ins = tokens[i].chars().next().unwrap_or('_');
            let end_ins = if end_ins == '_' { ' ' } else { end_ins };
            i += 1;

            let start = BrookhavenNumber::new(start_n, start_cid, start_ins);
            let end = BrookhavenNumber::new(end_n, end_cid, end_ins);
            segments.push(DomainSegment::range(start, end).with_reverse(reverse));
        }
    }

    if segments.is_empty() {
        segments.push(DomainSegment::all());
    }

    Ok(segments)
}

/// Loads domains from a domain specification file.
///
/// This is a convenience function that parses the domain file and loads
/// all specified domains.
///
/// # Arguments
///
/// * `path` - Path to the domain file
/// * `base_dir` - Optional base directory for resolving relative PDB paths
///
/// # Returns
///
/// Vector of loaded domains.
///
/// # Errors
///
/// Returns an error if the file cannot be read or any domain cannot be loaded.
pub fn load_domains<P: AsRef<Path>>(path: P, base_dir: Option<&Path>) -> StampResult<Vec<Domain>> {
    let (specs, _) = parse_domain_file(&path)?;

    let mut domains = Vec::with_capacity(specs.len());
    for spec in specs {
        let mut spec = spec;
        if let Some(base) = base_dir {
            if !Path::new(&spec.filename).is_absolute() {
                spec.filename = base.join(&spec.filename).to_string_lossy().to_string();
            }
        }
        domains.push(spec.load()?);
    }

    Ok(domains)
}

/// Writes coordinates to a PDB file.
///
/// # Arguments
///
/// * `path` - Output file path
/// * `domain` - Domain to write
/// * `transform` - Optional transformation to apply to coordinates
///
/// # Errors
///
/// Returns an error if the file cannot be written.
///
/// # Example
///
/// ```no_run
/// use stamp_core::io::{parse_pdb, write_pdb};
/// use stamp_core::types::Transform;
///
/// let domain = parse_pdb("1abc.pdb", Some('A'))?;
/// write_pdb("output.pdb", &domain, None)?;
/// # Ok::<(), stamp_core::types::StampError>(())
/// ```
pub fn write_pdb<P: AsRef<Path>>(
    path: P,
    domain: &Domain,
    transform: Option<&Transform>,
) -> StampResult<()> {
    let path = path.as_ref();
    let mut file = File::create(path)?;

    writeln!(file, "REMARK   Generated by STAMP-Rust")?;
    writeln!(file, "REMARK   Domain: {}", domain.id)?;

    for (i, residue) in domain.residues.iter().enumerate() {
        let coord = match transform {
            Some(t) => t.apply(&residue.ca_coord),
            None => residue.ca_coord,
        };

        // Format: ATOM  serial  name  resName chain resSeq  x y z occupancy tempFactor element
        writeln!(
            file,
            "ATOM  {:5}  CA  {:3} {:1}{:4}    {:8.3}{:8.3}{:8.3}  1.00  0.00           C",
            i + 1,
            one_to_three(residue.aa),
            if domain.chain == ' ' {
                '_'
            } else {
                domain.chain
            },
            residue.pdb_num,
            coord.x,
            coord.y,
            coord.z
        )?;
    }

    writeln!(file, "END")?;

    log::debug!("Wrote {} atoms to {}", domain.len(), path.display());

    Ok(())
}

/// Writes multiple domains to a PDB file with different chain IDs.
///
/// # Arguments
///
/// * `path` - Output file path
/// * `domains` - Domains to write
/// * `transforms` - Optional transformations to apply (one per domain)
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_pdb_multi<P: AsRef<Path>>(
    path: P,
    domains: &[Domain],
    transforms: Option<&[Transform]>,
) -> StampResult<()> {
    let path = path.as_ref();
    let mut file = File::create(path)?;

    writeln!(file, "REMARK   Generated by STAMP-Rust")?;
    writeln!(file, "REMARK   {} domains", domains.len())?;

    let chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut atom_num = 1usize;

    for (domain_idx, domain) in domains.iter().enumerate() {
        let chain = chain_ids
            .chars()
            .nth(domain_idx % chain_ids.len())
            .unwrap_or('X');
        let transform = transforms.and_then(|t| t.get(domain_idx));

        writeln!(
            file,
            "REMARK   Domain {}: {} (chain {})",
            domain_idx + 1,
            domain.id,
            chain
        )?;

        for residue in &domain.residues {
            let coord = match transform {
                Some(t) => t.apply(&residue.ca_coord),
                None => residue.ca_coord,
            };

            writeln!(
                file,
                "ATOM  {:5}  CA  {:3} {:1}{:4}    {:8.3}{:8.3}{:8.3}  1.00  0.00           C",
                atom_num,
                one_to_three(residue.aa),
                chain,
                residue.pdb_num,
                coord.x,
                coord.y,
                coord.z
            )?;

            atom_num += 1;
        }

        writeln!(file, "TER")?;
    }

    writeln!(file, "END")?;

    log::debug!(
        "Wrote {} domains ({} atoms total) to {}",
        domains.len(),
        atom_num - 1,
        path.display()
    );

    Ok(())
}

/// Writes a domain specification file.
///
/// # Arguments
///
/// * `path` - Output file path
/// * `specs` - Domain specifications to write
/// * `include_transforms` - Whether to include transformation matrices
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_domain_file<P: AsRef<Path>>(
    path: P,
    specs: &[DomainSpec],
    include_transforms: bool,
) -> StampResult<()> {
    let path = path.as_ref();
    let mut file = File::create(path)?;

    writeln!(file, "# STAMP domain file generated by STAMP-Rust")?;
    writeln!(file)?;

    for spec in specs {
        // Write filename and ID
        write!(file, "{} {} {{ ", spec.filename, spec.id)?;

        // Write segments
        for (i, segment) in spec.segments.iter().enumerate() {
            if i > 0 {
                write!(file, " ")?;
            }

            if segment.reverse {
                write!(file, "REVERSE ")?;
            }

            match segment.selection_type {
                DomainSelectionType::All => {
                    write!(file, "ALL")?;
                }
                DomainSelectionType::Chain => {
                    let cid = if segment.start.cid == ' ' {
                        '_'
                    } else {
                        segment.start.cid
                    };
                    write!(file, "CHAIN {}", cid)?;
                }
                DomainSelectionType::Range => {
                    write!(
                        file,
                        "{} TO {}",
                        segment.start.to_string_display(),
                        segment.end.to_string_display()
                    )?;
                }
            }
        }

        writeln!(file, " }}")?;

        // Write transformation matrix if present and requested
        if include_transforms {
            if let Some(ref transform) = spec.transform {
                writeln!(
                    file,
                    "{:.6} {:.6} {:.6} {:.6}",
                    transform.rotation[(0, 0)],
                    transform.rotation[(0, 1)],
                    transform.rotation[(0, 2)],
                    transform.translation.x
                )?;
                writeln!(
                    file,
                    "{:.6} {:.6} {:.6} {:.6}",
                    transform.rotation[(1, 0)],
                    transform.rotation[(1, 1)],
                    transform.rotation[(1, 2)],
                    transform.translation.y
                )?;
                writeln!(
                    file,
                    "{:.6} {:.6} {:.6} {:.6}",
                    transform.rotation[(2, 0)],
                    transform.rotation[(2, 1)],
                    transform.rotation[(2, 2)],
                    transform.translation.z
                )?;
            }
        }
    }

    log::debug!(
        "Wrote {} domain specifications to {}",
        specs.len(),
        path.display()
    );

    Ok(())
}

/// Converts three-letter amino acid code to one-letter code.
///
/// # Example
///
/// ```
/// use stamp_core::io::three_to_one;
///
/// assert_eq!(three_to_one("ALA"), 'A');
/// assert_eq!(three_to_one("GLY"), 'G');
/// assert_eq!(three_to_one("UNK"), 'X');
/// ```
#[must_use]
pub fn three_to_one(three: &str) -> char {
    match three.to_uppercase().as_str() {
        // Standard amino acids
        "ALA" => 'A',
        "ARG" => 'R',
        "ASN" => 'N',
        "ASP" => 'D',
        "CYS" => 'C',
        "GLN" => 'Q',
        "GLU" => 'E',
        "GLY" => 'G',
        "HIS" => 'H',
        "ILE" => 'I',
        "LEU" => 'L',
        "LYS" => 'K',
        "MET" => 'M',
        "PHE" => 'F',
        "PRO" => 'P',
        "SER" => 'S',
        "THR" => 'T',
        "TRP" => 'W',
        "TYR" => 'Y',
        "VAL" => 'V',
        // Non-standard amino acids
        "MSE" => 'M', // Selenomethionine
        "SEC" => 'U', // Selenocysteine
        "PYL" => 'O', // Pyrrolysine
        "ASX" => 'B', // Asn or Asp
        "GLX" => 'Z', // Gln or Glu
        "XLE" => 'J', // Leu or Ile
        // Nucleotides (for nucleic acids)
        "A" | "DA" | "ADE" => 'A',
        "C" | "DC" | "CYT" => 'C',
        "G" | "DG" | "GUA" => 'G',
        "T" | "DT" | "THY" => 'T',
        "U" | "URA" => 'U',
        _ => 'X',
    }
}

/// Converts one-letter amino acid code to three-letter code.
///
/// # Example
///
/// ```
/// use stamp_core::io::one_to_three;
///
/// assert_eq!(one_to_three('A'), "ALA");
/// assert_eq!(one_to_three('G'), "GLY");
/// assert_eq!(one_to_three('X'), "UNK");
/// ```
#[must_use]
pub fn one_to_three(one: char) -> &'static str {
    match one.to_ascii_uppercase() {
        'A' => "ALA",
        'R' => "ARG",
        'N' => "ASN",
        'D' => "ASP",
        'C' => "CYS",
        'Q' => "GLN",
        'E' => "GLU",
        'G' => "GLY",
        'H' => "HIS",
        'I' => "ILE",
        'L' => "LEU",
        'K' => "LYS",
        'M' => "MET",
        'F' => "PHE",
        'P' => "PRO",
        'S' => "SER",
        'T' => "THR",
        'W' => "TRP",
        'Y' => "TYR",
        'V' => "VAL",
        'U' => "SEC",
        'O' => "PYL",
        'B' => "ASX",
        'Z' => "GLX",
        'J' => "XLE",
        _ => "UNK",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_three_to_one() {
        assert_eq!(three_to_one("ALA"), 'A');
        assert_eq!(three_to_one("ala"), 'A');
        assert_eq!(three_to_one("GLY"), 'G');
        assert_eq!(three_to_one("UNK"), 'X');
        assert_eq!(three_to_one("MSE"), 'M');
    }

    #[test]
    fn test_one_to_three() {
        assert_eq!(one_to_three('A'), "ALA");
        assert_eq!(one_to_three('a'), "ALA");
        assert_eq!(one_to_three('G'), "GLY");
        assert_eq!(one_to_three('X'), "UNK");
    }

    #[test]
    fn test_brookhaven_number() {
        let num = BrookhavenNumber::new(42, 'A', ' ');
        assert_eq!(num.n, 42);
        assert_eq!(num.cid, 'A');
        assert_eq!(num.ins, ' ');

        let with_ins = BrookhavenNumber::new(42, 'A', 'B');
        assert_eq!(with_ins.to_string(), "A42B");

        let wildcard = BrookhavenNumber::chain_wildcard('A');
        assert!(wildcard.matches(&num));
        assert!(!wildcard.matches(&BrookhavenNumber::new(42, 'B', ' ')));
    }

    #[test]
    fn test_brookhaven_parse() {
        let parsed = BrookhavenNumber::parse("A 42 _").unwrap();
        assert_eq!(parsed.cid, 'A');
        assert_eq!(parsed.n, 42);
        assert_eq!(parsed.ins, ' ');

        let with_ins = BrookhavenNumber::parse("A 42 B").unwrap();
        assert_eq!(with_ins.ins, 'B');
    }

    #[test]
    fn test_domain_segment() {
        let all = DomainSegment::all();
        assert_eq!(all.selection_type, DomainSelectionType::All);
        assert!(!all.reverse);

        let chain = DomainSegment::chain('A');
        assert_eq!(chain.selection_type, DomainSelectionType::Chain);
        assert_eq!(chain.start.cid, 'A');

        let range = DomainSegment::range(
            BrookhavenNumber::new(1, 'A', ' '),
            BrookhavenNumber::new(100, 'A', ' '),
        );
        assert_eq!(range.selection_type, DomainSelectionType::Range);
        assert_eq!(range.start.n, 1);
        assert_eq!(range.end.n, 100);

        let reversed = DomainSegment::all().with_reverse(true);
        assert!(reversed.reverse);
    }

    #[test]
    fn test_domain_spec() {
        let spec = DomainSpec::new("test".to_string(), "test.pdb".to_string());
        assert_eq!(spec.id, "test");
        assert_eq!(spec.filename, "test.pdb");
        assert_eq!(spec.segments.len(), 1);
        assert_eq!(spec.segments[0].selection_type, DomainSelectionType::All);

        let with_chain = DomainSpec::with_chain("test".to_string(), "test.pdb".to_string(), 'A');
        assert_eq!(
            with_chain.segments[0].selection_type,
            DomainSelectionType::Chain
        );

        let with_range =
            DomainSpec::with_range("test".to_string(), "test.pdb".to_string(), 'A', 1, 100);
        assert_eq!(
            with_range.segments[0].selection_type,
            DomainSelectionType::Range
        );
    }

    #[test]
    fn test_parse_domain_descriptor() {
        let all = parse_domain_descriptor("ALL").unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].selection_type, DomainSelectionType::All);

        let chain = parse_domain_descriptor("CHAIN A").unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].selection_type, DomainSelectionType::Chain);
        assert_eq!(chain[0].start.cid, 'A');

        let range = parse_domain_descriptor("A 1 _ TO A 100 _").unwrap();
        assert_eq!(range.len(), 1);
        assert_eq!(range[0].selection_type, DomainSelectionType::Range);
        assert_eq!(range[0].start.n, 1);
        assert_eq!(range[0].end.n, 100);

        let reverse_chain = parse_domain_descriptor("REVERSE CHAIN A").unwrap();
        assert!(reverse_chain[0].reverse);

        let multi = parse_domain_descriptor("A 1 _ TO A 50 _ A 60 _ TO A 100 _").unwrap();
        assert_eq!(multi.len(), 2);
    }

    #[test]
    fn test_atom_type() {
        assert_eq!(AtomType::Ca.atom_name(), " CA ");
        assert_eq!(AtomType::P.atom_name(), " P  ");
        assert_eq!(AtomType::default(), AtomType::Ca);
    }

    #[test]
    fn test_pdb_parse_options() {
        let opts = PdbParseOptions::default();
        assert_eq!(opts.atom_type, AtomType::Ca);
        assert!(opts.first_alt_only);
        assert!(!opts.include_hetatm);
        assert!(opts.allowed_alt.contains(&' '));
        assert!(opts.allowed_alt.contains(&'A'));
    }
}
