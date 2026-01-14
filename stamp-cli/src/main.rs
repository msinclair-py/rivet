//! STAMP Command-Line Interface
//!
//! This binary provides command-line access to the STAMP structural
//! alignment functionality, matching the original STAMP options.

use clap::{ArgAction, Args, Parser, Subcommand};
use log::{error, info, warn};
use stamp_core::io::{
    parse_domain_file, parse_dssp, parse_pdb, write_domain_file, write_pdb, write_pdb_multi,
    DomainSpec,
};
use stamp_core::pairwise::align_pair;
use stamp_core::scan::{roughfit, DatabaseScanner};
use stamp_core::treewise::{multiple_align, refine_alignment};
use stamp_core::types::{Domain, Parameters};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

/// STAMP - Structural Alignment of Multiple Proteins
///
/// A program for structural alignment of proteins using C-alpha coordinates.
/// Implements the Rossmann-Argos scoring scheme with dynamic programming.
#[derive(Parser)]
#[command(name = "stamp")]
#[command(author = "STAMP Contributors")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "Structural Alignment of Multiple Proteins", long_about = None)]
struct Cli {
    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = ArgAction::Count, global = true)]
    verbose: u8,

    /// Output log file (default: stdout)
    #[arg(short = 'o', long = "output", global = true)]
    log_file: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Perform pairwise structural alignment
    #[command(alias = "pair")]
    Pairwise(PairwiseArgs),

    /// Perform tree-guided multiple structure alignment
    #[command(alias = "multi")]
    Treewise(TreewiseArgs),

    /// Scan query against a database of structures
    Scan(ScanArgs),

    /// Perform rough initial alignment from N-terminal ends
    Roughfit(RoughfitArgs),

    /// Show information about a PDB file
    Info(InfoArgs),

    /// Parse and validate a domain file
    Parse(ParseArgs),

    /// Convert between file formats
    Convert(ConvertArgs),
}

/// Alignment parameters shared across commands
#[derive(Args, Clone)]
struct AlignmentParams {
    /// E1 parameter for Pij calculation
    #[arg(long, default_value = "2.0")]
    e1: f64,

    /// E2 parameter for Pij calculation
    #[arg(long, default_value = "0.5")]
    e2: f64,

    /// Gap penalty
    #[arg(long = "pen", default_value = "0.0")]
    gap_penalty: f64,

    /// Cutoff distance for scoring
    #[arg(long, default_value = "10.0")]
    cutoff: f64,

    /// Number of fit passes
    #[arg(short = 'n', long, default_value = "2")]
    passes: u32,

    /// Use secondary structure in scoring
    #[arg(long)]
    use_secondary: bool,

    /// Maximum iterations for refinement
    #[arg(long, default_value = "100")]
    max_iter: u32,

    /// Convergence threshold
    #[arg(long, default_value = "0.01")]
    convergence: f64,
}

impl AlignmentParams {
    fn to_parameters(&self) -> Parameters {
        Parameters {
            n_passes: self.passes,
            e1: self.e1,
            e2: self.e2,
            pen_gap: self.gap_penalty,
            cutoff_dist: self.cutoff,
            use_secondary: self.use_secondary,
            max_iter: self.max_iter,
            convergence: self.convergence,
            ..Default::default()
        }
    }
}

#[derive(Args)]
struct PairwiseArgs {
    /// Domain list file containing two structures
    #[arg(short = 'f', long = "file")]
    domain_file: Option<PathBuf>,

    /// First PDB file (alternative to domain file)
    #[arg(short = '1', long = "pdb1", requires = "pdb2")]
    pdb1: Option<PathBuf>,

    /// Second PDB file
    #[arg(short = '2', long = "pdb2", requires = "pdb1")]
    pdb2: Option<PathBuf>,

    /// Chain for first structure
    #[arg(long, default_value = "A")]
    chain1: char,

    /// Chain for second structure
    #[arg(long, default_value = "A")]
    chain2: char,

    /// DSSP file for first structure
    #[arg(long)]
    dssp1: Option<PathBuf>,

    /// DSSP file for second structure
    #[arg(long)]
    dssp2: Option<PathBuf>,

    /// Output prefix for transformed structures
    #[arg(long = "prefix", default_value = "stamp_trans")]
    output_prefix: String,

    /// Output transformed PDB file
    #[arg(short = 'O', long = "out-pdb")]
    output_pdb: Option<PathBuf>,

    /// Output transformation matrix file
    #[arg(long = "matrix")]
    output_matrix: Option<PathBuf>,

    /// Parameter file (STAMP format)
    #[arg(short = 'P', long = "params")]
    param_file: Option<PathBuf>,

    /// Use sliding window scan mode (for structures in different orientations)
    #[arg(long = "scan-mode")]
    scan_mode: bool,

    /// Slide parameter for scan mode (residues to step)
    #[arg(long, default_value = "5")]
    slide: usize,

    /// Minimum length fraction for scan mode
    #[arg(long = "min-frac", default_value = "0.5")]
    min_frac: f64,

    #[command(flatten)]
    align_params: AlignmentParams,
}

#[derive(Args)]
struct TreewiseArgs {
    /// Domain list file (required)
    #[arg(short = 'f', long = "file")]
    domain_file: PathBuf,

    /// Output prefix for transformed structures
    #[arg(long = "prefix", default_value = "stamp_trans")]
    output_prefix: String,

    /// Output multi-domain PDB file
    #[arg(short = 'O', long = "out-pdb")]
    output_pdb: Option<PathBuf>,

    /// Output domain file with transformations
    #[arg(long = "out-domain")]
    output_domain: Option<PathBuf>,

    /// Parameter file (STAMP format)
    #[arg(short = 'P', long = "params")]
    param_file: Option<PathBuf>,

    /// DSSP directory for secondary structure files
    #[arg(long = "dssp-dir")]
    dssp_dir: Option<PathBuf>,

    /// Refine alignment after initial tree-guided alignment
    #[arg(long)]
    refine: bool,

    #[command(flatten)]
    align_params: AlignmentParams,
}

#[derive(Args)]
struct ScanArgs {
    /// Query PDB file
    #[arg(short = 'q', long = "query")]
    query: PathBuf,

    /// Database domain file
    #[arg(short = 'd', long = "database")]
    database: PathBuf,

    /// Chain for query structure
    #[arg(long, default_value = "A")]
    chain: char,

    /// DSSP file for query structure
    #[arg(long)]
    dssp: Option<PathBuf>,

    /// Slide parameter for sliding window scan (residues to step)
    #[arg(long, default_value = "5")]
    slide: usize,

    /// Score cutoff for reporting hits (Sc)
    #[arg(long = "scancut", default_value = "2.0")]
    score_cutoff: f64,

    /// Maximum number of hits to report
    #[arg(short = 'm', long = "max-hits", default_value = "100")]
    max_hits: usize,

    /// Number of top hits to refine with full alignment
    #[arg(long = "refine-top", default_value = "10")]
    refine_top: usize,

    /// Use quick alignment (faster but less accurate, no sliding window)
    #[arg(long)]
    quick: bool,

    /// Use sliding window scan mode (for structures of different lengths/orientations)
    #[arg(long = "scan-mode")]
    scan_mode: bool,

    /// Minimum length fraction for target structures (default: 0.5)
    #[arg(long = "min-frac", default_value = "0.5")]
    min_frac: f64,

    /// Truncate long target structures
    #[arg(long)]
    truncate: bool,

    /// Truncation factor (max target/query length ratio, default: 1.3)
    #[arg(long = "trunc-factor", default_value = "1.3")]
    trunc_factor: f64,

    /// Output file for scan results
    #[arg(short = 'O', long = "out")]
    output: Option<PathBuf>,

    #[command(flatten)]
    align_params: AlignmentParams,
}

#[derive(Args)]
struct RoughfitArgs {
    /// Domain list file (required)
    #[arg(short = 'f', long = "file")]
    domain_file: PathBuf,

    /// Output prefix for transformed structures
    #[arg(long = "prefix", default_value = "roughfit")]
    output_prefix: String,

    /// Output multi-domain PDB file
    #[arg(short = 'O', long = "out-pdb")]
    output_pdb: Option<PathBuf>,

    /// Output domain file with transformations
    #[arg(long = "out-domain")]
    output_domain: Option<PathBuf>,

    /// DSSP directory for secondary structure files
    #[arg(long = "dssp-dir")]
    dssp_dir: Option<PathBuf>,

    #[command(flatten)]
    align_params: AlignmentParams,
}

#[derive(Args)]
struct InfoArgs {
    /// PDB file to analyze
    #[arg(short = 'f', long = "file")]
    pdb: PathBuf,

    /// Chain to analyze (default: first chain)
    #[arg(short = 'c', long = "chain")]
    chain: Option<char>,

    /// DSSP file for secondary structure
    #[arg(long)]
    dssp: Option<PathBuf>,

    /// Show full sequence
    #[arg(long)]
    sequence: bool,

    /// Show secondary structure
    #[arg(long)]
    secondary: bool,

    /// Show coordinate statistics
    #[arg(long)]
    coords: bool,
}

#[derive(Args)]
struct ParseArgs {
    /// Domain file to parse
    #[arg(short = 'f', long = "file")]
    domain_file: PathBuf,

    /// Verify that all PDB files exist and are readable
    #[arg(long)]
    verify: bool,

    /// Load all domains and report statistics
    #[arg(long)]
    load: bool,
}

#[derive(Args)]
struct ConvertArgs {
    /// Input PDB file
    #[arg(short = 'i', long = "input")]
    input: PathBuf,

    /// Output PDB file
    #[arg(short = 'o', long = "output")]
    output: PathBuf,

    /// Input chain
    #[arg(long, default_value = "A")]
    chain: char,

    /// Apply transformation matrix file
    #[arg(long = "transform")]
    transform_file: Option<PathBuf>,
}

fn main() {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = match cli.verbose {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };

    env_logger::Builder::new()
        .filter_level(log_level)
        .format_timestamp(None)
        .init();

    // Set up output writer
    let mut output: Box<dyn Write> = match &cli.log_file {
        Some(path) => match File::create(path) {
            Ok(file) => Box::new(BufWriter::new(file)),
            Err(e) => {
                error!("Failed to create output file {:?}: {}", path, e);
                std::process::exit(1);
            }
        },
        None => Box::new(std::io::stdout()),
    };

    let result = match cli.command {
        Commands::Pairwise(args) => run_pairwise(args, &mut output),
        Commands::Treewise(args) => run_treewise(args, &mut output),
        Commands::Scan(args) => run_scan(args, &mut output),
        Commands::Roughfit(args) => run_roughfit(args, &mut output),
        Commands::Info(args) => run_info(args, &mut output),
        Commands::Parse(args) => run_parse(args, &mut output),
        Commands::Convert(args) => run_convert(args, &mut output),
    };

    if let Err(e) = result {
        error!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_pairwise(
    args: PairwiseArgs,
    output: &mut dyn Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let params = args.align_params.to_parameters();

    // Load domains either from domain file or from individual PDB files
    let (mut domain1, mut domain2) = if let Some(ref domain_file) = args.domain_file {
        info!("Loading domain file: {:?}", domain_file);
        let (specs, _) = parse_domain_file(domain_file)?;

        if specs.len() < 2 {
            return Err(
                "Domain file must contain at least 2 structures for pairwise alignment".into(),
            );
        }
        if specs.len() > 2 {
            warn!(
                "Domain file contains {} structures, using first two",
                specs.len()
            );
        }

        (specs[0].load()?, specs[1].load()?)
    } else if let (Some(ref pdb1), Some(ref pdb2)) = (&args.pdb1, &args.pdb2) {
        info!("Loading PDB files: {:?} and {:?}", pdb1, pdb2);
        (
            parse_pdb(pdb1, Some(args.chain1))?,
            parse_pdb(pdb2, Some(args.chain2))?,
        )
    } else {
        return Err("Either --file or both --pdb1 and --pdb2 must be specified".into());
    };

    // Load DSSP files if provided
    if let Some(ref dssp1) = args.dssp1 {
        info!("Loading DSSP file for structure 1: {:?}", dssp1);
        parse_dssp(dssp1, &mut domain1)?;
    }
    if let Some(ref dssp2) = args.dssp2 {
        info!("Loading DSSP file for structure 2: {:?}", dssp2);
        parse_dssp(dssp2, &mut domain2)?;
    }

    info!("Structure 1: {} ({} residues)", domain1.id, domain1.len());
    info!("Structure 2: {} ({} residues)", domain2.id, domain2.len());

    info!("Performing alignment with {} passes...", params.n_passes);
    let result = align_pair(&domain1, &domain2, &params)?;

    // Output results
    writeln!(output)?;
    writeln!(output, "=== Pairwise Alignment Results ===")?;
    writeln!(
        output,
        "Structure 1:       {} ({} residues)",
        domain1.id,
        domain1.len()
    )?;
    writeln!(
        output,
        "Structure 2:       {} ({} residues)",
        domain2.id,
        domain2.len()
    )?;
    writeln!(output, "-----------------------------------")?;
    writeln!(output, "Aligned residues:  {}", result.n_aligned)?;
    writeln!(output, "RMSD:              {:.3} A", result.rmsd)?;
    writeln!(output, "Sc score:          {:.4}", result.score)?;
    writeln!(
        output,
        "Sequence identity: {:.1}%",
        result.seq_identity * 100.0
    )?;

    // Output transformation matrix
    writeln!(output)?;
    writeln!(
        output,
        "Transformation matrix (to superpose {} onto {}):",
        domain2.id, domain1.id
    )?;
    for i in 0..3 {
        writeln!(
            output,
            "  {:10.6} {:10.6} {:10.6}  {:10.6}",
            result.transform.rotation[(i, 0)],
            result.transform.rotation[(i, 1)],
            result.transform.rotation[(i, 2)],
            result.transform.translation[i]
        )?;
    }

    // Write output files
    if let Some(ref output_path) = args.output_pdb {
        info!("Writing transformed structure to {:?}", output_path);
        write_pdb(output_path, &domain2, Some(&result.transform))?;
        writeln!(output)?;
        writeln!(
            output,
            "Transformed structure written to: {:?}",
            output_path
        )?;
    }

    if let Some(ref matrix_path) = args.output_matrix {
        info!("Writing transformation matrix to {:?}", matrix_path);
        let mut matrix_file = File::create(matrix_path)?;
        for i in 0..3 {
            writeln!(
                matrix_file,
                "{:.6} {:.6} {:.6} {:.6}",
                result.transform.rotation[(i, 0)],
                result.transform.rotation[(i, 1)],
                result.transform.rotation[(i, 2)],
                result.transform.translation[i]
            )?;
        }
        writeln!(
            output,
            "Transformation matrix written to: {:?}",
            matrix_path
        )?;
    }

    Ok(())
}

fn run_treewise(
    args: TreewiseArgs,
    output: &mut dyn Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let params = args.align_params.to_parameters();

    info!("Loading domain file: {:?}", args.domain_file);
    let (specs, _) = parse_domain_file(&args.domain_file)?;

    if specs.len() < 2 {
        return Err("Need at least 2 structures for multiple alignment".into());
    }

    info!("Loading {} structures...", specs.len());
    let mut domains: Vec<Domain> = Vec::new();
    for spec in &specs {
        match spec.load() {
            Ok(mut domain) => {
                // Try to load DSSP if directory provided
                if let Some(ref dssp_dir) = args.dssp_dir {
                    let dssp_path = dssp_dir.join(format!("{}.dssp", domain.id));
                    if dssp_path.exists() {
                        if let Err(e) = parse_dssp(&dssp_path, &mut domain) {
                            warn!("Failed to load DSSP for {}: {}", domain.id, e);
                        }
                    }
                }
                info!("  Loaded {} ({} residues)", domain.id, domain.len());
                domains.push(domain);
            }
            Err(e) => {
                error!("Failed to load {}: {}", spec.id, e);
                return Err(e.into());
            }
        }
    }

    info!("Performing tree-guided multiple alignment...");
    let mut result = multiple_align(&domains, &params)?;

    if args.refine {
        info!("Refining alignment...");
        refine_alignment(&domains, &mut result, &params)?;
    }

    // Output results
    writeln!(output)?;
    writeln!(output, "=== Multiple Alignment Results ===")?;
    writeln!(output, "Number of structures: {}", domains.len())?;
    writeln!(output, "Alignment columns:    {}", result.n_columns())?;
    writeln!(output, "Core positions:       {}", result.n_core())?;
    writeln!(output, "Average RMSD:         {:.3} A", result.avg_rmsd)?;
    writeln!(output, "Overall score:        {:.4}", result.score)?;

    writeln!(output)?;
    writeln!(output, "Structures:")?;
    for (i, domain) in domains.iter().enumerate() {
        writeln!(
            output,
            "  {:2}. {} ({} residues)",
            i + 1,
            domain.id,
            domain.len()
        )?;
    }

    // Write output files
    if let Some(ref output_path) = args.output_pdb {
        info!("Writing superposed structures to {:?}", output_path);
        write_pdb_multi(output_path, &domains, Some(&result.transforms))?;
        writeln!(output)?;
        writeln!(
            output,
            "Superposed structures written to: {:?}",
            output_path
        )?;
    }

    if let Some(ref output_path) = args.output_domain {
        info!(
            "Writing domain file with transformations to {:?}",
            output_path
        );
        let mut updated_specs: Vec<DomainSpec> = specs.clone();
        for (i, spec) in updated_specs.iter_mut().enumerate() {
            spec.transform = Some(result.transforms[i].clone());
        }
        write_domain_file(output_path, &updated_specs, true)?;
        writeln!(
            output,
            "Domain file with transformations written to: {:?}",
            output_path
        )?;
    }

    // Write individual transformed structures
    if args.output_pdb.is_none() {
        for (i, domain) in domains.iter().enumerate() {
            let output_path = format!("{}_{}.pdb", args.output_prefix, domain.id);
            write_pdb(&output_path, domain, Some(&result.transforms[i]))?;
            info!("Wrote {}", output_path);
        }
        writeln!(output)?;
        writeln!(
            output,
            "Transformed structures written with prefix: {}",
            args.output_prefix
        )?;
    }

    Ok(())
}

fn run_scan(args: ScanArgs, output: &mut dyn Write) -> Result<(), Box<dyn std::error::Error>> {
    let params = args.align_params.to_parameters();

    info!("Loading query structure: {:?}", args.query);
    let mut query = parse_pdb(&args.query, Some(args.chain))?;

    if let Some(ref dssp) = args.dssp {
        info!("Loading DSSP file: {:?}", dssp);
        parse_dssp(dssp, &mut query)?;
    }

    info!("Query: {} ({} residues)", query.id, query.len());

    info!("Loading database: {:?}", args.database);
    let (specs, _) = parse_domain_file(&args.database)?;

    let mut scanner = DatabaseScanner::new(params);
    scanner.set_quick_scan(args.quick);
    scanner.set_score_threshold(args.score_cutoff);
    scanner.set_max_hits(args.max_hits);
    scanner.set_slide(args.slide);
    scanner.set_scan_mode(args.scan_mode);

    let mut loaded = 0;
    for spec in &specs {
        match spec.load() {
            Ok(domain) => {
                scanner.add_target(domain);
                loaded += 1;
            }
            Err(e) => {
                warn!("Skipping {}: {}", spec.id, e);
            }
        }
    }

    info!(
        "Database contains {} structures ({} loaded)",
        specs.len(),
        loaded
    );
    if args.scan_mode {
        info!(
            "Scanning with slide={}, score cutoff={}...",
            args.slide, args.score_cutoff
        );
    } else {
        info!("Scanning with score cutoff {}...", args.score_cutoff);
    }

    let mut hits = scanner.scan(&query)?;

    // Refine top hits if requested
    if args.refine_top > 0 && (args.quick || args.scan_mode) {
        info!("Refining top {} hits...", args.refine_top);
        scanner.refine_top_hits(&query, &mut hits, args.refine_top)?;
    }

    // Output results
    writeln!(output)?;
    writeln!(output, "=== Scan Results ===")?;
    writeln!(output, "Query: {} ({} residues)", query.id, query.len())?;
    writeln!(output, "Database size: {}", scanner.database_size())?;
    if args.scan_mode {
        writeln!(output, "Scan mode: sliding window (slide={})", args.slide)?;
    }
    writeln!(output, "Score cutoff: {:.2}", args.score_cutoff)?;
    writeln!(output, "Hits above threshold: {}", hits.len())?;
    writeln!(output)?;

    writeln!(
        output,
        "{:>4} {:>20} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "Rank", "Target", "Score", "RMSD", "N_align", "Seq_ID", "Z-score"
    )?;
    writeln!(output, "{}", "-".repeat(76))?;

    for hit in &hits {
        let z_score = hit
            .z_score
            .map_or("N/A".to_string(), |z| format!("{:.2}", z));
        writeln!(
            output,
            "{:>4} {:>20} {:>8.4} {:>8.3} {:>8} {:>7.1}% {:>10}",
            hit.rank,
            hit.target_id,
            hit.alignment.score,
            hit.alignment.rmsd,
            hit.alignment.n_aligned,
            hit.alignment.seq_identity * 100.0,
            z_score
        )?;
    }

    // Write to file if specified
    if let Some(ref output_path) = args.output {
        let mut file = File::create(output_path)?;
        writeln!(file, "# STAMP scan results")?;
        writeln!(file, "# Query: {}", query.id)?;
        writeln!(file, "# Database: {:?}", args.database)?;
        writeln!(file, "# Score cutoff: {}", args.score_cutoff)?;
        writeln!(file, "#")?;
        writeln!(
            file,
            "# Rank\tTarget\tScore\tRMSD\tN_align\tSeq_ID\tZ-score"
        )?;
        for hit in &hits {
            let z_score = hit
                .z_score
                .map_or("NA".to_string(), |z| format!("{:.4}", z));
            writeln!(
                file,
                "{}\t{}\t{:.4}\t{:.3}\t{}\t{:.4}\t{}",
                hit.rank,
                hit.target_id,
                hit.alignment.score,
                hit.alignment.rmsd,
                hit.alignment.n_aligned,
                hit.alignment.seq_identity,
                z_score
            )?;
        }
        writeln!(output)?;
        writeln!(output, "Results written to: {:?}", output_path)?;
    }

    Ok(())
}

fn run_roughfit(
    args: RoughfitArgs,
    output: &mut dyn Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let params = args.align_params.to_parameters();

    info!("Loading domain file: {:?}", args.domain_file);
    let (specs, _) = parse_domain_file(&args.domain_file)?;

    if specs.len() < 2 {
        return Err("Need at least 2 structures for roughfit".into());
    }

    info!("Loading {} structures...", specs.len());
    let mut domains: Vec<Domain> = Vec::new();
    for spec in &specs {
        match spec.load() {
            Ok(mut domain) => {
                // Try to load DSSP if directory provided
                if let Some(ref dssp_dir) = args.dssp_dir {
                    let dssp_path = dssp_dir.join(format!("{}.dssp", domain.id));
                    if dssp_path.exists() {
                        if let Err(e) = parse_dssp(&dssp_path, &mut domain) {
                            warn!("Failed to load DSSP for {}: {}", domain.id, e);
                        }
                    }
                }
                info!("  Loaded {} ({} residues)", domain.id, domain.len());
                domains.push(domain);
            }
            Err(e) => {
                error!("Failed to load {}: {}", spec.id, e);
                return Err(e.into());
            }
        }
    }

    info!("Performing rough N-terminal alignment...");
    let results = roughfit(&mut domains, &params)?;

    // Output results
    writeln!(output)?;
    writeln!(output, "=== Roughfit Results ===")?;
    writeln!(output, "Number of structures: {}", domains.len())?;
    writeln!(
        output,
        "Reference structure: {} ({} residues)",
        domains[0].id,
        domains[0].len()
    )?;
    writeln!(output)?;

    writeln!(
        output,
        "{:>4} {:>20} {:>10} {:>10}",
        "Num", "Domain", "N_aligned", "RMSD"
    )?;
    writeln!(output, "{}", "-".repeat(50))?;

    for (i, result) in results.iter().enumerate() {
        writeln!(
            output,
            "{:>4} {:>20} {:>10} {:>10.3}",
            i + 1,
            result.domain_id,
            result.n_aligned,
            if result.rmsd.is_finite() {
                result.rmsd
            } else {
                -1.0
            }
        )?;
    }

    // Write output files
    if let Some(ref output_path) = args.output_pdb {
        info!("Writing superposed structures to {:?}", output_path);
        let transforms: Vec<_> = results.iter().map(|r| r.transform.clone()).collect();
        write_pdb_multi(output_path, &domains, Some(&transforms))?;
        writeln!(output)?;
        writeln!(
            output,
            "Superposed structures written to: {:?}",
            output_path
        )?;
    }

    if let Some(ref output_path) = args.output_domain {
        info!(
            "Writing domain file with transformations to {:?}",
            output_path
        );
        let mut updated_specs: Vec<DomainSpec> = specs.clone();
        for (i, spec) in updated_specs.iter_mut().enumerate() {
            spec.transform = Some(results[i].transform.clone());
        }
        write_domain_file(output_path, &updated_specs, true)?;
        writeln!(
            output,
            "Domain file with transformations written to: {:?}",
            output_path
        )?;
    }

    // Write individual transformed structures
    if args.output_pdb.is_none() {
        for (i, domain) in domains.iter().enumerate() {
            let output_path = format!("{}_{}.pdb", args.output_prefix, domain.id);
            write_pdb(&output_path, domain, Some(&results[i].transform))?;
            info!("Wrote {}", output_path);
        }
        writeln!(output)?;
        writeln!(
            output,
            "Transformed structures written with prefix: {}",
            args.output_prefix
        )?;
    }

    Ok(())
}

fn run_info(args: InfoArgs, output: &mut dyn Write) -> Result<(), Box<dyn std::error::Error>> {
    let mut domain = parse_pdb(&args.pdb, args.chain)?;

    if let Some(ref dssp) = args.dssp {
        parse_dssp(dssp, &mut domain)?;
    }

    writeln!(output)?;
    writeln!(output, "=== Structure Information ===")?;
    writeln!(output, "File:     {:?}", args.pdb)?;
    writeln!(output, "ID:       {}", domain.id)?;
    writeln!(output, "Chain:    {}", domain.chain)?;
    writeln!(output, "Residues: {}", domain.len())?;

    if args.sequence || (!args.secondary && !args.coords) {
        writeln!(output)?;
        writeln!(output, "Sequence:")?;
        let seq = domain.sequence();
        for (i, chunk) in seq.as_bytes().chunks(60).enumerate() {
            writeln!(
                output,
                "  {:>4}: {}",
                i * 60 + 1,
                std::str::from_utf8(chunk).unwrap_or("?")
            )?;
        }
    }

    if args.secondary || domain.has_dssp {
        writeln!(output)?;
        writeln!(output, "Secondary structure:")?;
        let ss = domain.secondary_structure();
        for (i, chunk) in ss.as_bytes().chunks(60).enumerate() {
            writeln!(
                output,
                "  {:>4}: {}",
                i * 60 + 1,
                std::str::from_utf8(chunk).unwrap_or("?")
            )?;
        }

        let (h, e, c) = stamp_core::secondary::ss_content(&domain);
        writeln!(output)?;
        writeln!(output, "Secondary structure content:")?;
        writeln!(output, "  Helix:  {:5.1}%", h * 100.0)?;
        writeln!(output, "  Strand: {:5.1}%", e * 100.0)?;
        writeln!(output, "  Coil:   {:5.1}%", c * 100.0)?;
    }

    if args.coords {
        writeln!(output)?;
        writeln!(output, "Coordinate statistics:")?;

        let coords: Vec<_> = domain.ca_coords().collect();
        if !coords.is_empty() {
            let mut min_x = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            let mut min_z = f64::INFINITY;
            let mut max_z = f64::NEG_INFINITY;

            for coord in &coords {
                min_x = min_x.min(coord.x);
                max_x = max_x.max(coord.x);
                min_y = min_y.min(coord.y);
                max_y = max_y.max(coord.y);
                min_z = min_z.min(coord.z);
                max_z = max_z.max(coord.z);
            }

            writeln!(
                output,
                "  X range: {:.2} to {:.2} ({:.2} A)",
                min_x,
                max_x,
                max_x - min_x
            )?;
            writeln!(
                output,
                "  Y range: {:.2} to {:.2} ({:.2} A)",
                min_y,
                max_y,
                max_y - min_y
            )?;
            writeln!(
                output,
                "  Z range: {:.2} to {:.2} ({:.2} A)",
                min_z,
                max_z,
                max_z - min_z
            )?;

            let centroid = stamp_core::math::centroid(coords.iter().copied());
            writeln!(
                output,
                "  Centroid: ({:.2}, {:.2}, {:.2})",
                centroid.x, centroid.y, centroid.z
            )?;
        }
    }

    Ok(())
}

fn run_parse(args: ParseArgs, output: &mut dyn Write) -> Result<(), Box<dyn std::error::Error>> {
    info!("Parsing domain file: {:?}", args.domain_file);
    let (specs, has_transforms) = parse_domain_file(&args.domain_file)?;

    writeln!(output)?;
    writeln!(output, "=== Domain File Contents ===")?;
    writeln!(output, "File: {:?}", args.domain_file)?;
    writeln!(output, "Domains found: {}", specs.len())?;
    writeln!(
        output,
        "Has transformations: {}",
        if has_transforms { "yes" } else { "no" }
    )?;
    writeln!(output)?;

    for (i, spec) in specs.iter().enumerate() {
        writeln!(output, "{}. {} (from {})", i + 1, spec.id, spec.filename)?;

        if args.verify {
            let path = std::path::Path::new(&spec.filename);
            if path.exists() {
                writeln!(output, "   Status: file exists")?;
            } else {
                writeln!(output, "   Status: FILE NOT FOUND")?;
            }
        }

        if args.load {
            match spec.load() {
                Ok(domain) => {
                    writeln!(
                        output,
                        "   Loaded: {} residues, chain {}",
                        domain.len(),
                        domain.chain
                    )?;
                }
                Err(e) => {
                    writeln!(output, "   Load error: {}", e)?;
                }
            }
        }
    }

    Ok(())
}

fn run_convert(
    args: ConvertArgs,
    output: &mut dyn Write,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Loading input PDB: {:?}", args.input);
    let domain = parse_pdb(&args.input, Some(args.chain))?;

    let transform = if let Some(ref transform_file) = args.transform_file {
        info!("Loading transformation from: {:?}", transform_file);
        // Parse transformation matrix file
        let content = std::fs::read_to_string(transform_file)?;
        let values: Vec<f64> = content
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        if values.len() < 12 {
            return Err("Transformation file must contain at least 12 values (3x4 matrix)".into());
        }

        let rotation = stamp_core::types::RotationMatrix::new(
            values[0], values[1], values[2], values[4], values[5], values[6], values[8], values[9],
            values[10],
        );
        let translation = stamp_core::types::Vec3::new(values[3], values[7], values[11]);

        Some(stamp_core::types::Transform {
            rotation,
            translation,
        })
    } else {
        None
    };

    info!("Writing output PDB: {:?}", args.output);
    write_pdb(&args.output, &domain, transform.as_ref())?;

    writeln!(output)?;
    writeln!(
        output,
        "Converted {} ({} residues) -> {:?}",
        domain.id,
        domain.len(),
        args.output
    )?;
    if transform.is_some() {
        writeln!(
            output,
            "Transformation applied from: {:?}",
            args.transform_file
        )?;
    }

    Ok(())
}
