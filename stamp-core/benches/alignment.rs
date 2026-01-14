//! Benchmarks for alignment operations.
//!
//! Run with: cargo bench --features parallel
//! Or without parallel: cargo bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use stamp_core::alignment::SmithWaterman;
use stamp_core::math::{
    distance_matrix_flat, kabsch, kabsch_soa, rmsd, rmsd_soa, superpose, superpose_soa,
};
use stamp_core::matrix::{CoordsSoA, Matrix, SymmetricMatrix};
use stamp_core::scoring::{calculate_probability_matrix, RossmannArgos, RossmannParams};
use stamp_core::types::{Coord3, Domain, Parameters, Residue};

fn make_test_domain(n: usize) -> Domain {
    let mut domain = Domain::new("bench".to_string());
    for i in 0..n {
        // Create a simple helix-like structure
        let t = i as f64 * 0.1;
        let x = 2.3 * t.cos();
        let y = 2.3 * t.sin();
        let z = 1.5 * i as f64;
        domain.residues.push(Residue::new(
            i as i32,
            i.to_string(),
            'A',
            Coord3::new(x, y, z),
        ));
    }
    domain
}

fn make_test_coords(n: usize) -> Vec<Coord3> {
    (0..n)
        .map(|i| {
            let t = i as f64 * 0.1;
            Coord3::new(2.3 * t.cos(), 2.3 * t.sin(), 1.5 * i as f64)
        })
        .collect()
}

fn make_test_coords_soa(n: usize) -> CoordsSoA {
    let coords = make_test_coords(n);
    CoordsSoA::from_coords(&coords)
}

// =============================================================================
// Matrix Operations Benchmarks
// =============================================================================

fn bench_matrix_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_creation");

    for size in [50, 100, 200, 500].iter() {
        group.bench_with_input(BenchmarkId::new("flat_matrix", size), size, |b, &size| {
            b.iter(|| Matrix::<f64>::new(black_box(size), black_box(size)))
        });

        group.bench_with_input(BenchmarkId::new("vec_vec", size), size, |b, &size| {
            b.iter(|| vec![vec![0.0f64; black_box(size)]; black_box(size)])
        });
    }

    group.finish();
}

fn bench_matrix_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_access");
    let size = 200;

    // Setup
    let flat: Matrix<f64> = Matrix::filled(size, size, 1.0);
    let nested: Vec<Vec<f64>> = vec![vec![1.0; size]; size];

    group.bench_function("flat_row_sum", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for row in 0..size {
                sum += flat.row(row).iter().sum::<f64>();
            }
            sum
        })
    });

    group.bench_function("nested_row_sum", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for row in &nested {
                sum += row.iter().sum::<f64>();
            }
            sum
        })
    });

    group.bench_function("flat_random_access", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for i in 0..1000 {
                let row = (i * 7) % size;
                let col = (i * 13) % size;
                sum += flat[(row, col)];
            }
            sum
        })
    });

    group.bench_function("nested_random_access", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for i in 0..1000 {
                let row = (i * 7) % size;
                let col = (i * 13) % size;
                sum += nested[row][col];
            }
            sum
        })
    });

    group.finish();
}

fn bench_symmetric_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("symmetric_matrix");

    for size in [100, 200, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("symmetric_creation", size),
            size,
            |b, &size| b.iter(|| SymmetricMatrix::<f64>::new(black_box(size))),
        );

        group.bench_with_input(
            BenchmarkId::new("full_matrix_creation", size),
            size,
            |b, &size| b.iter(|| Matrix::<f64>::new(black_box(size), black_box(size))),
        );
    }

    // Access patterns
    let sym: SymmetricMatrix<f64> = SymmetricMatrix::new(200);
    let full: Matrix<f64> = Matrix::new(200, 200);

    group.bench_function("symmetric_access_200", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for i in 0..199 {
                for j in (i + 1)..200 {
                    sum += sym.get(i, j);
                }
            }
            sum
        })
    });

    group.bench_function("full_access_200", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for i in 0..199 {
                for j in (i + 1)..200 {
                    sum += full[(i, j)];
                }
            }
            sum
        })
    });

    group.finish();
}

// =============================================================================
// Coordinate Layout Benchmarks (AoS vs SoA)
// =============================================================================

fn bench_coords_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("coords_conversion");

    for size in [100, 500, 1000].iter() {
        let coords = make_test_coords(*size);

        group.bench_with_input(
            BenchmarkId::new("aos_to_soa", size),
            &coords,
            |b, coords| b.iter(|| CoordsSoA::from_coords(black_box(coords))),
        );

        let soa = CoordsSoA::from_coords(&coords);
        group.bench_with_input(BenchmarkId::new("soa_to_aos", size), &soa, |b, soa| {
            b.iter(|| black_box(soa).to_coords())
        });
    }

    group.finish();
}

fn bench_centroid(c: &mut Criterion) {
    let mut group = c.benchmark_group("centroid");

    for size in [100, 500, 1000].iter() {
        let coords = make_test_coords(*size);
        let soa = CoordsSoA::from_coords(&coords);

        group.bench_with_input(BenchmarkId::new("aos", size), &coords, |b, coords| {
            b.iter(|| {
                let n = coords.len() as f64;
                let cx: f64 = coords.iter().map(|c| c.x).sum::<f64>() / n;
                let cy: f64 = coords.iter().map(|c| c.y).sum::<f64>() / n;
                let cz: f64 = coords.iter().map(|c| c.z).sum::<f64>() / n;
                (cx, cy, cz)
            })
        });

        group.bench_with_input(BenchmarkId::new("soa", size), &soa, |b, soa| {
            b.iter(|| black_box(soa).centroid())
        });
    }

    group.finish();
}

// =============================================================================
// RMSD and Superposition Benchmarks
// =============================================================================

fn bench_rmsd(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmsd");

    for size in [50, 100, 200, 500].iter() {
        let coords1 = make_test_coords(*size);
        let coords2: Vec<Coord3> = coords1
            .iter()
            .map(|c| Coord3::new(c.x + 0.1, c.y - 0.1, c.z + 0.05))
            .collect();

        let soa1 = CoordsSoA::from_coords(&coords1);
        let soa2 = CoordsSoA::from_coords(&coords2);

        group.bench_with_input(
            BenchmarkId::new("aos", size),
            &(&coords1, &coords2),
            |b, (c1, c2)| b.iter(|| rmsd(black_box(c1.iter()), black_box(c2.iter()))),
        );

        group.bench_with_input(
            BenchmarkId::new("soa", size),
            &(&soa1, &soa2),
            |b, (s1, s2)| b.iter(|| rmsd_soa(black_box(*s1), black_box(*s2))),
        );
    }

    group.finish();
}

fn bench_kabsch(c: &mut Criterion) {
    let mut group = c.benchmark_group("kabsch");

    for size in [50, 100, 200].iter() {
        let coords1 = make_test_coords(*size);
        let coords2: Vec<Coord3> = coords1
            .iter()
            .map(|c| Coord3::new(c.x + 1.0, c.y - 0.5, c.z + 0.2))
            .collect();

        let soa1 = CoordsSoA::from_coords(&coords1);
        let soa2 = CoordsSoA::from_coords(&coords2);

        group.bench_with_input(
            BenchmarkId::new("aos", size),
            &(&coords1, &coords2),
            |b, (c1, c2)| b.iter(|| kabsch(black_box(c1.iter()), black_box(c2.iter()))),
        );

        group.bench_with_input(
            BenchmarkId::new("soa", size),
            &(&soa1, &soa2),
            |b, (s1, s2)| b.iter(|| kabsch_soa(black_box(s1), black_box(s2))),
        );
    }

    group.finish();
}

fn bench_superpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("superpose");

    for size in [50, 100, 200].iter() {
        let coords1 = make_test_coords(*size);
        let coords2: Vec<Coord3> = coords1
            .iter()
            .map(|c| Coord3::new(c.x + 10.0, c.y + 5.0, c.z))
            .collect();

        let soa1 = CoordsSoA::from_coords(&coords1);
        let soa2 = CoordsSoA::from_coords(&coords2);

        group.bench_with_input(
            BenchmarkId::new("aos", size),
            &(&coords1, &coords2),
            |b, (c1, c2)| b.iter(|| superpose(black_box(*c1), black_box(*c2))),
        );

        group.bench_with_input(
            BenchmarkId::new("soa", size),
            &(&soa1, &soa2),
            |b, (s1, s2)| b.iter(|| superpose_soa(black_box(s1), black_box(s2))),
        );
    }

    group.finish();
}

// =============================================================================
// Distance Matrix Benchmarks
// =============================================================================

fn bench_distance_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_matrix");

    for size in [50, 100, 200].iter() {
        let coords1 = make_test_coords(*size);
        let coords2 = make_test_coords(*size);

        group.bench_with_input(
            BenchmarkId::new("flat", size),
            &(&coords1, &coords2),
            |b, (c1, c2)| b.iter(|| distance_matrix_flat(black_box(*c1), black_box(*c2))),
        );

        group.bench_with_input(
            BenchmarkId::new("nested_vec", size),
            &(&coords1, &coords2),
            |b, (c1, c2)| {
                b.iter(|| {
                    let result: Vec<Vec<f64>> = c1
                        .iter()
                        .map(|p1| {
                            c2.iter()
                                .map(|p2| {
                                    let dx = p1.x - p2.x;
                                    let dy = p1.y - p2.y;
                                    let dz = p1.z - p2.z;
                                    (dx * dx + dy * dy + dz * dz).sqrt()
                                })
                                .collect()
                        })
                        .collect();
                    result
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Scoring and Alignment Benchmarks
// =============================================================================

fn bench_score_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("score_matrix");
    let scorer = RossmannArgos::default_params();

    for size in [50, 100, 200].iter() {
        let domain1 = make_test_domain(*size);
        let domain2 = make_test_domain(*size);

        group.bench_with_input(
            BenchmarkId::new("rossmann_argos", size),
            &(&domain1, &domain2),
            |b, (d1, d2)| b.iter(|| scorer.score_matrix(black_box(*d1), black_box(*d2))),
        );
    }

    group.finish();
}

fn bench_smith_waterman(c: &mut Criterion) {
    let mut group = c.benchmark_group("smith_waterman");
    let params = Parameters::default();

    for size in [50, 100, 200].iter() {
        let scores: Vec<Vec<f64>> = (0..*size)
            .map(|i| {
                (0..*size)
                    .map(|j| {
                        if i == j {
                            5.0
                        } else {
                            -1.0 + 0.1 * (i as f64 - j as f64).abs()
                        }
                    })
                    .collect()
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("fill_traceback", size),
            &scores,
            |b, scores| {
                b.iter(|| {
                    let mut sw = SmithWaterman::new(*size, *size, &params);
                    sw.fill(black_box(scores));
                    sw.traceback()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Probability Matrix Benchmarks
// =============================================================================

fn bench_probability_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("probability_matrix");
    let params = RossmannParams::default();

    for size in [50, 100, 200].iter() {
        let coords1 = make_test_coords(*size);
        let coords2 = make_test_coords(*size);

        group.bench_with_input(
            BenchmarkId::new("sequential", size),
            &(&coords1, &coords2),
            |b, (c1, c2)| {
                b.iter(|| calculate_probability_matrix(black_box(*c1), black_box(*c2), &params))
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matrix_creation,
    bench_matrix_access,
    bench_symmetric_matrix,
    bench_coords_conversion,
    bench_centroid,
    bench_rmsd,
    bench_kabsch,
    bench_superpose,
    bench_distance_matrix,
    bench_score_matrix,
    bench_smith_waterman,
    bench_probability_matrix,
);
criterion_main!(benches);
