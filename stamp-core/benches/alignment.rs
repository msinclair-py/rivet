//! Benchmarks for alignment operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stamp_core::alignment::SmithWaterman;
use stamp_core::math::{kabsch, superpose};
use stamp_core::scoring::RossmannArgos;
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

fn bench_kabsch(c: &mut Criterion) {
    let points: Vec<Coord3> = (0..100)
        .map(|i| Coord3::new(i as f64, (i as f64).sin(), (i as f64).cos()))
        .collect();

    c.bench_function("kabsch_100", |b| {
        b.iter(|| kabsch(black_box(points.iter()), black_box(points.iter())))
    });
}

fn bench_superpose(c: &mut Criterion) {
    let points1: Vec<Coord3> = (0..100)
        .map(|i| Coord3::new(i as f64, (i as f64).sin(), (i as f64).cos()))
        .collect();
    let points2: Vec<Coord3> = points1
        .iter()
        .map(|p| Coord3::new(p.x + 10.0, p.y + 5.0, p.z))
        .collect();

    c.bench_function("superpose_100", |b| {
        b.iter(|| superpose(black_box(&points1), black_box(&points2)))
    });
}

fn bench_score_matrix(c: &mut Criterion) {
    let domain1 = make_test_domain(50);
    let domain2 = make_test_domain(50);
    let scorer = RossmannArgos::default_params();

    c.bench_function("score_matrix_50x50", |b| {
        b.iter(|| scorer.score_matrix(black_box(&domain1), black_box(&domain2)))
    });
}

fn bench_smith_waterman(c: &mut Criterion) {
    let params = Parameters::default();
    let scores: Vec<Vec<f64>> = (0..50)
        .map(|i| {
            (0..50)
                .map(|j| if i == j { 5.0 } else { -1.0 + 0.1 * (i as f64 - j as f64).abs() })
                .collect()
        })
        .collect();

    c.bench_function("smith_waterman_50x50", |b| {
        b.iter(|| {
            let mut sw = SmithWaterman::new(50, 50, &params);
            sw.fill(black_box(&scores));
            sw.traceback()
        })
    });
}

criterion_group!(
    benches,
    bench_kabsch,
    bench_superpose,
    bench_score_matrix,
    bench_smith_waterman
);
criterion_main!(benches);
