use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// Field type selection based on backend and curve
#[cfg(all(feature = "arkworks", feature = "bls12_381"))]
use ark_bls12_381::Fr as F;
#[cfg(all(feature = "arkworks", feature = "bn254"))]
use ark_bn254::Fr as F;

#[cfg(all(feature = "icicle", feature = "bls12_381"))]
use icicle_bls12_381::curve::ScalarField as F;
#[cfg(all(feature = "icicle", feature = "bn254"))]
use icicle_bn254::curve::ScalarField as F;
#[cfg(all(feature = "icicle", feature = "goldilocks"))]
use icicle_goldilocks::field::ScalarField as F;

use zktransformer::crypto::{LinearSumcheckProver, SumcheckProver};
use zktransformer::util::poly::{CryptoField, DenseMLPoly};
use zktransformer::util::transcript::Transcript;

fn generate_random_polynomials(num_vars: usize, num_polys: usize) -> Vec<DenseMLPoly<F>> {
  let domain_size = 1 << num_vars; // 2^num_vars
  let mut polynomials = Vec::new();

  for i in 0..num_polys {
    let mut poly = Vec::with_capacity(domain_size);
    for j in 0..domain_size {
      // Generate deterministic but varied values for reproducible benchmarks
      poly.push(<F as CryptoField>::from_u32((i * domain_size + j + 1) as u32));
    }
    polynomials.push(DenseMLPoly::from_evaluations(poly));
  }

  polynomials
}

fn bench_sumcheck_prover(c: &mut Criterion) {
  let mut group = c.benchmark_group("sumcheck_prover");

  // Test with 2 polynomials and varying number of variables
  let num_polys = 2;
  let variable_counts = vec![2, 3, 4, 5, 6, 7, 8];

  for num_vars in variable_counts {
    let polynomials = generate_random_polynomials(num_vars, num_polys);

    group.bench_with_input(BenchmarkId::new("variables", num_vars), &num_vars, |b, &num_vars| {
      b.iter(|| {
        let mut transcript = Transcript::new(b"sumcheck_bench");
        let mut prover = LinearSumcheckProver::new(black_box(num_vars), black_box(polynomials.len()), &mut transcript);
        let _ = prover.prove(black_box(&polynomials), black_box(&mut transcript));
      });
    });
  }

  group.finish();
}

criterion_group!(benches, bench_sumcheck_prover);
criterion_main!(benches);
