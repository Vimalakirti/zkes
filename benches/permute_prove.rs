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

use zktransformer::{
  basicblock::{BasicBlock, Permute},
  dag::{Claim, DataType, Role, Witness},
  util::poly::CryptoField,
  util::transcript::Transcript,
};

fn create_permute_test_data(size: usize) -> (Witness<F>, Permute, Vec<Claim<F>>) {
  // Create a square matrix input of size x size
  let input_data: Vec<F> = (0..size * size).map(|i| <F as CryptoField>::from_u32(i as u32 + 1)).collect();

  let input_witness = Witness::new(vec![size, size], input_data, DataType::Float, 1, Role::Input);

  // Create a transpose permutation (i,j) -> (j,i)
  let mut permutation = Vec::new();
  for i in 0..size {
    for j in 0..size {
      permutation.push((vec![i, j], vec![j, i]));
    }
  }

  let permute_op = Permute {
    permutation,
    output_shape: vec![size, size],
  };

  // Run the operation to get outputs
  let outputs = permute_op.run(&[&input_witness]);
  let output_witness = &outputs[0];
  //let aux_witness = &outputs[1];

  // Create a dummy claim for the output
  let point: Vec<F> = (0..output_witness.data.as_ref().unwrap().n()).map(|_| <F as CryptoField>::from_u32(1u32)).collect();
  let eval = output_witness.data.as_ref().unwrap().evaluate_at_point(&point);
  let claim = Claim {
    edge_id: 1, // dummy edge id
    point,
    eval,
  };

  (input_witness, permute_op, vec![claim])
}

fn bench_permute_prove(c: &mut Criterion) {
  let mut group = c.benchmark_group("permute_prove");

  // Test sizes: 2, 4, 8, 16, 32, 64, 128
  let sizes: Vec<usize> = (1..8).map(|i| 1 << i).collect(); // 2^1 to 2^7 (up to 128)

  for size in sizes {
    let (input_witness, permute_op, claims) = create_permute_test_data(size);
    let witnesses = vec![&input_witness];
    let edge_ids = vec![0, 1]; // input and output edge ids
    let out_claims: Vec<&Claim<F>> = claims.iter().collect();

    group.bench_with_input(BenchmarkId::new("matrix_transpose", format!("{}x{}", size, size)), &size, |b, _| {
      b.iter(|| {
        let mut transcript = Transcript::new(b"permute_bench");
        let result = permute_op.prove(
          black_box(&witnesses),
          black_box(&edge_ids),
          black_box(&out_claims),
          black_box(&mut transcript),
        );
        black_box(result);
      });
    });
  }

  group.finish();
}

fn bench_permute_prove_large(c: &mut Criterion) {
  let mut group = c.benchmark_group("permute_prove_large");
  group.sample_size(10); // Reduce sample size for large benchmarks

  // Test larger sizes with fewer samples
  let large_sizes = vec![512, 1024];

  for size in large_sizes {
    let (input_witness, permute_op, claims) = create_permute_test_data(size);
    let witnesses = vec![&input_witness];
    let edge_ids = vec![0, 1];
    let out_claims: Vec<&Claim<F>> = claims.iter().collect();

    group.bench_with_input(BenchmarkId::new("matrix_transpose_large", format!("{}x{}", size, size)), &size, |b, _| {
      b.iter(|| {
        let mut transcript = Transcript::new(b"permute_bench");
        let result = permute_op.prove(
          black_box(&witnesses),
          black_box(&edge_ids),
          black_box(&out_claims),
          black_box(&mut transcript),
        );
        black_box(result);
      });
    });
  }

  group.finish();
}

criterion_group!(benches, bench_permute_prove, bench_permute_prove_large);
criterion_main!(benches);
