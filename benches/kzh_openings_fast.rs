#![allow(non_snake_case)]
#![allow(unused_imports)]

#[cfg(feature = "arkworks")]
use ark_bn254::{Bn254, Fr};
#[cfg(feature = "arkworks")]
use ark_std::rand::{thread_rng, Rng};
#[cfg(feature = "arkworks")]
use ark_std::UniformRand;

#[cfg(feature = "icicle")]
use icicle_bn254::curve::ScalarField as Fr;
#[cfg(feature = "icicle")]
use icicle_core::traits::GenerateRandom;
#[cfg(feature = "icicle")]
use zktransformer::crypto::polycommit::IcicleBn254 as Bn254;

// use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::HashMap;
use std::time::Instant;
use zktransformer::crypto::polycommit::kzh2::{KZH2Commit, KZH2CommitKey, KZH2VerifierKey};
use zktransformer::crypto::polycommit::kzh3::{KZH3Commit, KZH3CommitKey, KZH3VerifierKey};
use zktransformer::crypto::polycommit::sparse_kzh2::{SparseKZH2Commit, SparseKZH2CommitKey, SparseKZH2VerifierKey};
use zktransformer::crypto::polycommit::sparse_kzh3::{SparseKZH3Commit, SparseKZH3CommitKey, SparseKZH3VerifierKey};
use zktransformer::crypto::polycommit::MLPolyCommit;
use zktransformer::util::poly::{usize_to_le_vec, CryptoField, DenseMLPoly, MLPoly, SparseMLPoly};

fn rand_poly(num_variables: usize) -> DenseMLPoly<Fr> {
  #[cfg(feature = "arkworks")]
  {
    let mut rng = thread_rng();
    let evaluations: Vec<Fr> = (0..(1 << num_variables)).map(|_| Fr::rand(&mut rng)).collect();
    DenseMLPoly::new(num_variables, evaluations)
  }
  #[cfg(feature = "icicle")]
  {
    let evaluations: Vec<Fr> = <Fr as GenerateRandom>::generate_random(1 << num_variables);
    DenseMLPoly::new(num_variables, evaluations)
  }
}

/// Create a sparse polynomial with sqrt(2^n) non-zero values for size 2^n
fn create_sparse_poly(num_variables: usize) -> SparseMLPoly<Fr> {
  let total_len = 1 << num_variables;
  let num_non_zero = (1 << (num_variables / 2)) as usize; // sqrt(2^n) non-zero values for polynomial of size 2^n

  let mut evaluations = HashMap::new();
  let mut indices = Vec::new();

  #[cfg(feature = "arkworks")]
  {
    let mut rng = thread_rng();
    let mut selected_indices = std::collections::HashSet::new();

    // Select n random indices
    while selected_indices.len() < num_non_zero {
      let idx = rng.gen_range(0..total_len);
      selected_indices.insert(idx);
    }

    // Assign random values to selected indices
    for idx in selected_indices {
      let idx_bytes = usize_to_le_vec(idx);
      let val = Fr::rand(&mut rng);
      evaluations.insert(idx_bytes.clone(), val);
      indices.push(idx_bytes);
    }
  }

  #[cfg(feature = "icicle")]
  {
    let random_scalars = <Fr as GenerateRandom>::generate_random(num_non_zero * 2);
    let mut selected_indices = std::collections::HashSet::new();

    // Select n random indices using random scalars
    let mut i = 0;
    while selected_indices.len() < num_non_zero && i < random_scalars.len() {
      let bytes = <Fr as CryptoField>::to_bytes_le(&random_scalars[i]);
      let mut idx_bytes = [0u8; 8];
      let copy_len = bytes.len().min(8);
      idx_bytes[..copy_len].copy_from_slice(&bytes[..copy_len]);
      let idx = (u64::from_le_bytes(idx_bytes) as usize) % total_len;
      selected_indices.insert(idx);
      i += 1;
    }

    // Assign random values to selected indices
    let random_values = <Fr as GenerateRandom>::generate_random(num_non_zero);
    for (i, idx) in selected_indices.into_iter().enumerate() {
      let idx_bytes = usize_to_le_vec(idx);
      evaluations.insert(idx_bytes.clone(), random_values[i]);
      indices.push(idx_bytes);
    }
  }

  // Sort indices for consistent ordering
  indices.sort();

  SparseMLPoly::new(num_variables, evaluations, indices.into())
}

fn bench_kzh2_openings() {
  println!("\n=== KZH2 Opening Benchmarks ===");
  let num_variables = vec![8, 10, 12, 15];

  for n in num_variables {
    println!("\nBenchmarking KZH2 opening for n={}", n);

    // Setup with real cryptographic parameters
    println!("  Setting up SRS (this may take a while)...");
    let kzh2: KZH2CommitKey<Bn254> = KZH2Commit::setup(n, 0, 0, 0);

    // Random polynomial
    let polynomial = rand_poly(n);

    // Commit to the polynomial
    let commitment = KZH2Commit::commit(&polynomial, &kzh2);

    // Random opening point
    #[cfg(feature = "arkworks")]
    let input: Vec<_> = {
      let mut rng = thread_rng();
      (0..n).map(|_| Fr::rand(&mut rng)).collect()
    };
    #[cfg(feature = "icicle")]
    let input: Vec<_> = <Fr as GenerateRandom>::generate_random(n);

    // Opening
    let start = Instant::now();
    let _opening = KZH2Commit::open(&commitment, &polynomial, &kzh2, &input);
    let duration = start.elapsed();
    println!("  Opening time: {:?}", duration);
  }
}

fn bench_sparse_kzh2_openings() {
  println!("\n=== Sparse KZH2 Opening Benchmarks (n non-zero values for size 2^n) ===");
  let num_variables = vec![8, 10, 12, 15];

  for n in num_variables {
    println!("\nBenchmarking Sparse KZH2 opening for n={} (2^{} non-zero values)", n, n / 2);

    // Setup with real cryptographic parameters
    println!("  Setting up SRS (this may take a while)...");
    let kzh2: SparseKZH2CommitKey<Bn254> = SparseKZH2Commit::setup(n, 0, 0, 0);

    // Sparse polynomial with 2^(n/2) non-zero
    let polynomial = create_sparse_poly(n);

    // Commit to the polynomial
    let commitment = SparseKZH2Commit::commit(&polynomial, &kzh2);

    // Random opening point
    #[cfg(feature = "arkworks")]
    let input: Vec<_> = {
      let mut rng = thread_rng();
      (0..n).map(|_| Fr::rand(&mut rng)).collect()
    };
    #[cfg(feature = "icicle")]
    let input: Vec<_> = <Fr as GenerateRandom>::generate_random(n);

    // Opening
    let start = Instant::now();
    let _opening = SparseKZH2Commit::open(&commitment, &polynomial, &kzh2, &input);
    let duration = start.elapsed();
    println!("  Opening time: {:?}", duration);
  }
}

fn bench_kzh3_openings() {
  println!("\n=== KZH3 Opening Benchmarks ===");
  let num_variables = vec![8, 10, 12, 15];

  for n in num_variables {
    println!("\nBenchmarking KZH3 opening for n={}", n);

    // Setup with real cryptographic parameters
    println!("  Setting up SRS (this may take a while)...");
    let kzh3: KZH3CommitKey<Bn254> = KZH3Commit::setup(n, 0, 0, 0);

    // Random polynomial
    let polynomial = rand_poly(n);

    // Commit to the polynomial
    let commitment = KZH3Commit::commit(&polynomial, &kzh3);

    // Random opening point
    #[cfg(feature = "arkworks")]
    let input: Vec<_> = {
      let mut rng = thread_rng();
      (0..n).map(|_| Fr::rand(&mut rng)).collect()
    };
    #[cfg(feature = "icicle")]
    let input: Vec<_> = <Fr as GenerateRandom>::generate_random(n);

    // Opening
    let start = Instant::now();
    let _opening = KZH3Commit::open(&commitment, &polynomial, &kzh3, &input);
    let duration = start.elapsed();
    println!("  Opening time: {:?}", duration);
  }
}

fn bench_sparse_kzh3_openings() {
  println!("\n=== Sparse KZH3 Opening Benchmarks (n non-zero values for size 2^n) ===");
  let num_variables = vec![8, 10, 12, 15];

  for n in num_variables {
    println!("\nBenchmarking Sparse KZH3 opening for n={} (2^{} non-zero values)", n, n / 2);

    // Setup with real cryptographic parameters
    println!("  Setting up SRS (this may take a while)...");
    let kzh3: SparseKZH3CommitKey<Bn254> = SparseKZH3Commit::setup(n, 0, 0, 0);

    // Sparse polynomial with 2^(n/2) non-zero
    let polynomial = create_sparse_poly(n);

    // Commit to the polynomial
    let commitment = SparseKZH3Commit::commit(&polynomial, &kzh3);

    // Random opening point
    #[cfg(feature = "arkworks")]
    let input: Vec<_> = {
      let mut rng = thread_rng();
      (0..n).map(|_| Fr::rand(&mut rng)).collect()
    };
    #[cfg(feature = "icicle")]
    let input: Vec<_> = <Fr as GenerateRandom>::generate_random(n);

    // Opening
    let start = Instant::now();
    let _opening = SparseKZH3Commit::open(&commitment, &polynomial, &kzh3, &input);
    let duration = start.elapsed();
    println!("  Opening time: {:?}", duration);
  }
}

fn main() {
  println!("Running KZH Opening Benchmarks with Real Cryptographic Setup");
  println!("=============================================================");
  println!("Note: Setup phase uses real SRS generation and may take time.");
  println!("The opening time is what we're benchmarking.\n");

  bench_kzh2_openings();
  bench_sparse_kzh2_openings();
  bench_kzh3_openings();
  bench_sparse_kzh3_openings();

  println!("\n\nBenchmarks completed!");
}
