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

use rand::Rng;
use zktransformer::util::poly::{fix_variables_zkgpt, DenseMLPoly};

fn main() {
  println!("Comparing fix_variables vs fix_variables_zkgpt\n");

  // Test parameters
  let num_vars = 20; // Total number of variables (log of total size)
  let r_len = 4; // Number of variables to fix (must be <= num_vars)
  let q_bits = 15; // Quantization bits

  let n = 1usize << r_len; // 2^r_len rows
  let m = 1usize << (num_vars - r_len); // 2^(num_vars - r_len) columns
  let total_size = 1usize << num_vars;

  println!("Parameters:");
  println!("  num_vars = {}", num_vars);
  println!("  r_len (vars to fix) = {}", r_len);
  println!("  q_bits = {}", q_bits);
  println!("  n (rows) = {}", n);
  println!("  m (cols) = {}", m);
  println!("  total_size = {}\n", total_size);

  let mut rng = rand::thread_rng();

  // Generate random integer indices in [-2^q_bits, 2^q_bits]
  let max_val = 1i128 << q_bits;
  let t_idx: Vec<i128> = (0..total_size).map(|_| rng.gen_range(-max_val..=max_val)).collect();

  // Convert to field elements for the standard fix_variables
  let field_evals: Vec<F> = t_idx.iter().map(|&x| F::from(x)).collect();

  // Generate random challenge point
  let r: Vec<F> = (0..r_len).map(|_| F::from(rng.gen::<i64>())).collect();

  println!("Running fix_variables (standard)...");
  let start = std::time::Instant::now();
  let poly = DenseMLPoly::new(num_vars, field_evals);
  let result_standard = poly.fix_variables(&r);
  let standard_time = start.elapsed();
  println!("  Time: {:?}", standard_time);

  println!("\nRunning fix_variables_zkgpt (optimized)...");
  let start = std::time::Instant::now();
  let result_zkgpt = fix_variables_zkgpt::<F>(num_vars, &t_idx, &r, q_bits + 3);
  let zkgpt_time = start.elapsed();
  println!("  Time: {:?}", zkgpt_time);

  // Compare results
  println!("\nComparing results...");
  let standard_evals = &result_standard.evaluations;
  let zkgpt_evals = &result_zkgpt.evaluations;

  assert_eq!(
    standard_evals.len(),
    zkgpt_evals.len(),
    "Output sizes differ: {} vs {}",
    standard_evals.len(),
    zkgpt_evals.len()
  );

  let mut diff_count = 0;
  for i in 0..standard_evals.len() {
    if standard_evals[i] != zkgpt_evals[i] {
      diff_count += 1;
      if diff_count <= 5 {
        println!(
          "  Difference at index {}: standard = {:?}, zkgpt = {:?}",
          i, standard_evals[i], zkgpt_evals[i]
        );
      }
    }
  }

  if diff_count == 0 {
    println!("  All {} evaluations match!", standard_evals.len());
    println!("\n✓ SUCCESS: Both implementations produce identical results");
  } else {
    println!("\n  Total differences: {} / {}", diff_count, standard_evals.len());
    println!("\n✗ FAILURE: Results differ between implementations");
  }

  println!("\nPerformance comparison:");
  println!("  Standard:  {:?}", standard_time);
  println!("  ZKGPT:     {:?}", zkgpt_time);
  if zkgpt_time < standard_time {
    let speedup = standard_time.as_secs_f64() / zkgpt_time.as_secs_f64();
    println!("  Speedup:   {:.2}x faster", speedup);
  } else {
    let slowdown = zkgpt_time.as_secs_f64() / standard_time.as_secs_f64();
    println!("  Slowdown:  {:.2}x slower", slowdown);
  }
}
