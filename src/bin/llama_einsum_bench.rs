//! Targeted benchmark measuring how much of `Einsum::prove` time is spent
//! inside `permute_evals_by_ranges`, using the einsum equations and witness
//! shapes from LLaMA-2-7B (pos=0, seq_len=1).
//!
//! This avoids running the full DAG (which requires KZH3 SRS setup for every
//! polynomial size in the model). Instead we exercise `Einsum::prove`
//! directly with synthetic witnesses / claims matching each unique einsum
//! call-site, multiplied by the number of times it is invoked per LLaMA-2-7B
//! forward pass.

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
use std::time::Instant;
use zk_torch_2::basicblock::einsum::{
  einsum_input_permutes, einsum_output_shape, permute_evals_by_ranges, report_einsum_timers, reset_einsum_timers, Einsum, EINSUM_PERMUTE_NS, EINSUM_PROVE_NS,
};
use zk_torch_2::basicblock::BasicBlock;
use zk_torch_2::dag::{Claim, DataType, Role, Witness};
use zk_torch_2::util::arith::get_n;
use zk_torch_2::util::poly::{CryptoField, DenseMLPoly};
use zk_torch_2::util::transcript::Transcript;
use zk_torch_2::SF_LOG;

fn rand_witness(shape: Vec<usize>) -> Witness<F> {
  let n_flat: usize = shape.iter().map(|s| s.next_power_of_two()).product();
  let mut rng = rand::thread_rng();
  let data: Vec<F> = (0..n_flat).map(|_| <F as CryptoField>::from_u32(rng.gen::<u32>() % 1000)).collect();
  Witness::new(shape, data, DataType::Float, *SF_LOG as usize, Role::Constant)
}

fn rand_point(n: usize) -> Vec<F> {
  let mut rng = rand::thread_rng();
  (0..n).map(|_| <F as CryptoField>::from_u32(rng.gen::<u32>() % 1000)).collect()
}

/// Build the (inputs + output) witness set and the output claim used to drive
/// `Einsum::prove`. Returns owned storage + claim so callers can either run the
/// baseline path or mutate witnesses to the pre-permuted path.
fn build_inputs(equation: &str, input_shapes: &[Vec<usize>]) -> (Vec<Witness<F>>, Claim<F>, Vec<usize>) {
  let out_shape = einsum_output_shape(equation, input_shapes).unwrap();
  let out_n = get_n(&out_shape);

  let mut witnesses_owned: Vec<Witness<F>> = input_shapes.iter().map(|s| rand_witness(s.clone())).collect();
  witnesses_owned.push(rand_witness(out_shape.clone()));

  let claim = Claim {
    edge_id: 0,
    sparse_id: 0,
    point: rand_point(out_n),
    eval: <F as CryptoField>::from_u32(0u32),
  };
  let edge_ids: Vec<usize> = (0..witnesses_owned.len()).collect();
  (witnesses_owned, claim, edge_ids)
}

/// Pre-permute input witnesses in-place and tag them with `is_permuted_with`.
/// This mirrors what `Dag::apply_prepermute` does for eligible edges.
fn apply_prepermute_to_inputs(equation: &str, input_shapes: &[Vec<usize>], witnesses: &mut [Witness<F>]) {
  let permutes = einsum_input_permutes(equation, input_shapes);
  for i in 0..input_shapes.len() {
    let perm = &permutes[i];
    if perm.is_empty() {
      continue;
    }
    let n = get_n(&witnesses[i].shape);
    // Permute the dense F-evaluations.
    {
      let poly = witnesses[i].data.as_ref().unwrap();
      let dense = poly.as_any().downcast_ref::<DenseMLPoly<F>>().expect("Expected DenseMLPoly");
      let new_evals = permute_evals_by_ranges(&dense.evaluations, n, perm);
      witnesses[i].data = Some(Box::new(DenseMLPoly::new(n, new_evals)) as Box<dyn zk_torch_2::util::poly::MLPoly<F>>);
    }
    // Permute the i128 shadow if present (used for n >= 18 path).
    if let Some(data_int) = witnesses[i].data_int.as_ref() {
      let new_int = permute_evals_by_ranges(data_int, n, perm);
      witnesses[i].data_int = Some(new_int);
    }
    witnesses[i].is_permuted_with = Some(perm.clone());
  }
}

/// Run `Einsum::prove` once for a given (equation, input-shapes) pair.
/// Returns (total_ns_for_this_call, permute_ns_accumulated_delta, claim_evals).
fn prove_once(equation: &str, input_shapes: &[Vec<usize>], pre_permute: bool) -> (u128, u64, Vec<F>) {
  let eq = Einsum { equation: equation.to_string() };
  let (mut witnesses_owned, claim, edge_ids) = build_inputs(equation, input_shapes);

  if pre_permute {
    apply_prepermute_to_inputs(equation, input_shapes, &mut witnesses_owned);
  }

  let witness_refs: Vec<&Witness<F>> = witnesses_owned.iter().collect();

  let permute_before = EINSUM_PERMUTE_NS.load(std::sync::atomic::Ordering::Relaxed);
  let mut ts = Transcript::<F>::new(b"bench");
  let t0 = Instant::now();
  let (_proofs, claims) = eq.prove(&witness_refs, &edge_ids, &[&claim], &mut ts);
  let elapsed = t0.elapsed().as_nanos();
  let permute_after = EINSUM_PERMUTE_NS.load(std::sync::atomic::Ordering::Relaxed);
  let evals: Vec<F> = claims.iter().map(|c| c.eval).collect();
  (elapsed, permute_after - permute_before, evals)
}

#[derive(Debug)]
struct OpSpec {
  tag: &'static str,
  equation: &'static str,
  input_shapes: Vec<Vec<usize>>,
  per_layer: usize,
}

fn llama_2_7b_einsum_specs() -> Vec<(OpSpec, usize)> {
  // (spec, total_count). Shapes correspond to pos=0 (seq_len=1) llama-2-7B.
  // The weight witness sizes are the ones that dominate `permute_evals_by_ranges`.

  // From src/dag/llama.rs:
  //   Q/K/V/O projection: bsi,ij->bsj with x(1,1,4096) × W(4096,4096)
  //   MLP proj_1/proj_2: bsi,ij->bsj with x(1,1,4096) × W(4096,11008)
  //   MLP proj_3:        bsi,ij->bsj with x(1,1,11008) × W(11008,4096)
  //   Attention: bshd->bhsd permute-only; scores bhsd,bhtd->bhst etc (small)
  //   Logits: bij,jk->ik with x(1,1,4096) × W(4096,32000)  (once, not per-layer)

  let num_layers = 32usize;

  let attn_qkvo = OpSpec {
    tag: "attn_qkvo (bsi,ij->bsj, W=(4096,4096))",
    equation: "bsi,ij->bsj",
    input_shapes: vec![vec![1, 1, 4096], vec![4096, 4096]],
    per_layer: 4,
  };
  let mlp_proj12 = OpSpec {
    tag: "mlp_proj12 (bsi,ij->bsj, W=(4096,11008))",
    equation: "bsi,ij->bsj",
    input_shapes: vec![vec![1, 1, 4096], vec![4096, 11008]],
    per_layer: 2,
  };
  let mlp_proj3 = OpSpec {
    tag: "mlp_proj3 (bsi,ij->bsj, W=(11008,4096))",
    equation: "bsi,ij->bsj",
    input_shapes: vec![vec![1, 1, 11008], vec![11008, 4096]],
    per_layer: 1,
  };
  // Non-weight-backed einsums (smaller witnesses) — included for completeness.
  let rmsnorm_sq = OpSpec {
    tag: "rmsnorm sq (bsi,bsi->bsi)",
    equation: "bsi,bsi->bsi",
    input_shapes: vec![vec![1, 1, 4096], vec![1, 1, 4096]],
    per_layer: 2, // x_sq in attn_norm and proj_norm
  };
  let rmsnorm_mul = OpSpec {
    tag: "rmsnorm mul (bsi,i->bsi)",
    equation: "bsi,i->bsi",
    input_shapes: vec![vec![1, 1, 4096], vec![4096]],
    per_layer: 2,
  };
  let qkv_perm = OpSpec {
    tag: "qkv transpose (bshd->bhsd)",
    equation: "bshd->bhsd",
    input_shapes: vec![vec![1, 1, 32, 128]],
    per_layer: 3,
  };
  let logits = OpSpec {
    tag: "logits (bij,jk->ik, W=(4096,32000))",
    equation: "bij,jk->ik",
    input_shapes: vec![vec![1, 1, 4096], vec![4096, 32000]],
    per_layer: 0, // counted once globally
  };

  let mut specs = vec![
    (attn_qkvo, num_layers * 4),
    (mlp_proj12, num_layers * 2),
    (mlp_proj3, num_layers * 1),
    (rmsnorm_sq, num_layers * 2),
    (rmsnorm_mul, num_layers * 2),
    (qkv_perm, num_layers * 3),
    (logits, 1),
  ];
  // Replace per_layer field with logical count in tag line; count field above.
  for (s, _) in specs.iter_mut() {
    let _ = s.per_layer;
  }
  specs
}

/// Run both the baseline path and the pre-permute path on the SAME random
/// witnesses + same claim, and assert the resulting claim evals match.
/// Returns (baseline_prove_ns, pre_prove_ns, baseline_permute_ns, pre_permute_ns).
fn prove_both_and_validate(
  equation: &str,
  input_shapes: &[Vec<usize>],
) -> (u64, u64, u64, u64) {
  let eq = Einsum { equation: equation.to_string() };
  let (witnesses_base, claim, edge_ids) = build_inputs(equation, input_shapes);

  // Clone witnesses for the pre-permute path, then permute them.
  let mut witnesses_pre: Vec<Witness<F>> = witnesses_base.iter().cloned().collect();
  apply_prepermute_to_inputs(equation, input_shapes, &mut witnesses_pre);

  // Baseline.
  let wrefs_base: Vec<&Witness<F>> = witnesses_base.iter().collect();
  let prove_before = EINSUM_PROVE_NS.load(std::sync::atomic::Ordering::Relaxed);
  let permute_before = EINSUM_PERMUTE_NS.load(std::sync::atomic::Ordering::Relaxed);
  let mut ts_a = Transcript::<F>::new(b"bench");
  let (_proofs_a, claims_a) = eq.prove(&wrefs_base, &edge_ids, &[&claim], &mut ts_a);
  let base_prove_ns = EINSUM_PROVE_NS.load(std::sync::atomic::Ordering::Relaxed) - prove_before;
  let base_permute_ns = EINSUM_PERMUTE_NS.load(std::sync::atomic::Ordering::Relaxed) - permute_before;

  // Pre-permute path.
  let wrefs_pre: Vec<&Witness<F>> = witnesses_pre.iter().collect();
  let prove_before = EINSUM_PROVE_NS.load(std::sync::atomic::Ordering::Relaxed);
  let permute_before = EINSUM_PERMUTE_NS.load(std::sync::atomic::Ordering::Relaxed);
  let mut ts_b = Transcript::<F>::new(b"bench");
  let (_proofs_b, claims_b) = eq.prove(&wrefs_pre, &edge_ids, &[&claim], &mut ts_b);
  let pre_prove_ns = EINSUM_PROVE_NS.load(std::sync::atomic::Ordering::Relaxed) - prove_before;
  let pre_permute_ns = EINSUM_PERMUTE_NS.load(std::sync::atomic::Ordering::Relaxed) - permute_before;

  // Correctness: sumcheck poly construction is mathematically identical in
  // both paths, so the sumcheck challenges and claim evals must match exactly.
  assert_eq!(claims_a.len(), claims_b.len(), "claim count differs");
  for i in 0..claims_a.len() {
    assert!(
      claims_a[i].eval == claims_b[i].eval,
      "eval mismatch for input {i} on equation '{equation}' shapes {:?}",
      input_shapes
    );
  }

  (base_prove_ns, pre_prove_ns, base_permute_ns, pre_permute_ns)
}

fn main() {
  env_logger::init();
  let specs = llama_2_7b_einsum_specs();

  // Warm up allocators / parallel pools on a small case first.
  let _ = prove_once("bsi,ij->bsj", &[vec![1, 1, 64], vec![64, 64]], false);
  reset_einsum_timers();

  println!("Benchmarking einsum prove on LLaMA-2-7B (pos=0, seq_len=1) einsum call-sites");
  println!("Feature flag F = (see cfg)");
  println!();
  println!(
    "{:<48} {:>6} {:>12} {:>12} {:>12} {:>10}",
    "op", "count", "base(ms)", "pre(ms)", "permute(ms)", "speedup"
  );

  let mut grand_base_ns: u64 = 0;
  let mut grand_pre_ns: u64 = 0;
  let mut grand_permute_ns: u64 = 0;
  let mut rows: Vec<(String, usize, f64, f64, f64, f64, f64, f64)> = Vec::new();

  for (spec, count) in &specs {
    if *count == 0 {
      continue;
    }
    let (prove_ns_call, pre_prove_ns_call, permute_ns_call, pre_permute_ns_call) =
      prove_both_and_validate(spec.equation, &spec.input_shapes);
    let _ = pre_permute_ns_call; // should be 0 on the pre-permute path

    // Project to full LLaMA-2-7B forward: count * per-op time.
    let total_base_ns = prove_ns_call as u128 * *count as u128;
    let total_pre_ns = pre_prove_ns_call as u128 * *count as u128;
    let total_permute_ns = permute_ns_call as u128 * *count as u128;
    grand_base_ns = grand_base_ns.saturating_add(total_base_ns as u64);
    grand_pre_ns = grand_pre_ns.saturating_add(total_pre_ns as u64);
    grand_permute_ns = grand_permute_ns.saturating_add(total_permute_ns as u64);

    let base_ms = prove_ns_call as f64 / 1e6;
    let pre_ms = pre_prove_ns_call as f64 / 1e6;
    let permute_ms = permute_ns_call as f64 / 1e6;
    let speedup = if pre_prove_ns_call > 0 {
      prove_ns_call as f64 / pre_prove_ns_call as f64
    } else {
      f64::INFINITY
    };
    let total_base_s = total_base_ns as f64 / 1e9;
    let total_pre_s = total_pre_ns as f64 / 1e9;
    let total_permute_s = total_permute_ns as f64 / 1e9;
    rows.push((
      spec.tag.to_string(),
      *count,
      base_ms,
      pre_ms,
      permute_ms,
      speedup,
      total_base_s,
      total_pre_s,
    ));
    let _ = total_permute_s;

    println!(
      "{:<48} {:>6} {:>12.2} {:>12.2} {:>12.2} {:>9.2}x",
      spec.tag, count, base_ms, pre_ms, permute_ms, speedup
    );
  }

  println!();
  println!("Projected totals over one full LLaMA-2-7B forward pass:");
  println!(
    "{:<48} {:>12} {:>12} {:>10}",
    "op", "base(s)", "pre(s)", "speedup"
  );
  for (tag, count, _b_ms, _p_ms, _pe_ms, _s, t_base_s, t_pre_s) in &rows {
    let speedup = if *t_pre_s > 0.0 { t_base_s / t_pre_s } else { f64::INFINITY };
    let _ = count;
    println!("{:<48} {:>12.3} {:>12.3} {:>9.2}x", tag, t_base_s, t_pre_s, speedup);
  }

  let grand_prove_s = grand_base_ns as f64 / 1e9;
  let grand_pre_s = grand_pre_ns as f64 / 1e9;
  let grand_permute_s = grand_permute_ns as f64 / 1e9;
  let grand_pct = if grand_base_ns > 0 {
    100.0 * grand_permute_ns as f64 / grand_base_ns as f64
  } else {
    0.0
  };
  let grand_speedup = if grand_pre_ns > 0 { grand_base_ns as f64 / grand_pre_ns as f64 } else { f64::INFINITY };
  println!();
  println!(
    "GRAND TOTAL (baseline): einsum prove = {:.3}s, permute = {:.3}s ({:.2}%)",
    grand_prove_s, grand_permute_s, grand_pct
  );
  println!(
    "GRAND TOTAL (pre-permute): einsum prove = {:.3}s, speedup = {:.2}x",
    grand_pre_s, grand_speedup
  );

  report_einsum_timers();
}
