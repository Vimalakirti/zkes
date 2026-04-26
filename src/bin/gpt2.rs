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

use plonky2::{timed, util::timing::TimingTree};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use zk_torch_2::util::transcript::Transcript;
use zk_torch_2::{
  crypto::polycommit::kzh3::{setup_kzh3_srs, KZH3Commit, KZH3CommitKey, KZH3Commitment, KZH3VerifierKey},
  crypto::polycommit::sparse_kzh3::{SparseKZH3Commit, SparseKZH3CommitKey, SparseKZH3VerifierKey},
  crypto::polycommit::{MLPolyCommit, NaiveMLPolyCommit},
  crypto::srs_storage::{load_kzh3_srs, store_kzh3_srs},
  dag::{
    extract_lookup_proof_only, extract_opening_proofs_only, extract_sumcheck_proofs_only, gpt2::gpt_2_small, DagBuilder, DataType, Role, Witness,
  },
  util::poly::{CryptoField, DenseMLPoly, MLPoly, SparseMLPoly},
  util::serialization::measure_total_proof_size,
  SF_LOG, SF_INT, TABLE_SIZE_LOG,
};

#[cfg(all(feature = "arkworks", feature = "bn254"))]
use zk_torch_2::crypto::polycommit::ArkBn254 as PairingType;
#[cfg(all(feature = "icicle", feature = "bn254"))]
use zk_torch_2::crypto::polycommit::IcicleBn254 as PairingType;

/// Signed random field elements in [-half_range, half_range].
/// Zero-mean values prevent activation blowup through layers.
fn gen_signed(rng: &mut StdRng, size: usize, half_range: u32) -> Vec<F> {
  let range = 2 * half_range + 1;
  (0..size).map(|_| {
    let v = (rng.gen::<u32>() % range) as i64 - half_range as i64;
    if v >= 0 {
      <F as CryptoField>::from_u32(v as u32)
    } else {
      <F as CryptoField>::zero() - <F as CryptoField>::from_u32((-v) as u32)
    }
  }).collect()
}

/// Norm γ centered around SF_INT (≈1.0 in real space).
fn gen_norm_weight(rng: &mut StdRng, size: usize) -> Vec<F> {
  let sf = *SF_INT;
  (0..size).map(|_| {
    let noise = (rng.gen::<u32>() % 65) as i64 - 32;
    <F as CryptoField>::from_u32((sf as i64 + noise) as u32)
  }).collect()
}

fn generate_gpt2_weights(rng: &mut StdRng, seq_len: usize) -> (
  Vec<Witness<F>>, // attn_norm_w_vec
  Vec<Witness<F>>, // attn_q_w_vec
  Vec<Witness<F>>, // attn_k_w_vec
  Vec<Witness<F>>, // attn_v_w_vec
  Vec<Witness<F>>, // attn_o_w_vec
  Vec<Witness<F>>, // attn_norm_b_vec
  Vec<Witness<F>>, // attn_q_b_vec
  Vec<Witness<F>>, // attn_k_b_vec
  Vec<Witness<F>>, // attn_v_b_vec
  Vec<Witness<F>>, // attn_o_b_vec
  Vec<Witness<F>>, // proj_norm_w_vec
  Vec<Witness<F>>, // proj_1_w_vec
  Vec<Witness<F>>, // proj_2_w_vec
  Vec<Witness<F>>, // proj_norm_b_vec
  Vec<Witness<F>>, // proj_1_b_vec
  Vec<Witness<F>>, // proj_2_b_vec
  Witness<F>,      // layer_norm_w
  Witness<F>,      // layer_norm_b
  Witness<F>,      // attention_mask
) {
  let sf = *SF_LOG as usize;
  let mut attn_norm_w_vec = Vec::new();
  let mut attn_q_w_vec = Vec::new();
  let mut attn_k_w_vec = Vec::new();
  let mut attn_v_w_vec = Vec::new();
  let mut attn_o_w_vec = Vec::new();
  let mut attn_norm_b_vec = Vec::new();
  let mut attn_q_b_vec = Vec::new();
  let mut attn_k_b_vec = Vec::new();
  let mut attn_v_b_vec = Vec::new();
  let mut attn_o_b_vec = Vec::new();
  let mut proj_norm_w_vec = Vec::new();
  let mut proj_1_w_vec = Vec::new();
  let mut proj_2_w_vec = Vec::new();
  let mut proj_norm_b_vec = Vec::new();
  let mut proj_1_b_vec = Vec::new();
  let mut proj_2_b_vec = Vec::new();

  // Xavier-like: matmul std ≈ SF/sqrt(fan_in). 768 inputs → half_range 64;
  // 3072 inputs (proj_2) → half_range 32. Norm γ near SF_INT (≈1.0).
  for _i in 0..12 {
    attn_norm_w_vec.push(Witness::new(vec![768], gen_norm_weight(rng, 1024), DataType::Float, sf, Role::Constant));
    attn_q_w_vec.push(Witness::new(vec![768, 768], gen_signed(rng, 1024 * 1024, 64), DataType::Float, sf, Role::Constant));
    attn_k_w_vec.push(Witness::new(vec![768, 768], gen_signed(rng, 1024 * 1024, 64), DataType::Float, sf, Role::Constant));
    attn_v_w_vec.push(Witness::new(vec![768, 768], gen_signed(rng, 1024 * 1024, 64), DataType::Float, sf, Role::Constant));
    attn_o_w_vec.push(Witness::new(vec![768, 768], gen_signed(rng, 1024 * 1024, 64), DataType::Float, sf, Role::Constant));
    attn_norm_b_vec.push(Witness::new(vec![768], gen_signed(rng, 1024, 64), DataType::Float, sf, Role::Constant));
    attn_q_b_vec.push(Witness::new(vec![768], gen_signed(rng, 1024, 64), DataType::Float, sf, Role::Constant));
    attn_k_b_vec.push(Witness::new(vec![768], gen_signed(rng, 1024, 64), DataType::Float, sf, Role::Constant));
    attn_v_b_vec.push(Witness::new(vec![768], gen_signed(rng, 1024, 64), DataType::Float, sf, Role::Constant));
    attn_o_b_vec.push(Witness::new(vec![768], gen_signed(rng, 1024, 64), DataType::Float, sf, Role::Constant));
    proj_norm_w_vec.push(Witness::new(vec![768], gen_norm_weight(rng, 1024), DataType::Float, sf, Role::Constant));
    proj_1_w_vec.push(Witness::new(vec![768, 3072], gen_signed(rng, 1024 * 4096, 64), DataType::Float, sf, Role::Constant));
    proj_2_w_vec.push(Witness::new(vec![3072, 768], gen_signed(rng, 4096 * 1024, 32), DataType::Float, sf, Role::Constant));
    proj_norm_b_vec.push(Witness::new(vec![768], gen_signed(rng, 1024, 64), DataType::Float, sf, Role::Constant));
    proj_1_b_vec.push(Witness::new(vec![3072], gen_signed(rng, 4096, 64), DataType::Float, sf, Role::Constant));
    proj_2_b_vec.push(Witness::new(vec![768], gen_signed(rng, 1024, 64), DataType::Float, sf, Role::Constant));
  }

  let layer_norm_w = Witness::new(vec![768], gen_norm_weight(rng, 1024), DataType::Float, sf, Role::Constant);
  let layer_norm_b = Witness::new(vec![768], gen_signed(rng, 1024, 64), DataType::Float, sf, Role::Constant);
  let seq_padded = seq_len.next_power_of_two();
  let attention_mask = Witness::new(vec![1, seq_len, seq_len], gen_signed(rng, seq_padded * seq_padded, 500), DataType::Float, sf, Role::Constant);

  (
    attn_norm_w_vec,
    attn_q_w_vec,
    attn_k_w_vec,
    attn_v_w_vec,
    attn_o_w_vec,
    attn_norm_b_vec,
    attn_q_b_vec,
    attn_k_b_vec,
    attn_v_b_vec,
    attn_o_b_vec,
    proj_norm_w_vec,
    proj_1_w_vec,
    proj_2_w_vec,
    proj_norm_b_vec,
    proj_1_b_vec,
    proj_2_b_vec,
    layer_norm_w,
    layer_norm_b,
    attention_mask,
  )
}

fn main() {
  let mut timing = TimingTree::default();
  env_logger::init();

  println!("usize bits {}", usize::BITS);

  #[cfg(feature = "arkworks")]
  println!("using arkworks");
  #[cfg(feature = "icicle")]
  println!("using icicle");

  let thread_num = rayon::current_num_threads();
  println!("using {} threads", thread_num);

  let seq_len: usize = std::env::var("SEQ_LEN")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or(1);
  let seed: u64 = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(42);
  println!("Generating GPT-2 Small calibrated random weights (seq_len={seq_len}, seed={seed})...");

  // --- Circuit compilation ---
  let mut g = DagBuilder::new();

  // Single seeded RNG drives all calibrated witnesses (weights + input).
  let mut rng = StdRng::seed_from_u64(seed);

  let (
    attn_norm_w_vec,
    attn_q_w_vec,
    attn_k_w_vec,
    attn_v_w_vec,
    attn_o_w_vec,
    attn_norm_b_vec,
    attn_q_b_vec,
    attn_k_b_vec,
    attn_v_b_vec,
    attn_o_b_vec,
    proj_norm_w_vec,
    proj_1_w_vec,
    proj_2_w_vec,
    proj_norm_b_vec,
    proj_1_b_vec,
    proj_2_b_vec,
    layer_norm_w,
    layer_norm_b,
    attention_mask,
  ) = generate_gpt2_weights(&mut rng, seq_len);

  // Input: (batch_size=1, seq_len, 768)
  let x = g.input(vec![1, seq_len, 768], DataType::Float);

  // Run GPT-2 Small model
  let output = g.pipe(
    &[x],
    gpt_2_small(
      attn_norm_w_vec,
      attn_q_w_vec,
      attn_k_w_vec,
      attn_v_w_vec,
      attn_o_w_vec,
      attn_norm_b_vec,
      attn_q_b_vec,
      attn_k_b_vec,
      attn_v_b_vec,
      attn_o_b_vec,
      proj_norm_w_vec,
      proj_1_w_vec,
      proj_2_w_vec,
      proj_norm_b_vec,
      proj_1_b_vec,
      proj_2_b_vec,
      layer_norm_w,
      layer_norm_b,
      attention_mask,
      seq_len,
    ),
  )[0];

  // Compile -> (Dag, initial edge values)
  println!("Compiling DAG...");
  let (dag, mut init) = g.compile();

  // --- Prover ---
  let mut transcript = Transcript::new(b"zkml");

  let mut dense_commitments: Vec<Option<KZH3Commitment<PairingType>>> = vec![None; dag.num_edges()];
  let mut sparse_commitments: Vec<Option<Vec<KZH3Commitment<PairingType>>>> = vec![None; dag.num_edges()];

  // Witness generation from input (must do this before collecting polynomial sizes).
  // Hidden state magnitude ≈ ±2000 mirrors post-embedding+positional scale used in oneshot_gpt2,
  // giving RMS/LayerNorm enough dynamic range for the mean/x_sq tolerance checks.
  println!("Generating witness from calibrated random input...");
  let seq_padded_bin = seq_len.next_power_of_two();
  let input = Witness::new(
    vec![1, seq_len, 768],
    gen_signed(&mut rng, seq_padded_bin * 1024, 2000),
    DataType::Float,
    *SF_LOG as usize,
    Role::Input,
  );

  dag.run(&mut init, &vec![(x, input)]);
  println!("GPT-2 output shape: {:?}", init[output][0].shape);

  // Pre-permute eligible Constant witnesses (weights consumed only by Einsum).
  // Must run before commit so commitments are to the permuted polynomial.
  timed!(timing, "apply_prepermute", dag.apply_prepermute(&mut init));

  // Collect polynomial sizes from the DAG after witness generation
  let polynomial_sizes = dag.collect_polynomial_sizes(&init);
  println!("\n=== Polynomial sizes in GPT-2 DAG ===");
  println!("Sizes needed: {:?}", polynomial_sizes);
  println!("Number of different sizes: {}", polynomial_sizes.len());
  println!("=====================================\n");

  // Load or generate SRS for all required polynomial sizes
  let mut srs_map = HashMap::new();
  for &size in &polynomial_sizes {
    let size_srs = if fs::metadata(&format!("{}.srs", size)).is_ok() {
      println!("Loading existing SRS for polynomial size {}", size);
      load_kzh3_srs(size).expect(&format!("Failed to load SRS for size {}", size))
    } else {
      println!("Generating SRS for polynomial size {}", size);
      #[cfg(feature = "arkworks")]
      let size_srs = {
        use rand::thread_rng;
        setup_kzh3_srs::<PairingType, _>(size, &mut thread_rng())
      };
      #[cfg(feature = "icicle")]
      let size_srs = setup_kzh3_srs::<PairingType, _>(size, &mut ());
      store_kzh3_srs(&size_srs, size).expect(&format!("Failed to store SRS for size {}", size));
      size_srs
    };
    srs_map.insert(size, Arc::new(size_srs));
  }
  let srs_map = Arc::new(srs_map);

  // Create commitment keys with the collected SRS sizes
  let kzh3 = KZH3CommitKey::<PairingType> { srs_map: srs_map.clone() };
  let sparse_kzh3 = SparseKZH3CommitKey::<PairingType> { srs_map: srs_map.clone() };

  // Create verifier keys with the same srs_map
  let dense_verifier_key = KZH3VerifierKey { srs_map: srs_map.clone() };
  let sparse_verifier_key = SparseKZH3VerifierKey { srs_map: srs_map.clone() };

  // Commit to all witnesses (constants, inputs, auxiliaries, and final outputs)
  println!("Committing to witnesses...");
  timed!(timing, "commit", {
    dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &kzh3,
      &sparse_kzh3,
      &init,
      &mut dense_commitments,
      &mut sparse_commitments,
      &mut timing,
    )
  });

  println!("Generating proof...");
  let (sumcheck_proofs, opening_proofs, range_proof, two_pow_proof, reducer_proofs) = timed!(
    timing,
    "prove",
    dag.prove::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &kzh3,
      &sparse_kzh3,
      &init,
      &dense_commitments,
      &sparse_commitments,
      &mut transcript,
      &mut timing
    )
  );

  // Clear the data from the witnesses before verification
  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));

  // --- Verifier ---
  let mut verifier_transcript = Transcript::new(b"zkml");
  let verified = timed!(
    timing,
    "verify",
    dag.verify::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &sumcheck_proofs,
      &opening_proofs,
      &range_proof,
      &two_pow_proof,
      &reducer_proofs,
      &init,
      &dense_verifier_key,
      &sparse_verifier_key,
      &dense_commitments,
      &sparse_commitments,
      &mut verifier_transcript,
    )
  );
  timing.print();
  println!("verified: {:?}", verified);

  // Measure proof sizes (excluding claims)
  let sumcheck_proofs_only = extract_sumcheck_proofs_only(&sumcheck_proofs);
  let opening_proofs_only = extract_opening_proofs_only::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(&opening_proofs);
  let range_proof_only = extract_lookup_proof_only(&range_proof);
  let two_pow_proof_only = extract_lookup_proof_only(&two_pow_proof);

  let proof_size = measure_total_proof_size(
    &sumcheck_proofs_only,
    &opening_proofs_only,
    &range_proof_only,
    &two_pow_proof_only,
    &reducer_proofs,
  );
  println!("proof size: {}", proof_size);
}
