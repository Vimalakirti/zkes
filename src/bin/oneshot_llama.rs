// One-shot transcript proving for LLaMA-2-7B.
//
// Full pipeline: embedding(W_E) → transformer (RoPE, no APE) + LM_head → argmax_check
//
// Differences from GPT-2 one-shot:
//   - LLaMA uses RoPE inside attention, so no additive positional encoding.
//   - LM head is a separate weight (not tied to embedding).
//
// Usage:
//   cargo +nightly-2025-07-01 run --release --bin oneshot_llama -- config.yaml
//   SEQ_LEN=8 VOCAB_SIZE=256 NUM_LAYERS=2 cargo +nightly-2025-07-01 run --release --bin oneshot_llama -- config.yaml

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
use zk_torch_2::{
  basicblock::einsum::{reset_einsum_timers, report_einsum_timers},
  crypto::polycommit::kzh3::{setup_kzh3_srs, KZH3Commit, KZH3CommitKey, KZH3Commitment, KZH3VerifierKey},
  crypto::polycommit::sparse_kzh3::{SparseKZH3Commit, SparseKZH3CommitKey, SparseKZH3VerifierKey},
  crypto::srs_storage::{load_kzh3_srs, store_kzh3_srs},
  dag::{extract_lookup_proof_only, extract_opening_proofs_only, extract_sumcheck_proofs_only, DagBuilder, DataType, Role, Witness},
  dag::llama::llama_2_7b,
  util::poly::CryptoField,
  util::serialization::measure_total_proof_size,
  util::transcript::Transcript,
  SF_LOG, SF_INT,
};

#[cfg(all(feature = "arkworks", feature = "bn254"))]
use zk_torch_2::crypto::polycommit::ArkBn254 as PairingType;
#[cfg(all(feature = "icicle", feature = "bn254"))]
use zk_torch_2::crypto::polycommit::IcicleBn254 as PairingType;

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

fn gen_norm_weight(rng: &mut StdRng, size: usize) -> Vec<F> {
  let sf = *SF_INT;
  (0..size).map(|_| {
    let noise = (rng.gen::<u32>() % 65) as i64 - 32;
    <F as CryptoField>::from_u32((sf as i64 + noise) as u32)
  }).collect()
}

fn generate_llama_weights(
  rng: &mut StdRng,
  num_layers: usize,
  hidden_dim: usize,
  mlp_dim: usize,
  vocab_size: usize,
) -> (
  Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>,
  Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>,
  Witness<F>, Witness<F>,
) {
  let sf = *SF_LOG as usize;
  let hidden_pad = hidden_dim.next_power_of_two();
  let mlp_pad = mlp_dim.next_power_of_two();
  let vocab_pad = vocab_size.next_power_of_two();

  let mut attn_norm_w = Vec::new();
  let mut attn_q_w = Vec::new();
  let mut attn_k_w = Vec::new();
  let mut attn_v_w = Vec::new();
  let mut attn_o_w = Vec::new();
  let mut proj_norm_w = Vec::new();
  let mut proj_1_w = Vec::new();
  let mut proj_2_w = Vec::new();
  let mut proj_3_w = Vec::new();

  for _ in 0..num_layers {
    attn_norm_w.push(Witness::new(vec![hidden_dim], gen_norm_weight(rng, hidden_pad), DataType::Float, sf, Role::Constant));
    attn_q_w.push(Witness::new(vec![hidden_dim, hidden_dim], gen_signed(rng, hidden_pad * hidden_pad, 64), DataType::Float, sf, Role::Constant));
    attn_k_w.push(Witness::new(vec![hidden_dim, hidden_dim], gen_signed(rng, hidden_pad * hidden_pad, 64), DataType::Float, sf, Role::Constant));
    attn_v_w.push(Witness::new(vec![hidden_dim, hidden_dim], gen_signed(rng, hidden_pad * hidden_pad, 64), DataType::Float, sf, Role::Constant));
    attn_o_w.push(Witness::new(vec![hidden_dim, hidden_dim], gen_signed(rng, hidden_pad * hidden_pad, 64), DataType::Float, sf, Role::Constant));
    proj_norm_w.push(Witness::new(vec![hidden_dim], gen_norm_weight(rng, hidden_pad), DataType::Float, sf, Role::Constant));
    proj_1_w.push(Witness::new(vec![hidden_dim, mlp_dim], gen_signed(rng, hidden_pad * mlp_pad, 64), DataType::Float, sf, Role::Constant));
    proj_2_w.push(Witness::new(vec![hidden_dim, mlp_dim], gen_signed(rng, hidden_pad * mlp_pad, 64), DataType::Float, sf, Role::Constant));
    proj_3_w.push(Witness::new(vec![mlp_dim, hidden_dim], gen_signed(rng, mlp_pad * hidden_pad, 32), DataType::Float, sf, Role::Constant));
  }
  let layer_norm_w = Witness::new(vec![hidden_dim], gen_norm_weight(rng, hidden_pad), DataType::Float, sf, Role::Constant);
  let logits_w = Witness::new(vec![hidden_dim, vocab_size], gen_signed(rng, hidden_pad * vocab_pad, 64), DataType::Float, sf, Role::Constant);

  (attn_norm_w, attn_q_w, attn_k_w, attn_v_w, attn_o_w,
   proj_norm_w, proj_1_w, proj_2_w, proj_3_w, layer_norm_w, logits_w)
}

/// Build the one-shot circuit for LLaMA.
/// Returns (dag, init, logits_edge, emb_selector_edge, argmax_selector_edge).
fn build_oneshot_circuit(
  seq_len: usize,
  vocab_size: usize,
  num_layers: usize,
  num_heads: usize,
  head_dim: usize,
  mlp_dim: usize,
  token_ids: &[usize],
  seed: u64,
) -> (zk_torch_2::dag::Dag, Vec<Vec<Witness<F>>>, usize, usize, usize) {
  let hidden_dim = num_heads * head_dim;
  let vocab_pad = vocab_size.next_power_of_two();
  let hidden_pad = hidden_dim.next_power_of_two();

  let mut rng = StdRng::seed_from_u64(seed);

  // Embedding matrix W_E: (vocab_size, hidden_dim). Larger magnitude so hidden activations
  // give enough dynamic range for RMSNorm's x_sq and the mean-tolerance check.
  let we_data = gen_signed(&mut rng, vocab_pad * hidden_pad, 2000);
  let w_e_witness = Witness::new(vec![vocab_size, hidden_dim], we_data, DataType::Float, *SF_LOG as usize, Role::Constant);

  let (anw, aqw, akw, avw, aow, pnw, p1w, p2w, p3w, lnw, logits_w) =
    generate_llama_weights(&mut rng, num_layers, hidden_dim, mlp_dim, vocab_size);

  let mut g = DagBuilder::new();

  // Commit W_E (embedding only; LLaMA LM head uses a separate weight).
  let w_e = g.param(w_e_witness);

  // 1. Embedding lookup: tokens → H_0 (seq_len, hidden_dim)
  let (h0, emb_selector_edge) = g.embedding_lookup(w_e, seq_len, vocab_size, token_ids);

  // 2. Reshape for transformer: (seq_len, hidden) → (1, seq_len, hidden).
  //    No positional encoding — LLaMA uses RoPE inside attention.
  let h_input = g.change_shape(h0, vec![1, seq_len, hidden_dim]);

  // 3. Transformer + LM head (logits: (seq_len, vocab_size)).
  let logits = g.pipe(
    &[h_input],
    llama_2_7b(anw, aqw, akw, avw, aow, pnw, p1w, p2w, p3w, lnw, logits_w,
               num_heads, head_dim, seq_len, vocab_size),
  )[0];

  // 4. Argmax check with dummy tokens (overwritten after forward pass).
  let dummy_tokens = vec![0usize; seq_len];
  let selector_edge = g.argmax_check(logits, seq_len, vocab_size, &dummy_tokens);

  let (dag, init) = g.compile();
  (dag, init, logits, emb_selector_edge, selector_edge)
}

fn extract_argmax_at(init: &[Vec<Witness<F>>], logits_edge: usize, pos: usize, seq_len: usize, vocab_size: usize) -> usize {
  let logits_w = &init[logits_edge][0];
  let data_int = logits_w.data_int.as_ref().expect("logits must have data_int after forward pass");
  let seq_pad = seq_len.next_power_of_two();
  let mut best_v = 0usize;
  let mut best_val = i128::MIN;
  for v in 0..vocab_size {
    let val = data_int[pos + v * seq_pad];
    if val > best_val {
      best_val = val;
      best_v = v;
    }
  }
  best_v
}

fn main() {
  let mut timing = TimingTree::default();
  env_logger::init();

  let seq_len: usize = std::env::var("SEQ_LEN").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
  let vocab_size: usize = std::env::var("VOCAB_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
  let num_layers: usize = std::env::var("NUM_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
  let num_heads: usize = std::env::var("NUM_HEADS").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
  let head_dim: usize = std::env::var("HEAD_DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
  let mlp_dim: usize = std::env::var("MLP_DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(11008);
  let hidden_dim = num_heads * head_dim;

  let prompt_len: usize = std::env::var("PROMPT_LEN").ok().and_then(|s| s.parse().ok())
    .unwrap_or((seq_len / 2).max(1));
  assert!(prompt_len >= 1 && prompt_len <= seq_len, "PROMPT_LEN must be in [1, SEQ_LEN]");
  let skip_ar: bool = std::env::var("SKIP_AUTOREGRESSIVE").ok().as_deref() == Some("1");
  let seed: u64 = 42;

  println!("=== One-Shot LLaMA-2 (Full Pipeline) ===");
  println!("seq_len={}, prompt_len={}, generated={}, vocab_size={}, hidden={}, heads={}, head_dim={}, mlp={}, layers={}",
    seq_len, prompt_len, seq_len - prompt_len, vocab_size, hidden_dim, num_heads, head_dim, mlp_dim, num_layers);
  println!("SF_LOG={}, threads={}", *SF_LOG, rayon::current_num_threads());

  let mut rng = StdRng::seed_from_u64(seed + 1000);
  let mut token_ids: Vec<usize> = vec![0usize; seq_len];
  for i in 0..prompt_len {
    token_ids[i] = rng.gen::<usize>() % vocab_size;
  }
  if skip_ar {
    for i in prompt_len..seq_len {
      token_ids[i] = rng.gen::<usize>() % vocab_size;
    }
    println!("SKIP_AUTOREGRESSIVE=1: using random tokens (generated positions are random, shift check skipped).");
  }
  println!("prompt token_ids[0..{}]: {:?}", prompt_len, &token_ids[..prompt_len]);

  println!("\n--- Building circuit ---");
  let (dag, mut init, logits_edge, emb_selector_edge, selector_edge) = timed!(timing, "build",
    build_oneshot_circuit(seq_len, vocab_size, num_layers, num_heads, head_dim, mlp_dim, &token_ids, seed));

  println!("Running initial forward pass...");
  timed!(timing, "forward_pass_init", dag.run(&mut init, &[]));

  if seq_len > prompt_len && !skip_ar {
    println!("Autoregressive generation for positions {}..{}:", prompt_len, seq_len);
    timed!(timing, "autoregressive_generate", {
      for pos in prompt_len..seq_len {
        let next_tok = extract_argmax_at(&init, logits_edge, pos - 1, seq_len, vocab_size);
        token_ids[pos] = next_tok;
        let new_s_emb = DagBuilder::<F>::build_one_hot_selector_pub(seq_len, vocab_size, &token_ids);
        init[emb_selector_edge] = vec![new_s_emb];
        dag.rerun_downstream(&mut init, &[emb_selector_edge]);
      }
    });
    println!("Final token_ids: {:?}", token_ids);
  }

  let mut next_token_ids = vec![0usize; seq_len];
  for i in 0..seq_len {
    next_token_ids[i] = extract_argmax_at(&init, logits_edge, i, seq_len, vocab_size);
  }
  println!("next_token_ids (argmax of logits): {:?}", next_token_ids);

  if !skip_ar {
    for i in (prompt_len - 1)..seq_len - 1 {
      assert_eq!(
        token_ids[i + 1], next_token_ids[i],
        "shift constraint violated at generated position {}: token_ids[{}]={} != argmax(logits[{}])={}",
        i, i + 1, token_ids[i + 1], i, next_token_ids[i]
      );
    }
    println!("Public shift-constraint check passed for generated positions [{}, {}).", prompt_len - 1, seq_len - 1);
  } else {
    println!("SKIP_AUTOREGRESSIVE=1: public shift-constraint check skipped (prover benchmarking only).");
  }

  println!("Fixing argmax selector and rerunning downstream...");
  timed!(timing, "argmax_fixup", {
    let correct_selector = DagBuilder::<F>::build_one_hot_selector_pub(seq_len, vocab_size, &next_token_ids);
    init[selector_edge] = vec![correct_selector];
    dag.rerun_downstream(&mut init, &[selector_edge]);
  });

  timed!(timing, "apply_prepermute", dag.apply_prepermute(&mut init));

  let polynomial_sizes = dag.collect_polynomial_sizes(&init);
  println!("Polynomial sizes: {:?}", polynomial_sizes);
  println!("Number of sizes: {}", polynomial_sizes.len());

  let srs_map = timed!(timing, "srs_setup", {
    let mut srs_map = HashMap::new();
    for &size in &polynomial_sizes {
      let size_srs = if fs::metadata(&format!("{}.srs", size)).is_ok() {
        println!("Loading SRS for size {}", size);
        load_kzh3_srs(size).expect(&format!("Failed to load SRS for size {}", size))
      } else {
        println!("Generating SRS for size {}", size);
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
    Arc::new(srs_map)
  });

  let kzh3 = KZH3CommitKey::<PairingType> { srs_map: srs_map.clone() };
  let sparse_kzh3 = SparseKZH3CommitKey::<PairingType> { srs_map: srs_map.clone() };
  let dense_verifier_key = KZH3VerifierKey { srs_map: srs_map.clone() };
  let sparse_verifier_key = SparseKZH3VerifierKey { srs_map: srs_map.clone() };

  let mut dense_commitments: Vec<Option<KZH3Commitment<PairingType>>> = vec![None; dag.num_edges()];
  let mut sparse_commitments: Vec<Option<Vec<KZH3Commitment<PairingType>>>> = vec![None; dag.num_edges()];

  println!("Committing...");
  timed!(timing, "commit", {
    dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &kzh3, &sparse_kzh3, &init, &mut dense_commitments, &mut sparse_commitments, &mut timing,
    )
  });

  let mut transcript = Transcript::new(b"zkml");
  println!("Proving...");
  reset_einsum_timers();
  let (sumcheck_proofs, opening_proofs, range_proof, two_pow_proof, reducer_proofs) = timed!(
    timing, "prove",
    dag.prove::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &kzh3, &sparse_kzh3, &init, &dense_commitments, &sparse_commitments, &mut transcript, &mut timing
    )
  );
  report_einsum_timers();

  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));

  let sumcheck_proofs_only = extract_sumcheck_proofs_only(&sumcheck_proofs);
  let opening_proofs_only = extract_opening_proofs_only::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(&opening_proofs);
  let range_proof_only = extract_lookup_proof_only(&range_proof);
  let two_pow_proof_only = extract_lookup_proof_only(&two_pow_proof);
  let proof_size = measure_total_proof_size(&sumcheck_proofs_only, &opening_proofs_only, &range_proof_only, &two_pow_proof_only, &reducer_proofs);

  let mut verifier_transcript = Transcript::new(b"zkml");
  println!("Verifying...");
  let verified = timed!(
    timing, "verify",
    dag.verify::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &sumcheck_proofs, &opening_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
      &init, &dense_verifier_key, &sparse_verifier_key, &dense_commitments, &sparse_commitments,
      &mut verifier_transcript,
    )
  );

  println!("\n=== Results ===");
  timing.print();
  println!("verified: {:?}", verified);
  println!("proof size: {}", proof_size);
}
