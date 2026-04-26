// One-shot transcript proving for GPT-2 Small.
//
// Full pipeline: embedding(W_E) → positional → transformer → LM_head(W_E) → argmax_check
//
// Uses two-pass approach:
//   Pass 1: Run without argmax to compute logits and extract argmax tokens
//   Pass 2: Build full circuit with argmax check, prove, verify
//
// Usage:
//   cargo +nightly-2025-07-01 run --release --bin oneshot_gpt2 -- config.yaml
//   SEQ_LEN=4 VOCAB_SIZE=256 cargo +nightly-2025-07-01 run --release --bin oneshot_gpt2 -- config.yaml

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
use zk_torch_2::{
  basicblock::einsum::{reset_einsum_timers, report_einsum_timers},
  crypto::polycommit::kzh3::{setup_kzh3_srs, KZH3Commit, KZH3CommitKey, KZH3Commitment, KZH3VerifierKey},
  crypto::polycommit::sparse_kzh3::{SparseKZH3Commit, SparseKZH3CommitKey, SparseKZH3VerifierKey},
  crypto::srs_storage::{load_kzh3_srs, store_kzh3_srs},
  dag::{extract_lookup_proof_only, extract_opening_proofs_only, extract_sumcheck_proofs_only, DagBuilder, DataType, Role, Witness},
  dag::gpt2::gpt_2_small,
  util::poly::CryptoField,
  util::serialization::measure_total_proof_size,
  util::transcript::Transcript,
  SF_LOG, SF_INT,
};

#[cfg(all(feature = "arkworks", feature = "bn254"))]
use zk_torch_2::crypto::polycommit::ArkBn254 as PairingType;
#[cfg(all(feature = "icicle", feature = "bn254"))]
use zk_torch_2::crypto::polycommit::IcicleBn254 as PairingType;

/// Generate signed random field elements in [-half_range, half_range].
/// Signed values prevent activation blowup through layers (zero-mean → cancellation).
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

/// Generate norm weights centered around SF_INT (representing 1.0).
/// LayerNorm/RMSNorm γ weights are initialized to 1.0 in real models.
fn gen_norm_weight(rng: &mut StdRng, size: usize) -> Vec<F> {
  let sf = *SF_INT;
  (0..size).map(|_| {
    let noise = (rng.gen::<u32>() % 65) as i64 - 32;
    <F as CryptoField>::from_u32((sf as i64 + noise) as u32)
  }).collect()
}

fn generate_gpt2_weights(rng: &mut StdRng, seq_len: usize) -> (
  Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>,
  Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>,
  Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>,
  Vec<Witness<F>>, Witness<F>, Witness<F>, Witness<F>,
) {
  let sf = *SF_LOG as usize;
  let mut attn_norm_w = Vec::new(); let mut attn_q_w = Vec::new();
  let mut attn_k_w = Vec::new(); let mut attn_v_w = Vec::new();
  let mut attn_o_w = Vec::new(); let mut attn_norm_b = Vec::new();
  let mut attn_q_b = Vec::new(); let mut attn_k_b = Vec::new();
  let mut attn_v_b = Vec::new(); let mut attn_o_b = Vec::new();
  let mut proj_norm_w = Vec::new(); let mut proj_1_w = Vec::new();
  let mut proj_2_w = Vec::new(); let mut proj_norm_b = Vec::new();
  let mut proj_1_b = Vec::new(); let mut proj_2_b = Vec::new();

  // Xavier-like initialization: weight std ≈ sf/sqrt(768) ≈ 37, half_range ≈ 64
  // Norm weights centered around sf (representing 1.0 in real)
  // Biases small and signed
  for _ in 0..12 {
    attn_norm_w.push(Witness::new(vec![768], gen_norm_weight(rng, 1024), DataType::Float, sf, Role::Constant));
    attn_q_w.push(Witness::new(vec![768, 768], gen_signed(rng, 1024 * 1024, 64), DataType::Float, sf, Role::Constant));
    attn_k_w.push(Witness::new(vec![768, 768], gen_signed(rng, 1024 * 1024, 64), DataType::Float, sf, Role::Constant));
    attn_v_w.push(Witness::new(vec![768, 768], gen_signed(rng, 1024 * 1024, 64), DataType::Float, sf, Role::Constant));
    attn_o_w.push(Witness::new(vec![768, 768], gen_signed(rng, 1024 * 1024, 64), DataType::Float, sf, Role::Constant));
    attn_norm_b.push(Witness::new(vec![768], gen_signed(rng, 1024, 64), DataType::Float, sf, Role::Constant));
    attn_q_b.push(Witness::new(vec![768], gen_signed(rng, 1024, 64), DataType::Float, sf, Role::Constant));
    attn_k_b.push(Witness::new(vec![768], gen_signed(rng, 1024, 64), DataType::Float, sf, Role::Constant));
    attn_v_b.push(Witness::new(vec![768], gen_signed(rng, 1024, 64), DataType::Float, sf, Role::Constant));
    attn_o_b.push(Witness::new(vec![768], gen_signed(rng, 1024, 64), DataType::Float, sf, Role::Constant));
    proj_norm_w.push(Witness::new(vec![768], gen_norm_weight(rng, 1024), DataType::Float, sf, Role::Constant));
    proj_1_w.push(Witness::new(vec![768, 3072], gen_signed(rng, 1024 * 4096, 64), DataType::Float, sf, Role::Constant));
    proj_2_w.push(Witness::new(vec![3072, 768], gen_signed(rng, 4096 * 1024, 32), DataType::Float, sf, Role::Constant)); // Xavier for 3072 inputs: std≈18≈sf/sqrt(3072)
    proj_norm_b.push(Witness::new(vec![768], gen_signed(rng, 1024, 64), DataType::Float, sf, Role::Constant));
    proj_1_b.push(Witness::new(vec![3072], gen_signed(rng, 4096, 64), DataType::Float, sf, Role::Constant));
    proj_2_b.push(Witness::new(vec![768], gen_signed(rng, 1024, 64), DataType::Float, sf, Role::Constant));
  }

  let layer_norm_w = Witness::new(vec![768], gen_norm_weight(rng, 1024), DataType::Float, sf, Role::Constant);
  let layer_norm_b = Witness::new(vec![768], gen_signed(rng, 1024, 64), DataType::Float, sf, Role::Constant);
  let seq_pad = seq_len.next_power_of_two();
  let attention_mask = Witness::new(vec![1, seq_len, seq_len], gen_signed(rng, seq_pad * seq_pad, 500), DataType::Float, sf, Role::Constant);

  (attn_norm_w, attn_q_w, attn_k_w, attn_v_w, attn_o_w,
   attn_norm_b, attn_q_b, attn_k_b, attn_v_b, attn_o_b,
   proj_norm_w, proj_1_w, proj_2_w, proj_norm_b, proj_1_b,
   proj_2_b, layer_norm_w, layer_norm_b, attention_mask)
}

/// Build the one-shot circuit (embedding → transformer → LM head → argmax check).
/// Uses dummy argmax tokens initially; caller overwrites after forward pass.
/// Returns (dag, init, logits_edge, emb_selector_edge, argmax_selector_edge).
fn build_oneshot_circuit(
  seq_len: usize,
  vocab_size: usize,
  token_ids: &[usize],
  seed: u64,
) -> (zk_torch_2::dag::Dag, Vec<Vec<Witness<F>>>, usize, usize, usize) {
  let hidden_dim: usize = 768;
  let vocab_pad = vocab_size.next_power_of_two();
  let hidden_pad = hidden_dim.next_power_of_two(); // 1024

  let mut rng = StdRng::seed_from_u64(seed);

  // Generate W_E: (vocab_size, hidden_dim) — embedding rows set activation scale
  // Use ±2000 so hidden states have |x| >> sqrt(sf), giving good x_sq resolution for RMS
  let we_data = gen_signed(&mut rng, vocab_pad * hidden_pad, 2000);
  let w_e_witness = Witness::new(vec![vocab_size, hidden_dim], we_data, DataType::Float, *SF_LOG as usize, Role::Constant);

  // Generate positional embeddings: (seq_len, hidden_dim)
  let seq_pad = seq_len.next_power_of_two();
  let pos_data = gen_signed(&mut rng, seq_pad * hidden_pad, 2000);
  let pos_embed = Witness::new(vec![seq_len, hidden_dim], pos_data, DataType::Float, *SF_LOG as usize, Role::Constant);

  // Generate transformer weights (deterministic with same rng)
  let (anw, aqw, akw, avw, aow, anb, aqb, akb, avb, aob,
       pnw, p1w, p2w, pnb, p1b, p2b, lnw, lnb, amask) =
    generate_gpt2_weights(&mut rng, seq_len);

  // --- Build circuit ---
  let mut g = DagBuilder::new();

  // Commit W_E (weight tying: embedding + LM head)
  let w_e = g.param(w_e_witness);

  // 1. Embedding lookup: tokens → H_0 (seq_len, hidden_dim)
  let (h0, emb_selector_edge) = g.embedding_lookup(w_e, seq_len, vocab_size, token_ids);

  // 2. Positional encoding: H_0 + P → H_pe (seq_len, hidden_dim)
  let h_pe = g.add_positional_encoding(h0, pos_embed);

  // 3. Reshape for transformer: (seq_len, 768) → (1, seq_len, 768)
  let h_input = g.change_shape(h_pe, vec![1, seq_len, hidden_dim]);

  // 4. Transformer (12 layers, causal masking)
  let h_out = g.pipe(
    &[h_input],
    gpt_2_small(anw, aqw, akw, avw, aow, anb, aqb, akb, avb, aob,
                pnw, p1w, p2w, pnb, p1b, p2b, lnw, lnb, amask, seq_len),
  )[0]; // (1, seq_len, 768)

  // 5. LM head (weight-tied): hidden → logits (seq_len, vocab_size)
  let logits = g.lm_head_weight_tied(h_out, w_e, seq_len, vocab_size);

  // 6. Argmax check with dummy tokens (will be overwritten after forward pass)
  let dummy_tokens = vec![0usize; seq_len];
  let selector_edge = g.argmax_check(logits, seq_len, vocab_size, &dummy_tokens);

  let (dag, init) = g.compile();
  (dag, init, logits, emb_selector_edge, selector_edge)
}

/// Extract argmax at a single sequence position.
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

  let seq_len: usize = std::env::var("SEQ_LEN").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
  let vocab_size: usize = std::env::var("VOCAB_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
  let prompt_len: usize = std::env::var("PROMPT_LEN").ok().and_then(|s| s.parse().ok())
    .unwrap_or((seq_len / 2).max(1));
  assert!(prompt_len >= 1 && prompt_len <= seq_len, "PROMPT_LEN must be in [1, SEQ_LEN]");
  // SKIP_AUTOREGRESSIVE=1 generates a random transcript without the AR loop. The
  // one-shot proof time is the same regardless; the AR loop only affects whether
  // the public shift-constraint check would succeed (real LLM) or fail (random).
  let skip_ar: bool = std::env::var("SKIP_AUTOREGRESSIVE").ok().as_deref() == Some("1");
  let seed: u64 = 42;

  println!("=== One-Shot GPT-2 Small (Full Pipeline) ===");
  println!("seq_len={}, prompt_len={}, generated={}, vocab_size={}, hidden_dim=768, layers=12",
    seq_len, prompt_len, seq_len - prompt_len, vocab_size);
  println!("SF_LOG={}, threads={}", *SF_LOG, rayon::current_num_threads());

  // Generate prompt token IDs. In AR mode, generated positions start as 0 and are
  // filled in by the autoregressive loop. In SKIP_AR mode, generated positions get
  // random tokens too (so the initial forward pass is the only forward pass we need).
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

  // ======== Build circuit ========
  println!("\n--- Building circuit ---");
  let (dag, mut init, logits_edge, emb_selector_edge, selector_edge) = timed!(timing, "build",
    build_oneshot_circuit(seq_len, vocab_size, &token_ids, seed));

  // ======== Initial forward pass ========
  // AR mode: gives logits for prompt, which drives the autoregressive loop below.
  // SKIP_AR mode: this is the ONLY forward pass — witness is complete after it.
  println!("Running initial forward pass...");
  timed!(timing, "forward_pass_init", dag.run(&mut init, &[]));

  if seq_len > prompt_len && !skip_ar {
    println!("Autoregressive generation for positions {}..{}:", prompt_len, seq_len);
    timed!(timing, "autoregressive_generate", {
      for pos in prompt_len..seq_len {
        // argmax at pos-1 gives the next token (teacher forced: goes into position pos of input)
        let next_tok = extract_argmax_at(&init, logits_edge, pos - 1, seq_len, vocab_size);
        token_ids[pos] = next_tok;

        // Update embedding selector with new token at position `pos` and rerun
        let new_s_emb = DagBuilder::<F>::build_one_hot_selector_pub(seq_len, vocab_size, &token_ids);
        init[emb_selector_edge] = vec![new_s_emb];
        dag.rerun_downstream(&mut init, &[emb_selector_edge]);
      }
    });
    println!("Final token_ids: {:?}", token_ids);
  }

  // ======== Shift constraint + argmax check ========
  // The argmax check inside the circuit uses next_token_ids[i] = argmax(logits[i])
  // for every position. This guarantees the range check passes: diffs[i,j] = logits[i,next[i]] - logits[i,j] >= 0.
  //
  // The verifier then checks the SHIFT CONSTRAINT on public data:
  //   next_token_ids[i] == token_ids[i+1]   for i in [prompt_len-1, seq_len-1)
  //
  // Since the circuit proves argmax(logits[i]) == next_token_ids[i] and the verifier checks
  // next_token_ids[i] == token_ids[i+1] publicly, together they enforce
  //   argmax(logits[i]) == token_ids[i+1]   for each generated position.
  // This is exactly the one-shot shift constraint from zkAgent §4.1.
  //
  // If a cheating prover uses arbitrary tokens (not autoregressive), next_token_ids[i]
  // (= argmax(logits[i])) will disagree with token_ids[i+1] and the verifier rejects publicly.
  let mut next_token_ids = vec![0usize; seq_len];
  for i in 0..seq_len {
    next_token_ids[i] = extract_argmax_at(&init, logits_edge, i, seq_len, vocab_size);
  }
  println!("next_token_ids (argmax of logits): {:?}", next_token_ids);

  // Public-data verifier check: for each generated position, input_token[i+1] == output_argmax[i].
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

  // Overwrite the argmax selector with the shifted tokens and rerun downstream nodes
  println!("Fixing argmax selector and rerunning downstream...");
  timed!(timing, "argmax_fixup", {
    let correct_selector = DagBuilder::<F>::build_one_hot_selector_pub(seq_len, vocab_size, &next_token_ids);
    init[selector_edge] = vec![correct_selector];
    dag.rerun_downstream(&mut init, &[selector_edge]);
  });

  // Pre-permute
  timed!(timing, "apply_prepermute", dag.apply_prepermute(&mut init));

  // SRS setup
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

  // Commit
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

  // Prove
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

  // Clear witness data
  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));

  // Proof size
  let sumcheck_proofs_only = extract_sumcheck_proofs_only(&sumcheck_proofs);
  let opening_proofs_only = extract_opening_proofs_only::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(&opening_proofs);
  let range_proof_only = extract_lookup_proof_only(&range_proof);
  let two_pow_proof_only = extract_lookup_proof_only(&two_pow_proof);
  let proof_size = measure_total_proof_size(&sumcheck_proofs_only, &opening_proofs_only, &range_proof_only, &two_pow_proof_only, &reducer_proofs);

  // Verify
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
