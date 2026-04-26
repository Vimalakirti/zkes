// nanoGPT proving binary. Builds the DAG for the ezkl nanoGPT config
// (n_layer=4, n_head=4, n_embd=64, bias=False) with random weights and
// runs the full commit/prove/verify pipeline, mirroring src/bin/gpt2.rs.

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
use rand::Rng;
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use zk_torch_2::util::transcript::Transcript;
use zk_torch_2::{
  crypto::polycommit::kzh3::{setup_kzh3_srs, KZH3Commit, KZH3CommitKey, KZH3Commitment, KZH3VerifierKey},
  crypto::polycommit::sparse_kzh3::{SparseKZH3Commit, SparseKZH3CommitKey, SparseKZH3VerifierKey},
  crypto::srs_storage::{load_kzh3_srs, store_kzh3_srs},
  dag::{
    extract_lookup_proof_only, extract_opening_proofs_only, extract_sumcheck_proofs_only,
    nanogpt::{nanogpt, NANO_HEAD_DIM, NANO_MLP_HIDDEN, NANO_N_EMBD, NANO_N_HEAD, NANO_N_LAYER, NANO_VOCAB_SIZE},
    DagBuilder, DataType, Role, Witness,
  },
  util::poly::CryptoField,
  util::serialization::measure_total_proof_size,
  SF_LOG,
};

#[cfg(all(feature = "arkworks", feature = "bn254"))]
use zk_torch_2::crypto::polycommit::ArkBn254 as PairingType;
#[cfg(all(feature = "icicle", feature = "bn254"))]
use zk_torch_2::crypto::polycommit::IcicleBn254 as PairingType;

fn generate_random_field_vec(size: usize) -> Vec<F> {
  let mut rng = rand::thread_rng();
  (0..size).map(|_| <F as CryptoField>::from_u32(rng.gen::<u32>() % 500)).collect()
}

fn generate_nanogpt_weights(seq_len: usize) -> (
  Witness<F>,      // wte (vocab_size, n_embd) — tied with lm_head
  Witness<F>,      // pos_emb (1, seq_len, n_embd) — wpe sliced to seq_len
  Vec<Witness<F>>, // attn_norm_w_vec
  Vec<Witness<F>>, // attn_q_w_vec
  Vec<Witness<F>>, // attn_k_w_vec
  Vec<Witness<F>>, // attn_v_w_vec
  Vec<Witness<F>>, // attn_o_w_vec
  Vec<Witness<F>>, // proj_norm_w_vec
  Vec<Witness<F>>, // proj_1_w_vec
  Vec<Witness<F>>, // proj_2_w_vec
  Witness<F>,      // layer_norm_w
  Witness<F>,      // attention_mask
) {
  let mut attn_norm_w_vec = Vec::new();
  let mut attn_q_w_vec = Vec::new();
  let mut attn_k_w_vec = Vec::new();
  let mut attn_v_w_vec = Vec::new();
  let mut attn_o_w_vec = Vec::new();
  let mut proj_norm_w_vec = Vec::new();
  let mut proj_1_w_vec = Vec::new();
  let mut proj_2_w_vec = Vec::new();

  let n_embd_pad = NANO_N_EMBD.next_power_of_two(); // 64
  let mlp_pad = NANO_MLP_HIDDEN.next_power_of_two(); // 256
  let vocab_pad = NANO_VOCAB_SIZE.next_power_of_two(); // 128
  let seq_pad = seq_len.next_power_of_two();

  // wte: (vocab_size, n_embd), weight-tied with lm_head in the PyTorch source.
  let wte = Witness::new(
    vec![NANO_VOCAB_SIZE, NANO_N_EMBD],
    generate_random_field_vec(vocab_pad * n_embd_pad),
    DataType::Float,
    *SF_LOG as usize,
    Role::Constant,
  );
  // pos_emb: wpe[0:seq_len] with shape (1, seq_len, n_embd).
  let pos_emb = Witness::new(
    vec![1, seq_len, NANO_N_EMBD],
    generate_random_field_vec(seq_pad * n_embd_pad),
    DataType::Float,
    *SF_LOG as usize,
    Role::Constant,
  );

  for _ in 0..NANO_N_LAYER {
    attn_norm_w_vec.push(Witness::new(
      vec![NANO_N_EMBD],
      generate_random_field_vec(n_embd_pad),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    for vec_ref in [&mut attn_q_w_vec, &mut attn_k_w_vec, &mut attn_v_w_vec, &mut attn_o_w_vec] {
      vec_ref.push(Witness::new(
        vec![NANO_N_EMBD, NANO_N_EMBD],
        generate_random_field_vec(n_embd_pad * n_embd_pad),
        DataType::Float,
        *SF_LOG as usize,
        Role::Constant,
      ));
    }

    proj_norm_w_vec.push(Witness::new(
      vec![NANO_N_EMBD],
      generate_random_field_vec(n_embd_pad),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    proj_1_w_vec.push(Witness::new(
      vec![NANO_N_EMBD, NANO_MLP_HIDDEN],
      generate_random_field_vec(n_embd_pad * mlp_pad),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    proj_2_w_vec.push(Witness::new(
      vec![NANO_MLP_HIDDEN, NANO_N_EMBD],
      generate_random_field_vec(mlp_pad * n_embd_pad),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
  }

  let layer_norm_w = Witness::new(
    vec![NANO_N_EMBD],
    generate_random_field_vec(n_embd_pad),
    DataType::Float,
    *SF_LOG as usize,
    Role::Constant,
  );

  let seq_padded = seq_len.next_power_of_two();
  let attention_mask = Witness::new(
    vec![1, seq_len, seq_len],
    generate_random_field_vec(seq_padded * seq_padded),
    DataType::Float,
    *SF_LOG as usize,
    Role::Constant,
  );

  (
    wte,
    pos_emb,
    attn_norm_w_vec,
    attn_q_w_vec,
    attn_k_w_vec,
    attn_v_w_vec,
    attn_o_w_vec,
    proj_norm_w_vec,
    proj_1_w_vec,
    proj_2_w_vec,
    layer_norm_w,
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
    .unwrap_or(8);
  println!(
    "Generating nanoGPT random weights (n_layer={}, n_head={}, n_embd={}, seq_len={})...",
    NANO_N_LAYER, NANO_N_HEAD, NANO_N_EMBD, seq_len
  );

  let mut g = DagBuilder::new();

  let (
    wte,
    pos_emb,
    attn_norm_w_vec,
    attn_q_w_vec,
    attn_k_w_vec,
    attn_v_w_vec,
    attn_o_w_vec,
    proj_norm_w_vec,
    proj_1_w_vec,
    proj_2_w_vec,
    layer_norm_w,
    attention_mask,
  ) = generate_nanogpt_weights(seq_len);

  // Input is a one-hot-style tensor over the vocab: (B, T, vocab_size).
  let x = g.input(vec![1, seq_len, NANO_VOCAB_SIZE], DataType::Float);

  let output = g.pipe(
    &[x],
    nanogpt(
      wte,
      pos_emb,
      attn_norm_w_vec,
      attn_q_w_vec,
      attn_k_w_vec,
      attn_v_w_vec,
      attn_o_w_vec,
      proj_norm_w_vec,
      proj_1_w_vec,
      proj_2_w_vec,
      layer_norm_w,
      attention_mask,
      NANO_N_LAYER,
      NANO_N_HEAD,
      NANO_HEAD_DIM,
      seq_len,
    ),
  )[0];

  println!("Compiling DAG...");
  let (dag, mut init) = g.compile();

  let mut transcript = Transcript::new(b"zkml");

  let mut dense_commitments: Vec<Option<KZH3Commitment<PairingType>>> = vec![None; dag.num_edges()];
  let mut sparse_commitments: Vec<Option<Vec<KZH3Commitment<PairingType>>>> = vec![None; dag.num_edges()];

  println!("Generating witness from random input...");
  let seq_padded_bin = seq_len.next_power_of_two();
  let vocab_pad = NANO_VOCAB_SIZE.next_power_of_two();
  let input = Witness::new(
    vec![1, seq_len, NANO_VOCAB_SIZE],
    generate_random_field_vec(seq_padded_bin * vocab_pad),
    DataType::Float,
    *SF_LOG as usize,
    Role::Input,
  );

  dag.run(&mut init, &vec![(x, input)]);
  println!("nanoGPT output shape: {:?}", init[output][0].shape);

  timed!(timing, "apply_prepermute", dag.apply_prepermute(&mut init));

  let polynomial_sizes = dag.collect_polynomial_sizes(&init);
  println!("\n=== Polynomial sizes in nanoGPT DAG ===");
  println!("Sizes needed: {:?}", polynomial_sizes);
  println!("Number of different sizes: {}", polynomial_sizes.len());
  println!("=======================================\n");

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

  let kzh3 = KZH3CommitKey::<PairingType> { srs_map: srs_map.clone() };
  let sparse_kzh3 = SparseKZH3CommitKey::<PairingType> { srs_map: srs_map.clone() };

  let dense_verifier_key = KZH3VerifierKey { srs_map: srs_map.clone() };
  let sparse_verifier_key = SparseKZH3VerifierKey { srs_map: srs_map.clone() };

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

  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));

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
