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
use zktransformer::util::transcript::Transcript;
use zktransformer::{
  crypto::polycommit::kzh3::{setup_kzh3_srs, KZH3Commit, KZH3CommitKey, KZH3Commitment, KZH3VerifierKey},
  crypto::polycommit::sparse_kzh3::{SparseKZH3Commit, SparseKZH3CommitKey, SparseKZH3VerifierKey},
  crypto::polycommit::{MLPolyCommit, NaiveMLPolyCommit},
  crypto::srs_storage::{load_kzh3_srs, store_kzh3_srs},
  dag::{bert::bert_large, DagBuilder, DataType, Role, Witness},
  util::poly::{CryptoField, DenseMLPoly, MLPoly, SparseMLPoly},
  SF_LOG, TABLE_SIZE_LOG,
};

#[cfg(all(feature = "arkworks", feature = "bn254"))]
use zktransformer::crypto::polycommit::ArkBn254 as PairingType;
#[cfg(all(feature = "icicle", feature = "bn254"))]
use zktransformer::crypto::polycommit::IcicleBn254 as PairingType;

fn generate_random_field_vec(size: usize) -> Vec<F> {
  let mut rng = rand::thread_rng();
  (0..size).map(|_| <F as CryptoField>::from_u32(rng.gen::<u32>() % 500)).collect()
}

fn generate_bert_weights() -> (
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
  Witness<F>,      // matmul_w
  Witness<F>,      // matmul_b
) {
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

  // Generate weights for 24 transformer layers in BERT-large
  for _i in 0..24 {
    // Attention norm weights: (1024)
    attn_norm_w_vec.push(Witness::new(
      vec![1024],
      generate_random_field_vec(1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // Attention Q, K, V, O weights: (1024, 1024)
    attn_q_w_vec.push(Witness::new(
      vec![1024, 1024],
      generate_random_field_vec(1024 * 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    attn_k_w_vec.push(Witness::new(
      vec![1024, 1024],
      generate_random_field_vec(1024 * 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    attn_v_w_vec.push(Witness::new(
      vec![1024, 1024],
      generate_random_field_vec(1024 * 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    attn_o_w_vec.push(Witness::new(
      vec![1024, 1024],
      generate_random_field_vec(1024 * 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // Attention bias terms: (1024)
    attn_norm_b_vec.push(Witness::new(
      vec![1024],
      generate_random_field_vec(1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    attn_q_b_vec.push(Witness::new(
      vec![1024],
      generate_random_field_vec(1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    attn_k_b_vec.push(Witness::new(
      vec![1024],
      generate_random_field_vec(1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    attn_v_b_vec.push(Witness::new(
      vec![1024],
      generate_random_field_vec(1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    attn_o_b_vec.push(Witness::new(
      vec![1024],
      generate_random_field_vec(1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // Projection norm weights: (1024)
    proj_norm_w_vec.push(Witness::new(
      vec![1024],
      generate_random_field_vec(1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // MLP weights
    // proj_1_w: (1024, 4096)
    proj_1_w_vec.push(Witness::new(
      vec![1024, 4096],
      generate_random_field_vec(1024 * 4096),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    // proj_2_w: (4096, 1024)
    proj_2_w_vec.push(Witness::new(
      vec![4096, 1024],
      generate_random_field_vec(4096 * 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // Projection bias terms
    proj_norm_b_vec.push(Witness::new(
      vec![1024],
      generate_random_field_vec(1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    proj_1_b_vec.push(Witness::new(
      vec![4096],
      generate_random_field_vec(4096),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    proj_2_b_vec.push(Witness::new(
      vec![1024],
      generate_random_field_vec(1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
  }

  // Layer norm weight: (1024)
  let layer_norm_w = Witness::new(
    vec![1024],
    generate_random_field_vec(1024),
    DataType::Float,
    *SF_LOG as usize,
    Role::Constant,
  );

  // Layer norm bias: (1024)
  let layer_norm_b = Witness::new(
    vec![1024],
    generate_random_field_vec(1024),
    DataType::Float,
    *SF_LOG as usize,
    Role::Constant,
  );

  // Attention mask: (batch_size=1, seq_len=384, seq_len=384)
  let attention_mask = Witness::new(
    vec![1, 1, 1],
    generate_random_field_vec(1 * 1 * 1),
    DataType::Float,
    *SF_LOG as usize,
    Role::Constant,
  );

  // Matmul weight: (1024, 2)
  let matmul_w = Witness::new(
    vec![1024, 2],
    generate_random_field_vec(1024 * 2),
    DataType::Float,
    *SF_LOG as usize,
    Role::Constant,
  );

  // Matmul bias: (2)
  let matmul_b = Witness::new(vec![2], generate_random_field_vec(2), DataType::Float, *SF_LOG as usize, Role::Constant);

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
    matmul_w,
    matmul_b,
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

  println!("Generating BERT-large random weights...");

  // --- Circuit compilation ---
  let mut g = DagBuilder::new();

  // Generate random weights for BERT-large
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
    matmul_w,
    matmul_b,
  ) = generate_bert_weights();

  // Input: (batch_size=1, seq_len=384, 1024)
  let x = g.input(vec![1, 1, 1024], DataType::Float);

  // Run BERT-large model
  let output = g.pipe(
    &[x],
    bert_large(
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
      matmul_w,
      matmul_b,
    ),
  )[0];

  // Compile -> (Dag, initial edge values)
  println!("Compiling DAG...");
  let (dag, mut init) = g.compile();

  // --- Prover ---
  let mut transcript = Transcript::new(b"zkml");

  let mut dense_commitments: Vec<Option<KZH3Commitment<PairingType>>> = vec![None; dag.num_edges()];
  let mut sparse_commitments: Vec<Option<Vec<KZH3Commitment<PairingType>>>> = vec![None; dag.num_edges()];

  // Witness generation from input (must do this before collecting polynomial sizes)
  println!("Generating witness from random input...");
  let input = Witness::new(
    vec![1, 1, 1024],
    generate_random_field_vec(1 * 1 * 1024), // Random input tokens
    DataType::Float,
    *SF_LOG as usize,
    Role::Input,
  );

  dag.run(&mut init, &vec![(x, input)]);
  println!("BERT output shape: {:?}", init[output][0].shape);

  // Collect polynomial sizes from the DAG after witness generation
  let polynomial_sizes = dag.collect_polynomial_sizes(&init);
  println!("\n=== Polynomial sizes in BERT DAG ===");
  println!("Sizes needed: {:?}", polynomial_sizes);
  println!("Number of different sizes: {}", polynomial_sizes.len());
  println!("====================================\n");

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
}
