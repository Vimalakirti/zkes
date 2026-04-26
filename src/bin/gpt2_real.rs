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
use std::fs::{self, File};
use std::io::Read as IoRead;
use std::path::Path;
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
  SF_LOG, TABLE_SIZE_LOG,
};

#[cfg(all(feature = "arkworks", feature = "bn254"))]
use zk_torch_2::crypto::polycommit::ArkBn254 as PairingType;
#[cfg(all(feature = "icicle", feature = "bn254"))]
use zk_torch_2::crypto::polycommit::IcicleBn254 as PairingType;

fn generate_random_field_vec(size: usize) -> Vec<F> {
  let mut rng = rand::thread_rng();
  //rng.gen::<u32>() % 500
  (0..size).map(|_| <F as CryptoField>::from_u32(0)).collect()
}

/// Load tensor data from a binary file containing int64 values (little-endian)
fn load_tensor_from_bin<P: AsRef<Path>>(path: P) -> Vec<i64> {
  let mut file = File::open(path).expect("Failed to open tensor file");
  let mut buffer = Vec::new();
  file.read_to_end(&mut buffer).expect("Failed to read tensor file");

  // Convert bytes to i64 (little-endian)
  buffer.chunks_exact(8).map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap())).collect()
}

/// Pad a 1D tensor from original_size to padded_size (pad with zeros)
/// e.g., [768] -> [1024]
fn pad_1d_tensor(data: &[i64], original_size: usize, padded_size: usize) -> Vec<F> {
  assert!(data.len() == original_size, "Data length mismatch");
  assert!(padded_size >= original_size, "Padded size must be >= original size");

  let mut result = Vec::with_capacity(padded_size);

  // Copy original data
  for &val in data.iter() {
    if val >= 0 {
      result.push(F::from(val as u64));
    } else {
      result.push(-F::from((-val) as u64));
    }
  }

  // Pad with zeros
  for _ in original_size..padded_size {
    result.push(<F as CryptoField>::zero());
  }

  result
}

/// Pad a 2D tensor from [rows, cols] to [padded_rows, padded_cols] (pad with zeros)
/// The data is in row-major order. Result is also row-major (flattened).
/// e.g., [768, 768] -> [1024, 1024] or [768, 3072] -> [1024, 4096]
fn pad_2d_tensor(data: &[i64], rows: usize, cols: usize, padded_rows: usize, padded_cols: usize) -> Vec<F> {
  assert!(data.len() == rows * cols, "Data length mismatch");
  assert!(padded_rows >= rows && padded_cols >= cols, "Padded dimensions must be >= original");

  let mut result = Vec::with_capacity(padded_rows * padded_cols);

  for r in 0..padded_rows {
    for c in 0..padded_cols {
      if r < rows && c < cols {
        let val = data[r * cols + c];
        if val >= 0 {
          result.push(F::from(val as u64));
        } else {
          result.push(-F::from((-val) as u64));
        }
      } else {
        result.push(<F as CryptoField>::zero());
      }
    }
  }

  result
}

/// Load and pad a 1D tensor from a bin file
fn load_1d_tensor(tensor_dir: &str, name: &str, original_size: usize, padded_size: usize) -> Vec<F> {
  let path = format!("{}/{}.bin", tensor_dir, name);
  let data = load_tensor_from_bin(&path);
  pad_1d_tensor(&data, original_size, padded_size)
}

/// Load and pad a 2D tensor from a bin file
fn load_2d_tensor(tensor_dir: &str, name: &str, rows: usize, cols: usize, padded_rows: usize, padded_cols: usize) -> Vec<F> {
  let path = format!("{}/{}.bin", tensor_dir, name);
  let data = load_tensor_from_bin(&path);
  pad_2d_tensor(&data, rows, cols, padded_rows, padded_cols)
}

/*
  | Code Variable  | Tensor File                | Shape → Padded       |
  |----------------|----------------------------|----------------------|
  | attn_norm_w    | h.{i}.ln_1.weight          | 768 → 1024           |
  | attn_norm_b    | h.{i}.ln_1.bias            | 768 → 1024           |
  | attn_q_w       | h.{i}.attn.c_attn.weight_q | 768×768 → 1024×1024  |
  | attn_k_w       | h.{i}.attn.c_attn.weight_k | 768×768 → 1024×1024  |
  | attn_v_w       | h.{i}.attn.c_attn.weight_v | 768×768 → 1024×1024  |
  | attn_o_w       | h.{i}.attn.c_proj.weight   | 768×768 → 1024×1024  |
  | proj_1_w       | h.{i}.mlp.c_fc.weight      | 768×3072 → 1024×4096 |
  | proj_2_w       | h.{i}.mlp.c_proj.weight    | 3072×768 → 4096×1024 |
  | layer_norm_w/b | ln_f.weight/bias           | 768 → 1024           |
*/

fn load_gpt2_weights() -> (
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
  let tensor_dir = "gpt2/tensors";

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

  // Load weights for 12 transformer layers
  for i in 0..12 {
    println!("Loading weights for layer {}...", i);

    // h.{i}.ln_1.weight -> attn_norm_w (768 -> 1024)
    attn_norm_w_vec.push(Witness::new(
      vec![768],
      load_1d_tensor(tensor_dir, &format!("h.{}.ln_1.weight", i), 768, 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // h.{i}.attn.c_attn.weight_q -> attn_q_w (768x768 -> 1024x1024)
    attn_q_w_vec.push(Witness::new(
      vec![768, 768],
      load_2d_tensor(tensor_dir, &format!("h.{}.attn.c_attn.weight_q", i), 768, 768, 1024, 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // h.{i}.attn.c_attn.weight_k -> attn_k_w (768x768 -> 1024x1024)
    attn_k_w_vec.push(Witness::new(
      vec![768, 768],
      load_2d_tensor(tensor_dir, &format!("h.{}.attn.c_attn.weight_k", i), 768, 768, 1024, 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // h.{i}.attn.c_attn.weight_v -> attn_v_w (768x768 -> 1024x1024)
    attn_v_w_vec.push(Witness::new(
      vec![768, 768],
      load_2d_tensor(tensor_dir, &format!("h.{}.attn.c_attn.weight_v", i), 768, 768, 1024, 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // h.{i}.attn.c_proj.weight -> attn_o_w (768x768 -> 1024x1024)
    attn_o_w_vec.push(Witness::new(
      vec![768, 768],
      load_2d_tensor(tensor_dir, &format!("h.{}.attn.c_proj.weight", i), 768, 768, 1024, 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // h.{i}.ln_1.bias -> attn_norm_b (768 -> 1024)
    attn_norm_b_vec.push(Witness::new(
      vec![768],
      load_1d_tensor(tensor_dir, &format!("h.{}.ln_1.bias", i), 768, 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // h.{i}.attn.c_attn.bias_q -> attn_q_b (768 -> 1024)
    attn_q_b_vec.push(Witness::new(
      vec![768],
      load_1d_tensor(tensor_dir, &format!("h.{}.attn.c_attn.bias_q", i), 768, 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // h.{i}.attn.c_attn.bias_k -> attn_k_b (768 -> 1024)
    attn_k_b_vec.push(Witness::new(
      vec![768],
      load_1d_tensor(tensor_dir, &format!("h.{}.attn.c_attn.bias_k", i), 768, 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // h.{i}.attn.c_attn.bias_v -> attn_v_b (768 -> 1024)
    attn_v_b_vec.push(Witness::new(
      vec![768],
      load_1d_tensor(tensor_dir, &format!("h.{}.attn.c_attn.bias_v", i), 768, 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // h.{i}.attn.c_proj.bias -> attn_o_b (768 -> 1024)
    attn_o_b_vec.push(Witness::new(
      vec![768],
      load_1d_tensor(tensor_dir, &format!("h.{}.attn.c_proj.bias", i), 768, 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // h.{i}.ln_2.weight -> proj_norm_w (768 -> 1024)
    proj_norm_w_vec.push(Witness::new(
      vec![768],
      load_1d_tensor(tensor_dir, &format!("h.{}.ln_2.weight", i), 768, 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // h.{i}.mlp.c_fc.weight -> proj_1_w (768x3072 -> 1024x4096)
    proj_1_w_vec.push(Witness::new(
      vec![768, 3072],
      load_2d_tensor(tensor_dir, &format!("h.{}.mlp.c_fc.weight", i), 768, 3072, 1024, 4096),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // h.{i}.mlp.c_proj.weight -> proj_2_w (3072x768 -> 4096x1024)
    proj_2_w_vec.push(Witness::new(
      vec![3072, 768],
      load_2d_tensor(tensor_dir, &format!("h.{}.mlp.c_proj.weight", i), 3072, 768, 4096, 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // h.{i}.ln_2.bias -> proj_norm_b (768 -> 1024)
    proj_norm_b_vec.push(Witness::new(
      vec![768],
      load_1d_tensor(tensor_dir, &format!("h.{}.ln_2.bias", i), 768, 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // h.{i}.mlp.c_fc.bias -> proj_1_b (3072 -> 4096)
    proj_1_b_vec.push(Witness::new(
      vec![3072],
      load_1d_tensor(tensor_dir, &format!("h.{}.mlp.c_fc.bias", i), 3072, 4096),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // h.{i}.mlp.c_proj.bias -> proj_2_b (768 -> 1024)
    proj_2_b_vec.push(Witness::new(
      vec![768],
      load_1d_tensor(tensor_dir, &format!("h.{}.mlp.c_proj.bias", i), 768, 1024),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
  }

  // ln_f.weight -> layer_norm_w (768 -> 1024)
  let layer_norm_w = Witness::new(
    vec![768],
    load_1d_tensor(tensor_dir, "ln_f.weight", 768, 1024),
    DataType::Float,
    *SF_LOG as usize,
    Role::Constant,
  );

  // ln_f.bias -> layer_norm_b (768 -> 1024)
  let layer_norm_b = Witness::new(
    vec![768],
    load_1d_tensor(tensor_dir, "ln_f.bias", 768, 1024),
    DataType::Float,
    *SF_LOG as usize,
    Role::Constant,
  );

  // Attention mask: (batch_size=1, seq_len=1, seq_len=1)
  let attention_mask = Witness::new(
    vec![1, 1, 1],
    generate_random_field_vec(1 * 1 * 1),
    DataType::Float,
    *SF_LOG as usize,
    Role::Constant,
  );

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

  println!("Loading GPT-2 Small weights from gpt2/tensors/...");

  // --- Circuit compilation ---
  let mut g = DagBuilder::new();

  // Load real weights for GPT-2 Small from binary files
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
  ) = load_gpt2_weights();

  // Input: (batch_size=1, seq_len=1, 768)
  let x = g.input(vec![1, 1, 768], DataType::Float);

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
    vec![1, 1, 768],
    generate_random_field_vec(1024), // Random input tokens
    DataType::Float,
    *SF_LOG as usize,
    Role::Input,
  );

  dag.run(&mut init, &vec![(x, input)]);
  println!("GPT-2 output shape: {:?}", init[output][0].shape);

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
