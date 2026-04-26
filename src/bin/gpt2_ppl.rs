//! GPT-2 small forward-pass over WikiText-2 sliding windows.
//!
//! Loads Q15 weights + per-window pre-embedded hidden states (produced by
//! `scripts/export_gpt2.py`), runs `dag.run` only (no commit/prove/verify),
//! and writes the post-final-layernorm hidden states for each window.
//! `scripts/aggregate_gpt2_ppl.py` then applies the LM head + cross-entropy
//! to produce a perplexity number we can compare against the FP32 / Python-Q15
//! references in `/scratch/bjchen4_icgpu/ppl/gpt2_quant.py`.
//!
//! Witness storage convention: zk-torch-2 stores tensors in column-major
//! ("MLE LSB-first") order — for shape [d_0, d_1, ...] the flat index of
//! element (i_0, i_1, ...) is `i_0 + i_1 * d_0_pad + i_2 * d_0_pad * d_1_pad + ...`.
//! See src/dag/mod.rs:88-95. Python writes its tensors in natural row-major
//! order; the loaders below transpose into column-major.

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
use std::fs::{self, File};
use std::io::{Read as IoRead, Write as IoWrite};
use std::path::Path;
use zk_torch_2::{
  dag::{gpt2::gpt_2_small, DagBuilder, DataType, Role, Witness},
  util::poly::CryptoField,
  SF_LOG,
};

const EMBED: usize = 768;
const EMBED_PAD: usize = 1024;
const FF: usize = 3072;
const FF_PAD: usize = 4096;
const N_LAYERS: usize = 12;

fn load_i64_bin<P: AsRef<Path>>(path: P) -> Vec<i64> {
  let mut file = File::open(&path).unwrap_or_else(|e| panic!("open {:?}: {}", path.as_ref(), e));
  let mut buf = Vec::new();
  file.read_to_end(&mut buf).expect("read");
  buf.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn save_i64_bin<P: AsRef<Path>>(path: P, data: &[i64]) {
  let mut bytes = Vec::with_capacity(data.len() * 8);
  for &v in data {
    bytes.extend_from_slice(&v.to_le_bytes());
  }
  let mut f = File::create(&path).unwrap_or_else(|e| panic!("create {:?}: {}", path.as_ref(), e));
  f.write_all(&bytes).expect("write");
}

fn i64_to_field(v: i64) -> F {
  if v >= 0 {
    F::from(v as u64)
  } else {
    -F::from((-v) as u64)
  }
}

/// Pad a 1D tensor [n] -> [n_pad] with trailing zeros.
fn pad_1d(data: &[i64], n: usize, n_pad: usize) -> Vec<F> {
  assert_eq!(data.len(), n);
  let mut out = Vec::with_capacity(n_pad);
  for &v in data {
    out.push(i64_to_field(v));
  }
  for _ in n..n_pad {
    out.push(<F as CryptoField>::zero());
  }
  out
}

/// Pad a 2D tensor [rows, cols] (row-major in `data`) into column-major
/// [rows_pad, cols_pad] storage. zk-torch-2 expects:
///   flat[r + c * rows_pad] = data[r, c]   for r<rows, c<cols
///   flat[...] = 0                          otherwise
fn pad_2d_col_major(data: &[i64], rows: usize, cols: usize, rows_pad: usize, cols_pad: usize) -> Vec<F> {
  assert_eq!(data.len(), rows * cols, "pad_2d_col_major shape mismatch");
  let mut out = vec![<F as CryptoField>::zero(); rows_pad * cols_pad];
  for c in 0..cols {
    for r in 0..rows {
      out[r + c * rows_pad] = i64_to_field(data[r * cols + c]);
    }
  }
  out
}

/// Pad a 3D tensor [b, s, d] (row-major in `data`) into column-major
/// [b_pad, s_pad, d_pad] storage. With b=1 (constant), the b dim is trivial.
///   flat[r + s * b_pad + d * b_pad * s_pad]  for the actual batch/seq/dim
fn pad_3d_col_major(data: &[i64], b: usize, s: usize, d: usize, b_pad: usize, s_pad: usize, d_pad: usize) -> Vec<F> {
  assert_eq!(data.len(), b * s * d);
  let mut out = vec![<F as CryptoField>::zero(); b_pad * s_pad * d_pad];
  for di in 0..d {
    for si in 0..s {
      for bi in 0..b {
        let row_major = (bi * s + si) * d + di;
        out[bi + si * b_pad + di * b_pad * s_pad] = i64_to_field(data[row_major]);
      }
    }
  }
  out
}

fn load_1d(tensor_dir: &str, name: &str, n: usize, n_pad: usize) -> Vec<F> {
  let raw = load_i64_bin(format!("{}/{}.bin", tensor_dir, name));
  pad_1d(&raw, n, n_pad)
}

fn load_2d(tensor_dir: &str, name: &str, rows: usize, cols: usize, rows_pad: usize, cols_pad: usize) -> Vec<F> {
  let raw = load_i64_bin(format!("{}/{}.bin", tensor_dir, name));
  pad_2d_col_major(&raw, rows, cols, rows_pad, cols_pad)
}

#[allow(clippy::type_complexity)]
fn load_gpt2_weights(tensor_dir: &str, seq_len: usize) -> (
  Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>,
  Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>,
  Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>,
  Vec<Witness<F>>, Vec<Witness<F>>, Vec<Witness<F>>,
  Witness<F>, Witness<F>, Witness<F>,
) {
  let mut attn_norm_w = Vec::new();
  let mut attn_norm_b = Vec::new();
  let mut attn_q_w = Vec::new();
  let mut attn_k_w = Vec::new();
  let mut attn_v_w = Vec::new();
  let mut attn_q_b = Vec::new();
  let mut attn_k_b = Vec::new();
  let mut attn_v_b = Vec::new();
  let mut attn_o_w = Vec::new();
  let mut attn_o_b = Vec::new();
  let mut proj_norm_w = Vec::new();
  let mut proj_norm_b = Vec::new();
  let mut proj_1_w = Vec::new();
  let mut proj_1_b = Vec::new();
  let mut proj_2_w = Vec::new();
  let mut proj_2_b = Vec::new();

  for i in 0..N_LAYERS {
    println!("loading layer {i}");
    attn_norm_w.push(Witness::new(
      vec![EMBED],
      load_1d(tensor_dir, &format!("h.{i}.ln_1.weight"), EMBED, EMBED_PAD),
      DataType::Float, *SF_LOG as usize, Role::Constant,
    ));
    attn_norm_b.push(Witness::new(
      vec![EMBED],
      load_1d(tensor_dir, &format!("h.{i}.ln_1.bias"), EMBED, EMBED_PAD),
      DataType::Float, *SF_LOG as usize, Role::Constant,
    ));
    attn_q_w.push(Witness::new(
      vec![EMBED, EMBED],
      load_2d(tensor_dir, &format!("h.{i}.attn.c_attn.weight_q"), EMBED, EMBED, EMBED_PAD, EMBED_PAD),
      DataType::Float, *SF_LOG as usize, Role::Constant,
    ));
    attn_k_w.push(Witness::new(
      vec![EMBED, EMBED],
      load_2d(tensor_dir, &format!("h.{i}.attn.c_attn.weight_k"), EMBED, EMBED, EMBED_PAD, EMBED_PAD),
      DataType::Float, *SF_LOG as usize, Role::Constant,
    ));
    attn_v_w.push(Witness::new(
      vec![EMBED, EMBED],
      load_2d(tensor_dir, &format!("h.{i}.attn.c_attn.weight_v"), EMBED, EMBED, EMBED_PAD, EMBED_PAD),
      DataType::Float, *SF_LOG as usize, Role::Constant,
    ));
    attn_q_b.push(Witness::new(
      vec![EMBED],
      load_1d(tensor_dir, &format!("h.{i}.attn.c_attn.bias_q"), EMBED, EMBED_PAD),
      DataType::Float, *SF_LOG as usize, Role::Constant,
    ));
    attn_k_b.push(Witness::new(
      vec![EMBED],
      load_1d(tensor_dir, &format!("h.{i}.attn.c_attn.bias_k"), EMBED, EMBED_PAD),
      DataType::Float, *SF_LOG as usize, Role::Constant,
    ));
    attn_v_b.push(Witness::new(
      vec![EMBED],
      load_1d(tensor_dir, &format!("h.{i}.attn.c_attn.bias_v"), EMBED, EMBED_PAD),
      DataType::Float, *SF_LOG as usize, Role::Constant,
    ));
    attn_o_w.push(Witness::new(
      vec![EMBED, EMBED],
      load_2d(tensor_dir, &format!("h.{i}.attn.c_proj.weight"), EMBED, EMBED, EMBED_PAD, EMBED_PAD),
      DataType::Float, *SF_LOG as usize, Role::Constant,
    ));
    attn_o_b.push(Witness::new(
      vec![EMBED],
      load_1d(tensor_dir, &format!("h.{i}.attn.c_proj.bias"), EMBED, EMBED_PAD),
      DataType::Float, *SF_LOG as usize, Role::Constant,
    ));
    proj_norm_w.push(Witness::new(
      vec![EMBED],
      load_1d(tensor_dir, &format!("h.{i}.ln_2.weight"), EMBED, EMBED_PAD),
      DataType::Float, *SF_LOG as usize, Role::Constant,
    ));
    proj_norm_b.push(Witness::new(
      vec![EMBED],
      load_1d(tensor_dir, &format!("h.{i}.ln_2.bias"), EMBED, EMBED_PAD),
      DataType::Float, *SF_LOG as usize, Role::Constant,
    ));
    proj_1_w.push(Witness::new(
      vec![EMBED, FF],
      load_2d(tensor_dir, &format!("h.{i}.mlp.c_fc.weight"), EMBED, FF, EMBED_PAD, FF_PAD),
      DataType::Float, *SF_LOG as usize, Role::Constant,
    ));
    proj_1_b.push(Witness::new(
      vec![FF],
      load_1d(tensor_dir, &format!("h.{i}.mlp.c_fc.bias"), FF, FF_PAD),
      DataType::Float, *SF_LOG as usize, Role::Constant,
    ));
    proj_2_w.push(Witness::new(
      vec![FF, EMBED],
      load_2d(tensor_dir, &format!("h.{i}.mlp.c_proj.weight"), FF, EMBED, FF_PAD, EMBED_PAD),
      DataType::Float, *SF_LOG as usize, Role::Constant,
    ));
    proj_2_b.push(Witness::new(
      vec![EMBED],
      load_1d(tensor_dir, &format!("h.{i}.mlp.c_proj.bias"), EMBED, EMBED_PAD),
      DataType::Float, *SF_LOG as usize, Role::Constant,
    ));
  }

  let layer_norm_w = Witness::new(
    vec![EMBED],
    load_1d(tensor_dir, "ln_f.weight", EMBED, EMBED_PAD),
    DataType::Float, *SF_LOG as usize, Role::Constant,
  );
  let layer_norm_b = Witness::new(
    vec![EMBED],
    load_1d(tensor_dir, "ln_f.bias", EMBED, EMBED_PAD),
    DataType::Float, *SF_LOG as usize, Role::Constant,
  );

  // attention_mask is referenced by gpt2_block but not actually used in the
  // attention math (the causal mask is applied via g.causal_mask). Provide a
  // dummy zeroed mask of the right shape.
  let seq_pad = seq_len.next_power_of_two();
  let attention_mask = Witness::new(
    vec![1, seq_len, seq_len],
    vec![<F as CryptoField>::zero(); seq_pad * seq_pad],
    DataType::Float, *SF_LOG as usize, Role::Constant,
  );

  (
    attn_norm_w, attn_q_w, attn_k_w, attn_v_w, attn_o_w,
    attn_norm_b, attn_q_b, attn_k_b, attn_v_b, attn_o_b,
    proj_norm_w, proj_1_w, proj_2_w,
    proj_norm_b, proj_1_b, proj_2_b,
    layer_norm_w, layer_norm_b, attention_mask,
  )
}

/// Convert a Witness's column-major-stored data back to row-major i64 in the
/// original (un-padded) shape, for writing as a `.bin` consumable by Python.
/// w.ndarray() returns the *padded* shape; slice back to the true shape.
fn witness_to_row_major_i64(w: &Witness<F>) -> Vec<i64> {
  let arr = w.ndarray(); // padded row-major ArrayD<i128>
  let true_shape = &w.shape;
  let slice_spec: Vec<ndarray::Slice> = true_shape.iter().map(|&s| ndarray::Slice::from(0..s)).collect();
  let mut view = arr.view();
  for (axis, sl) in slice_spec.iter().enumerate() {
    view.slice_axis_inplace(ndarray::Axis(axis), *sl);
  }
  view.iter().map(|&v| v as i64).collect()
}

fn main() {
  // dag.run inserts sparse-split logic for prove(); skip it here since we
  // never prove. NO_SPARSE_SPLIT=1 keeps witnesses in their canonical form.
  if std::env::var("NO_SPARSE_SPLIT").is_err() {
    std::env::set_var("NO_SPARSE_SPLIT", "1");
  }
  env_logger::init();

  let tensor_dir = std::env::var("TENSOR_DIR").unwrap_or_else(|_| "gpt2/tensors".to_string());
  let inputs_dir = std::env::var("INPUTS_DIR").unwrap_or_else(|_| "gpt2/inputs".to_string());
  let outputs_dir = std::env::var("OUTPUTS_DIR").unwrap_or_else(|_| "gpt2/outputs".to_string());
  let seq_len: usize = std::env::var("SEQ_LEN").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
  let n_windows: usize = std::env::var("N_WINDOWS")
    .ok().and_then(|s| s.parse().ok())
    .expect("N_WINDOWS env var required");
  let window_start: usize = std::env::var("WINDOW_START").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
  let window_end: usize = std::env::var("WINDOW_END").ok().and_then(|s| s.parse().ok()).unwrap_or(n_windows);

  fs::create_dir_all(&outputs_dir).expect("mkdir outputs");

  println!("=== gpt2_ppl: forward-pass-only PPL pipeline ===");
  println!("tensor_dir = {tensor_dir}");
  println!("inputs_dir = {inputs_dir}");
  println!("outputs_dir = {outputs_dir}");
  println!("seq_len = {seq_len}, n_windows = {n_windows}, range = [{window_start}, {window_end})");

  let mut timing = TimingTree::default();

  let mut g = DagBuilder::new();
  let (
    attn_norm_w, attn_q_w, attn_k_w, attn_v_w, attn_o_w,
    attn_norm_b, attn_q_b, attn_k_b, attn_v_b, attn_o_b,
    proj_norm_w, proj_1_w, proj_2_w,
    proj_norm_b, proj_1_b, proj_2_b,
    layer_norm_w, layer_norm_b, attention_mask,
  ) = timed!(timing, "load_weights", load_gpt2_weights(&tensor_dir, seq_len));

  let x = g.input(vec![1, seq_len, EMBED], DataType::Float);
  let output = g.pipe(
    &[x],
    gpt_2_small(
      attn_norm_w, attn_q_w, attn_k_w, attn_v_w, attn_o_w,
      attn_norm_b, attn_q_b, attn_k_b, attn_v_b, attn_o_b,
      proj_norm_w, proj_1_w, proj_2_w,
      proj_norm_b, proj_1_b, proj_2_b,
      layer_norm_w, layer_norm_b, attention_mask,
      seq_len,
    ),
  )[0];

  println!("Compiling DAG...");
  let (dag, mut init) = timed!(timing, "compile", g.compile());

  let seq_pad = seq_len.next_power_of_two();

  for win in window_start..window_end {
    let in_path = format!("{inputs_dir}/window_{win}.bin");
    let out_path = format!("{outputs_dir}/hidden_{win}.bin");
    println!("--- window {win}: {in_path} → {out_path} ---");

    let raw = load_i64_bin(&in_path);
    assert_eq!(
      raw.len(), seq_len * EMBED,
      "window {win}: expected {} elements, got {}", seq_len * EMBED, raw.len()
    );
    let input_data = pad_3d_col_major(&raw, 1, seq_len, EMBED, 1, seq_pad, EMBED_PAD);
    let input = Witness::new(
      vec![1, seq_len, EMBED],
      input_data,
      DataType::Float, *SF_LOG as usize, Role::Input,
    );

    timed!(timing, "run", dag.run(&mut init, &vec![(x, input)]));

    let out_w = &init[output][0];
    println!("  output shape (un-padded): {:?}", out_w.shape);
    let row_major = witness_to_row_major_i64(out_w);
    save_i64_bin(&out_path, &row_major);
    println!("  wrote {} i64 values to {out_path}", row_major.len());
  }

  timing.print();
  println!("=== done ===");
}
