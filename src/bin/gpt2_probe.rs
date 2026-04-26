//! Isolated GPT-2 forward-pass probe: run ONLY the first N transformer blocks,
//! optionally stopping before attention. Used to triangulate where accuracy
//! drift appears vs the reference Python Q15 pipeline.
//!
//! Env vars:
//!   PROBE=ln1_0          -> output after layer 0's ln_1
//!   PROBE=block_N        -> output after block N (0-indexed) full block
//!   PROBE=final          -> output after final ln_f of a block stack
//!   N_BLOCKS=N           -> # of blocks to stack before the final LN (used when PROBE=final)
//! Reads inputs/targets from INPUTS_DIR/window_0.bin and writes OUTPUTS_DIR/probe.bin.

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

use std::fs::File;
use std::io::{Read as IoRead, Write as IoWrite};
use std::path::Path;
use zk_torch_2::{
  dag::{
    gpt2::{gpt2_attention, gpt2_block, gpt2_layer_norm, gpt2_mlp},
    DagBuilder, DataType, Role, Witness,
  },
  util::poly::CryptoField,
  SF_LOG,
};

const EMBED: usize = 768;
const EMBED_PAD: usize = 1024;
const FF: usize = 3072;
const FF_PAD: usize = 4096;

fn load_i64<P: AsRef<Path>>(p: P) -> Vec<i64> {
  let mut f = File::open(p).expect("open");
  let mut buf = Vec::new();
  f.read_to_end(&mut buf).expect("read");
  buf.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
}
fn save_i64<P: AsRef<Path>>(p: P, d: &[i64]) {
  let mut b = Vec::with_capacity(d.len() * 8);
  for &v in d { b.extend_from_slice(&v.to_le_bytes()); }
  File::create(p).unwrap().write_all(&b).unwrap();
}
fn i64f(v: i64) -> F { if v >= 0 { F::from(v as u64) } else { -F::from((-v) as u64) } }
fn pad1(d: &[i64], n: usize, np: usize) -> Vec<F> {
  let mut o = Vec::with_capacity(np);
  for &v in d { o.push(i64f(v)); }
  for _ in n..np { o.push(<F as CryptoField>::zero()); }
  o
}
fn pad2(d: &[i64], r: usize, c: usize, rp: usize, cp: usize) -> Vec<F> {
  let mut o = vec![<F as CryptoField>::zero(); rp * cp];
  for cc in 0..c {
    for rr in 0..r {
      o[rr + cc * rp] = i64f(d[rr * c + cc]);
    }
  }
  o
}
fn pad3(d: &[i64], b: usize, s: usize, dd: usize, bp: usize, sp: usize, dp: usize) -> Vec<F> {
  let mut o = vec![<F as CryptoField>::zero(); bp * sp * dp];
  for di in 0..dd {
    for si in 0..s {
      for bi in 0..b {
        o[bi + si * bp + di * bp * sp] = i64f(d[(bi * s + si) * dd + di]);
      }
    }
  }
  o
}

fn load_const_1d(dir: &str, name: &str, n: usize, np: usize) -> Witness<F> {
  Witness::new(vec![n], pad1(&load_i64(format!("{dir}/{name}.bin")), n, np), DataType::Float, *SF_LOG as usize, Role::Constant)
}
fn load_const_2d(dir: &str, name: &str, r: usize, c: usize, rp: usize, cp: usize) -> Witness<F> {
  Witness::new(vec![r, c], pad2(&load_i64(format!("{dir}/{name}.bin")), r, c, rp, cp), DataType::Float, *SF_LOG as usize, Role::Constant)
}

fn witness_to_row_major_i64(w: &Witness<F>) -> Vec<i64> {
  let arr = w.ndarray();
  let true_shape = &w.shape;
  let mut view = arr.view();
  for (axis, &s) in true_shape.iter().enumerate() {
    view.slice_axis_inplace(ndarray::Axis(axis), ndarray::Slice::from(0..s));
  }
  view.iter().map(|&v| v as i64).collect()
}

fn main() {
  std::env::set_var("NO_SPARSE_SPLIT", "1");
  env_logger::init();

  let tensor_dir = std::env::var("TENSOR_DIR").unwrap_or_else(|_| "gpt2_smoke/tensors".into());
  let inputs_dir = std::env::var("INPUTS_DIR").unwrap_or_else(|_| "gpt2_smoke/inputs".into());
  let outputs_dir = std::env::var("OUTPUTS_DIR").unwrap_or_else(|_| "gpt2_smoke/outputs".into());
  let probe = std::env::var("PROBE").unwrap_or_else(|_| "ln1_0".into());
  let seq_len: usize = std::env::var("SEQ_LEN").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
  let n_blocks: usize = std::env::var("N_BLOCKS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);

  std::fs::create_dir_all(&outputs_dir).unwrap();
  let seq_pad = seq_len.next_power_of_two();
  println!("probe={probe}, seq_len={seq_len}, n_blocks={n_blocks}");

  let mut g = DagBuilder::<F>::new();
  let x = g.input(vec![1, seq_len, EMBED], DataType::Float);

  let attention_mask = {
    let w = Witness::new(
      vec![1, seq_len, seq_len],
      vec![<F as CryptoField>::zero(); seq_pad * seq_pad],
      DataType::Float, *SF_LOG as usize, Role::Constant,
    );
    g.param(w)
  };

  let out = match probe.as_str() {
    "ln1_0" => {
      let w = g.param(load_const_1d(&tensor_dir, "h.0.ln_1.weight", EMBED, EMBED_PAD));
      let b = g.param(load_const_1d(&tensor_dir, "h.0.ln_1.bias", EMBED, EMBED_PAD));
      g.pipe(&vec![x], gpt2_layer_norm(w, b))[0]
    }
    "attn_only_0" | "after_attn_0" | "after_ln2_0" | "after_mlp_only_0" | "block_0" => {
      let layer_params = |g: &mut DagBuilder<F>, i: usize| {
        (
          load_const_1d(&tensor_dir, &format!("h.{i}.ln_1.weight"), EMBED, EMBED_PAD),
          load_const_2d(&tensor_dir, &format!("h.{i}.attn.c_attn.weight_q"), EMBED, EMBED, EMBED_PAD, EMBED_PAD),
          load_const_2d(&tensor_dir, &format!("h.{i}.attn.c_attn.weight_k"), EMBED, EMBED, EMBED_PAD, EMBED_PAD),
          load_const_2d(&tensor_dir, &format!("h.{i}.attn.c_attn.weight_v"), EMBED, EMBED, EMBED_PAD, EMBED_PAD),
          load_const_2d(&tensor_dir, &format!("h.{i}.attn.c_proj.weight"), EMBED, EMBED, EMBED_PAD, EMBED_PAD),
          load_const_1d(&tensor_dir, &format!("h.{i}.ln_1.bias"), EMBED, EMBED_PAD),
          load_const_1d(&tensor_dir, &format!("h.{i}.attn.c_attn.bias_q"), EMBED, EMBED_PAD),
          load_const_1d(&tensor_dir, &format!("h.{i}.attn.c_attn.bias_k"), EMBED, EMBED_PAD),
          load_const_1d(&tensor_dir, &format!("h.{i}.attn.c_attn.bias_v"), EMBED, EMBED_PAD),
          load_const_1d(&tensor_dir, &format!("h.{i}.attn.c_proj.bias"), EMBED, EMBED_PAD),
          load_const_1d(&tensor_dir, &format!("h.{i}.ln_2.weight"), EMBED, EMBED_PAD),
          load_const_2d(&tensor_dir, &format!("h.{i}.mlp.c_fc.weight"), EMBED, FF, EMBED_PAD, FF_PAD),
          load_const_2d(&tensor_dir, &format!("h.{i}.mlp.c_proj.weight"), FF, EMBED, FF_PAD, EMBED_PAD),
          load_const_1d(&tensor_dir, &format!("h.{i}.ln_2.bias"), EMBED, EMBED_PAD),
          load_const_1d(&tensor_dir, &format!("h.{i}.mlp.c_fc.bias"), FF, FF_PAD),
          load_const_1d(&tensor_dir, &format!("h.{i}.mlp.c_proj.bias"), EMBED, EMBED_PAD),
        )
      };
      let (an_w, q_w, k_w, v_w, o_w, an_b, q_b, k_b, v_b, o_b, pn_w, p1_w, p2_w, pn_b, p1_b, p2_b) = layer_params(&mut g, 0);
      let an_w_e = g.param(an_w);
      let q_w_e = g.param(q_w); let k_w_e = g.param(k_w); let v_w_e = g.param(v_w); let o_w_e = g.param(o_w);
      let an_b_e = g.param(an_b);
      let q_b_e = g.param(q_b); let k_b_e = g.param(k_b); let v_b_e = g.param(v_b); let o_b_e = g.param(o_b);
      let pn_w_e = g.param(pn_w);
      let p1_w_e = g.param(p1_w); let p2_w_e = g.param(p2_w);
      let pn_b_e = g.param(pn_b);
      let p1_b_e = g.param(p1_b); let p2_b_e = g.param(p2_b);

      let ln1 = g.pipe(&vec![x], gpt2_layer_norm(an_w_e, an_b_e))[0];
      let attn = g.pipe(&vec![ln1, attention_mask], gpt2_attention(
        q_w_e, k_w_e, v_w_e, o_w_e, q_b_e, k_b_e, v_b_e, o_b_e, seq_len, 0,
      ))[0];
      if probe == "attn_only_0" { attn } else {
        let resid_attn = g.add(attn, x)[0];
        if probe == "after_attn_0" { resid_attn } else {
          let ln2 = g.pipe(&vec![resid_attn], gpt2_layer_norm(pn_w_e, pn_b_e))[0];
          if probe == "after_ln2_0" { ln2 } else {
            let mlp = g.pipe(&vec![ln2], gpt2_mlp(p1_w_e, p2_w_e, p1_b_e, p2_b_e))[0];
            if probe == "after_mlp_only_0" { mlp } else {
              // "block_0": full block = mlp + residual
              g.add(mlp, resid_attn)[0]
            }
          }
        }
      }
    }
    other => panic!("unknown PROBE: {other}"),
  };

  println!("compiling...");
  let (dag, mut init) = g.compile();

  let raw_in = load_i64(format!("{inputs_dir}/window_0.bin"));
  assert_eq!(raw_in.len(), seq_len * EMBED);
  let input = Witness::new(
    vec![1, seq_len, EMBED],
    pad3(&raw_in, 1, seq_len, EMBED, 1, seq_pad, EMBED_PAD),
    DataType::Float, *SF_LOG as usize, Role::Input,
  );

  println!("running...");
  dag.run(&mut init, &vec![(x, input)]);

  let out_w = &init[out][0];
  println!("out shape {:?}", out_w.shape);
  let row_major = witness_to_row_major_i64(out_w);
  let out_path = format!("{outputs_dir}/probe.bin");
  save_i64(&out_path, &row_major);
  println!("wrote {} values to {out_path}", row_major.len());
}
