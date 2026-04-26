// nanoGPT DAG builder. Mirrors src/dag/gpt2.rs but configured for the
// ezkl nanoGPT example (examples/onnx/nanoGPT/gen.py):
//   block_size=64, vocab_size=65, n_layer=4, n_head=4, n_embd=64, bias=False
//
// Since bias=False, LayerNorm/attention/MLP projections skip their bias terms.
// GELU is approximated by sigmoid(1.702x) as in gpt2.rs.

use crate::dag::llama::llama_rms_norm;
use crate::util::poly::CryptoField;
use crate::util::shape::pad_to_pow_of_two;
use crate::SF_FLOAT;
use crate::{
  dag::{DagBuilder, DataType, EdgeId, Role, Witness},
  SF_LOG,
};
use ndarray::ArrayD;

pub const NANO_N_LAYER: usize = 4;
pub const NANO_N_HEAD: usize = 4;
pub const NANO_N_EMBD: usize = 64;
pub const NANO_HEAD_DIM: usize = NANO_N_EMBD / NANO_N_HEAD; // 16
pub const NANO_MLP_HIDDEN: usize = 4 * NANO_N_EMBD; // 256
pub const NANO_VOCAB_SIZE: usize = 65;
pub const NANO_BLOCK_SIZE: usize = 64;

pub fn nanogpt_layer_norm<F: CryptoField + 'static>(w_e: EdgeId) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "nanogpt_layer_norm expects 1 input");
    let x = x[0]; // (batch_size, seq_len, n_embd)
    let x_sum = g.einsum("bsi->bs".to_string(), vec![x], false)[0];
    let x_shape = g.init_values[x].as_ref().unwrap().shape.clone();
    let seq = x_shape[1];
    let n = x_shape[x_shape.len() - 1]; // n_embd = 64
    let x_mean = g.div_const(x_sum, n)[0];
    let x_mean = g.change_shape(x_mean, vec![1, seq, 1]);

    let n_param: usize = g.param(Witness::new(vec![1], vec![F::from(n as u32)], DataType::Float, 0, Role::Constant));
    let mean_tolerance = g.param(Witness::new(
      vec![1],
      vec![F::from((n / 2) as u32)],
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    let x_mean_mul_n = g.einsum("bsi,i->bsi".to_string(), vec![x_mean, n_param], false)[0];

    let x_sum_3d = g.change_shape(x_sum, vec![1, seq, 1]);
    let x_sum_sub_x_mean_mul_n = g.sub(x_sum_3d, x_mean_mul_n)[0];
    let positive_1 = g.add(x_sum_sub_x_mean_mul_n, mean_tolerance)[0];
    let positive_2 = g.sub(mean_tolerance, x_sum_sub_x_mean_mul_n)[0];
    g.add_nonneg_node(positive_1);
    g.add_nonneg_node(positive_2);

    let x_minus_mean = g.sub(x, x_mean)[0];
    let x_minus_mean = g.mask(x_minus_mean, vec![1, seq, n]);

    // bias=False: skip the final bias add; the RMS step already applies weight.
    let out = g.pipe(&vec![x_minus_mean], llama_rms_norm(w_e))[0];
    vec![out]
  }
}

pub fn nanogpt_mlp<F: CryptoField + 'static>(
  w_1_e: EdgeId, // (n_embd, 4*n_embd)
  w_2_e: EdgeId, // (4*n_embd, n_embd)
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "nanogpt_mlp expects 1 input");
    let x = x[0]; // (batch_size, seq_len, n_embd)
    let h_1 = g.einsum("bsi,ij->bsj".to_string(), vec![x, w_1_e], true)[0]; // (B, S, 4N)

    // sigmoid(1.702x) GELU approximation (same as gpt2_mlp).
    let shape = g.init_values[h_1].as_ref().unwrap().shape.clone();
    let val_num = shape.iter().fold(1, |acc, x| acc * x);
    let vals: Vec<F> = (0..val_num).map(|_| F::from((1.702 * *SF_FLOAT).round() as u32)).collect();
    let vals = ArrayD::from_shape_vec(shape.clone(), vals).unwrap();
    let pad_vals = pad_to_pow_of_two(&vals, &<F as CryptoField>::zero());
    let col_major: Vec<_> = pad_vals.view().reversed_axes().iter().cloned().collect();
    let constant = Witness::new(shape.clone(), col_major, DataType::Float, *SF_LOG as usize, Role::Constant);
    let constant = g.param(constant);

    let h_1_scaled = g.einsum("bsi,bsi->bsi".to_string(), vec![h_1, constant], true)[0];
    let sigm = g.sigmoid(h_1_scaled)[0];
    let h_1 = g.einsum("bsi,bsi->bsi".to_string(), vec![h_1, sigm], true)[0];

    let h_2 = g.einsum("bsi,ij->bsj".to_string(), vec![h_1, w_2_e], true)[0]; // (B, S, N)
    vec![h_2]
  }
}

pub fn nanogpt_attention<F: CryptoField + 'static>(
  w_q_e: EdgeId, // (n_embd, n_embd)
  w_k_e: EdgeId,
  w_v_e: EdgeId,
  w_o_e: EdgeId,
  n_head: usize,
  head_dim: usize,
  seq_len: usize,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 2, "nanogpt_attention expects 2 inputs");
    let inp = x[0]; // (batch_size, seq_len, n_embd)
    let _attention_mask = x[1];
    let n_embd = n_head * head_dim;

    let q = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_q_e], true)[0];
    let k = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_k_e], true)[0];
    let v = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_v_e], true)[0];

    let q = g.reshape(q, vec![1, seq_len, n_head, head_dim])[0];
    let k = g.reshape(k, vec![1, seq_len, n_head, head_dim])[0];
    let v = g.reshape(v, vec![1, seq_len, n_head, head_dim])[0];

    let q = g.einsum("bshd->bhsd".to_string(), vec![q], false)[0];
    let k = g.einsum("bshd->bhsd".to_string(), vec![k], false)[0];
    let v = g.einsum("bshd->bhsd".to_string(), vec![v], false)[0];

    // 1/sqrt(head_dim) at scale SF_FLOAT.
    let d_sqrt_recip = (*SF_FLOAT as f64 / (head_dim as f64).sqrt()).round() as u32;
    let d_sqrt_recip = g.param(Witness::new(
      vec![1],
      vec![F::from(d_sqrt_recip)],
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    let scores = g.einsum("bhsd,bhtd->bhst".to_string(), vec![q, k], true)[0];
    let scores = g.einsum("bhst,z->bhst".to_string(), vec![scores, d_sqrt_recip], true)[0];
    let scores = if seq_len > 1 { g.causal_mask(scores, seq_len) } else { scores };
    let softmax_c = g.softmax_const(scores)[0];
    let scores = g.add(scores, softmax_c)[0];
    let scores = g.exp(scores)[0];

    let out = g.einsum("bhst,bhtd->bhsd".to_string(), vec![scores, v], true)[0];
    let out = g.einsum("bhsd->bshd".to_string(), vec![out], false)[0];
    let out = g.reshape(out, vec![1, seq_len, n_embd])[0];

    let out = g.einsum("bsi,ij->bsj".to_string(), vec![out, w_o_e], true)[0];
    vec![out]
  }
}

pub fn nanogpt_block<F: CryptoField + 'static>(
  attn_norm_w: Witness<F>,
  attn_q_w: Witness<F>,
  attn_k_w: Witness<F>,
  attn_v_w: Witness<F>,
  attn_o_w: Witness<F>,
  proj_norm_w: Witness<F>,
  proj_1_w: Witness<F>,
  proj_2_w: Witness<F>,
  attention_mask: usize,
  n_head: usize,
  head_dim: usize,
  seq_len: usize,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "nanogpt_block expects 1 input");
    let x = x[0];
    let attn_norm_w = g.param(attn_norm_w);
    let attn_q_w = g.param(attn_q_w);
    let attn_k_w = g.param(attn_k_w);
    let attn_v_w = g.param(attn_v_w);
    let attn_o_w = g.param(attn_o_w);
    let proj_norm_w = g.param(proj_norm_w);
    let proj_1_w = g.param(proj_1_w);
    let proj_2_w = g.param(proj_2_w);

    let attn_norm_out = g.pipe(&vec![x], nanogpt_layer_norm(attn_norm_w));
    let attn_inp = vec![attn_norm_out[0], attention_mask];
    let attn_out = g.pipe(
      &attn_inp,
      nanogpt_attention(attn_q_w, attn_k_w, attn_v_w, attn_o_w, n_head, head_dim, seq_len),
    );
    let residual_attn = g.add(attn_out[0], x)[0];

    let proj_norm_out = g.pipe(&vec![residual_attn], nanogpt_layer_norm(proj_norm_w));
    let proj_out = g.pipe(&proj_norm_out, nanogpt_mlp(proj_1_w, proj_2_w));
    let residual_proj = g.add(proj_out[0], residual_attn)[0];

    vec![residual_proj]
  }
}

// Token embedding via one-hot × weight: input one-hot has shape (B, T, vocab)
// and the embedding weight has shape (vocab, n_embd). Then broadcast-add the
// position embedding (pre-sliced to (1, seq_len, n_embd)).
pub fn nanogpt_embeddings<F: CryptoField + 'static>(
  wte_e: EdgeId,
  pos_emb_e: EdgeId,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "nanogpt_embeddings expects 1 input");
    let onehot = x[0]; // (B, T, vocab)
    let emb = g.einsum("btv,ve->bte".to_string(), vec![onehot, wte_e], true)[0];
    let out = g.add(emb, pos_emb_e)[0];
    vec![out]
  }
}

// Final LM head. Weight-tied to wte (shape (vocab, n_embd)), applied as
// x @ W.T via einsum "bte,ve->btv" to produce (B, T, vocab) logits.
pub fn nanogpt_lm_head<F: CryptoField + 'static>(lm_head_w_e: EdgeId) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "nanogpt_lm_head expects 1 input");
    let x = x[0];
    let out = g.einsum("bte,ve->btv".to_string(), vec![x, lm_head_w_e], true)[0];
    vec![out]
  }
}

pub fn nanogpt<F: CryptoField + 'static>(
  wte: Witness<F>,                  // (vocab_size, n_embd); tied with lm_head
  pos_emb: Witness<F>,              // (1, seq_len, n_embd); = wpe[0:seq_len]
  attn_norm_w_vec: Vec<Witness<F>>, // each (n_embd), length n_layer
  attn_q_w_vec: Vec<Witness<F>>,    // each (n_embd, n_embd)
  attn_k_w_vec: Vec<Witness<F>>,
  attn_v_w_vec: Vec<Witness<F>>,
  attn_o_w_vec: Vec<Witness<F>>,
  proj_norm_w_vec: Vec<Witness<F>>, // each (n_embd)
  proj_1_w_vec: Vec<Witness<F>>,    // each (n_embd, 4*n_embd)
  proj_2_w_vec: Vec<Witness<F>>,    // each (4*n_embd, n_embd)
  layer_norm_w: Witness<F>,         // (n_embd)
  attention_mask: Witness<F>,       // (batch_size, seq_len, seq_len)
  n_layer: usize,
  n_head: usize,
  head_dim: usize,
  seq_len: usize,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "nanogpt expects 1 input");
    let onehot = x[0];
    // Tie wte and lm_head weights (wte.weight is lm_head.weight in PyTorch source).
    let wte_e = g.param(wte);
    let pos_emb_e = g.param(pos_emb);
    let emb_out = g.pipe(&vec![onehot], nanogpt_embeddings(wte_e, pos_emb_e));
    let mut x = emb_out[0];

    let attention_mask = g.param(attention_mask);
    for i in 0..n_layer {
      let block = g.pipe(
        &vec![x],
        nanogpt_block(
          attn_norm_w_vec[i].to_owned(),
          attn_q_w_vec[i].to_owned(),
          attn_k_w_vec[i].to_owned(),
          attn_v_w_vec[i].to_owned(),
          attn_o_w_vec[i].to_owned(),
          proj_norm_w_vec[i].to_owned(),
          proj_1_w_vec[i].to_owned(),
          proj_2_w_vec[i].to_owned(),
          attention_mask,
          n_head,
          head_dim,
          seq_len,
        ),
      );
      x = block[0];
    }

    let layer_norm_w = g.param(layer_norm_w);
    let ln_out = g.pipe(&vec![x], nanogpt_layer_norm(layer_norm_w))[0];

    let logits = g.pipe(&vec![ln_out], nanogpt_lm_head(wte_e))[0];
    vec![logits]
  }
}
