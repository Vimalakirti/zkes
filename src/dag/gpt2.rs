use crate::dag::llama::llama_rms_norm;
use crate::util::poly::CryptoField;
use crate::util::shape::pad_to_pow_of_two;
use crate::SF_FLOAT;
use crate::{
  dag::{DagBuilder, DataType, EdgeId, Role, Witness},
  SF_LOG,
};
use ndarray::ArrayD;

pub fn gpt2_layer_norm<F: CryptoField + 'static>(w_e: EdgeId, b_e: EdgeId) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This custom LayerNorm layer expects 1 input");
    let x = x[0]; // (batch_size, seq_len, 768)
    let x_sum = g.einsum("bsi->bs".to_string(), vec![x], false)[0]; // (batch_size, seq_len)
    let x_shape = g.init_values[x].as_ref().unwrap().shape.clone();
    let seq = x_shape[1]; // seq_len (dynamic)
    let n = x_shape[x_shape.len() - 1]; // 768
    let x_mean = g.div_const(x_sum, n)[0]; // (batch_size, seq_len)
    let x_mean = g.change_shape(x_mean, vec![1, seq, 1]); // (batch_size, seq_len, 1)
                                                          // Check x_mean
    let n_param: usize = g.param(Witness::new(vec![1], vec![F::from(n as u32)], DataType::Float, 0, Role::Constant));
    let mean_tolerance = g.param(Witness::new(
      vec![1],
      vec![F::from((n / 2) as u32)],
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    let x_mean_mul_n = g.einsum("bsi,i->bsi".to_string(), vec![x_mean, n_param], false)[0]; // (batch_size, seq_len, 1)

    // Reshape x_sum from (batch, seq) → (batch, seq, 1) so broadcast matches x_mean_mul_n
    let x_sum_3d = g.change_shape(x_sum, vec![1, seq, 1]);
    let x_sum_sub_x_mean_mul_n = g.sub(x_sum_3d, x_mean_mul_n)[0]; // (batch_size, seq_len, 1)
    let positive_1 = g.add(x_sum_sub_x_mean_mul_n, mean_tolerance)[0];
    let positive_2 = g.sub(mean_tolerance, x_sum_sub_x_mean_mul_n)[0];
    g.add_nonneg_node(positive_1);
    g.add_nonneg_node(positive_2);

    let x_minus_mean = g.sub(x, x_mean)[0]; // (batch_size, seq_len, 768)
    let x_minus_mean = g.mask(x_minus_mean, vec![1, seq, 768]); // (batch_size, seq_len, 768)

    let x_rms = g.pipe(&vec![x_minus_mean], llama_rms_norm(w_e))[0]; // (batch_size, seq_len, 1)

    let out = g.add(x_rms, b_e)[0]; // (batch_size, seq_len, 768)
    vec![out]
  }
}

pub fn gpt2_mlp<F: CryptoField + 'static>(
  w_1_e: EdgeId, // (768, 3072)
  w_2_e: EdgeId, // (3072, 768)
  b_1_e: EdgeId, // (3072)
  b_2_e: EdgeId, // (768)
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This custom MLP layer expects 1 input");
    let x = x[0]; // (batch_size, seq_len, 768)
    let h_1 = g.einsum("bsi,ij->bsj".to_string(), vec![x, w_1_e], true)[0]; // (batch_size, seq_len, 3072)
    let h_1_x = g.add(h_1, b_1_e)[0]; // (batch_size, seq_len, 3072)

    // compute Gelu
    // A. compute 1.702
    let shape = g.init_values[h_1].as_ref().unwrap().shape.clone();
    let val_num = &shape.iter().fold(1, |acc, x| acc * x);
    let vals = (0..*val_num).map(|_| F::from((1.702 * *SF_FLOAT).round() as u32)).collect(); // 1.702 * SF_FLOAT
    let vals = ArrayD::from_shape_vec(shape.clone(), vals).unwrap();
    let pad_vals = pad_to_pow_of_two(&vals, &<F as CryptoField>::zero());
    let col_major: Vec<_> = pad_vals.clone().view().reversed_axes().iter().cloned().collect();
    let constant = Witness::new(shape.clone(), col_major, DataType::Float, *SF_LOG as usize, Role::Constant);
    let constant = g.param(constant);
    // B. compute 1.702x
    let h_1 = g.einsum("bsi,bsi->bsi".to_string(), vec![h_1_x, constant], true)[0]; // (batch_size, seq_len, 3072)
                                                                                    // C. approximate gelu by sigmoid(1.702x)
    let h_1 = g.sigmoid(h_1)[0]; // (batch_size, seq_len, 3072)
    let h_1 = g.einsum("bsi,bsi->bsi".to_string(), vec![h_1_x, h_1], true)[0]; // (batch_size, seq_len, 3072)

    let h_2 = g.einsum("bsi,ij->bsj".to_string(), vec![h_1, w_2_e], true)[0]; // (batch_size, seq_len, 768)
    let h_2 = g.add(h_2, b_2_e)[0]; // (batch_size, seq_len, 768)
    vec![h_2]
  }
}

pub fn gpt2_attention<F: CryptoField + 'static>(
  w_q_e: EdgeId, // (768, 768)
  w_k_e: EdgeId, // (768, 768)
  w_v_e: EdgeId, // (768, 768)
  w_o_e: EdgeId, // (768, 768)
  b_q_e: EdgeId, // (768)
  b_k_e: EdgeId, // (768)
  b_v_e: EdgeId, // (768)
  b_o_e: EdgeId, // (768)
  seq_len: usize,
  _attention_id: usize,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 2, "This custom Attention layer expects 2 inputs"); // [x, attention_mask]

    // in gpt-2 onnx, batch_size = 1, head_num = 12 and head_dim = 64
    let inp = x[0]; // (batch_size, seq_len, 768); head_num * head_dim = 768
    let _attention_mask = x[1]; // (batch_size, seq_len, seq_len)

    let q = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_q_e], true)[0]; // (batch_size, seq_len, 768)
    let k = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_k_e], true)[0]; // (batch_size, seq_len, 768)
    let v = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_v_e], true)[0]; // (batch_size, seq_len, 768)

    let q = g.add(q, b_q_e)[0]; // (batch_size, seq_len, 768)
    let k = g.add(k, b_k_e)[0]; // (batch_size, seq_len, 768)
    let v = g.add(v, b_v_e)[0]; // (batch_size, seq_len, 768)

    let q = g.reshape(q, vec![1, seq_len, 12, 64])[0];
    let k = g.reshape(k, vec![1, seq_len, 12, 64])[0];
    let v = g.reshape(v, vec![1, seq_len, 12, 64])[0];

    // (batch_size, seq_len, head_num, head_dim) -> (batch_size, head_num, seq_len, head_dim)
    let q = g.einsum("bshd->bhsd".to_string(), vec![q], false)[0];
    let k = g.einsum("bshd->bhsd".to_string(), vec![k], false)[0];
    let v = g.einsum("bshd->bhsd".to_string(), vec![v], false)[0];

    // (batch_size, head_num, seq_len, head_dim) * (batch_size, head_num, head_dim, Seq_len) -> (batch_size, head_num, seq_len, Seq_len)
    let d_sqrt_recip = (*SF_FLOAT as f64 / 8.0_f64).round() as u32;
    let d_sqrt_recip = g.param(Witness::new(
      vec![1],
      vec![F::from(d_sqrt_recip)],
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    let scores = g.einsum("bhsd,bhtd->bhst".to_string(), vec![q, k], true)[0];
    let scores = g.einsum("bhst,z->bhst".to_string(), vec![scores, d_sqrt_recip], true)[0];
    // Apply causal mask for seq_len > 1 (GPT-2 is autoregressive)
    let scores = if seq_len > 1 { g.causal_mask(scores, seq_len) } else { scores };
    // Log-sum-exp softmax: softmax_c is the per-row -logsumexp advice, so exp(scores + c)
    // equals the Q15 softmax directly — no 1/Σ rescale needed. Soundness in a full prove
    // path rests on enforcing Σ exp(scores + c) ≈ SF (softmax sums to 1).
    let softmax_c = g.softmax_const(scores)[0];
    let scores = g.add(scores, softmax_c)[0];
    let scores = g.exp(scores)[0]; // Q15 softmax

    let out = g.einsum("bhst,bhtd->bhsd".to_string(), vec![scores, v], true)[0];
    // Permute (b,h,s,d) -> (b,s,h,d) so data layout matches the shape label before
    // `reshape` flattens the last two dims. Without this, `change_shape` would only
    // relabel while leaving h/s strides mismatched (h_pad=16 vs seq_pad≥64).
    let out = g.einsum("bhsd->bshd".to_string(), vec![out], false)[0];
    let out = g.reshape(out, vec![1, seq_len, 768])[0];

    let out = g.einsum("bsi,ij->bsj".to_string(), vec![out, w_o_e], true)[0]; // (batch_size, seq_len, 4096)
    let out = g.add(out, b_o_e)[0]; // (batch_size, seq_len, 768)
    vec![out]
  }
}

pub fn gpt2_block<F: CryptoField + 'static>(
  // attn
  attn_norm_w: Witness<F>,
  attn_q_w: Witness<F>,
  attn_k_w: Witness<F>,
  attn_v_w: Witness<F>,
  attn_o_w: Witness<F>,
  attn_norm_b: Witness<F>,
  attn_q_b: Witness<F>,
  attn_k_b: Witness<F>,
  attn_v_b: Witness<F>,
  attn_o_b: Witness<F>,
  // proj
  proj_norm_w: Witness<F>,
  proj_1_w: Witness<F>,
  proj_2_w: Witness<F>,
  proj_norm_b: Witness<F>,
  proj_1_b: Witness<F>,
  proj_2_b: Witness<F>,
  attention_mask: usize,
  attention_id: usize,
  seq_len: usize,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This custom Block layer expects 1 input");
    let x = x[0]; // (batch_size, seq_len, 768)
    let attn_norm_w = g.param(attn_norm_w);
    let attn_q_w = g.param(attn_q_w);
    let attn_k_w = g.param(attn_k_w);
    let attn_v_w = g.param(attn_v_w);
    let attn_o_w = g.param(attn_o_w);
    let attn_norm_b = g.param(attn_norm_b);
    let attn_q_b = g.param(attn_q_b);
    let attn_k_b = g.param(attn_k_b);
    let attn_v_b = g.param(attn_v_b);
    let attn_o_b = g.param(attn_o_b);
    let proj_norm_w = g.param(proj_norm_w);
    let proj_1_w = g.param(proj_1_w);
    let proj_2_w = g.param(proj_2_w);
    let proj_norm_b = g.param(proj_norm_b);
    let proj_1_b = g.param(proj_1_b);
    let proj_2_b = g.param(proj_2_b);

    // h.{i}.ln_1.weight, h.{i}.ln_1.bias
    let attn_norm_out = g.pipe(&vec![x], gpt2_layer_norm(attn_norm_w, attn_norm_b));
    let attn_inp = vec![attn_norm_out[0], attention_mask];
    // h.{i}.attn.c_attn.weight_(q, k, v), h.{i}.attn.c_attn.bias_(q, k, v)
    // h.{i}.attn.c_proj.weight (o), h.{i}.attn.c_proj.bias (o)
    let attn_out = g.pipe(
      &attn_inp,
      gpt2_attention(
        attn_q_w,
        attn_k_w,
        attn_v_w,
        attn_o_w,
        attn_q_b,
        attn_k_b,
        attn_v_b,
        attn_o_b,
        seq_len,
        attention_id,
      ),
    );
    let residual_attn = g.add(attn_out[0], x)[0]; // (batch_size, seq_len, 768)

    // h.{i}.ln_2.weight, h.{i}.ln_2.bias
    let proj_norm_out = g.pipe(&vec![residual_attn], gpt2_layer_norm(proj_norm_w, proj_norm_b));
    // h.{i}.mlp.c_fc.weight (proj_1), h.{i}.mlp.c_fc.bias (proj_1)
    // h.{i}.mlp.c_proj.weight (proj_2), h.{i}.mlp.c_proj.bias (proj_2)
    let proj_out = g.pipe(&proj_norm_out, gpt2_mlp(proj_1_w, proj_2_w, proj_1_b, proj_2_b));
    let residual_proj = g.add(proj_out[0], residual_attn)[0]; // (batch_size, seq_len, 768)

    vec![residual_proj]
  }
}

pub fn gpt_2_small<F: CryptoField + 'static>(
  // attn
  attn_norm_w_vec: Vec<Witness<F>>, // each element is (768), length is 12
  attn_q_w_vec: Vec<Witness<F>>,    // each element is (768, 768), length is 12
  attn_k_w_vec: Vec<Witness<F>>,    // each element is (768, 768), length is 12
  attn_v_w_vec: Vec<Witness<F>>,    // each element is (768, 768), length is 12
  attn_o_w_vec: Vec<Witness<F>>,    // each element is (768, 768), length is 12
  attn_norm_b_vec: Vec<Witness<F>>, // each element is (768), length is 12
  attn_q_b_vec: Vec<Witness<F>>,    // each element is (768), length is 12
  attn_k_b_vec: Vec<Witness<F>>,    // each element is (768), length is 12
  attn_v_b_vec: Vec<Witness<F>>,    // each element is (768), length is 12
  attn_o_b_vec: Vec<Witness<F>>,    // each element is (768), length is 12
  // proj
  proj_norm_w_vec: Vec<Witness<F>>, // each element is (768), length is 12
  proj_1_w_vec: Vec<Witness<F>>,    // each element is (768, 3072), length is 12
  proj_2_w_vec: Vec<Witness<F>>,    // each element is (3072, 768), length is 12
  proj_norm_b_vec: Vec<Witness<F>>, // each element is (768), length is 12
  proj_1_b_vec: Vec<Witness<F>>,    // each element is (3072), length is 12
  proj_2_b_vec: Vec<Witness<F>>,    // each element is (768), length is 12
  layer_norm_w: Witness<F>,         // it is (768)
  layer_norm_b: Witness<F>,         // it is (768)
  attention_mask: Witness<F>,       // it is (batch_size, seq_len, seq_len)
  seq_len: usize,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This custom GPT-2 Small layer expects 1 input");
    let mut x = x[0]; // (batch_size, seq_len, 768)
    let attention_mask = g.param(attention_mask);
    for i in 0..12 {
      // 12 layers in GPT-2 Small
      // attn
      let attn_norm_w = attn_norm_w_vec[i].to_owned();
      let attn_q_w = attn_q_w_vec[i].to_owned();
      let attn_k_w = attn_k_w_vec[i].to_owned();
      let attn_v_w = attn_v_w_vec[i].to_owned();
      let attn_o_w = attn_o_w_vec[i].to_owned();
      let attn_norm_b = attn_norm_b_vec[i].to_owned();
      let attn_q_b = attn_q_b_vec[i].to_owned();
      let attn_k_b = attn_k_b_vec[i].to_owned();
      let attn_v_b = attn_v_b_vec[i].to_owned();
      let attn_o_b = attn_o_b_vec[i].to_owned();
      // proj
      let proj_norm_w = proj_norm_w_vec[i].to_owned();
      let proj_1_w = proj_1_w_vec[i].to_owned();
      let proj_2_w = proj_2_w_vec[i].to_owned();
      let proj_norm_b = proj_norm_b_vec[i].to_owned();
      let proj_1_b = proj_1_b_vec[i].to_owned();
      let proj_2_b = proj_2_b_vec[i].to_owned();
      let block = g.pipe(
        &vec![x],
        gpt2_block(
          attn_norm_w,
          attn_q_w,
          attn_k_w,
          attn_v_w,
          attn_o_w,
          attn_norm_b,
          attn_q_b,
          attn_k_b,
          attn_v_b,
          attn_o_b,
          proj_norm_w,
          proj_1_w,
          proj_2_w,
          proj_norm_b,
          proj_1_b,
          proj_2_b,
          attention_mask,
          i,
          seq_len,
        ),
      );
      x = block[0]; // (batch_size, seq_len, 768)
    }

    // ln_f.weight, ln_f.bias
    let layer_norm_w = g.param(layer_norm_w);
    let layer_norm_b = g.param(layer_norm_b);
    let layer_norm_out = g.pipe(&vec![x], gpt2_layer_norm(layer_norm_w, layer_norm_b)); // (batch_size, seq_len, 768)
    vec![layer_norm_out[0]]
  }
}
