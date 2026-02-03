use crate::dag::llama::llama_rms_norm;
use crate::util::poly::CryptoField;
use crate::util::shape::pad_to_pow_of_two;
use crate::SF_FLOAT;
use crate::{
  dag::{DagBuilder, DataType, EdgeId, Role, Witness},
  SF_LOG,
};
use ndarray::ArrayD;

pub fn gptj_layer_norm<F: CryptoField + 'static>(w_e: EdgeId, b_e: EdgeId) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This custom LayerNorm layer expects 1 input");
    let x = x[0]; // (batch_size, seq_len, 4096)
    let x_sum = g.einsum("bsi->bs".to_string(), vec![x], false)[0]; // (batch_size, seq_len)
    let x_shape = g.init_values[x].as_ref().unwrap().shape.clone();
    let n = x_shape[x_shape.len() - 1]; // 4096
    let x_mean = g.div_const(x_sum, n)[0]; // (batch_size, seq_len)
    let x_mean = g.change_shape(x_mean, vec![1, 1, 1]); // (batch_size, seq_len, 1)
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

    let x_sum_sub_x_mean_mul_n = g.sub(x_sum, x_mean_mul_n)[0]; // (batch_size, seq_len, 1)
    let positive_1 = g.add(x_sum_sub_x_mean_mul_n, mean_tolerance)[0];
    let positive_2 = g.sub(mean_tolerance, x_sum_sub_x_mean_mul_n)[0];
    g.add_nonneg_node(positive_1);
    g.add_nonneg_node(positive_2);

    let x_minus_mean = g.sub(x, x_mean)[0]; // (batch_size, seq_len, 4096)
    let x_minus_mean = g.mask(x_minus_mean, vec![1, 1, 4096]); // (batch_size, seq_len, 4096)

    let x_rms = g.pipe(&vec![x_minus_mean], llama_rms_norm(w_e))[0]; // (batch_size, seq_len, 1)

    let out = g.add(x_rms, b_e)[0]; // (batch_size, seq_len, 4096)
    vec![out]
  }
}

pub fn gptj_mlp<F: CryptoField + 'static>(
  w_1_e: EdgeId, // (4096, 16384)
  w_2_e: EdgeId, // (16384, 4096)
  b_1_e: EdgeId, // (16384)
  b_2_e: EdgeId, // (4096)
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This custom MLP layer expects 1 input");
    let x = x[0]; // (batch_size, seq_len, 4096)
    let h_1 = g.einsum("bsi,ij->bsj".to_string(), vec![x, w_1_e], true)[0]; // (batch_size, seq_len, 16384)
    let h_1 = g.add(h_1, b_1_e)[0]; // (batch_size, seq_len, 16384)

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
    let h_1 = g.einsum("bsi,bsi->bsi".to_string(), vec![h_1, constant], true)[0]; // (batch_size, seq_len, 16384)
                                                                                  // C. approximate gelu by sigmoid(1.702x)
    let h_1 = g.sigmoid(h_1)[0]; // (batch_size, seq_len, 16384)

    let h_2 = g.einsum("bsi,ij->bsj".to_string(), vec![h_1, w_2_e], true)[0]; // (batch_size, seq_len, 4096)
    let h_2 = g.add(h_2, b_2_e)[0]; // (batch_size, seq_len, 4096)
    vec![h_2]
  }
}

pub fn gptj_attention<F: CryptoField + 'static>(
  w_q_e: EdgeId,
  w_k_e: EdgeId,
  w_v_e: EdgeId,
  w_o_e: EdgeId,
  pos: usize,
  _attention_id: usize,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 2, "This custom Attention layer expects 2 inputs"); // [x, attention_mask]
    assert!(pos == 0, "We only support pos == 0 for now (cache is empty)");

    // in gptj onnx, batch_size = 1, seq_len = 1, head_num = 16 and head_dim = 256
    let inp = x[0]; // (batch_size, seq_len, 4096); head_num * head_dim = 4096
                    // use attention_id to index the cache
    let _attention_mask = x[1]; // (batch_size, 2048, 2048)

    let q = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_q_e], true)[0]; // (batch_size, seq_len, 4096)
    let k = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_k_e], true)[0]; // (batch_size, seq_len, 4096)
    let v = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_v_e], true)[0]; // (batch_size, seq_len, 4096)

    let q = g.reshape(q, vec![1, 1, 16, 256])[0];
    let k = g.reshape(k, vec![1, 1, 16, 256])[0];
    let v = g.reshape(v, vec![1, 1, 16, 256])[0];

    // apply rope embedding
    let q = g.rope(q, pos)[0];
    let k = g.rope(k, pos)[0];

    // (batch_size, seq_len, head_num, head_dim) -> (batch_size, head_num, seq_len, head_dim)
    let q = g.einsum("bshd->bhsd".to_string(), vec![q], false)[0];
    let k = g.einsum("bshd->bhsd".to_string(), vec![k], false)[0];
    let v = g.einsum("bshd->bhsd".to_string(), vec![v], false)[0];

    // (batch_size, head_num, seq_len, head_dim) * (batch_size, head_num, head_dim, Seq_len) -> (batch_size, head_num, seq_len, Seq_len)
    let d_sqrt_recip = ((*SF_FLOAT as f64) / (256.0_f64.sqrt())).round() as u32;
    let d_sqrt_recip = g.param(Witness::new(
      vec![1],
      vec![F::from(d_sqrt_recip)],
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    let scores = g.einsum("bhsd,bhtd->bhst".to_string(), vec![q, k], true)[0];
    let scores = g.einsum("bhst,s->bhst".to_string(), vec![scores, d_sqrt_recip], true)[0];
    // TODO: we skip the mask for now because we only support pos == 0 for now (cache is empty)
    let softmax_c = g.softmax_const(scores)[0]; // (batch_size, head_num, seq_len, Seq_len)
    let scores = g.add(scores, softmax_c)[0]; // (batch_size, head_num, seq_len, Seq_len)
    let scores = g.exp(scores)[0]; // (batch_size, head_num, seq_len, Seq_len)

    // TODO: check if scores sum to 1

    // (batch_size, head_num, seq_len, Seq_len) * (batch_size, head_num, Seq_len, head_dim) -> (batch_size, head_num, seq_len, head_dim)
    let out = g.einsum("bhst,bhtd->bhsd".to_string(), vec![scores, v], true)[0];
    let out = g.change_shape(out, vec![1, 1, 16, 256]);
    let out = g.reshape(out, vec![1, 1, 4096])[0];

    let out = g.einsum("bsi,ij->bsj".to_string(), vec![out, w_o_e], true)[0]; // (batch_size, seq_len, 4096)
    vec![out]
  }
}

pub fn gptj_block<F: CryptoField + 'static>(
  // attn
  attn_norm_w: Witness<F>,
  attn_q_w: Witness<F>,
  attn_k_w: Witness<F>,
  attn_v_w: Witness<F>,
  attn_o_w: Witness<F>,
  attn_norm_b: Witness<F>,
  // proj
  proj_1_w: Witness<F>,
  proj_2_w: Witness<F>,
  proj_1_b: Witness<F>,
  proj_2_b: Witness<F>,
  attention_mask: usize,
  attention_id: usize,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This custom Block layer expects 1 input");
    let x = x[0]; // (batch_size, seq_len, 4096)
    let attn_norm_w = g.param(attn_norm_w);
    let attn_q_w = g.param(attn_q_w);
    let attn_k_w = g.param(attn_k_w);
    let attn_v_w = g.param(attn_v_w);
    let attn_o_w = g.param(attn_o_w);
    let attn_norm_b = g.param(attn_norm_b);
    let proj_1_w = g.param(proj_1_w);
    let proj_2_w = g.param(proj_2_w);
    let proj_1_b = g.param(proj_1_b);
    let proj_2_b = g.param(proj_2_b);

    let attn_norm_out = g.pipe(&vec![x], gptj_layer_norm(attn_norm_w, attn_norm_b));
    let attn_inp = vec![attn_norm_out[0], attention_mask];
    let attn_out = g.pipe(
      &attn_inp,
      gptj_attention(
        attn_q_w,
        attn_k_w,
        attn_v_w,
        attn_o_w,
        0, // pos
        attention_id,
      ),
    );
    let mlp_inp = vec![attn_norm_out[0]];
    let mlp_out = g.pipe(&mlp_inp, gptj_mlp(proj_1_w, proj_2_w, proj_1_b, proj_2_b));
    let combine_attn_mlp = g.add(attn_out[0], mlp_out[0])[0]; // (batch_size, seq_len, 4096)

    let residual = g.add(combine_attn_mlp, x)[0]; // (batch_size, seq_len, 4096)

    vec![residual]
  }
}

pub fn gpt_j_6b<F: CryptoField + 'static>(
  // attn
  attn_norm_w_vec: Vec<Witness<F>>, // each element is (4096), length is 12
  attn_q_w_vec: Vec<Witness<F>>,    // each element is (4096, 4096), length is 12
  attn_k_w_vec: Vec<Witness<F>>,    // each element is (4096, 4096), length is 12
  attn_v_w_vec: Vec<Witness<F>>,    // each element is (4096, 4096), length is 12
  attn_o_w_vec: Vec<Witness<F>>,    // each element is (4096, 4096), length is 12
  attn_norm_b_vec: Vec<Witness<F>>, // each element is (4096), length is 12
  // proj
  proj_1_w_vec: Vec<Witness<F>>, // each element is (4096, 16384), length is 12
  proj_2_w_vec: Vec<Witness<F>>, // each element is (16384, 4096), length is 12
  proj_1_b_vec: Vec<Witness<F>>, // each element is (16384), length is 12
  proj_2_b_vec: Vec<Witness<F>>, // each element is (4096), length is 12
  layer_norm_w: Witness<F>,      // it is (4096)
  layer_norm_b: Witness<F>,      // it is (4096)
  // matmul
  matmul_w: Witness<F>,       // it is (4096, 50400)
  matmul_b: Witness<F>,       // it is (50400)
  attention_mask: Witness<F>, // it is (batch_size, seq_len, seq_len)
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This custom GPT-2 Small layer expects 1 input");
    let mut x = x[0]; // (batch_size, seq_len, 4096)
    let attention_mask = g.param(attention_mask);
    for i in 0..28 {
      // 28 layers in GPT-J
      // attn
      let attn_norm_w = attn_norm_w_vec[i].to_owned();
      let attn_q_w = attn_q_w_vec[i].to_owned();
      let attn_k_w = attn_k_w_vec[i].to_owned();
      let attn_v_w = attn_v_w_vec[i].to_owned();
      let attn_o_w = attn_o_w_vec[i].to_owned();
      let attn_norm_b = attn_norm_b_vec[i].to_owned();
      // proj
      let proj_1_w = proj_1_w_vec[i].to_owned();
      let proj_2_w = proj_2_w_vec[i].to_owned();
      let proj_1_b = proj_1_b_vec[i].to_owned();
      let proj_2_b = proj_2_b_vec[i].to_owned();
      let block = g.pipe(
        &vec![x],
        gptj_block(
          attn_norm_w,
          attn_q_w,
          attn_k_w,
          attn_v_w,
          attn_o_w,
          attn_norm_b,
          proj_1_w,
          proj_2_w,
          proj_1_b,
          proj_2_b,
          attention_mask,
          i,
        ),
      );
      x = block[0]; // (batch_size, seq_len, 4096)
    }
    let layer_norm_w = g.param(layer_norm_w);
    let layer_norm_b = g.param(layer_norm_b);
    let layer_norm_out = g.pipe(&vec![x], gptj_layer_norm(layer_norm_w, layer_norm_b)); // (batch_size, seq_len, 4096)
    let matmul_w = g.param(matmul_w);
    let matmul_b = g.param(matmul_b);
    let matmul_out = g.einsum("bsi,ij->bsj".to_string(), vec![layer_norm_out[0], matmul_w], true)[0]; // (batch_size, seq_len, 50400)
    let matmul_out = g.add(matmul_out, matmul_b)[0]; // (batch_size, seq_len, 50400)
    vec![matmul_out]
  }
}
