use crate::basicblock::BasicBlockType;
use crate::basicblock::{DivConst, RMSReciprocal, SoftmaxConst};
use crate::util::poly::CryptoField;
use crate::SF_FLOAT;
use crate::{
  dag::{DagBuilder, DataType, EdgeId, Role, Witness},
  SF_LOG,
};
use ndarray::Array2;

pub fn pair_swap_perm_matrix<F: CryptoField + 'static>(d: usize) -> Vec<F> {
  assert!(d % 2 == 0, "dimension d must be even");

  let mut m = Array2::from_elem((d, d), F::from(0));

  for i in (0..d).step_by(2) {
    m[[i, i + 1]] = <F as CryptoField>::one(); // y_i     = x_{i+1}
    m[[i + 1, i]] = <F as CryptoField>::one(); // y_{i+1} = x_i
  }

  m.into_dimensionality::<ndarray::Ix2>()
    .unwrap() // Still Array2, but ensures shape
    .into_dyn()
    .clone()
    .view()
    .reversed_axes()
    .iter()
    .map(|f| *f)
    .collect::<Vec<F>>()
}

/// Generate the cosine vector for RoPE:
/// [cos(mθ1), cos(mθ1), cos(mθ2), cos(mθ2), ...]
pub fn rope_cos_vec<F: CryptoField + 'static>(d: usize, m: f64, base: f64) -> Vec<F> {
  assert!(d % 2 == 0, "d must be even");
  let half = d / 2;
  let mut v = Vec::with_capacity(d);

  for i in 0..half {
    // θ_i = base^{-2i/d}
    let theta_i = base.powf(-2.0 * (i as f64) / (d as f64));
    let angle = m * theta_i;
    let c = angle.cos();
    let c = (c * *SF_FLOAT as f64).round() as i64;
    let c = if c > 0 {
      F::from(c as u32)
    } else {
      <F as CryptoField>::zero() - F::from(-c as u32)
    };
    v.push(c); // first time
    v.push(c); // repeat
  }

  v
}

/// Generate the sine vector for RoPE:
/// [sin(mθ1), -sin(mθ1), sin(mθ2), -sin(mθ2), ...]
pub fn rope_sin_vec<F: CryptoField + 'static>(d: usize, m: f64, base: f64) -> Vec<F> {
  assert!(d % 2 == 0, "d must be even");
  let half = d / 2;
  let mut v = Vec::with_capacity(d);

  for i in 0..half {
    // θ_i = base^{-2i/d}
    let theta_i = base.powf(-2.0 * (i as f64) / (d as f64));
    let angle = m * theta_i;
    let s = angle.sin();
    v.push((s * *SF_FLOAT as f64).round() as i64); // first time
    v.push((-s * *SF_FLOAT as f64).round() as i64); // second time
  }

  v.iter()
    .map(|f| {
      if *f > 0 {
        F::from(*f as u32)
      } else {
        <F as CryptoField>::zero() - F::from(-*f as u32)
      }
    })
    .collect::<Vec<F>>()
}

// good reference: https://github.com/aju22/LLaMA2/blob/main/model.py
impl<F: CryptoField + 'static> DagBuilder<F> {
  pub fn rms_reciprocal(&mut self, x: EdgeId) -> Vec<EdgeId> {
    let rms_reciprocal_basicblock = BasicBlockType::RMSReciprocal(RMSReciprocal);

    if self.init_values[x].is_some() {
      let mut shape = self.init_values[x].as_ref().unwrap().shape.clone();
      let shape_len = shape.len();
      let data_type = self.init_values[x].as_ref().unwrap().data_type;
      let sf = self.init_values[x].as_ref().unwrap().sf;
      shape[shape_len - 1] = 1;
      let out_value = Witness::new_wo_data(shape, data_type, sf, Role::Auxiliary);
      self.init_values.push(Some(out_value));
    } else {
      self.init_values.push(None);
    }

    self.add_gkr_node(vec![x], rms_reciprocal_basicblock)
  }

  pub fn div_const(&mut self, a: EdgeId, c: usize) -> Vec<EdgeId> {
    let div_const_basicblock = BasicBlockType::DivConst(DivConst { c: c });
    assert!(self.init_values[a].is_some(), "Input must be initialized");
    let inp_value = self.init_values[a].as_ref().unwrap();
    let shape = inp_value.shape.clone();
    let sf = inp_value.sf;
    let data_type = inp_value.data_type;
    let out_value = Witness::new_wo_data(shape, data_type, sf, Role::Output);
    self.init_values.push(Some(out_value));
    self.add_gkr_node(vec![a], div_const_basicblock)
  }

  pub fn softmax_const(&mut self, a: EdgeId) -> Vec<EdgeId> {
    let softmax_const_basicblock = BasicBlockType::SoftmaxConst(SoftmaxConst);
    assert!(self.init_values[a].is_some(), "Input must be initialized");
    let inp_value = self.init_values[a].as_ref().unwrap();
    let shape = inp_value.shape.clone();
    let sf = inp_value.sf;
    let data_type = inp_value.data_type;
    let out_value = Witness::new_wo_data(shape, data_type, sf, Role::Output);
    self.init_values.push(Some(out_value));
    self.add_gkr_node(vec![a], softmax_const_basicblock)
  }

  pub fn rope(&mut self, a: EdgeId, pos: usize) -> Vec<EdgeId> {
    assert!(self.init_values[a].is_some(), "Input must be initialized");
    let inp_value = self.init_values[a].as_ref().unwrap();
    let d = inp_value.shape[inp_value.shape.len() - 1];
    let perm_matrix = pair_swap_perm_matrix::<F>(d);
    let perm_matrix = Witness::new(vec![d, d], perm_matrix, DataType::Float, 0, Role::Constant);
    let perm_matrix = self.param(perm_matrix);
    let sin_branch = self.einsum("bshd,de->bshe".to_string(), vec![a, perm_matrix], false)[0];
    let cos_branch = a;

    let sin_param = rope_sin_vec::<F>(d, pos as f64, 10000.0);
    let cos_param = rope_cos_vec::<F>(d, pos as f64, 10000.0);
    let sin_param = Witness::new(vec![1, d], sin_param, DataType::Float, *SF_LOG as usize, Role::Constant);
    let cos_param = Witness::new(vec![1, d], cos_param, DataType::Float, *SF_LOG as usize, Role::Constant);
    let sin_param = self.param(sin_param);
    let cos_param = self.param(cos_param);
    let sin_branch = self.einsum("bshd,sd->bshd".to_string(), vec![sin_branch, sin_param], true)[0];
    let cos_branch = self.einsum("bshd,sd->bshd".to_string(), vec![cos_branch, cos_param], true)[0];
    let out = self.add(sin_branch, cos_branch)[0];
    vec![out]
  }
}

pub fn llama_rms_norm<F: CryptoField + 'static>(w_e: EdgeId) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This custom RMSNorm layer expects 1 input");
    let x = x[0]; // (batch_size, seq_len, 4096)
    let x_shape = g.init_values[x].as_ref().unwrap().shape.clone();
    let n = x_shape[x_shape.len() - 1]; // 4096
    let r = g.rms_reciprocal(x)[0]; // (batch_size, seq_len, 1)

    // Prove r is correctly computed
    let sf = g.param(Witness::new(
      vec![1],
      vec![F::from(*SF_FLOAT as u32)],
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    let tolerance = g.param(Witness::new(vec![1], vec![F::from(2)], DataType::Float, *SF_LOG as usize, Role::Constant));
    let x_sq = g.einsum("bsi,bsi->bsi".to_string(), vec![x, x], true)[0]; // (batch_size, seq_len, 4096)
    let x_sum = g.einsum("bsi->bs".to_string(), vec![x_sq], false)[0]; // (batch_size, seq_len)
    let x_mean = g.div_const(x_sum, n)[0]; // (batch_size, seq_len)
    let x_mean = g.change_shape(x_mean, vec![1, 1, 1]); // (batch_size, seq_len, 1)
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

    let r_sq = g.einsum("bsi,bsi->bsi".to_string(), vec![r, r], true)[0]; // (batch_size, seq_len, 1)
    let z = g.einsum("bsi,bsi->bsi".to_string(), vec![x_mean, r_sq], true)[0]; // (batch_size, seq_len, 1)
    let z_sf_diff = g.sub(z, sf)[0];
    let positive_3 = g.add(z_sf_diff, tolerance)[0];
    let positive_4 = g.sub(tolerance, z_sf_diff)[0];
    g.add_nonneg_node(positive_3);
    g.add_nonneg_node(positive_4);

    // Compute RMSNorm
    let r = g.change_shape(r, vec![1, 1]); // (batch_size, seq_len)
    let h = g.einsum("bsi,bs->bsi".to_string(), vec![x, r], true)[0]; // (batch_size, seq_len, 4096)
    let out = g.einsum("bsi,i->bsi".to_string(), vec![h, w_e], true)[0]; // (batch_size, seq_len, 4096)
    vec![out]
  }
}

pub fn llama_mlp<F: CryptoField + 'static>(w_1_e: EdgeId, w_2_e: EdgeId, w_3_e: EdgeId) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This custom MLP layer expects 1 input");
    let x = x[0]; // (batch_size, seq_len, 4096)
    let h_1 = g.einsum("bsi,ij->bsj".to_string(), vec![x, w_1_e], true)[0]; // (batch_size, seq_len, 11008)
    let sigmoid = g.sigmoid(h_1)[0]; // (batch_size, seq_len, 11008)
    let swish = g.einsum("bsi,bsi->bsi".to_string(), vec![h_1, sigmoid], true)[0]; // (batch_size, seq_len, 11008)
    let h_2 = g.einsum("bsi,ij->bsj".to_string(), vec![x, w_2_e], true)[0]; // (batch_size, seq_len, 11008)
    let mul = g.einsum("bsi,bsi->bsi".to_string(), vec![swish, h_2], true)[0]; // (batch_size, seq_len, 11008)
    let out = g.einsum("bsi,ij->bsj".to_string(), vec![mul, w_3_e], true)[0]; // (batch_size, seq_len, 4096)
    vec![out]
  }
}

pub fn llama_attention<F: CryptoField + 'static>(
  w_q_e: EdgeId,
  w_k_e: EdgeId,
  w_v_e: EdgeId,
  w_o_e: EdgeId,
  pos: usize,
  _attention_id: usize,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 4, "This custom Attention layer expects 4 inputs"); // [x, k_cache, v_cache, attention_mask]
    assert!(pos == 0, "We only support pos == 0 for now (cache is empty)");

    // in llama2-7b onnx, batch_size = 1, seq_len = 1, head_num = 32 and head_dim = 128
    let inp = x[0]; // (batch_size, seq_len, 4096); head_num * head_dim = 4096
                    // use attention_id to index the cache
    let _k_cache = x[1]; // (batch_size, attention_num, pos, head_num, head_dim)
    let _v_cache = x[2]; // (batch_size, attention_num, pos, head_num, head_dim)
    let _attention_mask = x[3]; // (batch_size, 2048, 2048)

    let q = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_q_e], true)[0]; // (batch_size, seq_len, 4096)
    let k = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_k_e], true)[0]; // (batch_size, seq_len, 4096)
    let v = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_v_e], true)[0]; // (batch_size, seq_len, 4096)

    let q = g.reshape(q, vec![1, 1, 32, 128])[0];
    let k = g.reshape(k, vec![1, 1, 32, 128])[0];
    let v = g.reshape(v, vec![1, 1, 32, 128])[0];

    // apply rope embedding
    let q = g.rope(q, pos)[0];
    let k = g.rope(k, pos)[0];

    // (batch_size, seq_len, head_num, head_dim) -> (batch_size, head_num, seq_len, head_dim)
    let q = g.einsum("bshd->bhsd".to_string(), vec![q], false)[0];
    let k = g.einsum("bshd->bhsd".to_string(), vec![k], false)[0];
    let v = g.einsum("bshd->bhsd".to_string(), vec![v], false)[0];

    // (batch_size, head_num, seq_len, head_dim) * (batch_size, head_num, head_dim, Seq_len) -> (batch_size, head_num, seq_len, Seq_len)
    let d_sqrt_recip = ((*SF_FLOAT as f64) / (128.0_f64.sqrt())).round() as u32;
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
    let out = g.change_shape(out, vec![1, 1, 32, 128]);
    let out = g.reshape(out, vec![1, 1, 4096])[0];

    let out = g.einsum("bsi,ij->bsj".to_string(), vec![out, w_o_e], true)[0]; // (batch_size, seq_len, 4096)
    vec![out]
  }
}

pub fn llama_block<F: CryptoField + 'static>(
  attn_norm_w: Witness<F>,
  attn_q_w: Witness<F>,
  attn_k_w: Witness<F>,
  attn_v_w: Witness<F>,
  attn_o_w: Witness<F>,
  proj_norm_w: Witness<F>,
  proj_1_w: Witness<F>,
  proj_2_w: Witness<F>,
  proj_3_w: Witness<F>,
  k_cache: usize,
  v_cache: usize,
  attention_mask: usize,
  pos: usize,
  attention_id: usize,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This custom Block layer expects 1 input");
    let x = x[0]; // (batch_size, seq_len, 4096)
    let attn_norm = g.param(attn_norm_w);
    let attn_q = g.param(attn_q_w);
    let attn_k = g.param(attn_k_w);
    let attn_v = g.param(attn_v_w);
    let attn_o = g.param(attn_o_w);
    let proj_norm = g.param(proj_norm_w);
    let proj_1 = g.param(proj_1_w);
    let proj_2 = g.param(proj_2_w);
    let proj_3 = g.param(proj_3_w);

    let attn_norm_out = g.pipe(&vec![x], llama_rms_norm(attn_norm));
    let attn_inp = vec![attn_norm_out[0], k_cache, v_cache, attention_mask];
    let attn_out = g.pipe(&attn_inp, llama_attention(attn_q, attn_k, attn_v, attn_o, pos, attention_id));
    let residual_attn = g.add(attn_out[0], x)[0]; // (batch_size, seq_len, 4096)

    let proj_norm_out = g.pipe(&vec![residual_attn], llama_rms_norm(proj_norm));
    let proj_out = g.pipe(&proj_norm_out, llama_mlp(proj_1, proj_2, proj_3));
    let residual_proj = g.add(proj_out[0], residual_attn)[0]; // (batch_size, seq_len, 4096)

    vec![residual_proj]
  }
}

pub fn llama_2_7b<F: CryptoField + 'static>(
  attn_norm_w_vec: Vec<Witness<F>>, // each element is (4096), length is 32
  attn_q_w_vec: Vec<Witness<F>>,    // each element is (4096, 4096), length is 32
  attn_k_w_vec: Vec<Witness<F>>,    // each element is (4096, 4096), length is 32
  attn_v_w_vec: Vec<Witness<F>>,    // each element is (4096, 4096), length is 32
  attn_o_w_vec: Vec<Witness<F>>,    // each element is (4096, 4096), length is 32
  proj_norm_w_vec: Vec<Witness<F>>, // each element is (4096), length is 32
  proj_1_w_vec: Vec<Witness<F>>,    // each element is (4096, 11008), length is 32
  proj_2_w_vec: Vec<Witness<F>>,    // each element is (4096, 11008), length is 32
  proj_3_w_vec: Vec<Witness<F>>,    // each element is (11008, 4096), length is 32
  layer_norm_w: Witness<F>,         // it is (4096)
  logits_w: Witness<F>,             // it is (4096, 32000)
  k_cache: Witness<F>,              // it is (batch_size, attention_num, pos, head_num, head_dim)
  v_cache: Witness<F>,              // it is (batch_size, attention_num, pos, head_num, head_dim)
  attention_mask: Witness<F>,       // it is (batch_size, 2048, 2048)
  pos: usize,                       // it is the position of the current token
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This custom LLaMA-2-7B layer expects 1 input");
    let mut x = x[0]; // (batch_size, seq_len, 4096)
    let k_cache = g.param(k_cache);
    let v_cache = g.param(v_cache);
    let attention_mask = g.param(attention_mask);
    for i in 0..32 {
      // 32 layers in LLaMA-2-7B
      let attn_norm_w = attn_norm_w_vec[i].to_owned();
      let attn_q_w = attn_q_w_vec[i].to_owned();
      let attn_k_w = attn_k_w_vec[i].to_owned();
      let attn_v_w = attn_v_w_vec[i].to_owned();
      let attn_o_w = attn_o_w_vec[i].to_owned();
      let proj_norm_w = proj_norm_w_vec[i].to_owned();
      let proj_1_w = proj_1_w_vec[i].to_owned();
      let proj_2_w = proj_2_w_vec[i].to_owned();
      let proj_3_w = proj_3_w_vec[i].to_owned();
      let block = g.pipe(
        &vec![x],
        llama_block(
          attn_norm_w,
          attn_q_w,
          attn_k_w,
          attn_v_w,
          attn_o_w,
          proj_norm_w,
          proj_1_w,
          proj_2_w,
          proj_3_w,
          k_cache,
          v_cache,
          attention_mask,
          pos,
          i,
        ),
      );
      x = block[0]; // (batch_size, seq_len, 4096)
    }
    let layer_norm_w = g.param(layer_norm_w);
    let layer_norm_out = g.pipe(&vec![x], llama_rms_norm(layer_norm_w)); // (batch_size, seq_len, 4096)
    let logits_w = g.param(logits_w); // (4096, 32000)
    let out = g.einsum("bij,jk->ik".to_string(), vec![layer_norm_out[0], logits_w], true)[0]; // (batch_size, seq_len, 32000)
    let out = g.change_shape(out, vec![1, 32000]); // (batch_size, 32000)
    vec![out]
  }
}
