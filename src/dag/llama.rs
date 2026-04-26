use crate::basicblock::BasicBlockType;
use crate::basicblock::{DivConst, Reciprocal, RMSReciprocal, SoftmaxConst};
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

  /// r[i] = round(SF^2 / x[i]), same shape as x. In Q15: r/SF ≈ 1/x_real.
  /// Used for softmax normalization (and any div-by-tensor), paired with `rescale(y * r)`.
  pub fn reciprocal(&mut self, a: EdgeId) -> Vec<EdgeId> {
    assert!(self.init_values[a].is_some(), "Input must be initialized");
    let inp_value = self.init_values[a].as_ref().unwrap();
    let shape = inp_value.shape.clone();
    let sf = inp_value.sf;
    let data_type = inp_value.data_type;
    let reciprocal_basicblock = BasicBlockType::Reciprocal(Reciprocal);
    let out_value = Witness::new_wo_data(shape.clone(), data_type, sf, Role::Auxiliary);
    self.init_values.push(Some(out_value));
    let r = self.add_gkr_node(vec![a], reciprocal_basicblock)[0];

    // Reciprocity soundness check: bind r to a via |a*r/SF - SF| <= tolerance.
    // Without it, Reciprocal::verify is a no-op and a malicious prover could
    // supply any r. Mirrors the tolerance check in `llama_rms_norm` for
    // RMSReciprocal. The extra einsum also provides a claim on `a`, which the
    // upstream op needs to pop off its output-claim list.
    assert!(shape.len() <= 26, "reciprocal: tensor rank too large for einsum letters");
    let letters: String = (0..shape.len()).map(|i| (b'a' + i as u8) as char).collect();
    let eq = format!("{0},{0}->{0}", letters);
    let sf_const = self.param(Witness::new(
      vec![1],
      vec![F::from(*SF_FLOAT as u32)],
      DataType::Float,
      sf,
      Role::Constant,
    ));
    // Tolerance matches the RMSReciprocal bound: 512 at SF_LOG=15 ≈ 1.5% of SF.
    // This accommodates round-off in r = round(SF^2/a) and accumulated scale_back error.
    let tolerance = self.param(Witness::new(
      vec![1],
      vec![F::from(512u32)],
      DataType::Float,
      sf,
      Role::Constant,
    ));
    let z = self.einsum(eq, vec![a, r], true)[0]; // a*r rescaled to sf
    let z_diff = self.sub(z, sf_const)[0];
    let positive_1 = self.add(z_diff, tolerance)[0];
    let positive_2 = self.sub(tolerance, z_diff)[0];
    self.add_nonneg_node(positive_1);
    self.add_nonneg_node(positive_2);

    vec![r]
  }

  pub fn softmax_const(&mut self, a: EdgeId) -> Vec<EdgeId> {
    assert!(self.init_values[a].is_some(), "Input must be initialized");
    let inp_value = self.init_values[a].as_ref().unwrap();
    let shape = inp_value.shape.clone();
    let sf = inp_value.sf;
    let data_type = inp_value.data_type;
    let dim = *shape.last().unwrap();
    let softmax_const_basicblock = BasicBlockType::SoftmaxConst(SoftmaxConst { dim });
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

  /// RoPE using pre-computed cos/sin matrices of shape [seq_len, d].
  /// Unlike `rope()`, this handles all sequence positions at once.
  pub fn rope_with_vecs(&mut self, a: EdgeId, cos_param: EdgeId, sin_param: EdgeId) -> Vec<EdgeId> {
    assert!(self.init_values[a].is_some(), "Input must be initialized");
    let inp_value = self.init_values[a].as_ref().unwrap();
    let d = inp_value.shape[inp_value.shape.len() - 1];
    let perm_matrix = pair_swap_perm_matrix::<F>(d);
    let perm_matrix = Witness::new(vec![d, d], perm_matrix, DataType::Float, 0, Role::Constant);
    let perm_matrix = self.param(perm_matrix);
    let sin_branch = self.einsum("bshd,de->bshe".to_string(), vec![a, perm_matrix], false)[0];
    let cos_branch = a;

    let sin_branch = self.einsum("bshd,sd->bshd".to_string(), vec![sin_branch, sin_param], true)[0];
    let cos_branch = self.einsum("bshd,sd->bshd".to_string(), vec![cos_branch, cos_param], true)[0];
    let out = self.add(sin_branch, cos_branch)[0];
    vec![out]
  }

  /// Add a lower-triangular causal attention mask to `scores`.
  /// Scores shape must end with [..., seq_len, seq_len].
  /// Future positions (key > query) get a large negative value so
  /// exp(score) ≈ 0 after softmax.
  pub fn causal_mask(&mut self, scores: EdgeId, seq_len: usize) -> EdgeId {
    let scores_shape = self.init_values[scores].as_ref().unwrap().shape.clone();
    let sf = self.init_values[scores].as_ref().unwrap().sf;
    let padded: Vec<usize> = scores_shape.iter().map(|&s| s.next_power_of_two()).collect();
    let total_padded: usize = padded.iter().product();

    // Must dominate the largest plausible raw attention score. GPT-2 block 4
    // head 11 is an outlier "attention-sink" head with scores reaching ±223
    // after Q·Kᵀ / √d, so the earlier -100·SF mask left masked entries above
    // the unmasked row max and leaked future tokens. -1000·SF guarantees
    // masked positions saturate the downstream ExpHelper clamp to ≈ 0.
    let big_neg_val = 1000u32 * (1u32 << sf);
    let big_neg: F = <F as CryptoField>::zero() - <F as CryptoField>::from_u32(big_neg_val);

    // Last two dims are [s, t] (query pos, key pos).
    // MLE layout: s has stride = product of all padded dims before it,
    //             t has stride = s_stride * s_pad.
    let ndim = scores_shape.len();
    let s_pad = padded[ndim - 2];
    let t_pad = padded[ndim - 1];
    let s_stride: usize = padded[..ndim - 2].iter().product();
    let t_stride = s_stride * s_pad;

    let mut mask_data = vec![<F as CryptoField>::zero(); total_padded];
    let num_groups: usize = padded[..ndim - 2].iter().product();
    for grp in 0..num_groups {
      for t in 0..t_pad {
        for s in 0..s_pad {
          let idx = grp + s * s_stride + t * t_stride;
          if t < seq_len && s < seq_len && t > s {
            mask_data[idx] = big_neg;
          }
        }
      }
    }

    let mask = Witness::new(scores_shape, mask_data, DataType::Float, sf, Role::Constant);
    let mask_id = self.param(mask);
    self.add(scores, mask_id)[0]
  }
}

/// Generate cosine matrix for RoPE: shape [seq_len, d].
/// Data in MLE order (s has stride 1, d has stride seq_padded).
pub fn rope_cos_mat<F: CryptoField + 'static>(d: usize, seq_len: usize, base: f64) -> Vec<F> {
  assert!(d % 2 == 0, "d must be even");
  let half = d / 2;
  let seq_padded = seq_len.next_power_of_two().max(1);
  let total = seq_padded * d; // d is already pow2 for head_dim=128
  let mut data = vec![<F as CryptoField>::zero(); total];

  for i in 0..half {
    let theta_i = base.powf(-2.0 * (i as f64) / (d as f64));
    for m in 0..seq_len {
      let angle = m as f64 * theta_i;
      let c = (angle.cos() * *SF_FLOAT as f64).round() as i64;
      let c_f = if c >= 0 {
        <F as CryptoField>::from_u32(c as u32)
      } else {
        <F as CryptoField>::zero() - <F as CryptoField>::from_u32((-c) as u32)
      };
      data[m + (2 * i) * seq_padded] = c_f;
      data[m + (2 * i + 1) * seq_padded] = c_f;
    }
  }
  data
}

/// Generate sine matrix for RoPE: shape [seq_len, d].
/// Data in MLE order (s has stride 1, d has stride seq_padded).
/// Pattern: sin[2i] = +sin(angle), sin[2i+1] = -sin(angle).
pub fn rope_sin_mat<F: CryptoField + 'static>(d: usize, seq_len: usize, base: f64) -> Vec<F> {
  assert!(d % 2 == 0, "d must be even");
  let half = d / 2;
  let seq_padded = seq_len.next_power_of_two().max(1);
  let total = seq_padded * d;
  let mut data = vec![<F as CryptoField>::zero(); total];

  for i in 0..half {
    let theta_i = base.powf(-2.0 * (i as f64) / (d as f64));
    for m in 0..seq_len {
      let angle = m as f64 * theta_i;
      let s = angle.sin();
      let pos_val = (s * *SF_FLOAT as f64).round() as i64;
      let neg_val = (-s * *SF_FLOAT as f64).round() as i64;
      let to_f = |v: i64| -> F {
        if v >= 0 {
          <F as CryptoField>::from_u32(v as u32)
        } else {
          <F as CryptoField>::zero() - <F as CryptoField>::from_u32((-v) as u32)
        }
      };
      data[m + (2 * i) * seq_padded] = to_f(pos_val);
      data[m + (2 * i + 1) * seq_padded] = to_f(neg_val);
    }
  }
  data
}

pub fn llama_rms_norm<F: CryptoField + 'static>(w_e: EdgeId) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This custom RMSNorm layer expects 1 input");
    let x = x[0]; // (batch_size, seq_len, hidden_dim)
    let x_shape = g.init_values[x].as_ref().unwrap().shape.clone();
    let seq = x_shape[1]; // seq_len (was hardcoded to 1)
    let n = x_shape[x_shape.len() - 1]; // hidden_dim
    let r = g.rms_reciprocal(x)[0]; // (batch_size, seq_len, 1)

    // Prove r is correctly computed
    let sf = g.param(Witness::new(
      vec![1],
      vec![F::from(*SF_FLOAT as u32)],
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    // Tolerance for |z - sf| where z = mean_sq * r^2 / sf^2. See comment on `r_sq` below:
    // a single end-of-chain ScaleDown preserves precision even for small r, so 512 is
    // ample headroom (bounds the prover's r within ~±25% of the true 1/rms).
    let tolerance = g.param(Witness::new(vec![1], vec![F::from(512u32)], DataType::Float, *SF_LOG as usize, Role::Constant));
    let x_sq = g.einsum("bsi,bsi->bsi".to_string(), vec![x, x], true)[0];
    let x_sum = g.einsum("bsi->bs".to_string(), vec![x_sq], false)[0]; // (batch_size, seq_len)
    let x_mean = g.div_const(x_sum, n)[0]; // (batch_size, seq_len)
    let x_mean = g.change_shape(x_mean, vec![1, seq, 1]); // (batch_size, seq_len, 1)
    let n_param: usize = g.param(Witness::new(vec![1], vec![F::from(n as u32)], DataType::Float, 0, Role::Constant));
    let mean_tolerance = g.param(Witness::new(
      vec![1],
      vec![F::from((n / 2) as u32)],
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    let x_mean_mul_n = g.einsum("bsi,i->bsi".to_string(), vec![x_mean, n_param], false)[0];

    // Reshape x_sum from (batch, seq) → (batch, seq, 1) so broadcast matches x_mean_mul_n
    let x_sum = g.change_shape(x_sum, vec![1, seq, 1]);
    let x_sum_sub_x_mean_mul_n = g.sub(x_sum, x_mean_mul_n)[0];
    let positive_1 = g.add(x_sum_sub_x_mean_mul_n, mean_tolerance)[0];
    let positive_2 = g.sub(mean_tolerance, x_sum_sub_x_mean_mul_n)[0];
    g.add_nonneg_node(positive_1);
    g.add_nonneg_node(positive_2);

    // Compute r_sq WITHOUT intermediate ScaleDown: rescaling r*r by sf would underflow
    // to 0 whenever r_sf^2 < sf/2 (i.e., rms > ~sqrt(2*sf) in real units), which is common
    // in deep residual stacks (LLaMA-2-7B hits this at ~layer 10+). Instead we keep r_sq
    // at scale sf^2 and let the outer ScaleDown (rescale_sf = 2*SF_LOG) produce z at sf
    // scale in a single rounding step, preserving all intermediate precision.
    let r_sq = g.einsum("bsi,bsi->bsi".to_string(), vec![r, r], false)[0];
    let z = g.einsum("bsi,bsi->bsi".to_string(), vec![x_mean, r_sq], true)[0];
    let z_sf_diff = g.sub(z, sf)[0];
    let positive_3 = g.add(z_sf_diff, tolerance)[0];
    let positive_4 = g.sub(tolerance, z_sf_diff)[0];
    g.add_nonneg_node(positive_3);
    g.add_nonneg_node(positive_4);

    // Compute RMSNorm
    let r = g.change_shape(r, vec![1, seq]); // (batch_size, seq_len)
    let h = g.einsum("bsi,bs->bsi".to_string(), vec![x, r], true)[0];
    let out = g.einsum("bsi,i->bsi".to_string(), vec![h, w_e], true)[0];
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
  num_heads: usize,
  head_dim: usize,
  seq_len: usize,
  cos_param: EdgeId,
  sin_param: EdgeId,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This custom Attention layer expects 1 input");

    let inp = x[0]; // (batch_size, seq_len, hidden_dim)

    let q = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_q_e], true)[0];
    let k = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_k_e], true)[0];
    let v = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_v_e], true)[0];

    let q = g.reshape(q, vec![1, seq_len, num_heads, head_dim])[0];
    let k = g.reshape(k, vec![1, seq_len, num_heads, head_dim])[0];
    let v = g.reshape(v, vec![1, seq_len, num_heads, head_dim])[0];

    let q = g.rope_with_vecs(q, cos_param, sin_param)[0];
    let k = g.rope_with_vecs(k, cos_param, sin_param)[0];

    let q = g.einsum("bshd->bhsd".to_string(), vec![q], false)[0];
    let k = g.einsum("bshd->bhsd".to_string(), vec![k], false)[0];
    let v = g.einsum("bshd->bhsd".to_string(), vec![v], false)[0];

    let d_sqrt_recip = ((*SF_FLOAT as f64) / ((head_dim as f64).sqrt())).round() as u32;
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
    // Log-sum-exp softmax: softmax_c is the per-row -logsumexp advice, so exp(scores + c)
    // is the Q15 softmax directly. Soundness check (sum ≈ SF) belongs in a later pass.
    let softmax_c = g.softmax_const(scores)[0];
    let scores = g.add(scores, softmax_c)[0];
    let scores = g.exp(scores)[0]; // Q15 softmax

    let out = g.einsum("bhst,bhtd->bhsd".to_string(), vec![scores, v], true)[0];
    // Permute (b,h,s,d) -> (b,s,h,d) so the subsequent reshape flattens h/d in the
    // correct order. A bare `change_shape` would only relabel, mixing up strides.
    let out = g.einsum("bhsd->bshd".to_string(), vec![out], false)[0];
    let out = g.reshape(out, vec![1, seq_len, num_heads * head_dim])[0];

    let out = g.einsum("bsi,ij->bsj".to_string(), vec![out, w_o_e], true)[0];
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
  num_heads: usize,
  head_dim: usize,
  seq_len: usize,
  cos_param: EdgeId,
  sin_param: EdgeId,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This custom Block layer expects 1 input");
    let x = x[0];
    let attn_norm = g.param(attn_norm_w);
    let attn_q = g.param(attn_q_w);
    let attn_k = g.param(attn_k_w);
    let attn_v = g.param(attn_v_w);
    let attn_o = g.param(attn_o_w);
    let proj_norm = g.param(proj_norm_w);
    let proj_1 = g.param(proj_1_w);
    let proj_2 = g.param(proj_2_w);
    let proj_3 = g.param(proj_3_w);

    let attn_norm_out = g.pipe(&[x], llama_rms_norm(attn_norm));
    let attn_out = g.pipe(
      &[attn_norm_out[0]],
      llama_attention(attn_q, attn_k, attn_v, attn_o, num_heads, head_dim, seq_len, cos_param, sin_param),
    );
    let residual_attn = g.add(attn_out[0], x)[0];

    let proj_norm_out = g.pipe(&[residual_attn], llama_rms_norm(proj_norm));
    let proj_out = g.pipe(&proj_norm_out, llama_mlp(proj_1, proj_2, proj_3));
    let residual_proj = g.add(proj_out[0], residual_attn)[0];

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
  logits_w: Witness<F>,             // it is (4096, vocab_size)
  num_heads: usize,
  head_dim: usize,
  seq_len: usize,
  vocab_size: usize,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This custom LLaMA-2-7B layer expects 1 input");
    let mut x = x[0]; // (batch_size, seq_len, 4096)

    // Precompute RoPE cos/sin matrices: shape [seq_len, head_dim]
    let theta = 10000.0;
    let cos_data = rope_cos_mat::<F>(head_dim, seq_len, theta);
    let sin_data = rope_sin_mat::<F>(head_dim, seq_len, theta);
    let cos_param = g.param(Witness::new(
      vec![seq_len, head_dim],
      cos_data,
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
    let sin_param = g.param(Witness::new(
      vec![seq_len, head_dim],
      sin_data,
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    let num_layers = attn_norm_w_vec.len();
    for i in 0..num_layers {
      let block = g.pipe(
        &[x],
        llama_block(
          attn_norm_w_vec[i].to_owned(),
          attn_q_w_vec[i].to_owned(),
          attn_k_w_vec[i].to_owned(),
          attn_v_w_vec[i].to_owned(),
          attn_o_w_vec[i].to_owned(),
          proj_norm_w_vec[i].to_owned(),
          proj_1_w_vec[i].to_owned(),
          proj_2_w_vec[i].to_owned(),
          proj_3_w_vec[i].to_owned(),
          num_heads,
          head_dim,
          seq_len,
          cos_param,
          sin_param,
        ),
      );
      x = block[0];
    }
    let layer_norm_w = g.param(layer_norm_w);
    let layer_norm_out = g.pipe(&[x], llama_rms_norm(layer_norm_w));
    let logits_w = g.param(logits_w);
    let out = g.einsum("bij,jk->ik".to_string(), vec![layer_norm_out[0], logits_w], true)[0];
    let out = g.change_shape(out, vec![seq_len, vocab_size]);
    vec![out]
  }
}
