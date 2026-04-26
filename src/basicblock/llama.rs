use crate::basicblock::BasicBlock;
use crate::crypto::SumcheckProof;
use crate::dag::{Claim, PolyType, Role, Witness};
use crate::util::arith::{f_to_int, get_n, next_pow};
use crate::util::poly::CryptoField;
use crate::util::poly::DenseMLPoly;
use crate::util::transcript::Transcript;

/// RMSReciprocal: computes 1/RMS(x) for RMSNorm.
/// Output shape: input shape with last dim = 1.
#[derive(Debug, Clone)]
pub struct RMSReciprocal;
impl<F: CryptoField> BasicBlock<F> for RMSReciprocal {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    let x = inputs[0];
    let mut y_shape = x.shape.clone();
    let ndim = y_shape.len();
    y_shape[ndim - 1] = 1;
    let n = get_n(&y_shape);
    let sf = (1u64 << x.sf) as f64;

    // Compute padded sizes and strides (column-major: first dim has stride 1).
    // Only sum/mean over the REAL last-dim size (x.shape[-1]); padded entries
    // must be treated as outside the tensor (dividing by d_pad inflates 1/rms
    // by sqrt(d_pad / d_real), which compounds over layers).
    let padded: Vec<usize> = x.shape.iter().map(|&s| next_pow(s as u32) as usize).collect();
    let d_pad = padded[ndim - 1];
    let d_real = x.shape[ndim - 1];
    let stride_d: usize = padded[..ndim - 1].iter().product();
    let num_groups = stride_d;

    let mut result = Vec::with_capacity(num_groups);

    for g in 0..num_groups {
      let nn = d_real as f64;
      let sum_sq: f64 = (0..d_real)
        .map(|d| {
          let idx = g + d * stride_d;
          let xv = f_to_int(x.data.as_ref().unwrap().index(idx)) as f64 / sf;
          xv * xv
        })
        .sum();
      let _ = d_pad; // padded dim size available if needed

      let rms = (sum_sq / nn).sqrt();
      let val = if rms == 0.0 {
        0i128
      } else {
        ((1.0 / rms) * sf).round() as i128
      };
      let val_f = if val >= 0 {
        F::from(val as u32)
      } else {
        <F as CryptoField>::zero() - F::from((-val) as u32)
      };
      result.push(val_f);
    }

    let total = 1usize << n;
    result.resize(total, <F as CryptoField>::zero());

    let y = Witness {
      shape: y_shape,
      data: Some(Box::new(DenseMLPoly::new(n, result)) as Box<dyn crate::util::poly::MLPoly<F>>),
      data_int: None,
      poly_type: PolyType::Dense,
      data_type: x.data_type,
      sf: x.sf,
      role: Role::Auxiliary,
      is_permuted_with: None,
    };
    vec![y]
  }

  fn prove(
    &self,
    _witnesses: &[&Witness<F>],
    _edge_ids: &[usize],
    _out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    (vec![], vec![])
  }

  fn verify(
    &self,
    _witnesses: &[&Witness<F>],
    _claims: &[&Claim<F>],
    _sumcheck_proofs: &[&SumcheckProof<F>],
    _transcript: &mut Transcript<F>,
  ) -> bool {
    true
  }
}

#[derive(Debug, Clone)]
pub struct DivConst {
  pub c: usize,
}
impl<F: CryptoField> BasicBlock<F> for DivConst {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    let x = inputs[0];
    let y_shape = x.shape.clone();
    let n = get_n(&y_shape);
    let c = self.c as f64;

    let x_shape = x.shape.iter().map(|&s| next_pow(s as u32) as usize).collect::<Vec<usize>>();
    let size: usize = x_shape.iter().product();
    let mut y_data = Vec::with_capacity(size);
    for i in 0..size {
      let x_i = x.data.as_ref().unwrap().index(i);
      let x_int = f_to_int(x_i) as f64;
      let y_int = (x_int / c).round() as i128;
      let y_f = if y_int >= 0 {
        F::from(y_int as u32)
      } else {
        <F as CryptoField>::zero() - F::from((-y_int) as u32)
      };
      y_data.push(y_f);
    }

    let y = Witness {
      shape: y_shape,
      data: Some(Box::new(DenseMLPoly::new(n, y_data)) as Box<dyn crate::util::poly::MLPoly<F>>),
      data_int: None,
      poly_type: PolyType::Dense,
      data_type: x.data_type,
      sf: x.sf,
      role: Role::Auxiliary,
      is_permuted_with: None,
    };
    vec![y]
  }

  fn prove(
    &self,
    _witnesses: &[&Witness<F>],
    _edge_ids: &[usize],
    _out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    (vec![], vec![])
  }

  fn verify(
    &self,
    _witnesses: &[&Witness<F>],
    _claims: &[&Claim<F>],
    _sumcheck_proofs: &[&SumcheckProof<F>],
    _transcript: &mut Transcript<F>,
  ) -> bool {
    true
  }
}

/// Reciprocal: prover-advice computation of r = round(SF^2 / x) elementwise.
/// Given a Q15 tensor x, produces a tensor of the same shape whose values represent
/// (in Q15 scale) the real reciprocal: r/SF ≈ SF/x_real = 1/(x_real_in_real_units) * SF.
/// Callers then use `rescale(y * r)` to implement division by x in Q15 arithmetic.
/// Zero/near-zero inputs produce 0 (safe for padded slots that never get used).
#[derive(Debug, Clone)]
pub struct Reciprocal;
impl<F: CryptoField> BasicBlock<F> for Reciprocal {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    let x = inputs[0];
    let y_shape = x.shape.clone();
    let n = get_n(&y_shape);
    let sf = (1u64 << x.sf) as f64;
    let sf_sq = sf * sf;

    let padded: Vec<usize> = x.shape.iter().map(|&s| next_pow(s as u32) as usize).collect();
    let size: usize = padded.iter().product();
    let mut y_data = Vec::with_capacity(size);
    for i in 0..size {
      let x_val = f_to_int(x.data.as_ref().unwrap().index(i)) as f64;
      let r_int = if x_val.abs() < 1.0 {
        0i128
      } else {
        (sf_sq / x_val).round() as i128
      };
      let r_f = if r_int >= 0 {
        F::from(r_int as u32)
      } else {
        <F as CryptoField>::zero() - F::from((-r_int) as u32)
      };
      y_data.push(r_f);
    }

    let y = Witness {
      shape: y_shape,
      data: Some(Box::new(DenseMLPoly::new(n, y_data)) as Box<dyn crate::util::poly::MLPoly<F>>),
      data_int: None,
      poly_type: PolyType::Dense,
      data_type: x.data_type,
      sf: x.sf,
      role: Role::Auxiliary,
      is_permuted_with: None,
    };
    vec![y]
  }

  fn prove(
    &self,
    _witnesses: &[&Witness<F>],
    _edge_ids: &[usize],
    _out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    // Input claims are produced by the reciprocity range check emitted by
    // `DagBuilder::reciprocal`, not here.
    (vec![], vec![])
  }

  fn verify(
    &self,
    _witnesses: &[&Witness<F>],
    _claims: &[&Claim<F>],
    _sumcheck_proofs: &[&SumcheckProof<F>],
    _transcript: &mut Transcript<F>,
  ) -> bool {
    true
  }
}

/// SoftmaxConst: per-row log-sum-exp shift (advice) for the softmax.
/// softmax_c[i] = -round(SF · ln(Σ_j exp(x_j / SF))) for all i in the same row
/// (last-dim group). Used as: scores' = exp(scores + softmax_c), which yields
/// the Q15 softmax directly, so no subsequent 1/Σ rescale is needed.
/// `dim` is the real (un-padded) length of the softmax axis.
/// Soundness in a full prove path relies on Σ_j exp(scores' + softmax_c) ≈ SF
/// (softmax sums to 1 in Q15) — that sum-check is the constraint that pins c.
#[derive(Debug, Clone)]
pub struct SoftmaxConst {
  pub dim: usize,
}
impl<F: CryptoField> BasicBlock<F> for SoftmaxConst {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    let x = inputs[0];
    let y_shape = x.shape.clone();
    let n = get_n(&y_shape);
    let ndim = x.shape.len();
    let padded: Vec<usize> = x.shape.iter().map(|&s| next_pow(s as u32) as usize).collect();
    let d_pad = padded[ndim - 1];
    let stride_d: usize = padded[..ndim - 1].iter().product();
    let num_groups = stride_d;
    let total_size: usize = padded.iter().product();

    let sf = (1u64 << x.sf) as f64;

    let mut y_data = vec![<F as CryptoField>::zero(); total_size];

    for g in 0..num_groups {
      // Find max_q for numerically-stable log-sum-exp.
      let mut max_val = i128::MIN;
      for d in 0..self.dim {
        let idx = g + d * stride_d;
        let val = f_to_int(x.data.as_ref().unwrap().index(idx));
        if val > max_val {
          max_val = val;
        }
      }
      // sum_shifted = Σ exp((x_q - max_q)/SF). Terms ∈ (0, 1], so sum_shifted ∈ [1, dim].
      let mut sum_shifted = 0.0_f64;
      for d in 0..self.dim {
        let idx = g + d * stride_d;
        let val = f_to_int(x.data.as_ref().unwrap().index(idx)) as f64;
        sum_shifted += ((val - max_val as f64) / sf).exp();
      }
      // logsumexp_q = max_q + round(SF · ln(sum_shifted))
      let lse_shift = (sf * sum_shifted.ln()).round() as i128;
      let c_q = -(max_val + lse_shift);

      let c_f = if c_q >= 0 {
        F::from(c_q as u64)
      } else {
        <F as CryptoField>::zero() - F::from((-c_q) as u64)
      };
      for d in 0..d_pad {
        let idx = g + d * stride_d;
        y_data[idx] = c_f;
      }
    }

    let y = Witness {
      shape: y_shape,
      data: Some(Box::new(DenseMLPoly::new(n, y_data)) as Box<dyn crate::util::poly::MLPoly<F>>),
      data_int: None,
      poly_type: PolyType::Dense,
      data_type: x.data_type,
      sf: x.sf,
      role: Role::Auxiliary,
      is_permuted_with: None,
    };
    vec![y]
  }

  fn prove(
    &self,
    _witnesses: &[&Witness<F>],
    _edge_ids: &[usize],
    _out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    (vec![], vec![])
  }

  fn verify(
    &self,
    _witnesses: &[&Witness<F>],
    _claims: &[&Claim<F>],
    _sumcheck_proofs: &[&SumcheckProof<F>],
    _transcript: &mut Transcript<F>,
  ) -> bool {
    true
  }
}

/// SigmoidConst: computes constant c such that exp((x+c)/sf) = sigmoid(x/sf).
/// sigmoid(t) = 1/(1+exp(-t)), so ln(sigmoid(t)) = -softplus(-t).
/// We need (x+c)/sf = ln(sigmoid(x/sf)), so c = -sf * softplus(-x/sf) - x.
#[derive(Debug, Clone)]
pub struct SigmoidConst;
impl<F: CryptoField> BasicBlock<F> for SigmoidConst {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    let x = inputs[0];
    let y_shape = x.shape.clone();
    let n = get_n(&y_shape);
    let sf = (1u64 << x.sf) as f64;

    let x_shape = x.shape.iter().map(|&s| next_pow(s as u32) as usize).collect::<Vec<usize>>();
    let size: usize = x_shape.iter().product();
    let mut y_data = Vec::with_capacity(size);

    for i in 0..size {
      let x_i = x.data.as_ref().unwrap().index(i);
      let x_int = f_to_int(x_i) as f64;
      let x_f = x_int / sf;
      // softplus(-x_f) = ln(1 + exp(-x_f)), numerically stable
      let t = -x_f;
      let sp = if t > 20.0 {
        t
      } else if t < -20.0 {
        0.0
      } else {
        (1.0_f64 + t.exp()).ln()
      };
      let c = -sf * sp - x_int;
      let c_int = c.round() as i128;
      let c_f = if c_int >= 0 {
        F::from(c_int as u32)
      } else {
        <F as CryptoField>::zero() - F::from((-c_int) as u32)
      };
      y_data.push(c_f);
    }

    let y = Witness {
      shape: y_shape,
      data: Some(Box::new(DenseMLPoly::new(n, y_data)) as Box<dyn crate::util::poly::MLPoly<F>>),
      data_int: None,
      poly_type: PolyType::Dense,
      data_type: x.data_type,
      sf: x.sf,
      role: Role::Auxiliary,
      is_permuted_with: None,
    };
    vec![y]
  }

  fn prove(
    &self,
    _witnesses: &[&Witness<F>],
    _edge_ids: &[usize],
    _out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    (vec![], vec![])
  }

  fn verify(
    &self,
    _witnesses: &[&Witness<F>],
    _claims: &[&Claim<F>],
    _sumcheck_proofs: &[&SumcheckProof<F>],
    _transcript: &mut Transcript<F>,
  ) -> bool {
    true
  }
}
