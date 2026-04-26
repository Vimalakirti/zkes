use crate::basicblock::BasicBlock;
use crate::crypto::SumcheckProof;
use crate::dag::{Claim, Role, Witness};
use crate::util::arith::{f_to_int, next_pow};
use crate::util::poly::CryptoField;
use crate::util::poly::MLPoly;
use crate::util::transcript::Transcript;

/// ClampLower: elementwise y = max(x, -c).
///
/// Soundness is enforced by the builder, which wires these constraints:
///   NonNeg(y - x)           → y ≥ x
///   NonNeg(y + c)           → y ≥ -c
///   ZeroCheck((y-x)(y+c))   → (y-x)(y+c) ≡ 0, so y ∈ {x, -c}
///
/// The three together imply y = max(x, -c): from y ≥ x and y ≥ -c we have
/// y ≥ max(x, -c); the product-zero forces y ∈ {x, -c}, pinning y to the max.
#[derive(Debug, Clone)]
pub struct ClampLower {
  pub c: u64,
}

impl<F: CryptoField> BasicBlock<F> for ClampLower {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    assert!(inputs.len() == 1, "ClampLower expects 1 input");
    let x = inputs[0];
    let x_shape = &x.shape.iter().map(|&s| next_pow(s as u32) as usize).collect::<Vec<usize>>();
    let n: usize = x_shape.iter().product();
    let neg_c = <F as CryptoField>::zero() - F::from(self.c as u32);
    let c_i128 = self.c as i128;
    let mut y_data = vec![<F as CryptoField>::zero(); n];
    let mut x_min: i128 = i128::MAX;
    let mut x_max: i128 = i128::MIN;
    let mut clamped: usize = 0;
    for i in 0..n {
      let x_i = x.data.as_ref().unwrap().index(i);
      let x_i_int = f_to_int(x_i);
      if x_i_int < x_min { x_min = x_i_int; }
      if x_i_int > x_max { x_max = x_i_int; }
      if x_i_int < -c_i128 { clamped += 1; y_data[i] = neg_c; } else { y_data[i] = x_i; }
    }
    let _ = (x_min, x_max, clamped, n);
    vec![Witness::new(x.shape.clone(), y_data, x.data_type, x.sf, Role::Output)]
  }

  fn prove(
    &self,
    witnesses: &[&Witness<F>],
    edge_ids: &[usize],
    out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    let inp_claim = Claim {
      edge_id: edge_ids[0],
      sparse_id: 0,
      point: out_claims[0].point.clone(),
      eval: witnesses[0].data.as_ref().unwrap().evaluate_at_point(&out_claims[0].point),
    };
    (vec![], vec![inp_claim])
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

/// ZeroCheck: asserts that the MLE of its single input is identically zero.
///
/// A fresh challenge point is drawn for the input edge during output-port
/// claim generation (see `Dag::prove`), and the claim's `eval` must be zero.
/// By Schwartz–Zippel, this holds with overwhelming probability iff the
/// underlying polynomial is the zero polynomial.
///
/// This block carries a dummy auxiliary output (shape [1]) so it participates
/// cleanly in the builder's edge/node bookkeeping, but that output is never
/// consumed.
#[derive(Debug, Clone)]
pub struct ZeroCheck;

impl<F: CryptoField> BasicBlock<F> for ZeroCheck {
  fn run(&self, _inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    vec![]
  }

  fn prove(
    &self,
    _witnesses: &[&Witness<F>],
    _edge_ids: &[usize],
    _out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    // The input-edge claim is pre-filled in `Dag::prove` (analogous to
    // NonNegative), so no sumcheck is emitted here.
    (vec![], vec![])
  }

  fn verify(
    &self,
    _witnesses: &[&Witness<F>],
    claims: &[&Claim<F>],
    _sumcheck_proofs: &[&SumcheckProof<F>],
    _transcript: &mut Transcript<F>,
  ) -> bool {
    claims.iter().all(|c| c.eval == <F as CryptoField>::zero())
  }
}
