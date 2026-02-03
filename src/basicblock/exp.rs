use crate::basicblock::BasicBlock;
use crate::crypto::SumcheckProof;
use crate::dag::{Claim, DataType, PolyType, Role, Witness};
use crate::{SF_FLOAT, SF_LOG};

use crate::util::arith::{f_to_int, get_n, next_pow};
use crate::util::poly::{SelectionPolynomial, SparseMLPoly};

use crate::util::poly::CryptoField;
use crate::util::transcript::Transcript;

// ExpHelper only accepts input in range roughly [-15 * sf, 0], use Clip before Exp if not in this range.
// ExpHelper makes input x into k * (-ln(2)*sf) + r, where k is a 4-bit unsigned integer and r is a float in range [-ln2/2, ln2/2].
#[derive(Debug, Clone)]
pub struct ExpHelper;
impl<F: CryptoField> BasicBlock<F> for ExpHelper {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    assert!(inputs.len() == 1, "ScaleBack expects 1 input");
    let x = inputs[0];
    let x_shape = &x.shape.iter().map(|&x| next_pow(x as u32) as usize).collect::<Vec<usize>>();
    let n: usize = x_shape.into_iter().product();
    let num_var = get_n(&x_shape);

    let ln2 = (2.0_f32.ln() * *SF_FLOAT).round() as f64;
    let neg_ln2_f = <F as CryptoField>::zero() - F::from(ln2 as u32);

    let mut r_data = vec![<F as CryptoField>::zero(); n];
    let mut selection = Vec::new();
    for i in 0..n {
      let x_i = x.data.as_ref().unwrap().index(i);
      let x_i_num = f_to_int(x_i) as f64;
      let k = ((-x_i_num) / ln2).round();

      r_data[i] = x_i - (F::from(k as u32) * neg_ln2_f);

      selection.push((i, k as usize));
    }
    let selection_polynomial = SelectionPolynomial::new(num_var, 4, selection);
    let r = Witness::new(inputs[0].shape.clone(), r_data, DataType::Float, *SF_LOG, Role::Output);
    let aux = Witness {
      shape: inputs[0].shape.clone(), // WORKAROUND: this shape is incorrect, but it is actually not used
      data: Some(Box::new(selection_polynomial.to_sparse()) as Box<dyn crate::util::poly::MLPoly<F>>),
      data_int: None,
      poly_type: PolyType::Sparse,
      data_type: x.data_type,
      sf: 0,
      role: Role::Auxiliary,
    };
    vec![r, aux]
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

// TwoPow makes input x into 2^(-x).
// sf is always 15
#[derive(Debug, Clone)]
pub struct TwoPow;
impl<F: CryptoField> BasicBlock<F> for TwoPow {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    assert!(inputs.len() == 1, "TwoPow expects 1 input");
    let x = inputs[0];
    let x_data = x.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap(); // the k in the above ExpHelper
    let inp_num_vars = x_data.selection.input_num_vars;
    let selection = &x_data.selection.selection;
    let mut y_data = vec![<F as CryptoField>::zero(); 1 << inp_num_vars];
    for (input_index, table_index) in selection {
      y_data[*input_index] = F::from((1 << (15 - table_index)) as u32);
    }
    let y = Witness::new(inputs[0].shape.clone(), y_data, DataType::Float, 15, Role::Output);
    vec![y]
  }

  fn prove(
    &self,
    _witnesses: &[&Witness<F>],
    _edge_ids: &[usize],
    _out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    // this will be batched proved later
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
