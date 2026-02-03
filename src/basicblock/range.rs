use crate::basicblock::BasicBlock;
use crate::crypto::SumcheckProof;
use crate::dag::{Claim, PolyType, Role, Witness};
use crate::util::arith::{f_to_int, get_n, next_pow};
use crate::util::poly::{SelectionPolynomial, SparseMLPoly};

use crate::util::poly::CryptoField;
use crate::util::transcript::Transcript;

#[derive(Debug, Clone)]
pub struct NonNegative {
  pub table_size_log: usize,
}

impl NonNegative {
  pub fn new(table_size_log: usize) -> Self {
    Self { table_size_log }
  }
}

impl<F: CryptoField> BasicBlock<F> for NonNegative {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    let x = inputs[0];
    let x_shape = &x.shape.iter().map(|&x| next_pow(x as u32) as usize).collect::<Vec<usize>>();
    let n: usize = x_shape.into_iter().product();
    let num_var = get_n(&x_shape);

    let mut selection = Vec::new();

    // currently only support non-negative range
    for i in 0..n {
      let x_i = x.data.as_ref().unwrap().index(i);
      let x_i_num = f_to_int(x_i);
      // if x_i_num != (x_i_num as usize as i128) {
      //   println!("x_i_num: {x_i_num}");
      // }
      selection.push((i, x_i_num as usize));
    }
    let selection_polynomial = SelectionPolynomial::new(num_var, self.table_size_log, selection);
    let aux_data: SparseMLPoly<F> = selection_polynomial.to_sparse();
    let aux = Witness {
      shape: x_shape.clone(), // WORKAROUND: this shape is incorrect, but it is actually not used
      data: Some(Box::new(aux_data) as Box<dyn crate::util::poly::MLPoly<F>>),
      data_int: None,
      poly_type: PolyType::Sparse,
      data_type: x.data_type,
      sf: 0,
      role: Role::Auxiliary,
    };
    vec![aux]
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
