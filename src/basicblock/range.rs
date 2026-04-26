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

    let table_size = 1usize << self.table_size_log;
    let mut oor_count = 0usize;
    let mut min_val: i128 = 0;
    let mut max_val: i128 = 0;
    let mut oor_samples: Vec<(usize, i128)> = Vec::new();
    for i in 0..n {
      let x_i = x.data.as_ref().unwrap().index(i);
      let x_i_num = f_to_int(x_i);
      if x_i_num < min_val { min_val = x_i_num; }
      if x_i_num > max_val { max_val = x_i_num; }
      // Clamp out-of-range values to 0 so the prover doesn't overflow.
      // The selection polynomial will be wrong for these entries, and the
      // verifier will reject the proof via the eval_to_check != eval_acc check.
      let table_index = if x_i_num >= 0 && (x_i_num as u128) < table_size as u128 {
        x_i_num as usize
      } else {
        oor_count += 1;
        if oor_samples.len() < 6 { oor_samples.push((i, x_i_num)); }
        0
      };
      selection.push((i, table_index));
    }
    if oor_count > 0 {
      eprintln!("[NonNeg] tsl={}, shape={:?}, n={}, min={}, max={}, oor={}/{}, samples={:?}",
        self.table_size_log, x.shape, n, min_val, max_val, oor_count, n, oor_samples);
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
      is_permuted_with: None,
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
