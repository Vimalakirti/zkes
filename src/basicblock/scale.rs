use crate::basicblock::BasicBlock;
use crate::crypto::SumcheckProof;
use crate::dag::{Claim, PolyType, Role, Witness};

use crate::util::arith::{f_to_int, get_n, next_pow};
use crate::util::poly::SelectionPolynomial;

use crate::util::poly::CryptoField;
use crate::util::transcript::Transcript;

#[derive(Debug, Clone)]
pub struct ScaleDown {
  pub input_sf: usize,
  pub output_sf: usize,
}
impl<F: CryptoField> BasicBlock<F> for ScaleDown {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    assert!(inputs.len() == 1, "ScaleDown expects 1 input");
    let x = inputs[0];
    let x_shape = &x.shape.iter().map(|&x| next_pow(x as u32) as usize).collect::<Vec<usize>>();
    let n: usize = x_shape.into_iter().product();
    let num_var = get_n(&x_shape);
    assert!(
      self.input_sf >= self.output_sf,
      "ScaleDown input SF must be greater than or equal to output SF"
    );
    let rescale_sf = self.input_sf - self.output_sf;
    let rescale_factor = 1 << rescale_sf;
    let rescale_factor_divided_by_2 = rescale_factor / 2;
    let rescale_factor_f = F::from(rescale_factor as u32);

    let mut y_data = vec![<F as CryptoField>::zero(); n];
    let mut selection = Vec::new();
    for i in 0..n {
      let x_i = x.data.as_ref().unwrap().index(i);
      let x_int = f_to_int(x_i);
      // Use integer floor division: y = floor((x + rescale/2) / rescale).
      // This guarantees aux = (x + rescale/2) mod rescale ∈ [0, rescale),
      // avoiding the boundary case where f64::round() half-away-from-zero
      // produces aux = rescale (out of range).
      let shifted = x_int + rescale_factor_divided_by_2 as i128;
      let y_int = shifted >> rescale_sf; // arithmetic right shift = floor div for power-of-2
      y_data[i] = F::from(y_int as i64);

      let aux_data = x_i - (y_data[i] * rescale_factor_f);
      let aux_num = f_to_int(aux_data) + rescale_factor_divided_by_2;
      // Clamp out-of-range values to 0 so the prover doesn't overflow.
      // The selection polynomial will be wrong for these entries, and the
      // verifier will reject the proof via the eval_to_check != eval_acc check.
      let table_index = if aux_num >= 0 && (aux_num as u128) < rescale_factor as u128 {
        aux_num as usize
      } else {
        0
      };
      selection.push((i, table_index));
    }
    let selection_polynomial = SelectionPolynomial::new(num_var, rescale_sf, selection);
    let y = Witness::new(inputs[0].shape.clone(), y_data, x.data_type, self.output_sf, Role::Output);
    let aux = Witness {
      shape: inputs[0].shape.clone(), // WORKAROUND: this shape is incorrect, but it is actually not used
      data: Some(Box::new(selection_polynomial.to_sparse()) as Box<dyn crate::util::poly::MLPoly<F>>),
      data_int: None,
      poly_type: PolyType::Sparse,
      data_type: x.data_type,
      sf: 0,
      role: Role::Auxiliary,
      is_permuted_with: None,
    };
    vec![y, aux]
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

#[derive(Debug, Clone)]
pub struct ScaleUp {
  pub input_sf: usize,
  pub output_sf: usize,
}
impl<F: CryptoField> BasicBlock<F> for ScaleUp {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    assert!(inputs.len() == 1, "ScaleUp expects 1 input");
    let x = inputs[0];
    let x_shape = &x.shape.iter().map(|&x| next_pow(x as u32) as usize).collect::<Vec<usize>>();
    let n: usize = x_shape.into_iter().product();
    let num_var = get_n(&x_shape);
    assert!(
      self.input_sf <= self.output_sf,
      "ScaleUp input SF must be less than or equal to output SF"
    );
    let rescale_sf = self.output_sf - self.input_sf;
    let rescale_factor = 1 << rescale_sf;
    let rescale_factor_divided_by_2 = rescale_factor / 2;
    let rescale_factor_f = F::from(rescale_factor as u32);

    let mut y_data = vec![<F as CryptoField>::zero(); n];
    let mut selection = Vec::new();
    for i in 0..n {
      let x_i = x.data.as_ref().unwrap().index(i);
      let mut x_i_num = f_to_int(x_i);
      x_i_num *= rescale_factor;

      y_data[i] = F::from(x_i_num);
      // if x_i_num < 0.0 {
      //   <F as CryptoField>::zero() - F::from((-x_i_num).round() as u32)
      // } else {
      //   F::from(x_i_num.round() as u32)
      // };

      let aux_data = x_i * rescale_factor_f - y_data[i];
      let aux_num = f_to_int(aux_data) + rescale_factor_divided_by_2;
      // Clamp out-of-range values to 0 so the prover doesn't overflow.
      // The selection polynomial will be wrong for these entries, and the
      // verifier will reject the proof via the eval_to_check != eval_acc check.
      let table_index = if aux_num >= 0 && (aux_num as u128) < rescale_factor as u128 {
        aux_num as usize
      } else {
        0
      };
      selection.push((i, table_index));
    }
    let selection_polynomial = SelectionPolynomial::new(num_var, rescale_sf, selection);
    let y = Witness::new(inputs[0].shape.clone(), y_data, x.data_type, self.output_sf, Role::Output);
    let aux = Witness {
      shape: inputs[0].shape.clone(), // WORKAROUND: this shape is incorrect, but it is actually not used
      data: Some(Box::new(selection_polynomial.to_sparse()) as Box<dyn crate::util::poly::MLPoly<F>>),
      data_int: None,
      poly_type: PolyType::Sparse,
      data_type: x.data_type,
      sf: 0,
      role: Role::Auxiliary,
      is_permuted_with: None,
    };
    vec![y, aux]
  }

  fn prove(
    &self,
    witnesses: &[&Witness<F>],
    edge_ids: &[usize],
    out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    // simply pass the claim
    let claim = vec![Claim {
      edge_id: edge_ids[0],
      sparse_id: 0,
      point: out_claims[0].point.clone(),
      eval: witnesses[0].data.as_ref().unwrap().evaluate_at_point(&out_claims[0].point),
    }];
    (vec![], claim)
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
