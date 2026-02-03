use crate::basicblock::BasicBlock;
use crate::crypto::{LinearSumcheckProver, SumcheckProof, SumcheckProver};
use crate::dag::{Claim, Role, Witness};
use crate::util::arith::next_pow;
use crate::util::poly::CryptoField;
use crate::util::poly::{evaluate_lagrange_basis, DenseMLPoly};
use crate::util::transcript::Transcript;

// In LLM models, we haven't used this basicblock yet. We can come back to fix TODOs later.

fn shape_pow(shape: &[usize]) -> Vec<usize> {
  shape.iter().map(|&s| next_pow(s as u32) as usize).collect::<Vec<usize>>()
}

fn shape_indices_to_eval_index(shape_pow: &[usize], indices: &[usize]) -> usize {
  let index = indices.iter().enumerate().map(|(i, &index)| index * shape_pow[..i].iter().fold(1, |acc, &x| acc * x)).sum();
  index
}

#[derive(Debug, Clone)]
pub struct Permute {
  pub permutation: Vec<(Vec<usize>, Vec<usize>)>, // Vec of (input_indices, output_indices)
  pub output_shape: Vec<usize>,
}

impl<F: CryptoField> BasicBlock<F> for Permute {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    let x = inputs[0];
    let input_shape = &x.shape;
    // concat output and input shape
    let _output_input_shape = self.output_shape.iter().chain(input_shape.iter()).map(|&x| next_pow(x as u32) as usize).collect::<Vec<usize>>();
    // y[i] = x[k] when (k, i) in permutation
    let y_data = vec![<F as CryptoField>::zero(); self.output_shape.iter().map(|&x| next_pow(x as u32) as usize).product()];
    let mut y = Witness::new(self.output_shape.clone(), y_data, x.data_type, x.sf, Role::Output);

    //let aux_data = vec![<F as CryptoField>::zero(); output_input_shape.iter().product()];
    //let mut aux = Witness::new(output_input_shape, aux_data, x.data_type, 1, Role::Auxiliary);

    for (input_indices, output_indices) in self.permutation.iter() {
      y.set(&output_indices, x.get(&input_indices));
      //let output_input_index = output_indices.iter().chain(input_indices.iter()).map(|&x| x).collect::<Vec<usize>>();
      //aux.set(&output_input_index, <F as CryptoField>::from_u32(1u32));
    }

    vec![y]
  }

  fn prove(
    &self,
    witnesses: &[&Witness<F>],
    edge_ids: &[usize],
    out_claims: &[&Claim<F>],
    transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    let output_shape_pow = shape_pow(&self.output_shape);
    let input_shape_pow = shape_pow(&witnesses[0].shape);
    let out_lagrange_basis = evaluate_lagrange_basis(&out_claims[0].point);
    let x = witnesses[0];
    let mut permute_poly = DenseMLPoly::new(
      x.data.as_ref().unwrap().n(),
      vec![<F as CryptoField>::zero(); x.data.as_ref().unwrap().len()],
    );
    for (input_indices, output_indices) in self.permutation.iter() {
      let output_index = shape_indices_to_eval_index(&output_shape_pow, output_indices);
      let input_index = shape_indices_to_eval_index(&input_shape_pow, input_indices);
      permute_poly[input_index] = permute_poly[input_index] + out_lagrange_basis[output_index];
    }
    let x_dense = x.data.as_ref().unwrap().as_any().downcast_ref::<DenseMLPoly<F>>().expect("Expected DenseMLPoly for x").clone();
    let mut sumcheck_prover = LinearSumcheckProver::new(x.data.as_ref().unwrap().n(), 2, transcript);
    let sumcheck_proof = sumcheck_prover.prove(&vec![x_dense, permute_poly], transcript);
    let challenges = sumcheck_prover.challenges;
    let claim_x = Claim {
      edge_id: edge_ids[0],
      sparse_id: 0,
      point: challenges.clone(),
      eval: x.data.as_ref().unwrap().evaluate_at_point(&challenges),
    };

    (vec![sumcheck_proof], vec![claim_x])
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
