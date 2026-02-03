use crate::basicblock::BasicBlock;
use crate::crypto::SumcheckProof;
use crate::dag::{Claim, Witness};

use crate::util::poly::CryptoField;
use crate::util::transcript::Transcript;

#[derive(Debug, Clone)]
pub struct ChangeShape {
  pub new_shape: Vec<usize>,
}
impl<F: CryptoField> BasicBlock<F> for ChangeShape {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    assert!(inputs.len() == 1, "ChangeShape expects 1 input");
    let mut output = inputs[0].clone();
    output.shape = self.new_shape.clone();
    return vec![output];
  }

  fn prove(
    &self,
    _witnesses: &[&Witness<F>],
    edge_ids: &[usize],
    out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    // simply pass the claim
    let claim = vec![Claim {
      edge_id: edge_ids[0],
      sparse_id: 0,
      point: out_claims[0].point.clone(),
      eval: out_claims[0].eval,
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
