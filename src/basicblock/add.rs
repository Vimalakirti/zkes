use crate::basicblock::BasicBlock;
use crate::crypto::SumcheckProof;
use crate::dag::{Claim, DataType, Role, Witness};
use crate::util::arith::log2_ceil;
use crate::util::poly::CryptoField;
use crate::util::shape::{broadcast_shape, matched_axes};
use crate::util::transcript::Transcript;

#[derive(Debug, Clone)]
pub struct Add;
impl<F: CryptoField> BasicBlock<F> for Add {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    let a = inputs[0];
    let b = inputs[1];
    assert_eq!(a.sf, b.sf, "Add expects inputs with the same scale factor");
    let a_shape = &a.shape;
    let b_shape = &b.shape;
    // c[i] = a[i] + b[i]
    let c_shape = broadcast_shape(a_shape, b_shape).unwrap();
    let a_ndarray = a.ndarray();
    let b_ndarray = b.ndarray();
    let c_ndarray = a_ndarray + b_ndarray;
    let col_major_output: Vec<_> = c_ndarray.clone().view().reversed_axes().iter().cloned().collect();
    let output_data = col_major_output
      .iter()
      .map(|f| {
        F::from(*f)
        //if *f > 0 {
        //  F::from(*f as u32)
        //} else {
        //  <F as CryptoField>::zero() - F::from(-*f as u32)
        //}
      })
      .collect::<Vec<F>>();
    let c = Witness::new(c_shape, output_data, DataType::Float, a.sf, Role::Output);
    vec![c]
  }

  fn prove(
    &self,
    witnesses: &[&Witness<F>],
    edge_ids: &[usize],
    out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    assert!(witnesses.len() == 3, "Add expects 2 inputs and 1 output");
    assert!(edge_ids.len() == 3, "Add expects 2 input edges and 1 output edge");
    assert!(out_claims.len() == 1, "Add expects 1 output claim");
    let claim = out_claims[0];
    let a_shape = &witnesses[0].shape;
    let b_shape = &witnesses[1].shape;
    let c_shape = &witnesses[2].shape;
    let a_matched = matched_axes(a_shape, c_shape).unwrap();
    let b_matched = matched_axes(b_shape, c_shape).unwrap();

    let mut claims = Vec::new();
    let mut a_start = 0;
    let mut a_point = Vec::new();
    for a_m in a_matched {
      let a_end = a_start + log2_ceil(c_shape[a_m]) as usize;
      let a_point_slice = claim.point[a_start..a_end].to_vec();
      a_point.extend(a_point_slice);
      a_start = a_end;
    }
    let mut b_start = 0;
    let mut b_point = Vec::new();
    for b_m in b_matched {
      let b_end = b_start + log2_ceil(c_shape[b_m]) as usize;
      let b_point_slice = claim.point[b_start..b_end].to_vec();
      b_point.extend(b_point_slice);
      b_start = b_end;
    }
    let a_claim = Claim {
      edge_id: edge_ids[0],
      sparse_id: 0,
      point: a_point.clone(),
      eval: witnesses[0].data.as_ref().unwrap().evaluate_at_point(&a_point),
    };
    let b_claim = Claim {
      edge_id: edge_ids[1],
      sparse_id: 0,
      point: b_point,
      eval: claim.eval - a_claim.eval,
    };
    claims.push(a_claim);
    claims.push(b_claim);
    (vec![], claims)
  }

  fn verify(&self, witnesses: &[&Witness<F>], claims: &[&Claim<F>], _sumcheck_proofs: &[&SumcheckProof<F>], _transcript: &mut Transcript<F>) -> bool {
    assert!(claims.len() == 3, "Add expects 3 claims"); // [a, b, c]
    let mut verified = true;
    let c_point = &claims[2].point;

    let a_shape = &witnesses[0].shape;
    let b_shape = &witnesses[1].shape;
    let c_shape = &witnesses[2].shape;
    let a_matched = matched_axes(a_shape, c_shape).unwrap();
    let b_matched = matched_axes(b_shape, c_shape).unwrap();
    let mut a_start = 0;
    let mut a_point = Vec::new();
    for a_m in a_matched {
      let a_end = a_start + log2_ceil(c_shape[a_m]) as usize;
      let a_point_slice = c_point[a_start..a_end].to_vec();
      a_point.extend(a_point_slice);
      a_start = a_end;
    }
    let mut b_start = 0;
    let mut b_point = Vec::new();
    for b_m in b_matched {
      let b_end = b_start + log2_ceil(c_shape[b_m]) as usize;
      let b_point_slice = c_point[b_start..b_end].to_vec();
      b_point.extend(b_point_slice);
      b_start = b_end;
    }
    let a_claim = claims[0];
    let b_claim = claims[1];
    let c_claim = claims[2];
    let a_eval = a_claim.eval;
    let b_eval = b_claim.eval;
    let c_eval = c_claim.eval;
    let a_point_proof = &a_claim.point;
    let b_point_proof = &b_claim.point;
    verified = verified && a_eval + b_eval == c_eval && *a_point_proof == a_point && *b_point_proof == b_point;
    verified
  }
}

#[derive(Debug, Clone)]
pub struct Sub;
impl<F: CryptoField> BasicBlock<F> for Sub {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    let a = inputs[0];
    let b = inputs[1];
    assert_eq!(a.sf, b.sf, "Sub expects inputs with the same scale factor");
    let a_shape = &a.shape;
    let b_shape = &b.shape;

    let c_shape = broadcast_shape(a_shape, b_shape).unwrap();
    let a_ndarray = a.ndarray();
    let b_ndarray = b.ndarray();
    let c_ndarray = a_ndarray - b_ndarray;
    let col_major_output: Vec<_> = c_ndarray.clone().view().reversed_axes().iter().cloned().collect();
    let output_data = col_major_output
      .iter()
      .map(|f| {
        F::from(*f)
        //if *f > 0 {
        //  F::from(*f as u32)
        //} else {
        //  <F as CryptoField>::zero() - F::from(-*f as u32)
        //}
      })
      .collect::<Vec<F>>();
    let c = Witness::new(c_shape, output_data, DataType::Float, a.sf, Role::Output);
    vec![c]
  }

  fn prove(
    &self,
    witnesses: &[&Witness<F>],
    edge_ids: &[usize],
    out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    assert!(witnesses.len() == 3, "Sub expects 2 inputs and 1 output");
    assert!(edge_ids.len() == 3, "Sub expects 2 input edges and 1 output edge");
    assert!(out_claims.len() == 1, "Sub expects 1 output claim");
    let claim = out_claims[0];
    let a_shape = &witnesses[0].shape;
    let b_shape = &witnesses[1].shape;
    let c_shape = &witnesses[2].shape;
    let a_matched = matched_axes(a_shape, c_shape).unwrap();
    let b_matched = matched_axes(b_shape, c_shape).unwrap();

    let mut claims = Vec::new();
    let mut a_start = 0;
    let mut a_point = Vec::new();
    for a_m in a_matched {
      let a_end = a_start + log2_ceil(c_shape[a_m]) as usize;
      let a_point_slice = claim.point[a_start..a_end].to_vec();
      a_point.extend(a_point_slice);
      a_start = a_end;
    }
    let mut b_start = 0;
    let mut b_point = Vec::new();
    for b_m in b_matched {
      let b_end = b_start + log2_ceil(c_shape[b_m]) as usize;
      let b_point_slice = claim.point[b_start..b_end].to_vec();
      b_point.extend(b_point_slice);
      b_start = b_end;
    }
    let a_claim = Claim {
      edge_id: edge_ids[0],
      sparse_id: 0,
      point: a_point.clone(),
      eval: witnesses[0].data.as_ref().unwrap().evaluate_at_point(&a_point),
    };
    let b_claim = Claim {
      edge_id: edge_ids[1],
      sparse_id: 0,
      point: b_point,
      eval: a_claim.eval - claim.eval,
    };
    claims.push(a_claim);
    claims.push(b_claim);
    (vec![], claims)
  }

  fn verify(&self, witnesses: &[&Witness<F>], claims: &[&Claim<F>], _sumcheck_proofs: &[&SumcheckProof<F>], _transcript: &mut Transcript<F>) -> bool {
    assert!(claims.len() == 3, "Sub expects 3 claims"); // [a, b, c]
    let mut verified = true;
    let c_point = &claims[2].point;

    let a_shape = &witnesses[0].shape;
    let b_shape = &witnesses[1].shape;
    let c_shape = &witnesses[2].shape;
    let a_matched = matched_axes(a_shape, c_shape).unwrap();
    let b_matched = matched_axes(b_shape, c_shape).unwrap();
    let mut a_start = 0;
    let mut a_point = Vec::new();
    for a_m in a_matched {
      let a_end = a_start + log2_ceil(c_shape[a_m]) as usize;
      let a_point_slice = c_point[a_start..a_end].to_vec();
      a_point.extend(a_point_slice);
      a_start = a_end;
    }
    let mut b_start = 0;
    let mut b_point = Vec::new();
    for b_m in b_matched {
      let b_end = b_start + log2_ceil(c_shape[b_m]) as usize;
      let b_point_slice = c_point[b_start..b_end].to_vec();
      b_point.extend(b_point_slice);
      b_start = b_end;
    }
    let a_claim = claims[0];
    let b_claim = claims[1];
    let c_claim = claims[2];
    let a_eval = a_claim.eval;
    let b_eval = b_claim.eval;
    let c_eval = c_claim.eval;
    let a_point_proof = &a_claim.point;
    let b_point_proof = &b_claim.point;
    verified = verified && a_eval - b_eval == c_eval && *a_point_proof == a_point && *b_point_proof == b_point;
    verified
  }
}
