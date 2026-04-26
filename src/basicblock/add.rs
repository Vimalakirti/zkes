use crate::basicblock::BasicBlock;
use crate::crypto::SumcheckProof;
use crate::dag::{Claim, DataType, Role, Witness};
use crate::util::arith::log2_ceil;
use crate::util::poly::CryptoField;
use crate::util::shape::{broadcast_shape, matched_axes};
use crate::util::transcript::Transcript;

/// Build a mapping from output dimension index to bit offset in claim.point.
/// For c_shape = [1, 4, 4], the offsets are [0, 0, 2] (dim0=0 bits, dim1=2 bits, dim2=2 bits).
fn dim_bit_offsets(c_shape: &[usize]) -> Vec<usize> {
  let mut offsets = Vec::with_capacity(c_shape.len());
  let mut offset = 0;
  for &dim in c_shape.iter() {
    offsets.push(offset);
    offset += log2_ceil(dim) as usize;
  }
  offsets
}

/// Extract the challenge point for an input from the output claim point,
/// using the matched output dimension indices.
fn extract_input_point<F: Clone>(claim_point: &[F], matched: &[usize], c_shape: &[usize], offsets: &[usize]) -> Vec<F> {
  let mut point = Vec::new();
  for &m in matched {
    let start = offsets[m];
    let end = start + log2_ceil(c_shape[m]) as usize;
    point.extend_from_slice(&claim_point[start..end]);
  }
  point
}

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

    let offsets = dim_bit_offsets(c_shape);
    let a_point = extract_input_point(&claim.point, &a_matched, c_shape, &offsets);
    let b_point = extract_input_point(&claim.point, &b_matched, c_shape, &offsets);

    let mut claims = Vec::new();
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

    let offsets = dim_bit_offsets(c_shape);
    let a_point = extract_input_point(c_point, &a_matched, c_shape, &offsets);
    let b_point = extract_input_point(c_point, &b_matched, c_shape, &offsets);

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

    let offsets = dim_bit_offsets(c_shape);
    let a_point = extract_input_point(&claim.point, &a_matched, c_shape, &offsets);
    let b_point = extract_input_point(&claim.point, &b_matched, c_shape, &offsets);

    let mut claims = Vec::new();
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

    let offsets = dim_bit_offsets(c_shape);
    let a_point = extract_input_point(c_point, &a_matched, c_shape, &offsets);
    let b_point = extract_input_point(c_point, &b_matched, c_shape, &offsets);

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
