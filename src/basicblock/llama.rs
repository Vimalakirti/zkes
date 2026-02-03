use crate::basicblock::BasicBlock;
use crate::crypto::SumcheckProof;
use crate::dag::{Claim, PolyType, Role, Witness};
use crate::util::arith::get_n;
use crate::util::poly::CryptoField;
use crate::util::poly::DenseMLPoly;
use crate::util::transcript::Transcript;
use ndarray::Axis;

// All basicblocks for LLaMA-2-7B here are advices and need no proof
#[derive(Debug, Clone)]
pub struct RMSReciprocal;
impl<F: CryptoField> BasicBlock<F> for RMSReciprocal {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    let x = inputs[0];
    let mut y_shape = x.shape.clone();
    y_shape[x.shape.len() - 1] = 1;
    let n = get_n(&y_shape);

    let x_ndarray = x.ndarray();
    let sf = (1 << x.sf) as f64;
    let last_axis = x_ndarray.ndim() - 1;

    // Map each 1D lane along the last axis
    let y_ndarray = x_ndarray.map_axis(Axis(last_axis), |lane| {
      let n = lane.len() as f64;

      // x = lane / sf
      let sum_sq: f64 = lane
        .iter()
        .map(|v| {
          let x = (*v as f64) / sf;
          x * x
        })
        .sum();

      let rms = (sum_sq / n).sqrt();
      ((1.0 / rms) * sf).round()
    });
    let y_data = y_ndarray.clone().view().reversed_axes().iter().map(|f| F::from(*f as u32)).collect::<Vec<F>>();

    let y_data = Some(Box::new(DenseMLPoly::new(n, y_data)) as Box<dyn crate::util::poly::MLPoly<F>>);
    let y = Witness {
      shape: y_shape,
      data: y_data,
      data_int: None,
      poly_type: PolyType::Dense,
      data_type: x.data_type,
      sf: x.sf,
      role: Role::Auxiliary,
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
    let x_ndarray = x.ndarray();
    let y_ndarray = x_ndarray.mapv(|v| (v as f64 / self.c as f64).round() as i64);
    let y_data = y_ndarray.clone().view().reversed_axes().iter().map(|f| F::from(*f as u32)).collect::<Vec<F>>();
    let y = Witness {
      shape: y_shape,
      data: Some(Box::new(DenseMLPoly::new(n, y_data)) as Box<dyn crate::util::poly::MLPoly<F>>),
      data_int: None,
      poly_type: PolyType::Dense,
      data_type: x.data_type,
      sf: x.sf,
      role: Role::Auxiliary,
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

// TO FIX: please implement softmax for array size > 1
#[derive(Debug, Clone)]
pub struct SoftmaxConst;
impl<F: CryptoField> BasicBlock<F> for SoftmaxConst {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    let x = inputs[0];
    let y_shape = x.shape.clone();
    let n = get_n(&y_shape);
    let x_ndarray = x.ndarray();
    let y_ndarray = x_ndarray.mapv(|v| -v);
    let y_data = y_ndarray
      .clone()
      .view()
      .reversed_axes()
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
    let y = Witness {
      shape: y_shape,
      data: Some(Box::new(DenseMLPoly::new(n, y_data)) as Box<dyn crate::util::poly::MLPoly<F>>),
      data_int: None,
      poly_type: PolyType::Dense,
      data_type: x.data_type,
      sf: x.sf,
      role: Role::Auxiliary,
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

// TO FIX: please implement the correct sigmoid constant
#[derive(Debug, Clone)]
pub struct SigmoidConst;
impl<F: CryptoField> BasicBlock<F> for SigmoidConst {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    let x = inputs[0];
    let y_shape = x.shape.clone();
    let n = get_n(&y_shape);
    let x_ndarray = x.ndarray();
    let y_ndarray = x_ndarray.mapv(|v| -v - 1); // FIX this for correctness
    let y_data = y_ndarray
      .clone()
      .view()
      .reversed_axes()
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
    let y = Witness {
      shape: y_shape,
      data: Some(Box::new(DenseMLPoly::new(n, y_data)) as Box<dyn crate::util::poly::MLPoly<F>>),
      data_int: None,
      poly_type: PolyType::Dense,
      data_type: x.data_type,
      sf: x.sf,
      role: Role::Auxiliary,
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
