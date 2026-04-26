//! Basic block module containing various ZKML basic blocks

use crate::crypto::SumcheckProof;
use crate::dag::{Claim, Witness};
use crate::util::poly::CryptoField;
use crate::util::transcript::Transcript;

pub mod add;
pub mod clamp;
pub mod einsum;
pub mod exp;
pub mod permute;
pub mod range;
pub mod reducer;
pub mod scale;
pub mod shape;

pub mod llama;

pub use add::{Add, Sub};
pub use clamp::{ClampLower, ZeroCheck};
pub use einsum::Einsum;
pub use exp::{ExpHelper, NonStructuredExp, TwoPow};
pub use permute::Permute;
pub use range::NonNegative;
pub use reducer::Reducer;
pub use scale::{ScaleDown, ScaleUp};
pub use shape::ChangeShape;

pub use llama::{DivConst, Reciprocal, RMSReciprocal, SigmoidConst, SoftmaxConst};
pub trait BasicBlock<F: CryptoField>: std::fmt::Debug + Send + Sync {
  // forward pass to compute the witnesses
  fn run(&self, _inputs: &[&Witness<F>]) -> Vec<Witness<F>>;

  // sumcheck proof to reduce claims
  fn prove(
    &self,
    _witnesses: &[&Witness<F>],
    _edge_ids: &[usize],
    _out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>);

  // verify the sumcheck proof reduces the claims correctly
  fn verify(
    &self,
    _witnesses: &[&Witness<F>],
    _claims: &[&Claim<F>],
    _sumcheck_proofs: &[&SumcheckProof<F>],
    _transcript: &mut Transcript<F>,
  ) -> bool;
}

/* =========================
Heterogeneous op enum
(arity lives here)
========================= */

#[derive(Clone, Debug)]
pub enum BasicBlockType {
  Add(Add),
  Sub(Sub),
  Einsum(Einsum),
  ExpHelper(ExpHelper),
  TwoPow(TwoPow),
  ChangeShape(ChangeShape),
  ScaleDown(ScaleDown),
  ScaleUp(ScaleUp),
  NonStructuredExp(NonStructuredExp),
  NonNegative(NonNegative),
  Reducer(Reducer),
  Permute(Permute),
  RMSReciprocal(RMSReciprocal),
  Reciprocal(Reciprocal),
  DivConst(DivConst),
  SoftmaxConst(SoftmaxConst),
  SigmoidConst(SigmoidConst),
  ClampLower(ClampLower),
  ZeroCheck(ZeroCheck),
}

impl BasicBlockType {
  pub fn out_arity(&self) -> usize {
    match self {
      BasicBlockType::Permute(_) => 2,
      BasicBlockType::ScaleDown(_) => 2,
      BasicBlockType::ScaleUp(_) => 2,
      BasicBlockType::ExpHelper(_) => 2,
      BasicBlockType::NonStructuredExp(_) => 2,
      BasicBlockType::ZeroCheck(_) => 0,
      _ => 1,
    }
  }

  pub fn check_inputs(&self, n: usize) {
    match self {
      BasicBlockType::Add(_) => assert!(n == 2, "Add expects 2 inputs (a,b), got {n}"),
      BasicBlockType::Sub(_) => assert!(n == 2, "Sub expects 2 inputs (a,b), got {n}"),
      BasicBlockType::NonStructuredExp(_) => assert!(n == 2, "NonStructuredExp expects 2 inputs (x,t), got {n}"),
      BasicBlockType::Einsum(_) => assert!(n >= 1, "Einsum expects at least 1 input, got {n}"),
      _ => assert!(n == 1, "Unary op expects 1 input, got {n}"),
    }
  }
}

impl<F: CryptoField> BasicBlock<F> for BasicBlockType {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    match self {
      BasicBlockType::Add(b) => b.run(inputs),
      BasicBlockType::Sub(b) => b.run(inputs),
      BasicBlockType::Einsum(b) => b.run(inputs),
      BasicBlockType::ExpHelper(b) => b.run(inputs),
      BasicBlockType::NonStructuredExp(b) => b.run(inputs),
      BasicBlockType::TwoPow(b) => b.run(inputs),
      BasicBlockType::ChangeShape(b) => b.run(inputs),
      BasicBlockType::ScaleDown(b) => b.run(inputs),
      BasicBlockType::ScaleUp(b) => b.run(inputs),
      BasicBlockType::NonNegative(b) => b.run(inputs),
      BasicBlockType::Permute(b) => b.run(inputs),
      BasicBlockType::Reducer(b) => b.run(inputs),
      BasicBlockType::ClampLower(b) => <ClampLower as BasicBlock<F>>::run(b, inputs),
      BasicBlockType::ZeroCheck(b) => <ZeroCheck as BasicBlock<F>>::run(b, inputs),
      // LLaMA specific
      BasicBlockType::RMSReciprocal(b) => b.run(inputs),
      BasicBlockType::Reciprocal(b) => b.run(inputs),
      BasicBlockType::DivConst(b) => b.run(inputs),
      BasicBlockType::SoftmaxConst(b) => b.run(inputs),
      BasicBlockType::SigmoidConst(b) => b.run(inputs),
    }
  }

  fn prove(
    &self,
    witnesses: &[&Witness<F>],
    edge_ids: &[usize],
    out_claims: &[&Claim<F>],
    transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    match self {
      BasicBlockType::Add(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
      BasicBlockType::Sub(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
      BasicBlockType::Einsum(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
      BasicBlockType::ExpHelper(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
      BasicBlockType::NonStructuredExp(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
      BasicBlockType::TwoPow(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
      BasicBlockType::ChangeShape(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
      BasicBlockType::ScaleDown(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
      BasicBlockType::ScaleUp(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
      BasicBlockType::NonNegative(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
      BasicBlockType::Permute(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
      BasicBlockType::Reducer(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
      BasicBlockType::ClampLower(b) => <ClampLower as BasicBlock<F>>::prove(b, witnesses, edge_ids, out_claims, transcript),
      BasicBlockType::ZeroCheck(b) => <ZeroCheck as BasicBlock<F>>::prove(b, witnesses, edge_ids, out_claims, transcript),
      // LLaMA specific
      BasicBlockType::RMSReciprocal(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
      BasicBlockType::Reciprocal(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
      BasicBlockType::DivConst(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
      BasicBlockType::SoftmaxConst(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
      BasicBlockType::SigmoidConst(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
    }
  }

  fn verify(&self, witnesses: &[&Witness<F>], claims: &[&Claim<F>], sumcheck_proofs: &[&SumcheckProof<F>], transcript: &mut Transcript<F>) -> bool {
    match self {
      BasicBlockType::Add(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
      BasicBlockType::Sub(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
      BasicBlockType::Einsum(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
      BasicBlockType::ExpHelper(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
      BasicBlockType::NonStructuredExp(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
      BasicBlockType::TwoPow(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
      BasicBlockType::ChangeShape(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
      BasicBlockType::ScaleDown(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
      BasicBlockType::ScaleUp(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
      BasicBlockType::NonNegative(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
      BasicBlockType::Permute(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
      BasicBlockType::Reducer(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
      BasicBlockType::ClampLower(b) => <ClampLower as BasicBlock<F>>::verify(b, witnesses, claims, sumcheck_proofs, transcript),
      BasicBlockType::ZeroCheck(b) => <ZeroCheck as BasicBlock<F>>::verify(b, witnesses, claims, sumcheck_proofs, transcript),
      // LLaMA specific
      BasicBlockType::RMSReciprocal(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
      BasicBlockType::Reciprocal(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
      BasicBlockType::DivConst(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
      BasicBlockType::SoftmaxConst(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
      BasicBlockType::SigmoidConst(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
    }
  }
}
