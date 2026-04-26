use crate::basicblock::BasicBlock;
use crate::crypto::{LinearSumcheckProver, SumcheckProof, SumcheckProver, SumcheckVerifier};
use crate::dag::{Claim, DataType, PolyType, Role, Witness};
use crate::{SF_FLOAT, SF_LOG};

use crate::util::arith::{f_to_int, get_n, next_pow};
use crate::util::poly::{DenseMLPoly, MLPoly, SelectionPolynomial, SparseMLPoly};

use crate::util::poly::CryptoField;
use crate::util::transcript::Transcript;

/// Number of bits for the exp decomposition selection polynomial.
/// Caller must guarantee exp input x ≤ 0 (softmax subtracts row-max; sigmoid
/// emits x + SigmoidConst ≤ 0). k = round(-x / (ln2*SF)) ∈ [0, 2^K_BITS - 1].
/// K_BITS=4 covers k ∈ [0, 15], i.e., exp(x) for x ∈ [-15·ln2, 0] at SF=15.
pub const K_BITS: usize = 4;

/// ExpHelper: decomposes input x (assumed ≤ 0) into unsigned k and remainder r
/// such that x = k * (-ln2 * SF) + r, with k ∈ [0, 15] and |r| ≤ ln2/2 * SF.
///
/// Output[0]: r (dense polynomial, remainder for Taylor series)
/// Output[1]: auxiliary sparse polynomial (selection polynomial encoding k values)
#[derive(Debug, Clone)]
pub struct ExpHelper;
impl<F: CryptoField> BasicBlock<F> for ExpHelper {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    assert!(inputs.len() == 1, "ExpHelper expects 1 input");
    let x = inputs[0];
    let x_shape = &x.shape.iter().map(|&x| next_pow(x as u32) as usize).collect::<Vec<usize>>();
    let n: usize = x_shape.into_iter().product();
    let num_var = get_n(&x_shape);

    let ln2 = (2.0_f64.ln() * (*SF_FLOAT as f64)).round() as i128;
    let neg_ln2_f = <F as CryptoField>::zero() - F::from(ln2 as u32);

    let table_size = 1usize << K_BITS;
    let mut r_data = vec![<F as CryptoField>::zero(); n];
    let mut selection = Vec::new();
    for i in 0..n {
      let x_i = x.data.as_ref().unwrap().index(i);
      let x_i_num = f_to_int(x_i) as f64;
      let k = ((-x_i_num) / (ln2 as f64)).round();

      r_data[i] = x_i - (F::from(k as u32) * neg_ln2_f);

      // Out-of-range entries (e.g. causal-masked scores) clamp to 0; their exp
      // is then incorrect, but soundness is unaffected and the masked positions
      // are still suppressed heavily by the 1.0 vs 2^15 ratio in softmax.
      let k_idx = k as i64;
      let table_index = if k_idx >= 0 && (k_idx as usize) < table_size {
        k_idx as usize
      } else {
        0
      };
      selection.push((i, table_index));
    }
    let selection_polynomial = SelectionPolynomial::new(num_var, K_BITS, selection);
    let r = Witness::new(inputs[0].shape.clone(), r_data, DataType::Float, *SF_LOG, Role::Output);
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

/// Non-structured exponentiation baseline for ablation.
/// Uses a committed lookup table instead of structured decomposition.
#[derive(Debug, Clone)]
pub struct NonStructuredExp;
impl<F: CryptoField> BasicBlock<F> for NonStructuredExp {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    assert!(inputs.len() == 2, "NonStructuredExp expects 2 inputs");
    let x = inputs[0];
    let x_shape = &x.shape.iter().map(|&x| next_pow(x as u32) as usize).collect::<Vec<usize>>();
    let n: usize = x_shape.into_iter().product();
    let num_var = get_n(&x_shape);

    let ln2 = (2.0_f32.ln() * *SF_FLOAT).round() as f64;
    let neg_ln2_f = <F as CryptoField>::zero() - F::from(ln2 as u32);

    let table_size = 1usize << 10;
    let mut r_data = vec![<F as CryptoField>::zero(); n];
    let mut selection = Vec::new();
    for i in 0..n {
      let x_i = x.data.as_ref().unwrap().index(i);
      let x_i_num = f_to_int(x_i) as f64;
      let k = ((-x_i_num) / ln2).round();

      r_data[i] = x_i - (F::from(k as u32) * neg_ln2_f);

      let k_idx = k as i64;
      let table_index = if k_idx >= 0 && (k_idx as usize) < table_size {
        k_idx as usize
      } else {
        0
      };
      selection.push((i, table_index));
    }
    let selection_polynomial = SelectionPolynomial::new(num_var, 10, selection);
    let r = Witness::new(inputs[0].shape.clone(), r_data, DataType::Float, *SF_LOG, Role::Output);
    let aux = Witness {
      shape: inputs[0].shape.clone(),
      data: Some(Box::new(selection_polynomial.to_sparse()) as Box<dyn crate::util::poly::MLPoly<F>>),
      data_int: None,
      poly_type: PolyType::Sparse,
      data_type: x.data_type,
      sf: 0,
      role: Role::Auxiliary,
      is_permuted_with: None,
    };
    vec![r, aux]
  }

  fn prove(
    &self,
    witnesses: &[&Witness<F>],
    edge_ids: &[usize],
    out_claims: &[&Claim<F>],
    transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    assert!(witnesses.len() == 4, "NonStructuredExp expects 2 inputs and 2 outputs");
    assert!(edge_ids.len() == 4, "NonStructuredExp expects 2 input edges and 2 output edges");
    let x = witnesses[0];
    let t = witnesses[1];
    let aux = witnesses[3];

    let mut sumcheck_prover = LinearSumcheckProver::new(t.data.as_ref().unwrap().n(), 2, transcript);

    let aux_poly = aux.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().expect("Expected SparseMLPoly for aux");
    let w_partial = aux_poly.fix_variables(&out_claims[0].point);
    let w_partial_dense = w_partial.to_dense();
    let t_poly = t.data.as_ref().unwrap().as_any().downcast_ref::<DenseMLPoly<F>>().expect("Expected DenseMLPoly for t").clone();

    let sumcheck_proof = sumcheck_prover.prove(&vec![w_partial_dense, t_poly], transcript);
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
    claims: &[&Claim<F>],
    sumcheck_proofs: &[&SumcheckProof<F>],
    transcript: &mut Transcript<F>,
  ) -> bool {
    assert!(sumcheck_proofs.len() == 1, "NonStructuredExp expects 1 sumcheck proof");
    let mut verifier = SumcheckVerifier::new(claims[0].point.len(), 2, transcript);
    let expected_sum = sumcheck_proofs[0].round_messages[0][0] + sumcheck_proofs[0].round_messages[0][1];
    let (verification_result, _challenges) = verifier.verify(transcript, sumcheck_proofs[0].round_messages.clone(), expected_sum);
    verification_result.is_some()
  }
}

/// TwoPow: table lookup for 2^(-k) at sf=15, i.e., table[k] = 2^(15 - k) for
/// k ∈ [0, 15]. Input is a SparseMLPoly from ExpHelper containing the selection.
/// Output scale factor is always 15.
#[derive(Debug, Clone)]
pub struct TwoPow;
impl<F: CryptoField> BasicBlock<F> for TwoPow {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    assert!(inputs.len() == 1, "TwoPow expects 1 input");
    let x = inputs[0];
    let x_data = x.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap();
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
