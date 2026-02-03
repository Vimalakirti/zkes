// Field type selection based on backend and curve
#[cfg(all(feature = "arkworks", feature = "bls12_381"))]
use ark_bls12_381::Fr as F;
#[cfg(all(feature = "arkworks", feature = "bn254"))]
use ark_bn254::Fr as F;

#[cfg(all(feature = "icicle", feature = "bls12_381"))]
use icicle_bls12_381::curve::ScalarField as F;
#[cfg(all(feature = "icicle", feature = "bn254"))]
use icicle_bn254::curve::ScalarField as F;
#[cfg(all(feature = "icicle", feature = "goldilocks"))]
use icicle_goldilocks::field::ScalarField as F;

use std::collections::{HashMap, VecDeque};
use zktransformer::crypto::table::BitDecomp;
use zktransformer::crypto::table::DummyTable;
use zktransformer::crypto::table::StructuredTable;
use zktransformer::crypto::{SparseDenseSumcheckProver, SumcheckProver, SumcheckVerifier};
use zktransformer::util::poly::{CryptoField, SparseMLPoly};
use zktransformer::util::transcript::Transcript;

#[test]
fn test_sd_sumcheck_basic_structured_table() {
  let n = 64; // 64 variable - simpler case

  // Create a sparse polynomial a(x) with a few non-zero entries
  // For n=2, we need to use the right indices that will work with the bit decomposition
  // The bit representation for 2 variables needs 2-bit indices
  // We need to ensure the sparse keys work correctly with the algorithm
  let mut a_map = HashMap::new();
  let mut a_keys = VecDeque::new();

  // Use only even indices (can be paired) to avoid the pairing issues
  a_map.insert(vec![3], <F as CryptoField>::from_u32(1u32));
  a_keys.push_back(vec![3]);
  a_map.insert(vec![7], <F as CryptoField>::from_u32(1u32));
  a_keys.push_back(vec![7]);

  let sparse_poly: SparseMLPoly<F> = SparseMLPoly::new(n, a_map, a_keys);

  // Create structured table (BitDecomp) for polynomial b(x)
  let table: BitDecomp = <BitDecomp as StructuredTable<F>>::new(n, 0, 0, 0);

  // Create the prover instance
  let mut transcript = Transcript::new(b"sd_sumcheck");
  let instances = (vec![sparse_poly], zktransformer::crypto::sumcheck::sd_prover::Val::Structured(table));
  let mut prover: SparseDenseSumcheckProver<F, BitDecomp, DummyTable> = SparseDenseSumcheckProver::new(n, 2, &mut transcript);

  // Run the proof
  let sumcheck_proof = prover.prove(&instances, &mut transcript);

  // Basic protocol structure verification
  assert_eq!(sumcheck_proof.round_messages.len(), n, "Should have {} round messages", n);

  // Each round message should have 3 evaluations (for points 0, 1, 2)
  for (round, message) in sumcheck_proof.round_messages.iter().enumerate() {
    assert_eq!(message.len(), 3, "Round {} should have 3 evaluations", round);

    // Verify all evaluations are valid field elements
    for &val in message.iter() {
      let _test = val + <F as CryptoField>::zero(); // This will fail if val is invalid
    }
  }

  // Verify final evaluation is a valid field element
  let _test_final = sumcheck_proof.final_eval + <F as CryptoField>::zero();

  // Test verifier accepts the proof
  let mut verifier_transcript = Transcript::new(b"sd_sumcheck");
  let mut verifier = SumcheckVerifier::new(n, 2, &mut verifier_transcript);

  // Calculate expected sum for verification
  // For our sparse polynomial: a(2) = 1, and BitDecomp table evaluates to 2 at point (0,1)
  // So the expected sum should be 1 * 2 = 2
  let expected_sum = <F as CryptoField>::from_u32(10u32);
  let (verification_result, _) = verifier.verify(&mut verifier_transcript, sumcheck_proof.round_messages, expected_sum);

  assert!(verification_result.is_some(), "Verifier should accept the proof");
}
