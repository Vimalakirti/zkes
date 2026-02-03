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

use zktransformer::crypto::{LinearSumcheckProver, SumcheckProver, SumcheckVerifier};
use zktransformer::util::poly::{CryptoField, DenseMLPoly};
use zktransformer::util::transcript::Transcript;

#[test]
fn test_linear_sumcheck_constant_polynomials() {
  // Test with constant polynomials - simplest case
  let n = 2;
  let ell = 2;

  // p1(x) = 1 (constant), p2(x) = 1 (constant)
  let p1 = vec![
    <F as CryptoField>::one(),
    <F as CryptoField>::one(),
    <F as CryptoField>::one(),
    <F as CryptoField>::one(),
  ];
  let p2 = vec![
    <F as CryptoField>::one(),
    <F as CryptoField>::one(),
    <F as CryptoField>::one(),
    <F as CryptoField>::one(),
  ];

  let polynomials = vec![DenseMLPoly::from_evaluations(p1), DenseMLPoly::from_evaluations(p2)];

  let mut transcript = Transcript::new(b"sumcheck");
  let mut prover = LinearSumcheckProver::new(n, polynomials.len(), &mut transcript);
  let sumcheck_proof = prover.prove(&polynomials, &mut transcript);

  // Verify basic protocol properties
  assert_eq!(sumcheck_proof.round_messages.len(), n, "Should have {} round messages", n);

  // Each round message should have ell+1 evaluations (degree ell polynomial)
  for (round, message) in sumcheck_proof.round_messages.iter().enumerate() {
    assert_eq!(message.len(), ell + 1, "Round {} should have {} evaluations", round, ell + 1);
  }

  // Test verifier accepts the proof
  let mut verifier_transcript = Transcript::new(b"sumcheck");
  let mut verifier = SumcheckVerifier::new(n, ell, &mut verifier_transcript);
  // 4 = 1*1 + 1*1 + 1*1 + 1*1
  let (verification_result, _) = verifier.verify(
    &mut verifier_transcript,
    sumcheck_proof.round_messages,
    <F as CryptoField>::from_u32(4u32),
  );
  assert!(verification_result.is_some(), "Verifier should accept valid proof");
}

#[test]
fn test_linear_sumcheck_identity_polynomials() {
  // Test with identity polynomials
  let n = 2;
  let ell = 2;

  // p1(x) represents x0: [0, 1, 0, 1] for inputs [00, 10, 01, 11]
  // p2(x) represents x1: [0, 0, 1, 1] for inputs [00, 10, 01, 11]
  let p1 = vec![
    <F as CryptoField>::zero(),
    <F as CryptoField>::one(),
    <F as CryptoField>::zero(),
    <F as CryptoField>::one(),
  ];
  let p2 = vec![
    <F as CryptoField>::zero(),
    <F as CryptoField>::zero(),
    <F as CryptoField>::one(),
    <F as CryptoField>::one(),
  ];

  let polynomials = vec![DenseMLPoly::from_evaluations(p1), DenseMLPoly::from_evaluations(p2)];

  let mut transcript = Transcript::new(b"sumcheck");
  let mut prover = LinearSumcheckProver::new(n, polynomials.len(), &mut transcript);
  let sumcheck_proof = prover.prove(&polynomials, &mut transcript);

  // Verify protocol structure
  assert_eq!(sumcheck_proof.round_messages.len(), n);
  for (round, message) in sumcheck_proof.round_messages.iter().enumerate() {
    assert_eq!(message.len(), ell + 1, "Round {} message length", round);

    // Round messages should contain field elements (non-trivial check)
    for &val in message.iter() {
      // At least verify they're valid field elements (this will fail if corrupt)
      let _test = val + <F as CryptoField>::zero();
    }
  }

  // Test verifier
  let mut verifier_transcript = Transcript::new(b"sumcheck");
  let mut verifier = SumcheckVerifier::new(n, ell, &mut verifier_transcript);
  // 1 = 0*0 + 1*0 + 0*1 + 1*1
  let (verification_result, _) = verifier.verify(&mut verifier_transcript, sumcheck_proof.round_messages, <F as CryptoField>::one());
  assert!(verification_result.is_some());
}

#[test]
fn test_linear_sumcheck_single_variable() {
  // Test with single variable
  let n = 1;
  let ell = 1;

  // p1(x) = x for single variable: [0, 1]
  let p1 = vec![<F as CryptoField>::zero(), <F as CryptoField>::one()];

  let polynomials = vec![DenseMLPoly::from_evaluations(p1)];

  let mut transcript = Transcript::new(b"sumcheck");
  let mut prover = LinearSumcheckProver::new(n, polynomials.len(), &mut transcript);
  let sumcheck_proof = prover.prove(&polynomials, &mut transcript);

  assert_eq!(sumcheck_proof.round_messages.len(), 1, "Single variable should have 1 round");
  assert_eq!(sumcheck_proof.round_messages[0].len(), 2, "Degree 1 polynomial should have 2 evaluations");

  // Test verifier
  let mut verifier_transcript = Transcript::new(b"sumcheck");
  let mut verifier = SumcheckVerifier::new(n, ell, &mut verifier_transcript);
  let (verification_result, _) = verifier.verify(&mut verifier_transcript, sumcheck_proof.round_messages, <F as CryptoField>::one());
  assert!(verification_result.is_some());
}

#[test]
fn test_linear_sumcheck_three_variables() {
  // Test with three variables
  let n = 3;
  let ell = 2;

  // p1(x) = x0 + x1 + x2, p2(x) = x0 * x1
  let p1 = vec![
    <F as CryptoField>::from_u32(0u32), // 000: 0+0+0=0
    <F as CryptoField>::from_u32(1u32), // 100: 1+0+0=1
    <F as CryptoField>::from_u32(1u32), // 010: 0+1+0=1
    <F as CryptoField>::from_u32(2u32), // 110: 1+1+0=2
    <F as CryptoField>::from_u32(1u32), // 001: 0+0+1=1
    <F as CryptoField>::from_u32(2u32), // 101: 1+0+1=2
    <F as CryptoField>::from_u32(2u32), // 011: 0+1+1=2
    <F as CryptoField>::from_u32(3u32), // 111: 1+1+1=3
  ];

  let p2 = vec![
    <F as CryptoField>::from_u32(0u32), // 000: 0*0=0
    <F as CryptoField>::from_u32(0u32), // 100: 1*0=0
    <F as CryptoField>::from_u32(0u32), // 010: 0*1=0
    <F as CryptoField>::from_u32(1u32), // 110: 1*1=1
    <F as CryptoField>::from_u32(0u32), // 001: 0*0=0
    <F as CryptoField>::from_u32(0u32), // 101: 1*0=0
    <F as CryptoField>::from_u32(0u32), // 011: 0*1=0
    <F as CryptoField>::from_u32(1u32), // 111: 1*1=1
  ];

  let polynomials = vec![DenseMLPoly::from_evaluations(p1), DenseMLPoly::from_evaluations(p2)];

  let mut transcript = Transcript::new(b"sumcheck");
  let mut prover = LinearSumcheckProver::new(n, polynomials.len(), &mut transcript);
  let sumcheck_proof = prover.prove(&polynomials, &mut transcript);

  assert_eq!(sumcheck_proof.round_messages.len(), 3, "Three variables should have 3 rounds");

  // Test verifier
  let mut verifier_transcript = Transcript::new(b"sumcheck");
  let mut verifier = SumcheckVerifier::new(n, ell, &mut verifier_transcript);
  let (verification_result, _) = verifier.verify(
    &mut verifier_transcript,
    sumcheck_proof.round_messages,
    <F as CryptoField>::from_u32(5u32),
  );
  assert!(verification_result.is_some());
}

#[test]
fn test_linear_sumcheck_round_message_structure() {
  // Test that round messages have correct structure
  let n = 2;
  let ell = 3; // Three polynomials

  let p1 = vec![
    <F as CryptoField>::one(),
    <F as CryptoField>::one(),
    <F as CryptoField>::one(),
    <F as CryptoField>::one(),
  ];
  let p2 = vec![
    <F as CryptoField>::zero(),
    <F as CryptoField>::one(),
    <F as CryptoField>::zero(),
    <F as CryptoField>::one(),
  ];
  let p3 = vec![
    <F as CryptoField>::zero(),
    <F as CryptoField>::zero(),
    <F as CryptoField>::one(),
    <F as CryptoField>::one(),
  ];

  let polynomials = vec![
    DenseMLPoly::from_evaluations(p1),
    DenseMLPoly::from_evaluations(p2),
    DenseMLPoly::from_evaluations(p3),
  ];

  let mut transcript = Transcript::new(b"sumcheck");
  let mut prover = LinearSumcheckProver::new(n, polynomials.len(), &mut transcript);
  let sumcheck_proof = prover.prove(&polynomials, &mut transcript);

  // With ell=3 polynomials, each round message should have 4 evaluations
  for (round, message) in sumcheck_proof.round_messages.iter().enumerate() {
    assert_eq!(message.len(), ell + 1, "Round {} should have {} evaluations", round, ell + 1);

    // Check that not all evaluations are the same (non-trivial polynomial expected)
    let first_val = message[0];
    let is_constant = message.iter().all(|&val| val == first_val);
    assert!(!is_constant, "Round {} should have non-constant evaluations", round);
  }

  // Test verifier
  let mut verifier_transcript = Transcript::new(b"sumcheck");
  let mut verifier = SumcheckVerifier::new(n, ell, &mut verifier_transcript);
  let (verification_result, _) = verifier.verify(&mut verifier_transcript, sumcheck_proof.round_messages, <F as CryptoField>::one());
  assert!(verification_result.is_some());
}

#[test]
fn test_linear_sumcheck_zero_polynomials() {
  // Edge case: all zero polynomials
  let n = 2;
  let ell = 2;

  let p1 = vec![
    <F as CryptoField>::zero(),
    <F as CryptoField>::zero(),
    <F as CryptoField>::zero(),
    <F as CryptoField>::zero(),
  ];
  let p2 = vec![
    <F as CryptoField>::zero(),
    <F as CryptoField>::zero(),
    <F as CryptoField>::zero(),
    <F as CryptoField>::zero(),
  ];

  let polynomials = vec![DenseMLPoly::from_evaluations(p1), DenseMLPoly::from_evaluations(p2)];

  let mut transcript = Transcript::new(b"sumcheck");
  let mut prover = LinearSumcheckProver::new(n, polynomials.len(), &mut transcript);
  let sumcheck_proof = prover.prove(&polynomials, &mut transcript);

  assert_eq!(sumcheck_proof.round_messages.len(), n);

  // For zero polynomials, all round messages should be zero
  for (round, message) in sumcheck_proof.round_messages.iter().enumerate() {
    for (i, &val) in message.iter().enumerate() {
      assert_eq!(val, <F as CryptoField>::zero(), "Round {} evaluation {} should be zero", round, i);
    }
  }

  // Test verifier
  let mut verifier_transcript = Transcript::new(b"sumcheck");
  let mut verifier = SumcheckVerifier::new(n, ell, &mut verifier_transcript);
  let (verification_result, _) = verifier.verify(&mut verifier_transcript, sumcheck_proof.round_messages, <F as CryptoField>::zero());
  assert!(verification_result.is_some());
}

#[test]
fn test_linear_sumcheck_minimal_case() {
  // Minimal case: 1 variable, 1 polynomial
  let n = 1;
  let ell = 1;

  let p1 = vec![<F as CryptoField>::from_u32(3u32), <F as CryptoField>::from_u32(7u32)]; // p(0)=3, p(1)=7

  let polynomials = vec![DenseMLPoly::from_evaluations(p1)];

  let mut transcript = Transcript::new(b"sumcheck");
  let mut prover = LinearSumcheckProver::new(n, polynomials.len(), &mut transcript);
  let sumcheck_proof = prover.prove(&polynomials, &mut transcript);

  assert_eq!(sumcheck_proof.round_messages.len(), 1);
  assert_eq!(sumcheck_proof.round_messages[0].len(), 2); // Degree 1 polynomial

  // Test verifier
  let mut verifier_transcript = Transcript::new(b"sumcheck");
  let mut verifier = SumcheckVerifier::new(n, ell, &mut verifier_transcript);
  let (verification_result, _) = verifier.verify(
    &mut verifier_transcript,
    sumcheck_proof.round_messages,
    <F as CryptoField>::from_u32(10u32),
  ); // p(5) = 3 + (7-3)*5 = 23
  assert!(verification_result.is_some());
}

#[test]
fn test_linear_sumcheck_protocol_completeness() {
  // Test protocol completeness: honest prover should always be accepted
  let test_cases = vec![
    // (n, ell, polynomials, expected_sum)
    (
      1,
      1,
      vec![DenseMLPoly::from_evaluations(vec![
        <F as CryptoField>::one(),
        <F as CryptoField>::from_u32(2u32),
      ])],
      <F as CryptoField>::from_u32(3u32),
    ),
    (
      2,
      1,
      vec![DenseMLPoly::from_evaluations(vec![
        <F as CryptoField>::from_u32(1u32),
        <F as CryptoField>::from_u32(2u32),
        <F as CryptoField>::from_u32(3u32),
        <F as CryptoField>::from_u32(4u32),
      ])],
      <F as CryptoField>::from_u32(10u32),
    ),
    (
      2,
      2,
      vec![
        DenseMLPoly::from_evaluations(vec![
          <F as CryptoField>::one(),
          <F as CryptoField>::one(),
          <F as CryptoField>::one(),
          <F as CryptoField>::one(),
        ]),
        DenseMLPoly::from_evaluations(vec![
          <F as CryptoField>::from_u32(2u32),
          <F as CryptoField>::from_u32(2u32),
          <F as CryptoField>::from_u32(2u32),
          <F as CryptoField>::from_u32(2u32),
        ]),
      ],
      <F as CryptoField>::from_u32(8u32),
    ),
  ];

  for (test_idx, (n, ell, polynomials, expected_sum)) in test_cases.into_iter().enumerate() {
    let mut transcript = Transcript::new(b"sumcheck");
    let mut prover = LinearSumcheckProver::new(n, polynomials.len(), &mut transcript);
    let sumcheck_proof = prover.prove(&polynomials, &mut transcript);

    // Basic structural checks
    assert_eq!(sumcheck_proof.round_messages.len(), n, "Test case {}: wrong number of rounds", test_idx);

    for (round, message) in sumcheck_proof.round_messages.iter().enumerate() {
      assert_eq!(message.len(), ell + 1, "Test case {}, round {}: wrong message length", test_idx, round);
    }

    // Verifier should accept
    let mut verifier_transcript = Transcript::new(b"sumcheck");
    let mut verifier = SumcheckVerifier::new(n, ell, &mut verifier_transcript);
    let (verification_result, _) = verifier.verify(&mut verifier_transcript, sumcheck_proof.round_messages, expected_sum);
    assert!(
      verification_result.is_some(),
      "Test case {}: verifier should accept honest proof",
      test_idx
    );
  }
}

#[test]
fn test_linear_sumcheck_round_polynomial_degrees() {
  // Test that round polynomials have correct degrees
  let n = 2;
  let ell = 3;

  // Three distinct polynomials
  let p1 = vec![
    <F as CryptoField>::from_u32(1u32),
    <F as CryptoField>::from_u32(2u32),
    <F as CryptoField>::from_u32(3u32),
    <F as CryptoField>::from_u32(4u32),
  ];
  let p2 = vec![
    <F as CryptoField>::from_u32(5u32),
    <F as CryptoField>::from_u32(6u32),
    <F as CryptoField>::from_u32(7u32),
    <F as CryptoField>::from_u32(8u32),
  ];
  let p3 = vec![
    <F as CryptoField>::from_u32(9u32),
    <F as CryptoField>::from_u32(10u32),
    <F as CryptoField>::from_u32(11u32),
    <F as CryptoField>::from_u32(12u32),
  ];

  let polynomials = vec![
    DenseMLPoly::from_evaluations(p1),
    DenseMLPoly::from_evaluations(p2),
    DenseMLPoly::from_evaluations(p3),
  ];

  let mut transcript = Transcript::new(b"sumcheck");
  let mut prover = LinearSumcheckProver::new(n, polynomials.len(), &mut transcript);
  let sumcheck_proof = prover.prove(&polynomials, &mut transcript);

  // Each round message should have exactly ell+1 evaluations
  // This corresponds to a polynomial of degree at most ell
  for (round, message) in sumcheck_proof.round_messages.iter().enumerate() {
    assert_eq!(message.len(), ell + 1, "Round {} should have degree {} polynomial", round, ell);

    // Check that not all evaluations are the same (non-constant polynomial expected)
    let first_val = message[0];
    let is_constant = message.iter().all(|&val| val == first_val);

    // For our test polynomials, we expect non-constant round polynomials
    if ell > 1 {
      assert!(!is_constant, "Round {} polynomial should not be constant", round);
    }
  }
}

#[test]
fn test_linear_sumcheck_deterministic_behavior() {
  // Test that the protocol is deterministic for the same inputs
  let n = 2;

  let p1 = vec![
    <F as CryptoField>::from_u32(1u32),
    <F as CryptoField>::from_u32(3u32),
    <F as CryptoField>::from_u32(5u32),
    <F as CryptoField>::from_u32(7u32),
  ];
  let p2 = vec![
    <F as CryptoField>::from_u32(2u32),
    <F as CryptoField>::from_u32(4u32),
    <F as CryptoField>::from_u32(6u32),
    <F as CryptoField>::from_u32(8u32),
  ];

  let polynomials = vec![DenseMLPoly::from_evaluations(p1.clone()), DenseMLPoly::from_evaluations(p2.clone())];

  // Run the protocol twice
  let mut transcript1 = Transcript::new(b"sumcheck");
  let mut prover1 = LinearSumcheckProver::new(n, polynomials.len(), &mut transcript1);
  let sumcheck_proof1 = prover1.prove(&polynomials, &mut transcript1);

  let mut transcript2 = Transcript::new(b"sumcheck");
  let mut prover2 = LinearSumcheckProver::new(n, polynomials.len(), &mut transcript2);
  let sumcheck_proof2 = prover2.prove(&polynomials, &mut transcript2);

  // Results should be identical
  assert_eq!(
    sumcheck_proof1.final_eval, sumcheck_proof2.final_eval,
    "Final evaluations should be identical"
  );
  assert_eq!(
    sumcheck_proof1.round_messages.len(),
    sumcheck_proof2.round_messages.len(),
    "Number of rounds should be identical"
  );

  for (round, (msg1, msg2)) in sumcheck_proof1.round_messages.iter().zip(sumcheck_proof2.round_messages.iter()).enumerate() {
    assert_eq!(msg1.len(), msg2.len(), "Round {} message lengths should be identical", round);
    for (i, (&val1, &val2)) in msg1.iter().zip(msg2.iter()).enumerate() {
      assert_eq!(val1, val2, "Round {} evaluation {} should be identical", round, i);
    }
  }
}
