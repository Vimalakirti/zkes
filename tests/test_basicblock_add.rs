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

use zktransformer::basicblock::{Add, BasicBlock};
use zktransformer::dag::{Claim, DataType, PolyType, Role, Witness};
use zktransformer::util::arith::get_n;
use zktransformer::util::poly::{CryptoField, DenseMLPoly};
use zktransformer::util::transcript::Transcript;

fn create_test_witness(shape: Vec<usize>, data: Vec<F>, sf: usize, role: Role) -> Witness<F> {
  let n = get_n(&shape);
  let poly_data = Some(Box::new(DenseMLPoly::new(n, data)) as Box<dyn zktransformer::util::poly::MLPoly<F>>);
  Witness {
    shape,
    data: poly_data,
    poly_type: PolyType::Dense,
    data_type: DataType::Float,
    sf,
    role,
  }
}

#[test]
fn test_add_run_basic() {
  let add = Add;

  // Create two simple 1D tensors: [1, 2] + [3, 4] = [4, 6]
  let a = create_test_witness(
    vec![2],
    vec![<F as CryptoField>::from_u32(1u32), <F as CryptoField>::from_u32(2u32)],
    1,
    Role::Input,
  );
  let b = create_test_witness(
    vec![2],
    vec![<F as CryptoField>::from_u32(3u32), <F as CryptoField>::from_u32(4u32)],
    1,
    Role::Input,
  );

  let inputs = vec![&a, &b];
  let results = add.run(&inputs);

  assert_eq!(results.len(), 1, "Add should return 1 output");
  let c = &results[0];

  assert_eq!(c.shape, vec![2], "Output shape should match input shape");
  assert_eq!(c.sf, 1, "Scale factor should be preserved");
  assert_eq!(c.role, Role::Output, "Output should have Output role");

  // Check actual computation: [1, 2] + [3, 4] = [4, 6]
  assert_eq!(c.data.as_ref().unwrap().index(0), <F as CryptoField>::from_u32(4u32));
  assert_eq!(c.data.as_ref().unwrap().index(1), <F as CryptoField>::from_u32(6u32));
}

#[test]
fn test_add_run_zero_inputs() {
  let add = Add;

  // Test with zero values: [0, 0] + [0, 0] = [0, 0]
  let a = create_test_witness(vec![2], vec![<F as CryptoField>::zero(), <F as CryptoField>::zero()], 0, Role::Input);
  let b = create_test_witness(vec![2], vec![<F as CryptoField>::zero(), <F as CryptoField>::zero()], 0, Role::Input);

  let inputs = vec![&a, &b];
  let results = add.run(&inputs);

  let c = &results[0];
  assert_eq!(c.data.as_ref().unwrap().index(0), <F as CryptoField>::zero());
  assert_eq!(c.data.as_ref().unwrap().index(1), <F as CryptoField>::zero());
  assert_eq!(c.sf, 0, "Scale factor should match inputs");
}

#[test]
fn test_add_run_single_element() {
  let add = Add;

  // Test with single element: [5] + [7] = [12]
  let a = create_test_witness(vec![1], vec![<F as CryptoField>::from_u32(5u32)], 2, Role::Input);
  let b = create_test_witness(vec![1], vec![<F as CryptoField>::from_u32(7u32)], 2, Role::Input);

  let inputs = vec![&a, &b];
  let results = add.run(&inputs);

  let c = &results[0];
  assert_eq!(c.shape, vec![1]);
  assert_eq!(c.data.as_ref().unwrap().index(0), <F as CryptoField>::from_u32(12u32));
  assert_eq!(c.sf, 2);
}

#[test]
fn test_add_run_large_tensors() {
  let add = Add;

  // Test with larger tensors
  let size = 8;
  let a_data: Vec<F> = (1..=size).map(|i| <F as CryptoField>::from_u32(i as u32)).collect();
  let b_data: Vec<F> = (1..=size).map(|i| <F as CryptoField>::from_u32((i * 2) as u32)).collect();

  let a = create_test_witness(vec![size], a_data.clone(), 1, Role::Input);
  let b = create_test_witness(vec![size], b_data.clone(), 1, Role::Input);

  let inputs = vec![&a, &b];
  let results = add.run(&inputs);

  let c = &results[0];
  assert_eq!(c.shape, vec![size]);

  // Check each element: a[i] + b[i] = i + 2*i = 3*i
  for i in 0..size {
    let expected = <F as CryptoField>::from_u32((3 * (i + 1)) as u32);
    assert_eq!(c.data.as_ref().unwrap().index(i), expected);
  }
}

#[test]
#[should_panic(expected = "Add expects inputs with the same scale factor")]
fn test_add_run_mismatched_scale_factors() {
  let add = Add;

  let a = create_test_witness(
    vec![2],
    vec![<F as CryptoField>::from_u32(1u32), <F as CryptoField>::from_u32(2u32)],
    1,
    Role::Input,
  );
  let b = create_test_witness(
    vec![2],
    vec![<F as CryptoField>::from_u32(3u32), <F as CryptoField>::from_u32(4u32)],
    2,
    Role::Input,
  ); // Different sf

  let inputs = vec![&a, &b];
  add.run(&inputs);
}

#[test]
#[should_panic]
fn test_add_run_mismatched_shapes() {
  let add = Add;

  let a = create_test_witness(
    vec![2],
    vec![<F as CryptoField>::from_u32(1u32), <F as CryptoField>::from_u32(2u32)],
    1,
    Role::Input,
  );
  let b = create_test_witness(
    vec![3],
    vec![
      <F as CryptoField>::from_u32(1u32),
      <F as CryptoField>::from_u32(2u32),
      <F as CryptoField>::from_u32(3u32),
    ],
    1,
    Role::Input,
  );

  let inputs = vec![&a, &b];
  add.run(&inputs);
}

#[test]
fn test_add_prove_verify_round_trip() {
  let add = Add;

  // Create test data
  let a = create_test_witness(
    vec![2],
    vec![<F as CryptoField>::from_u32(1u32), <F as CryptoField>::from_u32(2u32)],
    1,
    Role::Input,
  );
  let b = create_test_witness(
    vec![2],
    vec![<F as CryptoField>::from_u32(3u32), <F as CryptoField>::from_u32(4u32)],
    1,
    Role::Input,
  );
  let c = create_test_witness(
    vec![2],
    vec![<F as CryptoField>::from_u32(4u32), <F as CryptoField>::from_u32(6u32)],
    1,
    Role::Output,
  );

  let witnesses = vec![&a, &b, &c];
  let edge_ids = vec![0, 1, 2]; // edge IDs for a, b, c

  // Create claims for the output
  let out_claim = Claim {
    edge_id: 2,
    point: vec![<F as CryptoField>::from_u32(0u32)], // Evaluate at point [0]
    eval: <F as CryptoField>::from_u32(4u32),        // c[0] = 4
  };
  let out_claims = vec![&out_claim];

  let mut transcript = Transcript::new(b"add_test");

  // Generate proof
  let (proofs, claims) = add.prove(&witnesses, &edge_ids, &out_claims, &mut transcript);

  assert_eq!(proofs.len(), 1, "Should generate 1 proof for 1 output claim");
  assert_eq!(claims.len(), 2, "Should generate 2 input claims (for a and b)");

  // Verify the proof
  let mut verify_transcript = Transcript::new(b"add_test");
  let verify_claims = vec![&claims[0], &claims[1], &out_claim];
  let verify_proofs = vec![&proofs[0]];

  let verified = add.verify(&vec![], &verify_claims, &verify_proofs, &mut verify_transcript);
  assert!(verified, "Verification should succeed for valid proof");
}

#[test]
fn test_add_verify_claim_correctness() {
  let add = Add;

  // Test that verification checks a + b = c correctly
  let point = vec![<F as CryptoField>::from_u32(1u32)];

  let a_claim = Claim {
    edge_id: 0,
    point: point.clone(),
    eval: <F as CryptoField>::from_u32(5u32),
  };
  let b_claim = Claim {
    edge_id: 1,
    point: point.clone(),
    eval: <F as CryptoField>::from_u32(7u32),
  };
  let c_claim = Claim {
    edge_id: 2,
    point: point.clone(),
    eval: <F as CryptoField>::from_u32(12u32), // 5 + 7 = 12
  };

  let claims = vec![&a_claim, &b_claim, &c_claim];
  let dummy_proof = zktransformer::crypto::SumcheckProof {
    final_eval: <F as CryptoField>::zero(),
    round_messages: vec![],
  };
  let proofs = vec![&dummy_proof];

  let mut transcript = Transcript::new(b"add_verify");
  let verified = add.verify(&vec![], &claims, &proofs, &mut transcript);
  assert!(verified, "Should verify when a + b = c");
}

#[test]
fn test_add_verify_incorrect_sum() {
  let add = Add;

  let point = vec![<F as CryptoField>::from_u32(1u32)];

  let a_claim = Claim {
    edge_id: 0,
    point: point.clone(),
    eval: <F as CryptoField>::from_u32(5u32),
  };
  let b_claim = Claim {
    edge_id: 1,
    point: point.clone(),
    eval: <F as CryptoField>::from_u32(7u32),
  };
  let c_claim = Claim {
    edge_id: 2,
    point: point.clone(),
    eval: <F as CryptoField>::from_u32(13u32), // Wrong! Should be 12
  };

  let claims = vec![&a_claim, &b_claim, &c_claim];
  let dummy_proof = zktransformer::crypto::SumcheckProof {
    final_eval: <F as CryptoField>::zero(),
    round_messages: vec![],
  };
  let proofs = vec![&dummy_proof];

  let mut transcript = Transcript::new(b"add_verify");
  let verified = add.verify(&vec![], &claims, &proofs, &mut transcript);
  assert!(!verified, "Should fail verification when a + b ≠ c");
}

#[test]
fn test_add_verify_mismatched_points() {
  let add = Add;

  let a_claim = Claim {
    edge_id: 0,
    point: vec![<F as CryptoField>::from_u32(1u32)],
    eval: <F as CryptoField>::from_u32(5u32),
  };
  let b_claim = Claim {
    edge_id: 1,
    point: vec![<F as CryptoField>::from_u32(2u32)], // Different point!
    eval: <F as CryptoField>::from_u32(7u32),
  };
  let c_claim = Claim {
    edge_id: 2,
    point: vec![<F as CryptoField>::from_u32(1u32)],
    eval: <F as CryptoField>::from_u32(12u32),
  };

  let claims = vec![&a_claim, &b_claim, &c_claim];
  let dummy_proof = zktransformer::crypto::SumcheckProof {
    final_eval: <F as CryptoField>::zero(),
    round_messages: vec![],
  };
  let proofs = vec![&dummy_proof];

  let mut transcript = Transcript::new(b"add_verify");
  let verified = add.verify(&vec![], &claims, &proofs, &mut transcript);
  assert!(!verified, "Should fail verification when evaluation points don't match");
}

#[test]
fn test_add_verify_multiple_claims() {
  let add = Add;

  // Test with multiple output claims
  let point1 = vec![<F as CryptoField>::from_u32(0u32)];
  let point2 = vec![<F as CryptoField>::from_u32(1u32)];

  // First set of claims: a=1, b=2, c=3 at point [0]
  let a_claim1 = Claim {
    edge_id: 0,
    point: point1.clone(),
    eval: <F as CryptoField>::from_u32(1u32),
  };
  let b_claim1 = Claim {
    edge_id: 1,
    point: point1.clone(),
    eval: <F as CryptoField>::from_u32(2u32),
  };
  let c_claim1 = Claim {
    edge_id: 2,
    point: point1.clone(),
    eval: <F as CryptoField>::from_u32(3u32),
  };

  // Second set of claims: a=4, b=5, c=9 at point [1]
  let a_claim2 = Claim {
    edge_id: 0,
    point: point2.clone(),
    eval: <F as CryptoField>::from_u32(4u32),
  };
  let b_claim2 = Claim {
    edge_id: 1,
    point: point2.clone(),
    eval: <F as CryptoField>::from_u32(5u32),
  };
  let c_claim2 = Claim {
    edge_id: 2,
    point: point2.clone(),
    eval: <F as CryptoField>::from_u32(9u32),
  };

  // Claims are ordered: [a0, a1, b0, b1, c0, c1]
  let claims = vec![&a_claim1, &a_claim2, &b_claim1, &b_claim2, &c_claim1, &c_claim2];

  let dummy_proof = zktransformer::crypto::SumcheckProof {
    final_eval: <F as CryptoField>::zero(),
    round_messages: vec![],
  };
  let proofs = vec![&dummy_proof, &dummy_proof];

  let mut transcript = Transcript::new(b"add_verify");
  let verified = add.verify(&vec![], &claims, &proofs, &mut transcript);
  assert!(verified, "Should verify multiple correct claims");
}
