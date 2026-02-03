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
use zktransformer::util::poly::CryptoField;
use zktransformer::util::poly::{fix_variables_from_right, le_vec_to_usize, DenseMLPoly, SparseMLPoly};

#[test]
fn test_dense_fix_variables_single_variable() {
  // Test polynomial: f(x) = x (i.e., f(0) = 0, f(1) = 1)
  let evaluations = vec![<F as CryptoField>::zero(), <F as CryptoField>::one()];
  let poly = DenseMLPoly::from_evaluations(evaluations);

  // Fix x = 0, should get constant polynomial f = 0
  let partial_point = vec![<F as CryptoField>::zero()];
  let fixed_poly = poly.fix_variables(&partial_point);

  assert_eq!(fixed_poly.n, 0);
  assert_eq!(fixed_poly.evaluations.len(), 1);
  assert_eq!(fixed_poly.evaluations[0], <F as CryptoField>::zero());

  // Fix x = 1, should get constant polynomial f = 1
  let partial_point = vec![<F as CryptoField>::one()];
  let fixed_poly = poly.fix_variables(&partial_point);

  assert_eq!(fixed_poly.n, 0);
  assert_eq!(fixed_poly.evaluations.len(), 1);
  assert_eq!(fixed_poly.evaluations[0], <F as CryptoField>::one());

  // Fix x = 1/2, should get constant polynomial f = 1/2
  let half = <F as CryptoField>::from_u32(2u32).invert();
  let partial_point = vec![half];
  let fixed_poly = poly.fix_variables(&partial_point);

  assert_eq!(fixed_poly.n, 0);
  assert_eq!(fixed_poly.evaluations.len(), 1);
  assert_eq!(fixed_poly.evaluations[0], half);
}

#[test]
fn test_dense_fix_variables_two_variables() {
  // Test polynomial: f(x,y) with evaluations f(0,0)=1, f(1,0)=2, f(0,1)=3, f(1,1)=4
  let evaluations = vec![
    <F as CryptoField>::one(),
    <F as CryptoField>::from_u32(2u32),
    <F as CryptoField>::from_u32(3u32),
    <F as CryptoField>::from_u32(4u32),
  ];
  let poly = DenseMLPoly::from_evaluations(evaluations);

  // Fix x = 0, should get g(y) where g(0)=1, g(1)=3
  let partial_point = vec![<F as CryptoField>::zero()];
  let fixed_poly = poly.fix_variables(&partial_point);

  assert_eq!(fixed_poly.n, 1);
  assert_eq!(fixed_poly.evaluations.len(), 2);
  assert_eq!(fixed_poly.evaluations[0], <F as CryptoField>::one());
  assert_eq!(fixed_poly.evaluations[1], <F as CryptoField>::from_u32(3u32));

  // Fix x = 1, should get g(y) where g(0)=2, g(1)=4
  let partial_point = vec![<F as CryptoField>::one()];
  let fixed_poly = poly.fix_variables(&partial_point);

  assert_eq!(fixed_poly.n, 1);
  assert_eq!(fixed_poly.evaluations.len(), 2);
  assert_eq!(fixed_poly.evaluations[0], <F as CryptoField>::from_u32(2u32));
  assert_eq!(fixed_poly.evaluations[1], <F as CryptoField>::from_u32(4u32));

  // Fix both variables x = 0, y = 1, should get constant polynomial f = 3
  let partial_point = vec![<F as CryptoField>::zero(), <F as CryptoField>::one()];
  let fixed_poly = poly.fix_variables(&partial_point);

  assert_eq!(fixed_poly.n, 0);
  assert_eq!(fixed_poly.evaluations.len(), 1);
  assert_eq!(fixed_poly.evaluations[0], <F as CryptoField>::from_u32(3u32));
}

#[test]
fn test_dense_fix_variables_three_variables() {
  // Test polynomial: f(x,y,z) with evaluations at all 8 vertices
  let evaluations = vec![
    <F as CryptoField>::from_u32(1u32),
    <F as CryptoField>::from_u32(2u32),
    <F as CryptoField>::from_u32(3u32),
    <F as CryptoField>::from_u32(4u32),
    <F as CryptoField>::from_u32(5u32),
    <F as CryptoField>::from_u32(6u32),
    <F as CryptoField>::from_u32(7u32),
    <F as CryptoField>::from_u32(8u32),
  ];
  let poly = DenseMLPoly::from_evaluations(evaluations);

  // Fix x = 0, should get g(y,z) where g(0,0)=1, g(1,0)=3, g(0,1)=5, g(1,1)=7
  let partial_point = vec![<F as CryptoField>::zero()];
  let fixed_poly = poly.fix_variables(&partial_point);

  assert_eq!(fixed_poly.n, 2);
  assert_eq!(fixed_poly.evaluations.len(), 4);
  assert_eq!(fixed_poly.evaluations[0], <F as CryptoField>::from_u32(1u32)); // g(0,0) = f(0,0,0) = 1
  assert_eq!(fixed_poly.evaluations[1], <F as CryptoField>::from_u32(3u32)); // g(1,0) = f(0,1,0) = 3
  assert_eq!(fixed_poly.evaluations[2], <F as CryptoField>::from_u32(5u32)); // g(0,1) = f(0,0,1) = 5
  assert_eq!(fixed_poly.evaluations[3], <F as CryptoField>::from_u32(7u32)); // g(1,1) = f(0,1,1) = 7

  // Fix x = 0, y = 1, should get h(z) where h(0)=f(0,1,0)=3, h(1)=f(0,1,1)=7
  let partial_point = vec![<F as CryptoField>::zero(), <F as CryptoField>::one()];
  let fixed_poly = poly.fix_variables(&partial_point);

  assert_eq!(fixed_poly.n, 1);
  assert_eq!(fixed_poly.evaluations.len(), 2);
  assert_eq!(fixed_poly.evaluations[0], <F as CryptoField>::from_u32(3u32)); // h(0) = f(0,1,0) = 3
  assert_eq!(fixed_poly.evaluations[1], <F as CryptoField>::from_u32(7u32)); // h(1) = f(0,1,1) = 7
}

#[test]
fn test_dense_fix_variables_empty_partial_point() {
  let evaluations = vec![
    <F as CryptoField>::one(),
    <F as CryptoField>::from_u32(2u32),
    <F as CryptoField>::from_u32(3u32),
    <F as CryptoField>::from_u32(4u32),
  ];
  let poly = DenseMLPoly::from_evaluations(evaluations.clone());

  // Empty partial point should return the same polynomial
  let partial_point: Vec<F> = vec![];
  let fixed_poly = poly.fix_variables(&partial_point);

  assert_eq!(fixed_poly.n, poly.n);
  assert_eq!(fixed_poly.evaluations, evaluations);
}

#[test]
fn test_sparse_fix_variables_single_variable() {
  // Test sparse polynomial: f(x) = x, only non-zero at x=1
  let mut evaluations = HashMap::new();
  evaluations.insert(vec![1], <F as CryptoField>::one()); // f(1) = 1, f(0) = 0 (implicit)
  let indices: VecDeque<Vec<u8>> = [vec![1]].into();
  let poly = SparseMLPoly::new(1, evaluations, indices);

  // Fix x = 0, should get constant polynomial f = 0
  let partial_point = vec![<F as CryptoField>::zero()];
  let fixed_poly = poly.fix_variables(&partial_point);

  assert_eq!(fixed_poly.n, 0);
  // The result should have one entry at index 0 with value 0
  assert_eq!(fixed_poly.evaluations.len(), 1);
  assert_eq!(fixed_poly.evaluations[&vec![0]], <F as CryptoField>::zero());

  // Fix x = 1, should get constant polynomial f = 1
  let partial_point = vec![<F as CryptoField>::one()];
  let fixed_poly = poly.fix_variables(&partial_point);

  assert_eq!(fixed_poly.n, 0);
  assert_eq!(fixed_poly.evaluations.len(), 1);
  assert_eq!(fixed_poly.evaluations[&vec![0]], <F as CryptoField>::one());
}

#[test]
fn test_sparse_fix_variables_two_variables() {
  // Test sparse polynomial: f(x,y) with only a few non-zero entries
  let mut evaluations = HashMap::new();
  evaluations.insert(vec![0], <F as CryptoField>::from_u32(1u32)); // f(0,0) = 1
  evaluations.insert(vec![3], <F as CryptoField>::from_u32(4u32)); // f(1,1) = 4
  let indices: VecDeque<Vec<u8>> = [vec![0], vec![3]].into();
  let poly = SparseMLPoly::new(2, evaluations, indices);

  // Fix x = 0, should get g(y) where g(0)=f(0,0)=1, g(1)=f(0,1)=0
  let partial_point = vec![<F as CryptoField>::zero()];
  let fixed_poly = poly.fix_variables(&partial_point);

  assert_eq!(fixed_poly.n, 1);
  assert_eq!(fixed_poly.evaluations.len(), 2); // Both g(0) and g(1) are represented
  assert_eq!(fixed_poly.evaluations[&vec![0]], <F as CryptoField>::one()); // g(0) = f(0,0) = 1
  assert_eq!(fixed_poly.evaluations[&vec![1]], <F as CryptoField>::zero()); // g(1) = f(0,1) = 0

  // Fix x = 1, should get g(y) where g(0)=f(1,0)=0, g(1)=f(1,1)=4
  let partial_point = vec![<F as CryptoField>::one()];
  let fixed_poly = poly.fix_variables(&partial_point);

  assert_eq!(fixed_poly.n, 1);
  assert_eq!(fixed_poly.evaluations.len(), 2); // Both g(0) and g(1) need to be represented
  assert_eq!(fixed_poly.evaluations[&vec![0]], <F as CryptoField>::zero()); // g(0) = f(1,0) = 0
  assert_eq!(fixed_poly.evaluations[&vec![1]], <F as CryptoField>::from_u32(4u32));
  // g(1) = f(1,1) = 4
}

#[test]
fn test_sparse_fix_variables_interpolation() {
  // Test that sparse fix_variables gives same result as dense for the same polynomial
  let evaluations = vec![
    <F as CryptoField>::from_u32(1u32),
    <F as CryptoField>::from_u32(2u32),
    <F as CryptoField>::from_u32(3u32),
    <F as CryptoField>::from_u32(4u32),
  ];
  let dense_poly = DenseMLPoly::from_evaluations(evaluations.clone());

  // Create equivalent sparse polynomial
  let mut sparse_evaluations = HashMap::new();
  for (i, &val) in evaluations.iter().enumerate() {
    if val != <F as CryptoField>::zero() {
      sparse_evaluations.insert(vec![i as u8], val);
    }
  }
  let indices: VecDeque<Vec<u8>> = sparse_evaluations.keys().cloned().collect();
  let sparse_poly = SparseMLPoly::new(2, sparse_evaluations, indices);

  // Fix first variable for both polynomials
  let partial_point = vec![<F as CryptoField>::from_u32(3u32)];
  let dense_fixed = dense_poly.fix_variables(&partial_point);
  let sparse_fixed = sparse_poly.fix_variables(&partial_point);

  assert_eq!(dense_fixed.n, sparse_fixed.n);

  // Convert sparse result to dense for comparison
  let mut dense_from_sparse = vec![<F as CryptoField>::zero(); 1 << sparse_fixed.n];
  for (idx, &val) in sparse_fixed.evaluations.iter() {
    let idx = le_vec_to_usize(idx);
    println!("idx: {}", idx);
    dense_from_sparse[idx] = val;
  }

  // Check that both give the same result
  for (i, &expected) in dense_fixed.evaluations.iter().enumerate() {
    assert_eq!(dense_from_sparse[i], expected, "Mismatch at index {}", i);
  }
}

#[test]
fn test_sparse_fix_variables_empty_polynomial() {
  // Test with completely zero polynomial
  let evaluations = HashMap::new();
  let indices = VecDeque::new();
  let poly = SparseMLPoly::new(2, evaluations, indices);

  let partial_point = vec![<F as CryptoField>::from_u32(5u32)];
  let fixed_poly = poly.fix_variables(&partial_point);

  assert_eq!(fixed_poly.n, 1);
  assert!(fixed_poly.evaluations.is_empty());
}

#[test]
#[should_panic(expected = "invalid size of partial point")]
fn test_dense_fix_variables_invalid_partial_point_size() {
  let evaluations = vec![<F as CryptoField>::one(), <F as CryptoField>::from_u32(2u32)];
  let poly = DenseMLPoly::from_evaluations(evaluations);

  // Try to fix more variables than the polynomial has
  let partial_point = vec![<F as CryptoField>::one(), <F as CryptoField>::from_u32(2u32)];
  poly.fix_variables(&partial_point);
}

#[test]
#[should_panic(expected = "invalid partial point dimension")]
fn test_sparse_fix_variables_invalid_partial_point_size() {
  let mut evaluations = HashMap::new();
  evaluations.insert(vec![0], <F as CryptoField>::one());
  let indices: VecDeque<Vec<u8>> = [vec![0]].into();
  let poly = SparseMLPoly::new(1, evaluations, indices);

  // Try to fix more variables than the polynomial has
  let partial_point = vec![<F as CryptoField>::one(), <F as CryptoField>::from_u32(2u32)];
  poly.fix_variables(&partial_point);
}

#[test]
fn test_dense_fix_variables_from_right() {
  let evaluations = vec![
    <F as CryptoField>::from_u32(1u32),
    <F as CryptoField>::from_u32(2u32),
    <F as CryptoField>::from_u32(3u32),
    <F as CryptoField>::from_u32(4u32),
  ];
  let poly = DenseMLPoly::from_evaluations(evaluations.clone());

  let partial_suffix = vec![<F as CryptoField>::from_u32(5u32), <F as CryptoField>::from_u32(6u32)];
  let fixed_poly = fix_variables_from_right(&poly, &partial_suffix);

  let poly = poly.fix_variables(&partial_suffix);

  assert_eq!(fixed_poly.evaluations[0], poly.evaluations[0]);

  let new_poly = DenseMLPoly::from_evaluations(evaluations);
  let new_poly = fix_variables_from_right(&new_poly, &[<F as CryptoField>::from_u32(5u32)]);

  assert_eq!(
    new_poly.evaluations[0],
    (<F as CryptoField>::from_u32(1u32) - <F as CryptoField>::from_u32(5u32)) * <F as CryptoField>::from_u32(1u32)
      + <F as CryptoField>::from_u32(5u32) * <F as CryptoField>::from_u32(3u32)
  );
  assert_eq!(
    new_poly.evaluations[1],
    (<F as CryptoField>::from_u32(1u32) - <F as CryptoField>::from_u32(5u32)) * <F as CryptoField>::from_u32(2u32)
      + <F as CryptoField>::from_u32(5u32) * <F as CryptoField>::from_u32(4u32)
  );
}
