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
use zktransformer::util::poly::{evaluate_lagrange_basis, CryptoField};

#[test]
fn test_lagrange_basis_single_variable() {
  let r0 = <F as CryptoField>::from_u32(3u32);
  let r = vec![r0];

  let basis = evaluate_lagrange_basis(&r);

  assert_eq!(basis.len(), 2);
  assert_eq!(basis[0], <F as CryptoField>::one() - r0);
  assert_eq!(basis[1], r0);
}

#[test]
fn test_lagrange_basis_two_variables() {
  let r0 = <F as CryptoField>::from_u32(2u32);
  let r1 = <F as CryptoField>::from_u32(5u32);
  let r = vec![r0, r1];

  let basis = evaluate_lagrange_basis(&r);

  // For 2 variables, should have 2^2 = 4 basis elements
  assert_eq!(basis.len(), 4);

  // Check specific values for L_{00}, L_{10}, L_{01}, L_{11}
  assert_eq!(basis[0], (<F as CryptoField>::one() - r0) * (<F as CryptoField>::one() - r1)); // L_{00}
  assert_eq!(basis[1], r0 * (<F as CryptoField>::one() - r1)); // L_{10}
  assert_eq!(basis[2], (<F as CryptoField>::one() - r0) * r1); // L_{01}
  assert_eq!(basis[3], r0 * r1); // L_{11}
}

#[test]
fn test_lagrange_basis_three_variables() {
  let r0 = <F as CryptoField>::from_u32(1u32);
  let r1 = <F as CryptoField>::from_u32(2u32);
  let r2 = <F as CryptoField>::from_u32(3u32);
  let r = vec![r0, r1, r2];

  let basis = evaluate_lagrange_basis(&r);

  // For 3 variables, should have 2^3 = 8 basis elements
  assert_eq!(basis.len(), 8);

  // Check a few specific values
  assert_eq!(
    basis[0],
    (<F as CryptoField>::one() - r0) * (<F as CryptoField>::one() - r1) * (<F as CryptoField>::one() - r2)
  ); // L_{000}
  assert_eq!(basis[7], r0 * r1 * r2); // L_{111}
}

#[test]
fn test_lagrange_basis_partition_of_unity() {
  // Test that the basis functions sum to 1 (partition of unity property)
  let r = vec![<F as CryptoField>::from_u32(7u32), <F as CryptoField>::from_u32(11u32)];
  let basis = evaluate_lagrange_basis(&r);

  let sum: F = basis.iter().fold(<F as CryptoField>::zero(), |acc, &val| acc + val);
  assert_eq!(sum, <F as CryptoField>::one());
}

#[test]
fn test_lagrange_basis_edge_cases() {
  // Test with zero values
  let r = vec![<F as CryptoField>::zero(), <F as CryptoField>::zero()];
  let basis = evaluate_lagrange_basis(&r);

  assert_eq!(basis.len(), 4);
  assert_eq!(basis[0], <F as CryptoField>::one()); // L_{00} = (1-0)*(1-0) = 1
  assert_eq!(basis[1], <F as CryptoField>::zero()); // L_{10} = 0*(1-0) = 0
  assert_eq!(basis[2], <F as CryptoField>::zero()); // L_{01} = (1-0)*0 = 0
  assert_eq!(basis[3], <F as CryptoField>::zero()); // L_{11} = 0*0 = 0

  // Test with one values
  let r = vec![<F as CryptoField>::one()];
  let basis = evaluate_lagrange_basis(&r);

  assert_eq!(basis.len(), 2);
  assert_eq!(basis[0], <F as CryptoField>::zero()); // L_0 = 1-1 = 0
  assert_eq!(basis[1], <F as CryptoField>::one()); // L_1 = 1
}

#[test]
fn test_lagrange_basis_interpolation_property() {
  // Test at each hypercube vertex
  let vertices = vec![
    vec![<F as CryptoField>::zero(), <F as CryptoField>::zero()],
    vec![<F as CryptoField>::one(), <F as CryptoField>::zero()],
    vec![<F as CryptoField>::zero(), <F as CryptoField>::one()],
    vec![<F as CryptoField>::one(), <F as CryptoField>::one()],
  ];

  for (i, vertex) in vertices.iter().enumerate() {
    let basis = evaluate_lagrange_basis(vertex);

    // L_i should be 1 at vertex i, 0 at all other vertices
    for (j, &basis_val) in basis.iter().enumerate() {
      if i == j {
        assert_eq!(basis_val, <F as CryptoField>::one(), "L_{} should be 1 at vertex {}", j, i);
      } else {
        assert_eq!(basis_val, <F as CryptoField>::zero(), "L_{} should be 0 at vertex {}", j, i);
      }
    }
  }
}

#[test]
fn test_lagrange_basis_empty_input() {
  // Edge case: empty input should return single basis element of 1
  let r: Vec<F> = vec![];
  let basis = evaluate_lagrange_basis(&r);

  assert_eq!(basis.len(), 1);
  assert_eq!(basis[0], <F as CryptoField>::one());
}
