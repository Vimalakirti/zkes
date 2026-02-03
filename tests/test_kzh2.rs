/// Macro to generate KZH2 tests for a given backend
/// Handles differences between arkworks (direct Fr::from) and icicle (CryptoField trait)
#[allow(unused_macros)]
macro_rules! kzh2_test_suite {
  (
    $mod_name:ident,
    $cfg_attr:meta,
    $curve:ty,
    $fr:ty,
    $use_crypto_field:expr
  ) => {
    #[$cfg_attr]
    #[cfg(test)]
    mod $mod_name {
      use zktransformer::crypto::polycommit::kzh2::{kzh2_commit, kzh2_open, kzh2_verify, KZH2Commit, KZH2CommitKey, KZH2VerifierKey};
      use zktransformer::crypto::polycommit::MLPolyCommit;
      use zktransformer::util::poly::{CryptoField, DenseMLPoly, MLPoly};

      // Helper to create field elements - works for both arkworks and icicle
      macro_rules! fr_from {
        ($val:expr) => {
          if $use_crypto_field {
            <$fr as CryptoField>::from_u32($val)
          } else {
            <$fr>::from($val as u32)
          }
        };
      }

      macro_rules! fr_zero {
        () => {
          if $use_crypto_field {
            <$fr as CryptoField>::zero()
          } else {
            <$fr>::from(0u32)
          }
        };
      }

      /// Test core KZH2 protocol
      #[test]
      fn test_basic_kzh2() {
        let max_degree_log = 4;
        let key: KZH2CommitKey<$curve> = KZH2Commit::<$curve>::setup(max_degree_log, 0, 0, 0);

        let poly = DenseMLPoly::new(4, (0..16).map(|_| fr_from!(1)).collect());

        let srs = key.srs_map.get(&poly.n).expect("No SRS found for polynomial size");

        let (commitment, aux) = kzh2_commit::<$curve>(srs, &poly);

        let point = vec![fr_from!(1), fr_from!(0), fr_from!(0), fr_from!(0)];
        let eval = poly.evaluate_at_point(&point);

        let opening = kzh2_open::<$curve>(srs, &point, &commitment, &aux, &poly);

        let verified = kzh2_verify::<$curve>(srs, &point, &eval, &commitment, &opening);

        assert!(verified, "Core KZH2 protocol failed verification");
      }

      /// Test KZH2 commit/open/verify cycle with MLPolyCommit interface
      #[test]
      fn test_kzh2_basic_commit_open_verify() {
        let max_degree_log = 4;
        let key: KZH2CommitKey<$curve> = KZH2Commit::setup(max_degree_log, 0, 0, 0);
        let vk = KZH2VerifierKey {
          srs_map: key.srs_map.clone(),
        };

        let poly = DenseMLPoly::new(
          4,
          vec![
            fr_from!(1),
            fr_from!(2),
            fr_from!(3),
            fr_from!(4),
            fr_from!(5),
            fr_from!(6),
            fr_from!(7),
            fr_from!(8),
            fr_from!(9),
            fr_from!(10),
            fr_from!(11),
            fr_from!(12),
            fr_from!(13),
            fr_from!(14),
            fr_from!(15),
            fr_from!(16),
          ],
        );

        let commitment = KZH2Commit::commit(&poly, &key);

        let point = vec![fr_from!(0), fr_from!(1), fr_from!(1), fr_from!(0)];
        let proof = KZH2Commit::open(&commitment, &poly, &key, &point);

        let verified = KZH2Commit::verify(&commitment, &proof, &vk, &point);

        assert!(verified, "KZH2 protocol failed verification");
      }

      /// Test KZH2 with different polynomial values
      #[test]
      fn test_kzh2_random_poly() {
        let max_degree_log = 6;
        let key: KZH2CommitKey<$curve> = KZH2Commit::setup(max_degree_log, 0, 0, 0);
        let vk = KZH2VerifierKey {
          srs_map: key.srs_map.clone(),
        };

        let poly = DenseMLPoly::new(6, (0..64).map(|i| fr_from!(i + 1)).collect());

        let commitment = KZH2Commit::commit(&poly, &key);

        let point1 = vec![fr_from!(0), fr_from!(0), fr_from!(0), fr_from!(0), fr_from!(0), fr_from!(0)];
        let proof1 = KZH2Commit::open(&commitment, &poly, &key, &point1);
        assert!(KZH2Commit::verify(&commitment, &proof1, &vk, &point1), "Verification failed for point1");

        let point2 = vec![fr_from!(1), fr_from!(1), fr_from!(0), fr_from!(1), fr_from!(0), fr_from!(0)];
        let proof2 = KZH2Commit::open(&commitment, &poly, &key, &point2);
        assert!(KZH2Commit::verify(&commitment, &proof2, &vk, &point2), "Verification failed for point2");
      }

      /// Test KZH2 with a small polynomial
      #[test]
      fn test_kzh2_small_poly() {
        let max_degree_log = 2;
        let key: KZH2CommitKey<$curve> = KZH2Commit::setup(max_degree_log, 0, 0, 0);
        let vk = KZH2VerifierKey {
          srs_map: key.srs_map.clone(),
        };

        let poly = DenseMLPoly::new(2, vec![fr_from!(10), fr_from!(20), fr_from!(30), fr_from!(40)]);

        let commitment = KZH2Commit::commit(&poly, &key);

        let point = vec![fr_from!(1), fr_from!(0)];
        let proof = KZH2Commit::open(&commitment, &poly, &key, &point);

        assert!(
          KZH2Commit::verify(&commitment, &proof, &vk, &point),
          "Small polynomial verification failed"
        );
      }

      /// Test that KZH2 correctly handles polynomial evaluations
      #[test]
      fn test_kzh2_evaluation_correctness() {
        let max_degree_log = 4;
        let key: KZH2CommitKey<$curve> = KZH2Commit::setup(max_degree_log, 0, 0, 0);
        let vk = KZH2VerifierKey {
          srs_map: key.srs_map.clone(),
        };

        let poly = DenseMLPoly::new(4, (0..16).map(|i| fr_from!(i)).collect());

        let commitment = KZH2Commit::commit(&poly, &key);

        let zero_point = vec![fr_from!(0), fr_from!(0), fr_from!(0), fr_from!(0)];
        let expected_eval_zero = poly.evaluate_at_point(&zero_point);
        let proof_zero = KZH2Commit::open(&commitment, &poly, &key, &zero_point);

        use zktransformer::crypto::polycommit::kzh2::split_input;
        let srs = &commitment.srs;
        let split_input = split_input(srs, &zero_point, fr_zero!());
        let r_y = &split_input[0];
        let computed_eval = proof_zero.f_star.evaluate_at_point(r_y);

        assert_eq!(computed_eval, expected_eval_zero, "Evaluation at origin should match polynomial value");
        assert!(
          KZH2Commit::verify(&commitment, &proof_zero, &vk, &zero_point),
          "Verification at origin failed"
        );
      }
    }
  };
}

// Generate tests for arkworks backend
#[cfg(feature = "arkworks")]
kzh2_test_suite!(arkworks_kzh2_tests, cfg(feature = "arkworks"), ark_bn254::Bn254, ark_bn254::Fr, false);

// Generate tests for icicle BLS12_381 backend
#[cfg(all(feature = "icicle", feature = "bls12_381"))]
kzh2_test_suite!(
  icicle_kzh2_tests,
  cfg(all(feature = "icicle", feature = "bls12_381")),
  zktransformer::crypto::polycommit::IcicleBls12_381,
  icicle_bls12_381::curve::ScalarField,
  true
);

// Generate tests for icicle BN254 backend
#[cfg(all(feature = "icicle", feature = "bn254"))]
kzh2_test_suite!(
  icicle_bn254_kzh2_tests,
  cfg(all(feature = "icicle", feature = "bn254")),
  zktransformer::crypto::polycommit::IcicleBn254,
  icicle_bn254::curve::ScalarField,
  true
);

/// Macro to generate sparse KZH2 tests for a given backend
#[allow(unused_macros)]
macro_rules! sparse_kzh2_test_suite {
  (
    $mod_name:ident,
    $cfg_attr:meta,
    $curve:ty,
    $fr:ty,
    $use_crypto_field:expr
  ) => {
    #[$cfg_attr]
    #[cfg(test)]
    mod $mod_name {
      use std::collections::HashMap;
      use zktransformer::crypto::polycommit::kzh2::kzh2_verify;
      use zktransformer::crypto::polycommit::sparse_kzh2::{
        sparse_kzh2_commit, sparse_kzh2_open, SparseKZH2Commit, SparseKZH2CommitKey, SparseKZH2VerifierKey,
      };
      use zktransformer::crypto::polycommit::MLPolyCommit;
      use zktransformer::util::poly::{usize_to_le_vec, CryptoField, MLPoly, SparseMLPoly};

      // Helper to create field elements
      macro_rules! fr_from {
        ($val:expr) => {
          if $use_crypto_field {
            <$fr as CryptoField>::from_u32($val)
          } else {
            <$fr>::from($val as u32)
          }
        };
      }

      macro_rules! fr_zero {
        () => {
          if $use_crypto_field {
            <$fr as CryptoField>::zero()
          } else {
            <$fr>::from(0u32)
          }
        };
      }

      /// Creates a sparse polynomial from a list of (index, value) pairs
      fn create_sparse_poly(num_vars: usize, entries: Vec<(usize, $fr)>) -> SparseMLPoly<$fr> {
        let mut evaluations = HashMap::new();
        let mut indices = Vec::new();

        for (idx, val) in entries {
          let idx_bytes = usize_to_le_vec(idx);
          evaluations.insert(idx_bytes.clone(), val);
          indices.push(idx_bytes);
        }

        indices.sort();

        SparseMLPoly::new(num_vars, evaluations, indices.into())
      }

      /// Test sparse KZH2 commit/open/verify cycle
      #[test]
      fn test_sparse_kzh2_basic_commit_open_verify() {
        let max_degree_log = 4;
        let key: SparseKZH2CommitKey<$curve> = SparseKZH2Commit::setup(max_degree_log, 0, 0, 0);
        let vk = SparseKZH2VerifierKey {
          srs_map: key.srs_map.clone(),
        };

        let poly = create_sparse_poly(4, vec![(0, fr_from!(1)), (2, fr_from!(3)), (7, fr_from!(8)), (15, fr_from!(16))]);

        let commitment = SparseKZH2Commit::commit(&poly, &key);

        let point = vec![fr_from!(0), fr_from!(1), fr_from!(1), fr_from!(0)];

        let proof = SparseKZH2Commit::open(&commitment, &poly, &key, &point);

        let verified = SparseKZH2Commit::verify(&commitment, &proof, &vk, &point);

        assert!(verified, "Sparse KZH2 protocol failed verification");
      }

      /// Test sparse KZH2 with different polynomial values
      #[test]
      fn test_sparse_kzh2_random_poly() {
        let max_degree_log = 6;
        let key: SparseKZH2CommitKey<$curve> = SparseKZH2Commit::setup(max_degree_log, 0, 0, 0);
        let vk = SparseKZH2VerifierKey {
          srs_map: key.srs_map.clone(),
        };

        let poly = create_sparse_poly(
          6,
          vec![
            (1, fr_from!(2)),
            (10, fr_from!(11)),
            (25, fr_from!(26)),
            (50, fr_from!(51)),
            (63, fr_from!(64)),
          ],
        );

        let commitment = SparseKZH2Commit::commit(&poly, &key);

        let point = vec![fr_from!(1), fr_from!(1), fr_from!(0), fr_from!(0), fr_from!(0), fr_from!(0)];
        let proof = SparseKZH2Commit::open(&commitment, &poly, &key, &point);
        assert!(SparseKZH2Commit::verify(&commitment, &proof, &vk, &point), "Verification failed");
      }

      /// Test sparse KZH2 with core functions
      #[test]
      fn test_basic_sparse_kzh2_core() {
        let max_degree_log = 4;
        let key: SparseKZH2CommitKey<$curve> = SparseKZH2Commit::<$curve>::setup(max_degree_log, 0, 0, 0);

        let poly = create_sparse_poly(4, vec![(0, fr_from!(1)), (5, fr_from!(6)), (10, fr_from!(11))]);

        let srs = key.srs_map.get(&poly.n()).expect("No SRS found for polynomial size");

        let (commitment, aux) = sparse_kzh2_commit::<$curve>(srs, &poly);

        let point = vec![fr_from!(1), fr_from!(0), fr_from!(0), fr_from!(0)];
        let eval = poly.evaluate_at_point(&point);

        let opening = sparse_kzh2_open::<$curve>(srs, &point, &commitment, &aux, &poly);

        let verified = kzh2_verify::<$curve>(srs, &point, &eval, &commitment, &opening);

        assert!(verified, "Core sparse KZH2 protocol failed verification");
      }

      /// Test sparse KZH2 with small polynomial
      #[test]
      fn test_sparse_kzh2_small_poly() {
        let max_degree_log = 2;
        let key: SparseKZH2CommitKey<$curve> = SparseKZH2Commit::setup(max_degree_log, 0, 0, 0);
        let vk = SparseKZH2VerifierKey {
          srs_map: key.srs_map.clone(),
        };

        let poly = create_sparse_poly(2, vec![(0, fr_from!(10)), (3, fr_from!(40))]);

        let commitment = SparseKZH2Commit::commit(&poly, &key);

        let point = vec![fr_from!(1), fr_from!(0)];
        let proof = SparseKZH2Commit::open(&commitment, &poly, &key, &point);

        assert!(
          SparseKZH2Commit::verify(&commitment, &proof, &vk, &point),
          "Small sparse polynomial verification failed"
        );
      }

      /// Test sparse KZH2 evaluation correctness
      #[test]
      fn test_sparse_kzh2_evaluation_correctness() {
        let max_degree_log = 4;
        let key: SparseKZH2CommitKey<$curve> = SparseKZH2Commit::setup(max_degree_log, 0, 0, 0);
        let vk = SparseKZH2VerifierKey {
          srs_map: key.srs_map.clone(),
        };

        let poly = create_sparse_poly(4, vec![(0, fr_from!(5)), (7, fr_from!(12)), (15, fr_from!(20))]);

        let commitment = SparseKZH2Commit::commit(&poly, &key);

        let zero_point = vec![fr_from!(0), fr_from!(0), fr_from!(0), fr_from!(0)];
        let expected_eval_zero = poly.evaluate_at_point(&zero_point);
        let proof_zero = SparseKZH2Commit::open(&commitment, &poly, &key, &zero_point);

        use zktransformer::crypto::polycommit::kzh2::split_input;
        let srs = &commitment.srs;
        let split_input = split_input(srs, &zero_point, fr_zero!());
        let r_y = &split_input[0];
        let computed_eval = proof_zero.f_star.evaluate_at_point(r_y);

        assert_eq!(computed_eval, expected_eval_zero, "Evaluation at origin should match polynomial value");
        assert!(
          SparseKZH2Commit::verify(&commitment, &proof_zero, &vk, &zero_point),
          "Verification at origin failed"
        );
      }
    }
  };
}

// Generate sparse tests for arkworks backend
#[cfg(feature = "arkworks")]
sparse_kzh2_test_suite!(
  arkworks_sparse_kzh2_tests,
  cfg(feature = "arkworks"),
  ark_bn254::Bn254,
  ark_bn254::Fr,
  false
);

// Generate sparse tests for icicle BN254 backend
#[cfg(all(feature = "icicle", feature = "bn254"))]
sparse_kzh2_test_suite!(
  icicle_bn254_sparse_kzh2_tests,
  cfg(all(feature = "icicle", feature = "bn254")),
  zktransformer::crypto::polycommit::IcicleBn254,
  icicle_bn254::curve::ScalarField,
  true
);
