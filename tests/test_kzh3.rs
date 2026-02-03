/// Macro to generate non-ZK KZH3 tests for a given backend
#[allow(unused_macros)]
macro_rules! kzh3_test_suite {
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
      use zktransformer::crypto::polycommit::kzh3::{KZH3Commit, KZH3CommitKey, KZH3VerifierKey};
      use zktransformer::crypto::polycommit::MLPolyCommit;
      use zktransformer::util::poly::{CryptoField, DenseMLPoly};

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

      /// Test non-ZK KZH3 commit/open/verify cycle
      #[test]
      fn test_kzh3_basic_commit_open_verify() {
        let max_degree_log = 3;
        let kzh3: KZH3CommitKey<$curve> = KZH3Commit::setup(max_degree_log, 0, 0, 0);

        let vk = KZH3VerifierKey {
          srs_map: kzh3.srs_map.clone(),
        };

        let poly = DenseMLPoly::new(
          3,
          vec![
            fr_from!(1),
            fr_from!(2),
            fr_from!(3),
            fr_from!(4),
            fr_from!(5),
            fr_from!(6),
            fr_from!(7),
            fr_from!(8),
          ],
        );

        let commitment = KZH3Commit::commit(&poly, &kzh3);

        let point = vec![fr_from!(0), fr_from!(1), fr_from!(1)];

        let proof = KZH3Commit::open(&commitment, &poly, &kzh3, &point);

        let verified = KZH3Commit::verify(&commitment, &proof, &vk, &point);

        assert!(verified, "Non-ZK KZH3 protocol failed verification");
      }

      /// Test KZH3 with different polynomial values
      #[test]
      fn test_kzh3_random_poly() {
        let max_degree_log = 4;
        let kzh3: KZH3CommitKey<$curve> = KZH3Commit::setup(max_degree_log, 0, 0, 0);

        let vk = KZH3VerifierKey {
          srs_map: kzh3.srs_map.clone(),
        };

        let poly = DenseMLPoly::new(4, (0..16).map(|i| fr_from!(i + 1)).collect());

        let commitment = KZH3Commit::commit(&poly, &kzh3);

        let point1 = vec![fr_from!(0), fr_from!(0), fr_from!(0), fr_from!(0)];
        let proof1 = KZH3Commit::open(&commitment, &poly, &kzh3, &point1);
        assert!(KZH3Commit::verify(&commitment, &proof1, &vk, &point1), "Verification failed for point1");

        let point2 = vec![fr_from!(1), fr_from!(1), fr_from!(0), fr_from!(0)];
        let proof2 = KZH3Commit::open(&commitment, &poly, &kzh3, &point2);
        assert!(KZH3Commit::verify(&commitment, &proof2, &vk, &point2), "Verification failed for point2");
      }
    }
  };
}

// Generate non-ZK KZH3 tests for arkworks backend
#[cfg(feature = "arkworks")]
kzh3_test_suite!(arkworks_kzh3_tests, cfg(feature = "arkworks"), ark_bn254::Bn254, ark_bn254::Fr, false);

// Generate non-ZK KZH3 tests for icicle BLS12_381 backend
#[cfg(feature = "icicle")]
kzh3_test_suite!(
  icicle_non_zk_kzh3_tests,
  cfg(feature = "icicle"),
  zktransformer::crypto::polycommit::IcicleBls12_381,
  icicle_bls12_381::curve::ScalarField,
  true
);

/// Macro to generate sparse KZH3 tests for a given backend
#[allow(unused_macros)]
macro_rules! sparse_kzh3_test_suite {
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
      use zktransformer::crypto::polycommit::sparse_kzh3::{sparse_kzh3_commit, sparse_kzh3_open, sparse_kzh3_verify};
      use zktransformer::crypto::polycommit::sparse_kzh3::{SparseKZH3Commit, SparseKZH3CommitKey, SparseKZH3VerifierKey};
      use zktransformer::crypto::polycommit::MLPolyCommit;
      use zktransformer::util::poly::{usize_to_le_vec, CryptoField, DenseMLPoly, MLPoly, SparseMLPoly};

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

      /// Test non-ZK sparse KZH3 commit/open/verify cycle
      #[test]
      fn test_sparse_kzh3_basic_commit_open_verify() {
        let max_degree_log = 3;
        let kzh3: SparseKZH3CommitKey<$curve> = SparseKZH3Commit::setup(max_degree_log, 0, 0, 0);

        let vk = SparseKZH3VerifierKey {
          srs_map: kzh3.srs_map.clone(),
        };

        let poly = create_sparse_poly(3, vec![(0, fr_from!(1)), (2, fr_from!(3)), (5, fr_from!(6)), (7, fr_from!(8))]);

        let commitment = SparseKZH3Commit::commit(&poly, &kzh3);

        let point = vec![fr_from!(0), fr_from!(1), fr_from!(1)];

        let proof = SparseKZH3Commit::open(&commitment, &poly, &kzh3, &point);

        let verified = SparseKZH3Commit::verify(&commitment, &proof, &vk, &point);

        assert!(verified, "Non-ZK sparse KZH3 protocol failed verification");
      }

      /// Test non-ZK sparse KZH3 with different polynomial values
      #[test]
      fn test_sparse_kzh3_random_poly() {
        let max_degree_log = 4;
        let kzh3: SparseKZH3CommitKey<$curve> = SparseKZH3Commit::setup(max_degree_log, 0, 0, 0);

        let vk = SparseKZH3VerifierKey {
          srs_map: kzh3.srs_map.clone(),
        };

        let poly = create_sparse_poly(4, vec![(1, fr_from!(2)), (5, fr_from!(6)), (9, fr_from!(10)), (14, fr_from!(15))]);

        let commitment = SparseKZH3Commit::commit(&poly, &kzh3);

        let point = vec![fr_from!(1), fr_from!(1), fr_from!(0), fr_from!(0)];
        let proof = SparseKZH3Commit::open(&commitment, &poly, &kzh3, &point);
        assert!(SparseKZH3Commit::verify(&commitment, &proof, &vk, &point), "Verification failed");
      }

      /// Test the core KZH3 protocol using sparse polynomials
      #[test]
      fn test_basic_sparse_kzh3_core_functions() {
        let max_degree_log = 3;
        let pcs: SparseKZH3CommitKey<$curve> = SparseKZH3Commit::<$curve>::setup(max_degree_log, 0, 0, 0);

        let poly = create_sparse_poly(3, vec![(0, fr_from!(1)), (2, fr_from!(3)), (5, fr_from!(6))]);

        let srs = pcs.srs_map.get(&max_degree_log).expect("SRS not found");
        let (commitment, aux) = sparse_kzh3_commit::<$curve>(srs, &poly);
        let point = vec![fr_from!(1), fr_from!(0), fr_from!(0)];
        let eval = poly.evaluate_at_point(&point);

        let opening = sparse_kzh3_open::<$curve>(srs, &point, &commitment, &aux, &poly);
        let verified = sparse_kzh3_verify::<$curve>(srs, &point, &eval, &commitment, &opening);

        assert!(verified, "Core Sparse KZH3 protocol failed verification");
      }

      /// Test comparing sparse and dense representations
      #[test]
      fn test_sparse_vs_dense_consistency() {
        let _max_degree_log = 4;

        // Create identical polynomial in both sparse and dense forms
        let mut dense_evals = (0..16).map(|_| fr_from!(0)).collect::<Vec<_>>();
        let mut sparse_entries = Vec::new();

        // Set a few random values
        #[cfg(feature = "arkworks")]
        {
          use ark_ff::UniformRand;
          use ark_std::rand::thread_rng;
          let mut rng = thread_rng();
          for idx in [1, 5, 9, 14] {
            let val = <$fr>::rand(&mut rng);
            dense_evals[idx] = val;
            sparse_entries.push((idx, val));
          }
          let point = vec![<$fr>::rand(&mut rng), <$fr>::rand(&mut rng), <$fr>::rand(&mut rng), <$fr>::rand(&mut rng)];
          let dense_poly = DenseMLPoly::new(4, dense_evals);
          let sparse_poly = create_sparse_poly(4, sparse_entries);

          let dense_eval = dense_poly.evaluate_at_point(&point);
          let sparse_eval = sparse_poly.evaluate_at_point(&point);

          assert_eq!(
            dense_eval, sparse_eval,
            "Sparse and dense representations should evaluate to the same value"
          );
        }

        #[cfg(feature = "icicle")]
        {
          use icicle_core::traits::GenerateRandom;
          for idx in [1, 5, 9, 14] {
            let val = <$fr>::generate_random(1)[0];
            dense_evals[idx] = val;
            sparse_entries.push((idx, val));
          }
          let point = vec![
            <$fr>::generate_random(1)[0],
            <$fr>::generate_random(1)[0],
            <$fr>::generate_random(1)[0],
            <$fr>::generate_random(1)[0],
          ];
          let dense_poly = DenseMLPoly::new(4, dense_evals);
          let sparse_poly = create_sparse_poly(4, sparse_entries);

          let dense_eval = dense_poly.evaluate_at_point(&point);
          let sparse_eval = sparse_poly.evaluate_at_point(&point);

          assert_eq!(
            dense_eval, sparse_eval,
            "Sparse and dense representations should evaluate to the same value"
          );
        }
      }
    }
  };
}

// Generate sparse KZH3 tests for arkworks backend
#[cfg(feature = "arkworks")]
sparse_kzh3_test_suite!(
  arkworks_sparse_kzh3_tests,
  cfg(feature = "arkworks"),
  ark_bn254::Bn254,
  ark_bn254::Fr,
  false
);

// Generate sparse KZH3 tests for icicle BLS12_381 backend
#[cfg(feature = "icicle")]
sparse_kzh3_test_suite!(
  icicle_sparse_kzh3_tests,
  cfg(feature = "icicle"),
  zktransformer::crypto::polycommit::IcicleBls12_381,
  icicle_bls12_381::curve::ScalarField,
  true
);
