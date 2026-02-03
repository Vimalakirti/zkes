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

use plonky2::{timed, util::timing::TimingTree};
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use zktransformer::{
  crypto::polycommit::kzh3::{setup_kzh3_srs, KZH3Commit, KZH3CommitKey, KZH3Commitment, KZH3VerifierKey},
  crypto::polycommit::sparse_kzh3::{SparseKZH3Commit, SparseKZH3CommitKey, SparseKZH3VerifierKey},
  crypto::srs_storage::{load_kzh3_srs, store_kzh3_srs},
  dag::{dense_add_relu, extract_lookup_proof_only, extract_opening_proofs_only, extract_sumcheck_proofs_only, DagBuilder, DataType, Role, Witness},
  util::poly::CryptoField,
  util::serialization::measure_total_proof_size,
  util::transcript::Transcript,
  SF_LOG,
};

#[cfg(all(feature = "arkworks", feature = "bn254"))]
use zktransformer::crypto::polycommit::ArkBn254 as PairingType;
#[cfg(all(feature = "icicle", feature = "bn254"))]
use zktransformer::crypto::polycommit::IcicleBn254 as PairingType;

fn main() {
  let mut timing = TimingTree::default();
  env_logger::init();

  // --- Circuit compilation ---
  let mut g = DagBuilder::new();
  let w = Witness::new(
    vec![4, 4],
    vec![
      <F as CryptoField>::from_u32(0u32 * 1000),
      <F as CryptoField>::from_u32(1u32 * 1000),
      <F as CryptoField>::from_u32(2u32 * 1000),
      <F as CryptoField>::from_u32(3u32 * 1000),
      <F as CryptoField>::from_u32(4u32 * 1000),
      <F as CryptoField>::from_u32(5u32 * 1000),
      <F as CryptoField>::from_u32(6u32 * 1000),
      <F as CryptoField>::from_u32(7u32 * 1000),
      <F as CryptoField>::from_u32(8u32 * 1000),
      <F as CryptoField>::from_u32(7u32 * 1000),
      <F as CryptoField>::from_u32(6u32 * 1000),
      <F as CryptoField>::from_u32(5u32 * 1000),
      <F as CryptoField>::from_u32(4u32 * 1000),
      <F as CryptoField>::from_u32(3u32 * 1000),
      <F as CryptoField>::from_u32(4u32 * 1000),
      <F as CryptoField>::from_u32(5u32 * 1000),
    ],
    DataType::Float,
    *SF_LOG as usize,
    Role::Constant,
  );
  let b = Witness::new(
    vec![4],
    vec![
      <F as CryptoField>::from_u32(1u32 * 1000),
      <F as CryptoField>::from_u32(0u32) - <F as CryptoField>::from_u32(2u32 * 1000),
      <F as CryptoField>::from_u32(0u32) - <F as CryptoField>::from_u32(3u32 * 1000),
      <F as CryptoField>::from_u32(0u32) - <F as CryptoField>::from_u32(4u32 * 1000),
    ],
    DataType::Float,
    *SF_LOG as usize,
    Role::Constant,
  );
  let w2 = w.clone();
  let b2 = b.clone();
  let x = g.input(vec![4], DataType::Float);
  let z = g.pipe(&[x], dense_add_relu(w, b))[0];
  let output = g.pipe(&[z], dense_add_relu(w2, b2))[0];
  let output = g.add(z, output)[0]; // residual

  // Compile -> (Dag, initial edge values)
  let (dag, mut init) = g.compile();

  // --- Prover ---
  let mut transcript = Transcript::new(b"zkml");

  let mut dense_commitments: Vec<Option<KZH3Commitment<PairingType>>> = vec![None; dag.num_edges()];
  let mut sparse_commitments: Vec<Option<Vec<KZH3Commitment<PairingType>>>> = vec![None; dag.num_edges()];

  // Witness generation from input (must do this before collecting polynomial sizes
  // because auxiliary witnesses are created during run())
  let input = Witness::new(
    vec![4],
    vec![
      <F as CryptoField>::from_u32(1u32 * 1000),
      <F as CryptoField>::from_u32(2u32 * 1000),
      <F as CryptoField>::from_u32(3u32 * 1000),
      <F as CryptoField>::from_u32(4u32 * 1000),
    ],
    DataType::Float,
    *SF_LOG as usize,
    Role::Input,
  );
  dag.run(&mut init, &vec![(x, input)]);
  println!("output: {:?}", init[output]);

  // Collect polynomial sizes from the DAG after witness generation
  // to capture all polynomial sizes including auxiliaries
  let polynomial_sizes = dag.collect_polynomial_sizes(&init);
  println!("Polynomial sizes needed: {:?}", polynomial_sizes);

  // Load or generate SRS for all required polynomial sizes
  let mut srs_map = HashMap::new();
  for &size in &polynomial_sizes {
    let size_srs = if fs::metadata(&format!("{}.srs", size)).is_ok() {
      println!("Loading existing SRS for polynomial size {}", size);
      load_kzh3_srs(size).expect(&format!("Failed to load SRS for size {}", size))
    } else {
      println!("Generating SRS for polynomial size {}", size);
      #[cfg(feature = "arkworks")]
      let size_srs = {
        use rand::thread_rng;
        setup_kzh3_srs::<PairingType, _>(size, &mut thread_rng())
      };
      #[cfg(feature = "icicle")]
      let size_srs = setup_kzh3_srs::<PairingType, _>(size, &mut ());
      store_kzh3_srs(&size_srs, size).expect(&format!("Failed to store SRS for size {}", size));
      size_srs
    };
    srs_map.insert(size, Arc::new(size_srs));
  }
  let srs_map = Arc::new(srs_map);

  // Create commitment keys with the collected SRS sizes
  let kzh3 = KZH3CommitKey::<PairingType> { srs_map: srs_map.clone() };
  let sparse_kzh3 = SparseKZH3CommitKey::<PairingType> { srs_map: srs_map.clone() };

  // Create verifier keys with the same srs_map
  let dense_verifier_key = KZH3VerifierKey { srs_map: srs_map.clone() };
  let sparse_verifier_key = SparseKZH3VerifierKey { srs_map: srs_map.clone() };

  // Commit to all witnesses (constants, inputs, auxiliaries, and final outputs)
  dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &kzh3,
    &sparse_kzh3,
    &init,
    &mut dense_commitments,
    &mut sparse_commitments,
    &mut timing,
  );

  let (sumcheck_proofs, opening_proofs, range_proof, two_pow_proof, reducer_proofs) = timed!(
    timing,
    "prove",
    dag.prove::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &kzh3,
      &sparse_kzh3,
      &init,
      &dense_commitments,
      &sparse_commitments,
      &mut transcript,
      &mut timing
    )
  );

  // clear the data from the witnesses before verification
  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));

  // Measure proof sizes (excluding claims and middle_claims)
  let sumcheck_proofs_only = extract_sumcheck_proofs_only(&sumcheck_proofs);
  let opening_proofs_only = extract_opening_proofs_only::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(&opening_proofs);
  let range_proof_only = extract_lookup_proof_only(&range_proof);
  let two_pow_proof_only = extract_lookup_proof_only(&two_pow_proof);
  let proof_size = measure_total_proof_size(
    &sumcheck_proofs_only,
    &opening_proofs_only,
    &range_proof_only,
    &two_pow_proof_only,
    &reducer_proofs,
  );

  // --- Verifier ---
  let mut verifier_transcript = Transcript::new(b"zkml");
  let verified = timed!(
    timing,
    "verify",
    dag.verify::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &sumcheck_proofs,
      &opening_proofs,
      &range_proof,
      &two_pow_proof,
      &reducer_proofs,
      &init,
      &dense_verifier_key,
      &sparse_verifier_key,
      &dense_commitments,
      &sparse_commitments,
      &mut verifier_transcript,
    )
  );
  timing.print();
  println!("verified: {:?}", verified);
  println!("proof size: {}", proof_size);
}
