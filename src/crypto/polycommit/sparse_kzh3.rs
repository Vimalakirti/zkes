use crate::crypto::polycommit::kzh3::{kzh3_verify, scalar_field_zero, setup_kzh3_srs, split_input, KZH3Aux, KZH3Commitment, KZH3Opening, KZH3SRS};
use crate::crypto::polycommit::{MLPolyCommit, PairingTrait};
use crate::util::arith::Math;
use crate::util::poly::{evaluate_lagrange_basis, CryptoField, DenseMLPoly, MLPoly, SparseMLPoly};

#[cfg(feature = "arkworks")]
use ark_ec::CurveGroup;
#[cfg(feature = "arkworks")]
use rayon::prelude::*;

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct SparseKZH3Commit<P: PairingTrait> {
  pub srs: Arc<KZH3SRS<P>>,
  pub aux: Arc<RefCell<Option<KZH3Aux<P>>>>,
}

#[derive(Clone, Debug)]
pub struct SparseKZH3CommitKey<P: PairingTrait> {
  // HashMap of polynomial size (n) -> SRS for that size
  pub srs_map: Arc<HashMap<usize, Arc<KZH3SRS<P>>>>,
}

#[derive(Clone)]
pub struct SparseKZH3VerifierKey<P: PairingTrait> {
  // HashMap of polynomial size (n) -> SRS for that size
  pub srs_map: Arc<HashMap<usize, Arc<KZH3SRS<P>>>>,
}
pub fn sparse_kzh3_commit<P: PairingTrait>(srs: &KZH3SRS<P>, poly: &SparseMLPoly<P::ScalarField>) -> (P::G1Affine, KZH3Aux<P>)
where
  P::ScalarField: CryptoField,
{
  let z_log = srs.degree_z.log_2();
  let y_log = srs.degree_y.log_2();
  let x_log = srs.degree_x.log_2();
  let z_mask = (1 << z_log) - 1;
  let y_mask = (1 << y_log) - 1;
  let x_mask = (1 << x_log) - 1;

  // Compute commitment with sparse MSM - h_xyz is now in DenseMLPoly order
  let (indices, scalars): (Vec<_>, Vec<_>) = poly.evaluations.iter().map(|(idx, &val)| (*idx, val)).unzip();

  let c = {
    #[cfg(feature = "arkworks")]
    {
      use crate::crypto::polycommit::sparse_msm;
      sparse_msm::<P::G1, _>(&srs.h_xyz, &indices, &scalars).unwrap().into()
    }
    #[cfg(feature = "icicle")]
    {
      use crate::crypto::polycommit::PairingG1;
      <P::G1 as PairingG1>::sparse_msm(&srs.h_xyz, &indices, &scalars).unwrap().into()
    }
  };

  // Batch compute d_x: Group sparse entries by x-index in a single pass
  let yz_size = srs.degree_y * srs.degree_z;

  // Group sparse entries by their x-index - store indices instead of cloning curve points
  let mut x_groups: Vec<(Vec<usize>, Vec<P::ScalarField>)> = vec![(Vec::new(), Vec::new()); srs.degree_x];

  for (denseml_idx, &val) in poly.evaluations.iter() {
    // Extract x, y, z from DenseMLPoly index
    let z = denseml_idx & z_mask;
    let y = (denseml_idx >> z_log) & y_mask;
    let x = (denseml_idx >> (z_log + y_log)) & x_mask;

    if x < srs.degree_x {
      // Index into h_yz using DenseMLPoly order (z in low bits, y in high bits)
      let yz_idx = z + (y << z_log);
      if yz_idx < yz_size {
        x_groups[x].0.push(yz_idx); // Store index, not cloned curve point
        x_groups[x].1.push(val);
      }
    }
  }

  // Compute MSM for each x-slice using indices
  #[cfg(feature = "arkworks")]
  let d_x: Vec<_> = x_groups
    .into_par_iter()
    .map(|(indices, scalars)| {
      if indices.is_empty() {
        P::G1::from(P::G1Affine::default())
      } else {
        use crate::crypto::polycommit::sparse_msm;
        sparse_msm::<P::G1, _>(&srs.h_yz, &indices, &scalars).unwrap()
      }
    })
    .collect();

  #[cfg(not(feature = "arkworks"))]
  let d_x: Vec<_> = x_groups
    .into_iter()
    .map(|(indices, scalars)| {
      if indices.is_empty() {
        use crate::crypto::polycommit::PairingG1;
        <P::G1 as PairingG1>::zero()
      } else {
        use crate::crypto::polycommit::PairingG1;
        <P::G1 as PairingG1>::sparse_msm(&srs.h_yz, &indices, &scalars).unwrap()
      }
    })
    .collect();

  (c, KZH3Aux { d_x })
}

pub fn sparse_kzh3_open<P: PairingTrait>(
  srs: &KZH3SRS<P>,
  input: &[P::ScalarField],
  _com: &P::G1Affine,
  aux: &KZH3Aux<P>,
  poly: &SparseMLPoly<P::ScalarField>,
) -> KZH3Opening<P>
where
  P::ScalarField: CryptoField,
{
  let x_log = srs.degree_x.log_2();
  let y_log = srs.degree_y.log_2();
  let z_log = srs.degree_z.log_2();
  let len = x_log + y_log + z_log;

  // Pad input if needed (DenseMLPoly uses LSB-first ordering)
  let zero = <P::ScalarField as CryptoField>::zero();
  let padded_input = if input.len() < len {
    let mut padded = vec![zero; len];
    padded[..input.len()].copy_from_slice(input);
    padded
  } else {
    input.to_vec()
  };

  // Precompute eq polynomials for both f_prime and f_star
  // Extract x variables directly (bits (y_log+z_log)..(y_log+z_log+x_log) in LSB-first DenseMLPoly order)
  let x_input = &padded_input[(y_log + z_log)..(y_log + z_log + x_log)];
  let eq_x_evals = evaluate_lagrange_basis(x_input);

  // Extract x and y variables for f_star (bits z_log..(z_log+y_log+x_log) in LSB-first DenseMLPoly order)
  let xy_input = &padded_input[z_log..(z_log + y_log + x_log)];
  let eq_xy_evals = evaluate_lagrange_basis(xy_input);

  let yz_size = srs.degree_y * srs.degree_z;
  let z_size = srs.degree_z;
  let mut f_prime_evals = vec![<P::ScalarField as CryptoField>::zero(); yz_size];
  let mut f_star_evals = vec![<P::ScalarField as CryptoField>::zero(); z_size];

  let z_mask = (1 << z_log) - 1;
  let y_mask = (1 << y_log) - 1;
  let x_mask = (1 << x_log) - 1;

  // Iterate through sparse evaluations in DenseMLPoly's native ordering
  for (denseml_idx, &val) in poly.evaluations.iter() {
    // Extract z (lowest z_log bits in LSB-first DenseML ordering)
    let z = *denseml_idx & z_mask;

    // Extract y bits (next y_log bits) - already in correct order
    let y = (denseml_idx >> z_log) & y_mask;

    // Extract x bits (next x_log bits) - already in correct order
    let x = (denseml_idx >> (z_log + y_log)) & x_mask;

    // Compute f_prime contribution: f' = f(x, Y, Z) fixing x
    if x < srs.degree_x && y < srs.degree_y && z < srs.degree_z {
      // Use DenseMLPoly order for yz_idx (z in low bits, y in high bits)
      let yz_idx = z + (y << z_log);
      let eq_val = eq_x_evals[x];
      f_prime_evals[yz_idx] = f_prime_evals[yz_idx] + eq_val * val;
    }

    // Compute f_star contribution: f* = f(x, y, Z) fixing x and y
    if x < srs.degree_x && y < srs.degree_y && z < srs.degree_z {
      // Use DenseMLPoly order for xy_idx (y in low bits relative to x)
      let xy_idx = y + (x << y_log);
      let eq_val = eq_xy_evals[xy_idx];
      f_star_evals[z] = f_star_evals[z] + eq_val * val;
    }
  }

  let f_prime = DenseMLPoly::new(srs.degree_y.log_2() + srs.degree_z.log_2(), f_prime_evals);
  let f_star = DenseMLPoly::new(srs.degree_z.log_2(), f_star_evals);

  // Compute D_y using slices to avoid unnecessary allocations
  #[cfg(feature = "arkworks")]
  let d_y: Vec<_> = (0..srs.degree_y)
    .into_par_iter()
    .map(|i| {
      let start = srs.degree_z * i;
      let end = start + srs.degree_z;
      let partial_evals = &f_prime.evaluations[start..end];
      use ark_ec::VariableBaseMSM;
      <P::G1 as VariableBaseMSM>::msm(&srs.h_z, partial_evals).unwrap()
    })
    .collect();

  #[cfg(not(feature = "arkworks"))]
  let d_y: Vec<_> = (0..srs.degree_y)
    .map(|i| {
      let start = srs.degree_z * i;
      let end = start + srs.degree_z;
      let partial_evals = &f_prime.evaluations[start..end];
      use crate::crypto::polycommit::PairingG1;
      <P::G1 as PairingG1>::msm(&srs.h_z, partial_evals).unwrap()
    })
    .collect();

  // Compute C_y using eq_x_evals (same as earlier)
  let eq_evals = eq_x_evals.clone();
  let d_x_affine: Vec<_> = aux.d_x.iter().map(|e| (*e).into()).collect();
  let c_y = {
    #[cfg(feature = "arkworks")]
    {
      use ark_ec::VariableBaseMSM;
      <P::G1 as VariableBaseMSM>::msm(&d_x_affine, &eq_evals).unwrap().into()
    }
    #[cfg(feature = "icicle")]
    {
      use crate::crypto::polycommit::PairingG1;
      <P::G1 as PairingG1>::msm(&d_x_affine, &eq_evals).unwrap().into()
    }
  };

  KZH3Opening {
    d_x: aux.d_x.clone(),
    d_y,
    c_y,
    f_star,
  }
}

// Sparse KZH3 verify - optimized with parallel precomputation
pub fn sparse_kzh3_verify<P: PairingTrait>(
  srs: &KZH3SRS<P>,
  input: &[P::ScalarField],
  output: &P::ScalarField,
  com: &P::G1Affine,
  open: &KZH3Opening<P>,
) -> bool
where
  P::ScalarField: CryptoField,
{
  use crate::util::poly::pad_at_start;
  let z_log = srs.degree_z.log_2();
  let y_log = srs.degree_y.log_2();
  let x_log = srs.degree_x.log_2();
  let len = x_log + y_log + z_log;
  let padded_input = pad_at_start(input, len);

  let x_input = &padded_input[(y_log + z_log)..(y_log + z_log + x_log)];
  let y_input = &padded_input[z_log..(z_log + y_log)];

  // Parallelize affine conversions and lagrange basis evaluations
  #[cfg(feature = "arkworks")]
  let ((d_x_affine, d_y_affine), (eq_evals_x, eq_evals_y)) = rayon::join(
    || {
      rayon::join(
        || open.d_x.par_iter().map(|g| (*g).into()).collect::<Vec<P::G1Affine>>(),
        || open.d_y.par_iter().map(|g| (*g).into()).collect::<Vec<P::G1Affine>>(),
      )
    },
    || rayon::join(|| evaluate_lagrange_basis(x_input), || evaluate_lagrange_basis(y_input)),
  );

  #[cfg(all(feature = "icicle", not(feature = "arkworks")))]
  let (d_x_affine, d_y_affine, eq_evals_x, eq_evals_y) = {
    let d_x_affine: Vec<P::G1Affine> = open.d_x.iter().map(|g| (*g).into()).collect();
    let d_y_affine: Vec<P::G1Affine> = open.d_y.iter().map(|g| (*g).into()).collect();
    let eq_evals_x = evaluate_lagrange_basis(x_input);
    let eq_evals_y = evaluate_lagrange_basis(y_input);
    (d_x_affine, d_y_affine, eq_evals_x, eq_evals_y)
  };

  // Phase 2: Run ALL expensive operations (pairings + MSMs) in parallel
  // This maximizes parallelism for the common case where verification succeeds
  #[cfg(feature = "arkworks")]
  let ((pairing1_lhs, pairing1_rhs), ((computed_c_y, (pairing3_lhs, pairing3_rhs)), (msm_lhs, msm_rhs))) = rayon::join(
    // Check 1: D_x well-formatted
    || (P::multi_pairing(&d_x_affine, &srs.v_x), P::pairing(com, &srs.v)),
    || {
      rayon::join(
        || {
          rayon::join(
            // Check 2: c_y computation
            || {
              use ark_ec::VariableBaseMSM;
              P::G1::msm(&d_x_affine, &eq_evals_x).unwrap().into_affine()
            },
            // Check 3: D_y well-formatted
            || (P::multi_pairing(&d_y_affine, &srs.v_y), P::pairing(&open.c_y, &srs.v)),
          )
        },
        // Check 4: Final MSM comparison
        || {
          use ark_ec::VariableBaseMSM;
          rayon::join(
            || P::G1::msm(&srs.h_z, &open.f_star.evaluations()).unwrap(),
            || P::G1::msm(&d_y_affine, &eq_evals_y).unwrap(),
          )
        },
      )
    },
  );

  #[cfg(all(feature = "icicle", not(feature = "arkworks")))]
  let (pairing1_lhs, pairing1_rhs, computed_c_y, pairing3_lhs, pairing3_rhs, msm_lhs, msm_rhs) = {
    use crate::crypto::polycommit::PairingG1;
    let pairing1_lhs = P::multi_pairing(&d_x_affine, &srs.v_x);
    let pairing1_rhs = P::pairing(com, &srs.v);
    let computed_c_y = <P::G1 as PairingG1>::msm(&d_x_affine, &eq_evals_x).unwrap().into_affine();
    let pairing3_lhs = P::multi_pairing(&d_y_affine, &srs.v_y);
    let pairing3_rhs = P::pairing(&open.c_y, &srs.v);
    let msm_lhs = <P::G1 as PairingG1>::msm(&srs.h_z, &open.f_star.evaluations()).unwrap();
    let msm_rhs = <P::G1 as PairingG1>::msm(&d_y_affine, &eq_evals_y).unwrap();
    (pairing1_lhs, pairing1_rhs, computed_c_y, pairing3_lhs, pairing3_rhs, msm_lhs, msm_rhs)
  };

  // Phase 3: Verify all results
  if pairing1_lhs != pairing1_rhs {
    return false;
  }
  if computed_c_y != open.c_y {
    return false;
  }
  if pairing3_lhs != pairing3_rhs {
    return false;
  }
  if msm_lhs != msm_rhs {
    return false;
  }

  // Check 5: Final polynomial evaluation
  let z_input = &padded_input[0..z_log];
  let computed_output = open.f_star.evaluate_at_point(z_input);
  computed_output == *output
}

impl<P: PairingTrait> SparseKZH3Commit<P> {
  /// Fast fake setup for benchmarking purposes only.
  /// Uses deterministic values instead of cryptographically secure randomness.
  /// WARNING: Do NOT use for production - not cryptographically secure!
  #[cfg(feature = "arkworks")]
  pub fn fake_setup(n: usize) -> SparseKZH3CommitKey<P>
  where
    P::ScalarField: CryptoField,
    P::G1: ark_ec::CurveGroup,
    P::G1Affine: Send + Sync,
  {
    use crate::crypto::polycommit::kzh3::KZH3Commit;
    let kzh3_key = KZH3Commit::<P>::fake_setup(n);
    SparseKZH3CommitKey { srs_map: kzh3_key.srs_map }
  }

  /// Fast fake setup for icicle backend
  #[cfg(all(feature = "icicle", not(feature = "arkworks")))]
  pub fn fake_setup(n: usize) -> SparseKZH3CommitKey<P>
  where
    P::ScalarField: CryptoField,
    P::G1Affine: icicle_core::traits::GenerateRandom,
    P::G2Affine: icicle_core::traits::GenerateRandom,
  {
    use crate::crypto::polycommit::kzh3::KZH3Commit;
    let kzh3_key = KZH3Commit::<P>::fake_setup(n);
    SparseKZH3CommitKey { srs_map: kzh3_key.srs_map }
  }
}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> MLPolyCommit<P::ScalarField, SparseMLPoly<P::ScalarField>> for SparseKZH3Commit<P>
where
  P::ScalarField: CryptoField,
  P::G1: ark_ec::CurveGroup,
  P::G1Affine: Send + Sync,
{
  type CommitmentKey = SparseKZH3CommitKey<P>;
  type VerifierKey = SparseKZH3VerifierKey<P>;
  type Commitment = KZH3Commitment<P>;
  type Proof = KZH3Opening<P>;
  type BatchProof = ();

  fn setup(n: usize, _sf_log: usize, _offset: i128, _size: usize) -> Self::CommitmentKey {
    let maximum_degree_log2 = n;
    use ark_std::rand::thread_rng;
    let mut rng = thread_rng();
    let srs = setup_kzh3_srs::<P, _>(maximum_degree_log2, &mut rng);
    let mut srs_map = HashMap::new();
    srs_map.insert(n, Arc::new(srs));
    SparseKZH3CommitKey { srs_map: Arc::new(srs_map) }
  }

  fn commit(poly: &SparseMLPoly<P::ScalarField>, key: &Self::CommitmentKey) -> Self::Commitment {
    let srs = key.srs_map.get(&poly.n()).expect(&format!("No SRS found for polynomial size {}", poly.n()));
    let (com, aux) = sparse_kzh3_commit(srs, poly);
    KZH3Commitment {
      srs: Arc::clone(srs),
      c: com,
      aux,
    }
  }

  fn open(commitment: &Self::Commitment, poly: &SparseMLPoly<P::ScalarField>, _key: &Self::CommitmentKey, point: &[P::ScalarField]) -> Self::Proof {
    let srs = &commitment.srs;
    sparse_kzh3_open(srs, point, &commitment.c, &commitment.aux, poly)
  }

  fn verify(commitment: &Self::Commitment, proof: &Self::Proof, _key: &Self::VerifierKey, point: &[P::ScalarField]) -> bool {
    let srs = &commitment.srs;
    let split_input = split_input(srs, point, scalar_field_zero::<P::ScalarField>());
    let r_z = &split_input[0];
    let claimed_eval = proof.f_star.evaluate_at_point(r_z);
    kzh3_verify(srs, point, &claimed_eval, &commitment.c, proof)
  }

  fn verify_and_extract(commitment: &Self::Commitment, proof: &Self::Proof, _key: &Self::VerifierKey, point: &[P::ScalarField]) -> (bool, P::ScalarField) {
    let srs = &commitment.srs;
    let split_input = split_input(srs, point, scalar_field_zero::<P::ScalarField>());
    let r_z = &split_input[0];
    let claimed_eval = proof.f_star.evaluate_at_point(r_z);
    let ok = kzh3_verify(srs, point, &claimed_eval, &commitment.c, proof);
    (ok, claimed_eval)
  }

  fn batch_open(
    _commitments: &[Self::Commitment],
    _polys: &[SparseMLPoly<P::ScalarField>],
    _keys: &[Self::CommitmentKey],
    _point: &[P::ScalarField],
  ) -> Self::BatchProof {
    todo!()
  }

  fn batch_verify(_commitments: &[Self::Commitment], _proofs: &[Self::Proof], _keys: &[Self::VerifierKey], _point: &[P::ScalarField]) -> bool {
    todo!()
  }
}

#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl<P: PairingTrait> MLPolyCommit<P::ScalarField, SparseMLPoly<P::ScalarField>> for SparseKZH3Commit<P>
where
  P::ScalarField: CryptoField,
  P::ScalarField: icicle_core::traits::GenerateRandom,
  P::G1Affine: icicle_core::traits::GenerateRandom,
  P::G2Affine: icicle_core::traits::GenerateRandom,
{
  type CommitmentKey = SparseKZH3CommitKey<P>;
  type VerifierKey = SparseKZH3VerifierKey<P>;
  type Commitment = KZH3Commitment<P>;
  type Proof = KZH3Opening<P>;
  type BatchProof = ();

  fn setup(n: usize, _sf_log: usize, _offset: i128, _size: usize) -> Self::CommitmentKey {
    let maximum_degree_log2 = n;
    let mut dummy_rng = ();
    let srs = setup_kzh3_srs::<P, _>(maximum_degree_log2, &mut dummy_rng);
    let mut srs_map = HashMap::new();
    srs_map.insert(n, Arc::new(srs));
    SparseKZH3CommitKey { srs_map: Arc::new(srs_map) }
  }

  fn commit(poly: &SparseMLPoly<P::ScalarField>, key: &Self::CommitmentKey) -> Self::Commitment {
    let srs = key.srs_map.get(&poly.n()).expect(&format!("No SRS found for polynomial size {}", poly.n()));
    let (com, aux) = sparse_kzh3_commit(srs, poly);
    KZH3Commitment {
      srs: Arc::clone(srs),
      c: com,
      aux,
    }
  }

  fn open(commitment: &Self::Commitment, poly: &SparseMLPoly<P::ScalarField>, _key: &Self::CommitmentKey, point: &[P::ScalarField]) -> Self::Proof {
    let srs = &commitment.srs;
    sparse_kzh3_open(srs, point, &commitment.c, &commitment.aux, poly)
  }

  fn verify(commitment: &Self::Commitment, proof: &Self::Proof, _key: &Self::VerifierKey, point: &[P::ScalarField]) -> bool {
    let srs = &commitment.srs;
    let split_input = split_input(srs, point, scalar_field_zero::<P::ScalarField>());
    let r_z = &split_input[0];
    let claimed_eval = proof.f_star.evaluate_at_point(r_z);
    kzh3_verify(srs, point, &claimed_eval, &commitment.c, proof)
  }

  fn verify_and_extract(commitment: &Self::Commitment, proof: &Self::Proof, _key: &Self::VerifierKey, point: &[P::ScalarField]) -> (bool, P::ScalarField) {
    let srs = &commitment.srs;
    let split_input = split_input(srs, point, scalar_field_zero::<P::ScalarField>());
    let r_z = &split_input[0];
    let claimed_eval = proof.f_star.evaluate_at_point(r_z);
    let ok = kzh3_verify(srs, point, &claimed_eval, &commitment.c, proof);
    (ok, claimed_eval)
  }

  fn batch_open(
    _commitments: &[Self::Commitment],
    _polys: &[SparseMLPoly<P::ScalarField>],
    _keys: &[Self::CommitmentKey],
    _point: &[P::ScalarField],
  ) -> Self::BatchProof {
    todo!()
  }

  fn batch_verify(_commitments: &[Self::Commitment], _proofs: &[Self::Proof], _keys: &[Self::VerifierKey], _point: &[P::ScalarField]) -> bool {
    todo!()
  }
}
