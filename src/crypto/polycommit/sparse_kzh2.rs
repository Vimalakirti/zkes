use crate::crypto::polycommit::kzh2::{
  kzh2_verify, scalar_field_zero, setup_kzh2_srs, split_input, KZH2Aux, KZH2Commit, KZH2Commitment, KZH2Opening, KZH2SRS,
};
use crate::crypto::polycommit::{MLPolyCommit, PairingTrait};
use crate::util::arith::Math;
use crate::util::poly::{evaluate_lagrange_basis, CryptoField, DenseMLPoly, MLPoly, SparseMLPoly};

#[cfg(feature = "arkworks")]
use rayon::prelude::*;

use std::cell::RefCell;
use std::sync::Arc;

pub fn sparse_kzh2_commit<P: PairingTrait>(srs: &KZH2SRS<P>, poly: &SparseMLPoly<P::ScalarField>) -> (P::G1Affine, KZH2Aux<P>)
where
  P::ScalarField: CryptoField,
{
  let _len = srs.degree_x.log_2() + srs.degree_y.log_2();
  let x_log = srs.degree_x.log_2();
  let y_log = srs.degree_y.log_2();
  let y_mask = (1 << y_log) - 1;
  let x_mask = (1 << x_log) - 1;

  let (indices, scalars): (Vec<_>, Vec<_>) = poly.evaluations.iter().map(|(idx, &val)| (*idx, val)).unzip();

  let c = {
    #[cfg(feature = "arkworks")]
    {
      use crate::crypto::polycommit::sparse_msm;
      sparse_msm::<P::G1, _>(&srs.h_xy, &indices, &scalars).unwrap().into()
    }
    #[cfg(feature = "icicle")]
    {
      use crate::crypto::polycommit::PairingG1;
      <P::G1 as PairingG1>::sparse_msm(&srs.h_xy, &indices, &scalars).unwrap().into()
    }
  };

  // Batch compute d_x: Group sparse entries by x-index
  // Store indices instead of cloning curve points
  let mut x_groups: Vec<(Vec<usize>, Vec<P::ScalarField>)> = vec![(Vec::new(), Vec::new()); srs.degree_x];

  for (idx, &val) in poly.evaluations.iter() {
    let y = *idx & y_mask;
    let x = (*idx >> y_log) & x_mask;

    if x < srs.degree_x && y < srs.degree_y {
      x_groups[x].0.push(y); // Store index, not cloned curve point
      x_groups[x].1.push(val);
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
        sparse_msm::<P::G1, _>(&srs.h_y, &indices, &scalars).unwrap()
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
        <P::G1 as PairingG1>::sparse_msm(&srs.h_y, &indices, &scalars).unwrap()
      }
    })
    .collect();

  (c, KZH2Aux { d_x })
}

pub fn sparse_kzh2_open<P: PairingTrait>(
  srs: &KZH2SRS<P>,
  input: &[P::ScalarField],
  _com: &P::G1Affine,
  aux: &KZH2Aux<P>,
  poly: &SparseMLPoly<P::ScalarField>,
) -> KZH2Opening<P>
where
  P::ScalarField: CryptoField,
{
  let len = srs.degree_x.log_2() + srs.degree_y.log_2();
  assert!(poly.n() <= len, "Polynomial has more variables than SRS supports");

  let split_input = split_input(srs, input, scalar_field_zero::<P::ScalarField>());
  let r_x = &split_input[1];

  // Extract x variables and compute eq polynomial evaluations
  let eq_x_evals = evaluate_lagrange_basis(r_x);

  let mut f_prime_evals = vec![scalar_field_zero::<P::ScalarField>(); srs.degree_y];

  let y_log = srs.degree_y.log_2();
  let x_log = srs.degree_x.log_2();
  let y_mask = (1 << y_log) - 1;
  let x_mask = (1 << x_log) - 1;

  for (denseml_idx, &val) in poly.evaluations.iter() {
    let y = denseml_idx & y_mask;

    let x = (denseml_idx >> y_log) & x_mask;

    if x < srs.degree_x && y < srs.degree_y {
      let eq_val = eq_x_evals[x];
      f_prime_evals[y] = f_prime_evals[y] + eq_val * val;
    }
  }

  let f_prime = DenseMLPoly::new(srs.degree_y.log_2(), f_prime_evals);

  KZH2Opening {
    d_x: aux.d_x.clone(),
    f_star: f_prime,
  }
}

#[derive(Clone, Debug)]
pub struct SparseKZH2Commit<P: PairingTrait> {
  pub srs: Arc<KZH2SRS<P>>,
  pub aux: Arc<RefCell<Option<KZH2Aux<P>>>>,
}

#[derive(Clone, Debug)]
pub struct SparseKZH2CommitKey<P: PairingTrait> {
  // HashMap of polynomial size (n) -> SRS for that size
  pub srs_map: Arc<std::collections::HashMap<usize, Arc<KZH2SRS<P>>>>,
}

#[derive(Clone)]
pub struct SparseKZH2VerifierKey<P: PairingTrait> {
  // HashMap of polynomial size (n) -> SRS for that size
  pub srs_map: Arc<std::collections::HashMap<usize, Arc<KZH2SRS<P>>>>,
}

impl<P: PairingTrait> SparseKZH2Commit<P> {
  /// Fast fake setup for benchmarking purposes only.
  /// Uses deterministic values instead of cryptographically secure randomness.
  /// WARNING: Do NOT use for production - not cryptographically secure!
  #[cfg(feature = "arkworks")]
  pub fn fake_setup(n: usize) -> SparseKZH2CommitKey<P>
  where
    P::ScalarField: CryptoField,
    P::G1: ark_ec::CurveGroup,
    P::G1Affine: Send + Sync,
  {
    let kzh2_key = KZH2Commit::<P>::fake_setup(n);
    SparseKZH2CommitKey { srs_map: kzh2_key.srs_map }
  }

  /// Fast fake setup for icicle backend
  #[cfg(all(feature = "icicle", not(feature = "arkworks")))]
  pub fn fake_setup(n: usize) -> SparseKZH2CommitKey<P>
  where
    P::ScalarField: CryptoField,
    P::G1Affine: icicle_core::traits::GenerateRandom,
    P::G2Affine: icicle_core::traits::GenerateRandom,
  {
    let kzh2_key = KZH2Commit::<P>::fake_setup(n);
    SparseKZH2CommitKey { srs_map: kzh2_key.srs_map }
  }
}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> MLPolyCommit<P::ScalarField, SparseMLPoly<P::ScalarField>> for SparseKZH2Commit<P>
where
  P::ScalarField: CryptoField,
  P::G1: ark_ec::CurveGroup,
  P::G1Affine: Send + Sync,
{
  type CommitmentKey = SparseKZH2CommitKey<P>;
  type VerifierKey = SparseKZH2VerifierKey<P>;
  type Commitment = KZH2Commitment<P>;
  type Proof = KZH2Opening<P>;
  type BatchProof = ();

  fn setup(n: usize, _sf_log: usize, _offset: i128, _size: usize) -> Self::CommitmentKey {
    let maximum_degree_log2 = n;
    use ark_std::rand::thread_rng;
    let mut rng = thread_rng();
    let srs = setup_kzh2_srs::<P, _>(maximum_degree_log2, &mut rng);
    let mut srs_map = std::collections::HashMap::new();
    srs_map.insert(n, Arc::new(srs));
    SparseKZH2CommitKey { srs_map: Arc::new(srs_map) }
  }

  fn commit(poly: &SparseMLPoly<P::ScalarField>, key: &Self::CommitmentKey) -> Self::Commitment {
    let srs = key.srs_map.get(&poly.n()).expect(&format!("No SRS found for polynomial size {}", poly.n()));
    let (com, aux) = sparse_kzh2_commit(srs, poly);
    KZH2Commitment {
      srs: Arc::clone(srs),
      c: com,
      aux,
    }
  }

  fn open(commitment: &Self::Commitment, poly: &SparseMLPoly<P::ScalarField>, _key: &Self::CommitmentKey, point: &[P::ScalarField]) -> Self::Proof {
    let srs = &commitment.srs;
    sparse_kzh2_open(srs, point, &commitment.c, &commitment.aux, poly)
  }

  fn verify(commitment: &Self::Commitment, proof: &Self::Proof, _key: &Self::VerifierKey, point: &[P::ScalarField]) -> bool {
    let srs = &commitment.srs;
    let split_input = split_input(srs, point, scalar_field_zero::<P::ScalarField>());
    let r_y = &split_input[0];
    let claimed_eval = proof.f_star.evaluate_at_point(r_y);
    kzh2_verify(srs, point, &claimed_eval, &commitment.c, proof)
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
impl<P: PairingTrait> MLPolyCommit<P::ScalarField, SparseMLPoly<P::ScalarField>> for SparseKZH2Commit<P>
where
  P::ScalarField: CryptoField,
  P::ScalarField: icicle_core::traits::GenerateRandom,
  P::G1Affine: icicle_core::traits::GenerateRandom,
  P::G2Affine: icicle_core::traits::GenerateRandom,
{
  type CommitmentKey = SparseKZH2CommitKey<P>;
  type VerifierKey = SparseKZH2VerifierKey<P>;
  type Commitment = KZH2Commitment<P>;
  type Proof = KZH2Opening<P>;
  type BatchProof = ();

  fn setup(n: usize, _sf_log: usize, _offset: i128, _size: usize) -> Self::CommitmentKey {
    let maximum_degree_log2 = n;
    let mut dummy_rng = ();
    let srs = setup_kzh2_srs::<P, _>(maximum_degree_log2, &mut dummy_rng);
    let mut srs_map = std::collections::HashMap::new();
    srs_map.insert(n, Arc::new(srs));
    SparseKZH2CommitKey { srs_map: Arc::new(srs_map) }
  }

  fn commit(poly: &SparseMLPoly<P::ScalarField>, key: &Self::CommitmentKey) -> Self::Commitment {
    let srs = key.srs_map.get(&poly.n()).expect(&format!("No SRS found for polynomial size {}", poly.n()));
    let (com, aux) = sparse_kzh2_commit(srs, poly);
    KZH2Commitment {
      srs: Arc::clone(srs),
      c: com,
      aux,
    }
  }

  fn open(commitment: &Self::Commitment, poly: &SparseMLPoly<P::ScalarField>, _key: &Self::CommitmentKey, point: &[P::ScalarField]) -> Self::Proof {
    let srs = &commitment.srs;
    sparse_kzh2_open(srs, point, &commitment.c, &commitment.aux, poly)
  }

  fn verify(commitment: &Self::Commitment, proof: &Self::Proof, _key: &Self::VerifierKey, point: &[P::ScalarField]) -> bool {
    let srs = &commitment.srs;
    let split_input = split_input(srs, point, scalar_field_zero::<P::ScalarField>());
    let r_y = &split_input[0];
    let claimed_eval = proof.f_star.evaluate_at_point(r_y);
    kzh2_verify(srs, point, &claimed_eval, &commitment.c, proof)
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
