use crate::crypto::polycommit::{Commitment, MLPolyCommit, PairingTrait};
use crate::util::arith::Math;
use crate::util::poly::{evaluate_lagrange_basis, fix_variables_from_right, CryptoField, DenseMLPoly, MLPoly};

#[cfg(feature = "arkworks")]
use rayon::prelude::*;

#[cfg(feature = "arkworks")]
use ark_std::UniformRand;

#[cfg(feature = "icicle")]
use icicle_core::traits::GenerateRandom;

use std::cell::RefCell;
use std::sync::Arc;

pub fn pad_at_start<F: CryptoField>(input: &[F], len: usize) -> Vec<F> {
  if input.len() >= len {
    return input.to_vec();
  }
  let mut result = vec![<F as CryptoField>::zero(); len - input.len()];
  result.extend_from_slice(input);
  result
}

/// KZH2 SRS - 2-dimensional structured reference string
#[derive(Clone, Debug)]
pub struct KZH2SRS<P: PairingTrait> {
  pub degree_x: usize,
  pub degree_y: usize,
  pub h_xy: Vec<P::G1Affine>,
  pub h_y: Vec<P::G1Affine>,
  pub v_x: Vec<P::G2Affine>,
  pub v_prime: P::G2Affine,
}

/// KZH2 Auxiliary data
#[derive(Clone, Debug)]
pub struct KZH2Aux<P: PairingTrait> {
  pub d_x: Vec<P::G1>,
}

impl<P: PairingTrait> KZH2Aux<P> {
  pub fn scale_by_r(&mut self, r: &P::ScalarField) {
    #[cfg(feature = "arkworks")]
    {
      self.d_x.par_iter_mut().for_each(|d| {
        *d = *d * *r;
      });
    }
    #[cfg(not(feature = "arkworks"))]
    {
      self.d_x.iter_mut().for_each(|d| {
        *d = *d * *r;
      });
    }
  }

  /// Create a dummy aux for deserialization/default (not cryptographically valid)
  pub fn dummy() -> Self {
    Self { d_x: vec![] }
  }
}

impl<P: PairingTrait> std::ops::Add for KZH2Aux<P> {
  type Output = Self;
  fn add(self, other: Self) -> Self {
    assert_eq!(self.d_x.len(), other.d_x.len());
    Self {
      d_x: self.d_x.into_iter().zip(other.d_x).map(|(a, b)| a + b).collect(),
    }
  }
}

impl<P: PairingTrait> std::ops::Sub for KZH2Aux<P> {
  type Output = Self;
  fn sub(self, other: Self) -> Self {
    assert_eq!(self.d_x.len(), other.d_x.len());
    Self {
      d_x: self.d_x.into_iter().zip(other.d_x).map(|(a, b)| a - b).collect(),
    }
  }
}

impl<P: PairingTrait> std::ops::Mul<P::ScalarField> for KZH2Aux<P> {
  type Output = Self;
  fn mul(self, scalar: P::ScalarField) -> Self {
    Self {
      d_x: self.d_x.into_iter().map(|d| d * scalar).collect(),
    }
  }
}

/// KZH2 Opening proof
#[derive(Clone, Debug)]
pub struct KZH2Opening<P: PairingTrait> {
  pub d_x: Vec<P::G1>,
  pub f_star: DenseMLPoly<P::ScalarField>,
}

/// KZH2 Commitment (non-ZK)
#[derive(Clone, Debug)]
pub struct KZH2Commitment<P: PairingTrait> {
  pub srs: Arc<KZH2SRS<P>>,
  pub c: P::G1Affine,
  pub aux: KZH2Aux<P>,
}

// Manual PartialEq implementation that only compares the commitment point,
// not the SRS (since different Arc pointers to the same SRS data should be considered equal)
impl<P: PairingTrait> PartialEq for KZH2Commitment<P> {
  fn eq(&self, other: &Self) -> bool {
    self.c == other.c
  }
}

impl<P: PairingTrait> Eq for KZH2Commitment<P> {}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> ark_serialize::CanonicalSerialize for KZH2Commitment<P>
where
  P::G1Affine: ark_serialize::CanonicalSerialize,
  P::G1: ark_serialize::CanonicalSerialize,
{
  fn serialize_with_mode<W: ark_serialize::Write>(
    &self,
    mut writer: W,
    compress: ark_serialize::Compress,
  ) -> Result<(), ark_serialize::SerializationError> {
    self.c.serialize_with_mode(&mut writer, compress)?;
    self.aux.d_x.serialize_with_mode(&mut writer, compress)?;
    Ok(())
  }

  fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
    self.c.serialized_size(compress) + self.aux.d_x.serialized_size(compress)
  }
}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> ark_serialize::CanonicalDeserialize for KZH2Commitment<P>
where
  P::ScalarField: CryptoField,
  P::G1Affine: ark_serialize::CanonicalDeserialize + ark_serialize::Valid,
  P::G1: ark_serialize::CanonicalDeserialize + ark_serialize::Valid,
{
  fn deserialize_with_mode<R: ark_serialize::Read>(
    mut reader: R,
    compress: ark_serialize::Compress,
    validate: ark_serialize::Validate,
  ) -> Result<Self, ark_serialize::SerializationError> {
    let c = P::G1Affine::deserialize_with_mode(&mut reader, compress, validate)?;
    let d_x = Vec::<P::G1>::deserialize_with_mode(&mut reader, compress, validate)?;
    Ok(Self {
      srs: create_dummy_srs::<P>(),
      c,
      aux: KZH2Aux { d_x },
    })
  }
}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> ark_serialize::Valid for KZH2Commitment<P>
where
  P::G1Affine: ark_serialize::Valid,
{
  fn check(&self) -> Result<(), ark_serialize::SerializationError> {
    self.c.check()
  }
}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> std::ops::Add for KZH2Commitment<P>
where
  P::ScalarField: CryptoField,
  P::G1: ark_ec::CurveGroup,
{
  type Output = Self;

  fn add(self, other: Self) -> Self {
    // Both commitments should use the same SRS
    assert!(Arc::ptr_eq(&self.srs, &other.srs), "Cannot add commitments with different SRS");

    let g1_self: P::G1 = self.c.into();
    let g1_other: P::G1 = other.c.into();
    use ark_ec::CurveGroup;
    let sum = (g1_self + g1_other).into_affine();
    Self {
      srs: self.srs,
      c: sum,
      aux: self.aux + other.aux,
    }
  }
}

#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl<P: PairingTrait> std::ops::Add for KZH2Commitment<P>
where
  P::ScalarField: CryptoField,
{
  type Output = Self;

  fn add(self, other: Self) -> Self {
    // Both commitments should use the same SRS
    assert!(Arc::ptr_eq(&self.srs, &other.srs), "Cannot add commitments with different SRS");

    use crate::crypto::polycommit::PairingG1;
    let g1_self: P::G1 = self.c.into();
    let g1_other: P::G1 = other.c.into();
    let sum = (g1_self + g1_other).into_affine();
    Self {
      srs: self.srs,
      c: sum,
      aux: self.aux + other.aux,
    }
  }
}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> std::ops::Add<&KZH2Commitment<P>> for &KZH2Commitment<P>
where
  P::ScalarField: CryptoField,
  P::G1: ark_ec::CurveGroup,
{
  type Output = KZH2Commitment<P>;

  fn add(self, other: &KZH2Commitment<P>) -> Self::Output {
    self.clone() + other.clone()
  }
}

#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl<P: PairingTrait> std::ops::Add<&KZH2Commitment<P>> for &KZH2Commitment<P>
where
  P::ScalarField: CryptoField,
{
  type Output = KZH2Commitment<P>;

  fn add(self, other: &KZH2Commitment<P>) -> Self::Output {
    self.clone() + other.clone()
  }
}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> std::ops::Sub for KZH2Commitment<P>
where
  P::ScalarField: CryptoField,
  P::G1: ark_ec::CurveGroup,
{
  type Output = Self;

  fn sub(self, other: Self) -> Self {
    // Both commitments should use the same SRS
    assert!(Arc::ptr_eq(&self.srs, &other.srs), "Cannot subtract commitments with different SRS");

    use ark_ec::CurveGroup;
    let g1_self: P::G1 = self.c.into();
    let g1_other: P::G1 = other.c.into();
    let neg_other: P::G1 = -g1_other;
    let diff = (g1_self + neg_other).into_affine();
    Self {
      srs: self.srs,
      c: diff,
      aux: self.aux - other.aux,
    }
  }
}

#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl<P: PairingTrait> std::ops::Sub for KZH2Commitment<P>
where
  P::ScalarField: CryptoField,
{
  type Output = Self;

  fn sub(self, other: Self) -> Self {
    // Both commitments should use the same SRS
    assert!(Arc::ptr_eq(&self.srs, &other.srs), "Cannot subtract commitments with different SRS");

    use crate::crypto::polycommit::PairingG1;
    let g1_self: P::G1 = self.c.into();
    let g1_other: P::G1 = other.c.into();
    let diff = (g1_self - g1_other).into_affine();
    Self {
      srs: self.srs,
      c: diff,
      aux: self.aux - other.aux,
    }
  }
}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> std::ops::Sub<&KZH2Commitment<P>> for &KZH2Commitment<P>
where
  P::ScalarField: CryptoField,
  P::G1: ark_ec::CurveGroup,
{
  type Output = KZH2Commitment<P>;

  fn sub(self, other: &KZH2Commitment<P>) -> Self::Output {
    self.clone() - other.clone()
  }
}

#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl<P: PairingTrait> std::ops::Sub<&KZH2Commitment<P>> for &KZH2Commitment<P>
where
  P::ScalarField: CryptoField,
{
  type Output = KZH2Commitment<P>;

  fn sub(self, other: &KZH2Commitment<P>) -> Self::Output {
    self.clone() - other.clone()
  }
}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> std::ops::Mul<P::ScalarField> for KZH2Commitment<P>
where
  P::ScalarField: CryptoField,
  P::G1: ark_ec::CurveGroup,
{
  type Output = Self;

  fn mul(self, scalar: P::ScalarField) -> Self {
    let g1: P::G1 = self.c.into();
    use ark_ec::CurveGroup;
    let scaled = (g1 * scalar).into_affine();
    Self {
      srs: self.srs,
      c: scaled,
      aux: self.aux * scalar,
    }
  }
}

#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl<P: PairingTrait> std::ops::Mul<P::ScalarField> for KZH2Commitment<P>
where
  P::ScalarField: CryptoField,
{
  type Output = Self;

  fn mul(self, scalar: P::ScalarField) -> Self {
    use crate::crypto::polycommit::PairingG1;
    let g1: P::G1 = self.c.into();
    let scaled = (g1 * scalar).into_affine();
    Self {
      srs: self.srs,
      c: scaled,
      aux: self.aux * scalar,
    }
  }
}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> std::ops::Mul<&P::ScalarField> for &KZH2Commitment<P>
where
  P::ScalarField: CryptoField,
  P::G1: ark_ec::CurveGroup,
{
  type Output = KZH2Commitment<P>;

  fn mul(self, scalar: &P::ScalarField) -> Self::Output {
    self.clone() * scalar.clone()
  }
}

#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl<P: PairingTrait> std::ops::Mul<&P::ScalarField> for &KZH2Commitment<P>
where
  P::ScalarField: CryptoField,
{
  type Output = KZH2Commitment<P>;

  fn mul(self, scalar: &P::ScalarField) -> Self::Output {
    self.clone() * scalar.clone()
  }
}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> Default for KZH2Commitment<P>
where
  P::ScalarField: CryptoField,
  P::G1: ark_ec::CurveGroup,
{
  fn default() -> Self {
    Self {
      srs: create_dummy_srs::<P>(),
      c: P::G1Affine::default(),
      aux: KZH2Aux::dummy(),
    }
  }
}

#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl<P: PairingTrait> Default for KZH2Commitment<P>
where
  P::ScalarField: CryptoField,
{
  fn default() -> Self {
    use crate::crypto::polycommit::PairingG1;
    let zero_projective: P::G1 = <P::G1 as PairingG1>::zero();
    Self {
      srs: create_dummy_srs::<P>(),
      c: zero_projective.into_affine(),
      aux: KZH2Aux::dummy(),
    }
  }
}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> Commitment<P::ScalarField> for KZH2Commitment<P>
where
  P::ScalarField: CryptoField,
  P::G1: ark_ec::CurveGroup,
{
}

#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl<P: PairingTrait> Commitment<P::ScalarField> for KZH2Commitment<P> where P::ScalarField: CryptoField {}

fn get_degree_from_maximum_supported_degree(n: usize) -> (usize, usize) {
  match n % 2 {
    0 => (n / 2, n / 2),
    1 => (n / 2, n / 2 + 1),
    _ => unreachable!(),
  }
}

// Helper function to create a minimal dummy SRS for Default/deserialization
// This SRS is not cryptographically valid and should not be used for actual cryptographic operations
#[cfg(feature = "arkworks")]
fn create_dummy_srs<P: PairingTrait>() -> Arc<KZH2SRS<P>>
where
  P::ScalarField: CryptoField,
{
  Arc::new(KZH2SRS {
    degree_x: 1,
    degree_y: 1,
    h_xy: vec![P::G1Affine::default()],
    h_y: vec![P::G1Affine::default()],
    v_x: vec![P::G2Affine::default()],
    v_prime: P::G2Affine::default(),
  })
}

#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
fn create_dummy_srs<P: PairingTrait>() -> Arc<KZH2SRS<P>>
where
  P::ScalarField: CryptoField,
  P::G2: icicle_core::projective::Projective<Affine = P::G2Affine>,
{
  use crate::crypto::polycommit::PairingG1;
  use icicle_core::projective::Projective;
  // Create zero/identity G2Affine value safely using Projective::zero
  // This is only used for Default/deserialization, not for actual cryptographic operations
  let g2_zero: P::G2 = <P::G2 as Projective>::zero();
  let dummy_g2_affine: P::G2Affine = g2_zero.to_affine();
  Arc::new(KZH2SRS {
    degree_x: 1,
    degree_y: 1,
    h_xy: vec![<P::G1 as PairingG1>::zero().into_affine()],
    h_y: vec![<P::G1 as PairingG1>::zero().into_affine()],
    v_x: vec![dummy_g2_affine.clone()],
    v_prime: dummy_g2_affine,
  })
}

pub fn scalar_field_zero<F: CryptoField>() -> F {
  <F as CryptoField>::zero()
}

/// the function receives an input r and splits into two sub-vectors y and x to be used for PCS
///
/// It's used later when we have a constant SRS, and we pad the polynomial so we can commit to it via SRS
/// This function in fact pads to polynomial inputs by appends necessary zeros and split the input into y and x input
/// The ordering follows LSB-first (y in lowest bits, x in highest bits) to match DenseMLPoly ordering
pub fn split_input<P: PairingTrait, T: Clone>(srs: &KZH2SRS<P>, input: &[T], default: T) -> Vec<Vec<T>> {
  let y_log = srs.degree_y.log_2();
  let x_log = srs.degree_x.log_2();
  let total_length = x_log + y_log;

  // If r is smaller than the required length, extend it with zeros at the beginning
  let mut extended_r = input.to_vec();
  if input.len() < total_length {
    let mut zeros = vec![default; total_length - input.len()];
    zeros.extend(extended_r); // Prepend zeros to the beginning
    extended_r = zeros;
  }

  // Split the vector into two parts following LSB-first ordering: [y, x]
  // y is in bits 0..y_log (lowest bits)
  // x is in bits y_log..y_log+x_log (highest bits)
  let r_y = extended_r[..y_log].to_vec();
  let r_x = extended_r[y_log..(y_log + x_log)].to_vec();
  vec![r_y, r_x]
}

/// Setup KZH2 SRS
#[cfg(feature = "arkworks")]
pub fn setup_kzh2_srs<P: PairingTrait, R: ark_std::rand::Rng>(maximum_degree: usize, rng: &mut R) -> KZH2SRS<P>
where
  P::ScalarField: CryptoField,
  P::G1Affine: ark_ec::AffineRepr,
  P::G2Affine: ark_ec::AffineRepr,
{
  let (degree_x_log, degree_y_log) = get_degree_from_maximum_supported_degree(maximum_degree);
  let degree_x = 1 << degree_x_log;
  let degree_y = 1 << degree_y_log;

  let (g, v) = (P::G1Affine::rand(rng), P::G2Affine::rand(rng));
  let g_proj: P::G1 = g.into();
  let v_proj: P::G2 = v.into();

  // Sample trapdoors
  let tau_x: Vec<P::ScalarField> = (0..degree_x).map(|_| P::ScalarField::rand(rng)).collect();
  let tau_y: Vec<P::ScalarField> = (0..degree_y).map(|_| P::ScalarField::rand(rng)).collect();
  let alpha = P::ScalarField::rand(rng);

  // Generate h_xy: g^{tau_x[i] * tau_y[j]} for all i, j
  // Indexed in DenseMLPoly order (y in low bits, x in high bits)
  let x_log = degree_x.log_2();
  let y_log = degree_y.log_2();
  let y_mask = (1 << y_log) - 1;
  let x_mask = (1 << x_log) - 1;

  let h_xy: Vec<_> = (0..degree_x * degree_y)
    .into_par_iter()
    .map(|denseml_idx| {
      let y = denseml_idx & y_mask;
      let x = (denseml_idx >> y_log) & x_mask;
      let scalar = tau_x[x] * tau_y[y];
      (g_proj * scalar).into()
    })
    .collect();

  // Generate h_y: g^{alpha * tau_y[j]} for all j
  let h_y: Vec<_> = (0..degree_y).into_par_iter().map(|i| (g_proj * (alpha * tau_y[i])).into()).collect();

  // Generate v_x: v^{tau_x[i]} for all i
  let v_x: Vec<_> = (0..degree_x).into_par_iter().map(|i| (v_proj * tau_x[i]).into()).collect();

  // Generate v_prime: v^{alpha}
  let v_prime = (v_proj * alpha).into();

  KZH2SRS {
    degree_x,
    degree_y,
    h_xy,
    h_y,
    v_x,
    v_prime,
  }
}

/// Setup KZH2 SRS for icicle backend
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
pub fn setup_kzh2_srs<P: PairingTrait, R>(maximum_degree: usize, _rng: &mut R) -> KZH2SRS<P>
where
  P::ScalarField: CryptoField + GenerateRandom,
  P::G1Affine: icicle_core::traits::GenerateRandom,
  P::G2Affine: icicle_core::traits::GenerateRandom,
{
  let (degree_x_log, degree_y_log) = get_degree_from_maximum_supported_degree(maximum_degree);
  let degree_x = 1 << degree_x_log;
  let degree_y = 1 << degree_y_log;

  use icicle_core::traits::GenerateRandom;
  let g_vec = P::G1Affine::generate_random(1);
  let v_vec = P::G2Affine::generate_random(1);
  let g = g_vec[0].clone();
  let v = v_vec[0].clone();

  let g_proj: P::G1 = g.clone().into();
  let v_proj: P::G2 = v.clone().into();

  // Sample trapdoors
  let tau_x: Vec<P::ScalarField> = P::ScalarField::generate_random(degree_x);
  let tau_y: Vec<P::ScalarField> = P::ScalarField::generate_random(degree_y);
  let alpha = P::ScalarField::generate_random(1)[0];

  // Generate h_xy: g^{tau_x[i] * tau_y[j]} for all i, j
  // Indexed in DenseMLPoly order (y in low bits, x in high bits)
  let x_log = degree_x.log_2();
  let y_log = degree_y.log_2();
  let y_mask = (1 << y_log) - 1;
  let x_mask = (1 << x_log) - 1;

  let h_xy: Vec<_> = (0..degree_x * degree_y)
    .map(|denseml_idx| {
      let y = denseml_idx & y_mask;
      let x = (denseml_idx >> y_log) & x_mask;
      let scalar = tau_x[x] * tau_y[y];
      (g_proj * scalar).into()
    })
    .collect();

  // Generate h_y: g^{alpha * tau_y[j]} for all j
  let h_y: Vec<_> = (0..degree_y).map(|i| (g_proj * (alpha * tau_y[i])).into()).collect();

  // Generate v_x: v^{tau_x[i]} for all i
  let v_x: Vec<_> = (0..degree_x).map(|i| (v_proj * tau_x[i]).into()).collect();

  // Generate v_prime: v^{alpha}
  let v_prime = (v_proj * alpha).into();

  KZH2SRS {
    degree_x,
    degree_y,
    h_xy,
    h_y,
    v_x,
    v_prime,
  }
}

/// Commit to a polynomial using KZH2
pub fn kzh2_commit<P: PairingTrait>(srs: &KZH2SRS<P>, poly: &DenseMLPoly<P::ScalarField>) -> (P::G1Affine, KZH2Aux<P>)
where
  P::ScalarField: CryptoField,
{
  let len = srs.degree_x.log_2() + srs.degree_y.log_2();

  // Only extend if needed to avoid unnecessary cloning
  let extended_poly;
  let poly_ref: &DenseMLPoly<P::ScalarField> = if poly.n == len {
    poly // No clone needed!
  } else {
    extended_poly = poly.clone().extend_number_of_variables(len);
    &extended_poly
  };

  // Compute commitment: C = sum_i f[i] * h_xy[i]
  // h_xy is now indexed by DenseMLPoly order, so direct MSM works
  let c = {
    #[cfg(feature = "arkworks")]
    {
      use ark_ec::VariableBaseMSM;
      P::G1::msm(&srs.h_xy, &poly_ref.evaluations).unwrap().into()
    }
    #[cfg(feature = "icicle")]
    {
      use crate::crypto::polycommit::PairingG1;
      <P::G1 as PairingG1>::msm(&srs.h_xy, &poly_ref.evaluations).unwrap().into()
    }
  };

  // Compute auxiliary data: d_x[i] = sum_j f[i,j] * h_y[j]
  // Use get_partial_evaluation_for_boolean_input to directly access polynomial slices
  // without creating an intermediate 2D structure, avoiding data movement
  #[cfg(feature = "arkworks")]
  let d_x: Vec<_> = (0..srs.degree_x)
    .into_par_iter()
    .map(|i| {
      use ark_ec::VariableBaseMSM;
      let partial_evals = poly_ref.get_partial_evaluation_for_boolean_input(i, srs.degree_y);
      P::G1::msm(&srs.h_y, partial_evals).unwrap()
    })
    .collect();

  #[cfg(not(feature = "arkworks"))]
  let d_x: Vec<_> = (0..srs.degree_x)
    .into_iter()
    .map(|i| {
      use crate::crypto::polycommit::PairingG1;
      let partial_evals = poly_ref.get_partial_evaluation_for_boolean_input(i, srs.degree_y);
      <P::G1 as PairingG1>::msm(&srs.h_y, partial_evals).unwrap()
    })
    .collect();

  (c, KZH2Aux { d_x })
}

/// Open a commitment at a point
pub fn kzh2_open<P: PairingTrait>(
  srs: &KZH2SRS<P>,
  input: &[P::ScalarField],
  _com: &P::G1Affine,
  aux: &KZH2Aux<P>,
  poly: &DenseMLPoly<P::ScalarField>,
) -> KZH2Opening<P>
where
  P::ScalarField: CryptoField,
{
  let len = srs.degree_x.log_2() + srs.degree_y.log_2();

  // Only extend if needed to avoid unnecessary cloning
  let extended_poly;
  let poly_ref: &DenseMLPoly<P::ScalarField> = if poly.n == len {
    poly // No clone needed!
  } else {
    extended_poly = poly.clone().extend_number_of_variables(len);
    &extended_poly
  };

  assert_eq!(poly_ref.n, len);
  assert_eq!(poly_ref.evaluations.len(), 1 << poly_ref.n);

  let split_input = split_input(srs, input, scalar_field_zero::<P::ScalarField>());
  let r_x = &split_input[1];

  // Compute f_star by fixing x variables from the right (highest bits)
  let f_star = fix_variables_from_right(poly_ref, r_x);

  KZH2Opening {
    d_x: aux.d_x.clone(),
    f_star,
  }
}

/// Verify an opening proof
#[cfg(feature = "arkworks")]
pub fn kzh2_verify<P: PairingTrait>(
  srs: &KZH2SRS<P>,
  input: &[P::ScalarField],
  output: &P::ScalarField,
  com: &P::G1Affine,
  open: &KZH2Opening<P>,
) -> bool
where
  P::ScalarField: CryptoField,
  P::G1: ark_ec::CurveGroup,
  P::G2: ark_ec::CurveGroup,
  P::TargetField: PartialEq,
{
  let split_input = split_input(srs, input, scalar_field_zero::<P::ScalarField>());
  let r_x = &split_input[1];
  let r_y = &split_input[0];

  // Convert d_x to affine once and reuse for both pairing and MSM checks
  let d_x_affine: Vec<P::G1Affine> = {
    use ark_ec::CurveGroup;
    open.d_x.iter().map(|g| (*g).into_affine()).collect()
  };

  // Step 1: pairing check
  {
    use ark_ec::CurveGroup;
    use ark_std::Zero;
    // Combine the pairings into a single multi-pairing
    let g1_elems: Vec<_> = std::iter::once(com.clone())
      .chain(d_x_affine.iter().map(|g1_affine| {
        let g1_proj: P::G1 = g1_affine.clone().into();
        (P::G1::zero() - g1_proj).into_affine()
      }))
      .collect();

    let g2_elems: Vec<_> = std::iter::once(srs.v_prime.clone()).chain(srs.v_x.iter().cloned()).collect();

    let pairing_result = P::multi_pairing(&g1_elems, &g2_elems);
    // Check if pairing result is identity by pairing with identity elements
    let identity_g1_proj = P::G1::zero();
    let identity_g2_proj = P::G2::zero();
    let identity_g1 = identity_g1_proj.into_affine();
    let identity_g2 = identity_g2_proj.into_affine();
    let identity_pairing = P::pairing(&identity_g1, &identity_g2);
    if pairing_result != identity_pairing {
      return false;
    }
  }

  // Step 2: MSM check
  // Use r_x for eq polynomial evaluation (fixing x variables)
  let eq_evals: Vec<_> = evaluate_lagrange_basis(r_x);
  let negated_eq_evals: Vec<_> = eq_evals
    .iter()
    .map(|scalar| {
      let zero = scalar_field_zero::<P::ScalarField>();
      zero - *scalar
    })
    .collect();

  let scalars: Vec<_> = open.f_star.evaluations.iter().chain(negated_eq_evals.iter()).cloned().collect();

  // Reuse d_x_affine from earlier
  let bases: Vec<_> = srs.h_y.iter().chain(d_x_affine.iter()).cloned().collect();

  let result = {
    use ark_ec::VariableBaseMSM;
    P::G1::msm(&bases, &scalars).unwrap()
  };

  {
    use ark_std::Zero;
    if result != P::G1::zero() {
      return false;
    }
  }

  // Step 3: verify the claimed evaluation
  let computed_eval = open.f_star.evaluate_at_point(r_y);
  if computed_eval != *output {
    return false;
  }

  true
}

#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
pub fn kzh2_verify<P: PairingTrait>(
  srs: &KZH2SRS<P>,
  input: &[P::ScalarField],
  output: &P::ScalarField,
  com: &P::G1Affine,
  open: &KZH2Opening<P>,
) -> bool
where
  P::ScalarField: CryptoField,
{
  let split_input = split_input(srs, input, scalar_field_zero::<P::ScalarField>());
  let r_x = &split_input[1];
  let r_y = &split_input[0];

  // Convert d_x to affine once and reuse for both pairing and MSM checks
  let d_x_affine: Vec<P::G1Affine> = open.d_x.iter().map(|g| (*g).into()).collect();

  // Step 1: pairing check
  {
    // For icicle, we need to check if the pairing result is the identity
    // This is a simplified check - you may need to adjust based on your GT implementation
    // For now, we'll use a different approach - check individual pairings
    let lhs = P::pairing(com, &srs.v_prime);
    let rhs = P::multi_pairing(&d_x_affine, &srs.v_x);
    if lhs != rhs {
      return false;
    }
  }

  // Step 2: MSM check
  // Use r_x for eq polynomial evaluation (fixing x variables)
  let eq_evals: Vec<_> = evaluate_lagrange_basis(r_x);
  let negated_eq_evals: Vec<_> = eq_evals
    .iter()
    .map(|scalar| {
      let zero = scalar_field_zero::<P::ScalarField>();
      zero - *scalar
    })
    .collect();

  let scalars: Vec<_> = open.f_star.evaluations.iter().chain(negated_eq_evals.iter()).cloned().collect();

  // Reuse d_x_affine from earlier
  let bases: Vec<_> = srs.h_y.iter().chain(d_x_affine.iter()).cloned().collect();

  let result = {
    use crate::crypto::polycommit::PairingG1;
    <P::G1 as PairingG1>::msm(&bases, &scalars).unwrap()
  };

  {
    use crate::crypto::polycommit::PairingG1;
    if result != <P::G1 as PairingG1>::zero() {
      return false;
    }
  }

  // Step 3: verify the claimed evaluation
  let computed_eval = open.f_star.evaluate_at_point(r_y);
  if computed_eval != *output {
    return false;
  }

  true
}

/// Main KZH2 commitment scheme struct
#[derive(Clone, Debug)]
pub struct KZH2Commit<P: PairingTrait> {
  pub srs: Arc<KZH2SRS<P>>,
  pub aux: Arc<RefCell<Option<KZH2Aux<P>>>>,
}

#[derive(Clone, Debug)]
pub struct KZH2CommitKey<P: PairingTrait> {
  // HashMap of polynomial size (n) -> SRS for that size
  pub srs_map: Arc<std::collections::HashMap<usize, Arc<KZH2SRS<P>>>>,
}

#[derive(Clone)]
pub struct KZH2VerifierKey<P: PairingTrait> {
  // HashMap of polynomial size (n) -> SRS for that size
  pub srs_map: Arc<std::collections::HashMap<usize, Arc<KZH2SRS<P>>>>,
}

impl<P: PairingTrait> KZH2Commit<P> {
  /// Fast fake setup for benchmarking purposes only.
  /// Uses default/identity values to skip expensive elliptic curve operations.
  /// WARNING: Do NOT use for production - not cryptographically secure!
  #[cfg(feature = "arkworks")]
  pub fn fake_setup(n: usize) -> KZH2CommitKey<P>
  where
    P::ScalarField: CryptoField,
    P::G1: ark_ec::CurveGroup,
    P::G1Affine: Send + Sync,
  {
    let (degree_x_log, degree_y_log) = get_degree_from_maximum_supported_degree(n);
    let degree_x = 1 << degree_x_log;
    let degree_y = 1 << degree_y_log;

    // Use default/identity elements to avoid expensive curve operations
    let default_g1 = P::G1Affine::default();
    let default_g2 = P::G2Affine::default();

    // Instantly allocate with default values instead of computing MSMs
    let h_xy = vec![default_g1; degree_x * degree_y];
    let h_y = vec![default_g1; degree_y];
    let v_x = vec![default_g2; degree_x];
    let v_prime = default_g2;

    let srs = KZH2SRS {
      degree_x,
      degree_y,
      h_xy,
      h_y,
      v_x,
      v_prime,
    };

    let mut srs_map = std::collections::HashMap::new();
    srs_map.insert(n, Arc::new(srs));
    KZH2CommitKey { srs_map: Arc::new(srs_map) }
  }

  /// Fast fake setup for icicle backend
  #[cfg(all(feature = "icicle", not(feature = "arkworks")))]
  pub fn fake_setup(n: usize) -> KZH2CommitKey<P>
  where
    P::ScalarField: CryptoField,
    P::G1Affine: icicle_core::traits::GenerateRandom,
    P::G2Affine: icicle_core::traits::GenerateRandom,
  {
    use crate::crypto::polycommit::{generate_random_g1_affine, generate_random_g2_affine};

    let (degree_x_log, degree_y_log) = get_degree_from_maximum_supported_degree(n);
    let degree_x = 1 << degree_x_log;
    let degree_y = 1 << degree_y_log;

    // Generate one random point and reuse it (fast, no MSMs)
    let g1_template = generate_random_g1_affine::<P>(1)[0].clone();
    let g2_template = generate_random_g2_affine::<P>(1)[0].clone();

    // Instantly allocate with same value instead of computing MSMs
    let h_xy = vec![g1_template.clone(); degree_x * degree_y];
    let h_y = vec![g1_template; degree_y];
    let v_x = vec![g2_template.clone(); degree_x];
    let v_prime = g2_template;

    let srs = KZH2SRS {
      degree_x,
      degree_y,
      h_xy,
      h_y,
      v_x,
      v_prime,
    };

    let mut srs_map = std::collections::HashMap::new();
    srs_map.insert(n, Arc::new(srs));
    KZH2CommitKey { srs_map: Arc::new(srs_map) }
  }
}

/// MLPolyCommit implementation for arkworks backend
#[cfg(feature = "arkworks")]
impl<P: PairingTrait> MLPolyCommit<P::ScalarField, DenseMLPoly<P::ScalarField>> for KZH2Commit<P>
where
  P::ScalarField: CryptoField,
  P::G1: ark_ec::CurveGroup,
  P::G1Affine: Send + Sync,
{
  type CommitmentKey = KZH2CommitKey<P>;
  type VerifierKey = KZH2VerifierKey<P>;
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
    KZH2CommitKey { srs_map: Arc::new(srs_map) }
  }

  fn commit(poly: &DenseMLPoly<P::ScalarField>, key: &Self::CommitmentKey) -> Self::Commitment {
    let srs = key.srs_map.get(&poly.n).expect(&format!("No SRS found for polynomial size {}", poly.n));
    let (com, aux) = kzh2_commit(srs, poly);
    KZH2Commitment {
      srs: Arc::clone(srs),
      c: com,
      aux,
    }
  }

  fn open(commitment: &Self::Commitment, poly: &DenseMLPoly<P::ScalarField>, _key: &Self::CommitmentKey, point: &[P::ScalarField]) -> Self::Proof {
    let srs = &commitment.srs;
    kzh2_open(srs, point, &commitment.c, &commitment.aux, poly)
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
    _polys: &[DenseMLPoly<P::ScalarField>],
    _keys: &[Self::CommitmentKey],
    _point: &[P::ScalarField],
  ) -> Self::BatchProof {
    todo!()
  }

  fn batch_verify(_commitments: &[Self::Commitment], _proofs: &[Self::Proof], _keys: &[Self::VerifierKey], _point: &[P::ScalarField]) -> bool {
    todo!()
  }
}

/// MLPolyCommit implementation for icicle backend
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl<P: PairingTrait> MLPolyCommit<P::ScalarField, DenseMLPoly<P::ScalarField>> for KZH2Commit<P>
where
  P::ScalarField: CryptoField + GenerateRandom,
  P::G1Affine: icicle_core::traits::GenerateRandom,
  P::G2Affine: icicle_core::traits::GenerateRandom,
{
  type CommitmentKey = KZH2CommitKey<P>;
  type VerifierKey = KZH2VerifierKey<P>;
  type Commitment = KZH2Commitment<P>;
  type Proof = KZH2Opening<P>;
  type BatchProof = ();

  fn setup(n: usize, _sf_log: usize, _offset: i128, _size: usize) -> Self::CommitmentKey {
    let maximum_degree_log2 = n;
    let mut dummy_rng = ();
    let srs = setup_kzh2_srs::<P, _>(maximum_degree_log2, &mut dummy_rng);
    let mut srs_map = std::collections::HashMap::new();
    srs_map.insert(n, Arc::new(srs));
    KZH2CommitKey { srs_map: Arc::new(srs_map) }
  }

  fn commit(poly: &DenseMLPoly<P::ScalarField>, key: &Self::CommitmentKey) -> Self::Commitment {
    let srs = key.srs_map.get(&poly.n).expect(&format!("No SRS found for polynomial size {}", poly.n));
    let (com, aux) = kzh2_commit::<P>(srs, poly);
    KZH2Commitment {
      srs: Arc::clone(srs),
      c: com,
      aux,
    }
  }

  fn open(commitment: &Self::Commitment, poly: &DenseMLPoly<P::ScalarField>, _key: &Self::CommitmentKey, point: &[P::ScalarField]) -> Self::Proof {
    let srs = &commitment.srs;
    kzh2_open::<P>(srs, point, &commitment.c, &commitment.aux, poly)
  }

  fn verify(commitment: &Self::Commitment, proof: &Self::Proof, _key: &Self::VerifierKey, point: &[P::ScalarField]) -> bool {
    let srs = &commitment.srs;
    let split_input = split_input(srs, point, scalar_field_zero::<P::ScalarField>());
    let r_y = &split_input[0];
    let claimed_eval = proof.f_star.evaluate_at_point(r_y);
    kzh2_verify::<P>(srs, point, &claimed_eval, &commitment.c, proof)
  }

  fn batch_open(
    _commitments: &[Self::Commitment],
    _polys: &[DenseMLPoly<P::ScalarField>],
    _keys: &[Self::CommitmentKey],
    _point: &[P::ScalarField],
  ) -> Self::BatchProof {
    todo!()
  }

  fn batch_verify(_commitments: &[Self::Commitment], _proofs: &[Self::Proof], _keys: &[Self::VerifierKey], _point: &[P::ScalarField]) -> bool {
    todo!()
  }
}
