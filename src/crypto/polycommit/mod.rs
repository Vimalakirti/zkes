use crate::util::poly::{CryptoField, MLPoly};
pub mod kzh2;
pub mod kzh3;
pub mod sparse_kzh2;
pub mod sparse_kzh3;
pub use kzh3::{KZH3Commit, KZH3Commitment, KZH3VerifierKey};

/// Sparse MSM helper for arkworks: directly sum all_bases[idx] in parallel using rayon
#[cfg(feature = "arkworks")]
pub fn sparse_msm<G, F>(all_bases: &[G::Affine], indices: &[usize], scalars: &[F]) -> Result<G, String>
where
  G: ark_ec::VariableBaseMSM<MulBase = G::Affine> + ark_ec::CurveGroup<ScalarField = F>,
  F: Copy + Send + Sync,
  G::Affine: Send + Sync,
{
  if indices.len() != scalars.len() {
    return Err(format!(
      "Sparse MSM: indices and scalars must have the same length, got {} and {}",
      indices.len(),
      scalars.len()
    ));
  }

  if indices.is_empty() {
    return Ok(G::zero());
  }

  // Gather bases at indices - more efficient than individual clones
  let gathered_bases: Vec<G::Affine> = indices.iter().map(|&idx| all_bases[idx].clone()).collect();
  G::msm(&gathered_bases, scalars).map_err(|e| format!("MSM failed: {:?}", e))
}

/// Unified pairing trait that works with either arkworks or icicle
#[cfg(feature = "arkworks")]
pub use ark_ec::pairing::Pairing as PairingTrait;

#[cfg(feature = "icicle")]
pub trait PairingTrait: Sized + Send + Sync + Clone + std::fmt::Debug {
  type G1Affine: Clone + std::fmt::Debug + PartialEq + Into<Self::G1> + PairingAffine;
  type G2Affine: Clone + std::fmt::Debug + Into<Self::G2>;
  type G1: Clone
    + std::fmt::Debug
    + From<Self::G1Affine>
    + Into<Self::G1Affine>
    + std::ops::Add<Output = Self::G1>
    + std::ops::Sub<Output = Self::G1>
    + std::ops::Mul<Self::ScalarField, Output = Self::G1>
    + Copy
    + PartialEq
    + PairingG1<ScalarField = Self::ScalarField, Affine = Self::G1Affine>;
  type G2: Clone
    + std::fmt::Debug
    + From<Self::G2Affine>
    + Into<Self::G2Affine>
    + std::ops::Add<Output = Self::G2>
    + std::ops::Sub<Output = Self::G2>
    + std::ops::Mul<Self::ScalarField, Output = Self::G2>
    + Copy
    + icicle_core::projective::Projective<Affine = Self::G2Affine>;
  type GT: Clone + std::fmt::Debug + PartialEq;
  type ScalarField: CryptoField + PartialEq;

  /// Pairing operation: e : G1 × G2 → GT
  fn pairing(p: &Self::G1Affine, q: &Self::G2Affine) -> Self::GT;

  /// Multi-pairing: computes the product of pairings
  fn multi_pairing(ps: &[Self::G1Affine], qs: &[Self::G2Affine]) -> Self::GT;
}

/// Helper trait for G1 group operations (MSM, zero, etc.) for icicle
#[cfg(feature = "icicle")]
pub trait PairingG1: Sized {
  type ScalarField: CryptoField;
  type Affine: Into<Self> + Clone + PairingAffine;

  /// Multi-scalar multiplication: computes sum of bases[i] * scalars[i]
  /// Returns Result for consistency with arkworks API
  fn msm(bases: &[Self::Affine], scalars: &[Self::ScalarField]) -> Result<Self, String>;

  /// Multi-scalar multiplication without error checking
  fn msm_unchecked(bases: &[Self::Affine], scalars: &[Self::ScalarField]) -> Self;

  /// Sparse MSM: computes sum of all_bases[indices[i]] * scalars[i]
  /// Avoids cloning curve points by indexing directly into the base array
  fn sparse_msm(all_bases: &[Self::Affine], indices: &[usize], scalars: &[Self::ScalarField]) -> Result<Self, String>;

  /// Identity element
  fn zero() -> Self;

  /// Convert to affine representation
  fn into_affine(self) -> Self::Affine;
}

/// Helper trait for affine point operations
#[cfg(feature = "icicle")]
pub trait PairingAffine: Sized {
  type BaseField: crate::util::poly::CryptoField;

  fn x(&self) -> Self::BaseField;
  fn y(&self) -> Self::BaseField;
}

#[cfg(all(feature = "arkworks", feature = "icicle"))]
compile_error!("Features 'arkworks' and 'icicle' cannot be enabled simultaneously. Choose one.");

// Macro to implement PairingG1 for icicle projective types
// This reduces duplication if we need to support multiple curves
#[cfg(feature = "icicle")]
macro_rules! impl_pairing_g1_for_icicle {
  ($projective_type:ty, $scalar_field:ty, $affine_type:ty) => {
    impl PairingG1 for $projective_type {
      type ScalarField = $scalar_field;
      type Affine = $affine_type;

      fn msm(bases: &[Self::Affine], scalars: &[Self::ScalarField]) -> Result<Self, String> {
        use icicle_core::msm::{MSMConfig, MSM};
        use icicle_runtime::memory::HostSlice;

        if bases.len() != scalars.len() {
          return Err(format!(
            "MSM: bases and scalars must have the same length, got {} and {}",
            bases.len(),
            scalars.len()
          ));
        }

        if bases.is_empty() {
          use icicle_core::projective::Projective;
          return Ok(<$projective_type as Projective>::zero());
        }

        let cfg = MSMConfig::default();
        use icicle_core::projective::Projective;
        let mut result = vec![<$projective_type as Projective>::zero(); 1];
        let scalars_slice = HostSlice::from_slice(scalars);
        let bases_slice = HostSlice::from_slice(bases);
        let result_slice = HostSlice::from_mut_slice(&mut result);

        match <Self as MSM<Self>>::msm(scalars_slice, bases_slice, &cfg, result_slice) {
          Ok(_) => Ok(result[0]),
          Err(e) => Err(format!("MSM failed: {:?}", e)),
        }
      }

      fn sparse_msm(all_bases: &[Self::Affine], indices: &[usize], scalars: &[Self::ScalarField]) -> Result<Self, String> {
        if indices.len() != scalars.len() {
          return Err(format!(
            "Sparse MSM: indices and scalars must have the same length, got {} and {}",
            indices.len(),
            scalars.len()
          ));
        }

        if indices.is_empty() {
          use icicle_core::projective::Projective;
          return Ok(<$projective_type as Projective>::zero());
        }

        // Gather bases at specified indices (unavoidable, but more efficient than user doing it)
        let gathered_bases: Vec<Self::Affine> = indices.iter().map(|&idx| all_bases[idx]).collect();
        Self::msm(&gathered_bases, scalars)
      }

      fn msm_unchecked(bases: &[Self::Affine], scalars: &[Self::ScalarField]) -> Self {
        use icicle_core::msm::{MSMConfig, MSM};
        use icicle_runtime::memory::HostSlice;

        if bases.is_empty() {
          use icicle_core::projective::Projective;
          return <$projective_type as Projective>::zero();
        }

        let cfg = MSMConfig::default();
        use icicle_core::projective::Projective;
        let mut result = vec![<$projective_type as Projective>::zero(); 1];
        let scalars_slice = HostSlice::from_slice(scalars);
        let bases_slice = HostSlice::from_slice(bases);
        let result_slice = HostSlice::from_mut_slice(&mut result);

        let _ = <Self as MSM<Self>>::msm(scalars_slice, bases_slice, &cfg, result_slice);
        result[0]
      }

      fn zero() -> Self {
        use icicle_core::projective::Projective;
        <$projective_type as Projective>::zero()
      }

      fn into_affine(self) -> Self::Affine {
        self.into()
      }
    }
  };
}

// Implement PairingG1 for icicle BLS12-381 G1
#[cfg(feature = "icicle")]
impl_pairing_g1_for_icicle!(
  icicle_bls12_381::curve::G1Projective,
  icicle_bls12_381::curve::ScalarField,
  icicle_bls12_381::curve::G1Affine
);

// Implement PairingG1 for icicle BN254 G1
#[cfg(feature = "icicle")]
impl_pairing_g1_for_icicle!(
  icicle_bn254::curve::G1Projective,
  icicle_bn254::curve::ScalarField,
  icicle_bn254::curve::G1Affine
);

// Macro to implement PairingAffine for icicle affine types
#[cfg(feature = "icicle")]
macro_rules! impl_pairing_affine_for_icicle {
  // Simple case: affine type and base field are the same (no conversion needed)
  ($affine_type:ty, $base_field:ty) => {
    impl PairingAffine for $affine_type {
      type BaseField = $base_field;

      fn x(&self) -> Self::BaseField {
        use icicle_core::affine::Affine;
        Affine::x(self)
      }

      fn y(&self) -> Self::BaseField {
        use icicle_core::affine::Affine;
        Affine::y(self)
      }
    }
  };
  // Conversion case: affine type uses a different field type that needs conversion
  // Format: impl_pairing_affine_for_icicle_with_conversion!(AffineType, SourceField, TargetField)
  ($affine_type:ty, $source_field:ty, $target_field:ty) => {
    impl PairingAffine for $affine_type {
      type BaseField = $target_field;

      fn x(&self) -> Self::BaseField {
        use icicle_core::affine::Affine;
        use icicle_core::bignum::BigNum;
        let source_x = Affine::x(self);
        // Convert via bytes using BigNum trait methods directly to avoid recursion
        // We can't use CryptoField trait methods here as they might call zero()
        let bytes = <$source_field as BigNum>::to_bytes_le(&source_x);
        <$target_field as BigNum>::from_bytes_le(&bytes)
      }

      fn y(&self) -> Self::BaseField {
        use icicle_core::affine::Affine;
        use icicle_core::bignum::BigNum;
        let source_y = Affine::y(self);
        // Convert via bytes using BigNum trait methods directly to avoid recursion
        let bytes = <$source_field as BigNum>::to_bytes_le(&source_y);
        <$target_field as BigNum>::from_bytes_le(&bytes)
      }
    }
  };
}

// Implement PairingAffine for icicle BLS12-381 G1Affine
#[cfg(feature = "icicle")]
impl_pairing_affine_for_icicle!(icicle_bls12_381::curve::G1Affine, icicle_bls12_381::curve::BaseField);

// Implement PairingAffine for icicle BLS12-381 G2Affine
// Note: G2Affine uses G2BaseField, but we need to convert to BaseField for compatibility
#[cfg(feature = "icicle")]
impl_pairing_affine_for_icicle!(
  icicle_bls12_381::curve::G2Affine,
  icicle_bls12_381::curve::G2BaseField,
  icicle_bls12_381::curve::BaseField
);

// Implement PairingAffine for icicle BN254 G1Affine
#[cfg(feature = "icicle")]
impl_pairing_affine_for_icicle!(icicle_bn254::curve::G1Affine, icicle_bn254::curve::BaseField);

// Implement PairingAffine for icicle BN254 G2Affine
#[cfg(feature = "icicle")]
impl_pairing_affine_for_icicle!(
  icicle_bn254::curve::G2Affine,
  icicle_bn254::curve::G2BaseField,
  icicle_bn254::curve::BaseField
);

// Helper functions to generate random G1Affine and G2Affine for a PairingTrait
#[cfg(feature = "icicle")]
pub(crate) fn generate_random_g1_affine<P: PairingTrait>(count: usize) -> Vec<P::G1Affine>
where
  P::G1Affine: icicle_core::traits::GenerateRandom,
{
  use icicle_core::traits::GenerateRandom;
  P::G1Affine::generate_random(count)
}

#[cfg(feature = "icicle")]
pub(crate) fn generate_random_g2_affine<P: PairingTrait>(count: usize) -> Vec<P::G2Affine>
where
  P::G2Affine: icicle_core::traits::GenerateRandom,
{
  use icicle_core::traits::GenerateRandom;
  P::G2Affine::generate_random(count)
}

// Zero-sized struct to implement PairingTrait for icicle BLS12-381
#[cfg(feature = "icicle")]
#[derive(Clone, Copy, Debug)]
pub struct IcicleBls12_381;

// Zero-sized struct to implement PairingTrait for icicle BN254
#[cfg(feature = "icicle")]
#[derive(Clone, Copy, Debug)]
pub struct IcicleBn254;

// Arkworks uses the curve types directly as pairing trait implementors
#[cfg(feature = "arkworks")]
pub use ark_bls12_381::Bls12_381 as ArkBls12_381;
#[cfg(feature = "arkworks")]
pub use ark_bn254::Bn254 as ArkBn254;

// Macro to implement PairingTrait for icicle pairing types
#[cfg(feature = "icicle")]
macro_rules! impl_pairing_trait_for_icicle {
  (
    $pairing_type:ty,
    $g1_affine:ty,
    $g2_affine:ty,
    $g1_projective:ty,
    $g2_projective:ty,
    $gt:ty,
    $scalar_field:ty
  ) => {
    impl PairingTrait for $pairing_type {
      type G1Affine = $g1_affine;
      type G2Affine = $g2_affine;
      type G1 = $g1_projective;
      type G2 = $g2_projective;
      type GT = $gt;
      type ScalarField = $scalar_field;

      fn pairing(p: &Self::G1Affine, q: &Self::G2Affine) -> Self::GT {
        use icicle_core::pairing::Pairing;
        <$g1_projective as Pairing<$g1_projective, $g2_projective, Self::GT>>::pairing(p, q).expect("Pairing should not fail")
      }

      fn multi_pairing(ps: &[Self::G1Affine], qs: &[Self::G2Affine]) -> Self::GT {
        use icicle_core::pairing::Pairing;
        assert_eq!(ps.len(), qs.len(), "multi_pairing requires equal length vectors");
        if ps.is_empty() {
          // For empty input, return identity element by pairing zero points
          // Pairing of identity elements gives identity in GT
          use icicle_core::projective::Projective;
          let g1_zero = <$g1_projective as Projective>::zero();
          let g2_zero = <$g2_projective as Projective>::zero();
          let g1_affine: Self::G1Affine = g1_zero.into();
          let g2_affine: Self::G2Affine = g2_zero.into();
          return <$g1_projective as Pairing<$g1_projective, $g2_projective, Self::GT>>::pairing(&g1_affine, &g2_affine)
            .expect("Pairing should not fail");
        }
        // Compute product of pairings
        // In pairing-based crypto, multi_pairing computes the product (multiplication in GT)
        let mut result =
          <$g1_projective as Pairing<$g1_projective, $g2_projective, Self::GT>>::pairing(&ps[0], &qs[0]).expect("Pairing should not fail");
        for (p, q) in ps[1..].iter().zip(qs[1..].iter()) {
          let pair_result = <$g1_projective as Pairing<$g1_projective, $g2_projective, Self::GT>>::pairing(p, q).expect("Pairing should not fail");
          result = result * pair_result;
        }
        result
      }
    }
  };
}

// Implement PairingTrait for icicle BLS12-381 pairing
#[cfg(feature = "icicle")]
impl_pairing_trait_for_icicle!(
  IcicleBls12_381,
  icicle_bls12_381::curve::G1Affine,
  icicle_bls12_381::curve::G2Affine,
  icicle_bls12_381::curve::G1Projective,
  icicle_bls12_381::curve::G2Projective,
  icicle_bls12_381::pairing::PairingTargetField,
  icicle_bls12_381::curve::ScalarField
);

// Implement PairingTrait for icicle BN254 pairing
#[cfg(feature = "icicle")]
impl_pairing_trait_for_icicle!(
  IcicleBn254,
  icicle_bn254::curve::G1Affine,
  icicle_bn254::curve::G2Affine,
  icicle_bn254::curve::G1Projective,
  icicle_bn254::curve::G2Projective,
  icicle_bn254::pairing::PairingTargetField,
  icicle_bn254::curve::ScalarField
);

/// Enables commitment arithmetic
pub trait Commitment<F: CryptoField>:
  Clone + Sized + std::fmt::Debug + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + std::ops::Mul<F, Output = Self> + Default
{
  /// Identity element (commitment to zero polynomial)
  ///
  /// This is a convenience method that delegates to `Default::default()`.
  /// Prefer using `Default::default()` or `Self::zero()` directly.
  fn zero() -> Self {
    Self::default()
  }
}

pub trait MLPolyCommit<F: CryptoField, P: MLPoly<F>> {
  type CommitmentKey;
  type VerifierKey;
  type Commitment: Commitment<F>;
  type Proof;
  type BatchProof;
  fn setup(n: usize, sf_log: usize, offset: i128, size: usize) -> Self::CommitmentKey;
  fn commit(poly: &P, key: &Self::CommitmentKey) -> Self::Commitment;
  fn open(commitment: &Self::Commitment, poly: &P, key: &Self::CommitmentKey, point: &[F]) -> Self::Proof;
  fn verify(commitment: &Self::Commitment, proof: &Self::Proof, key: &Self::VerifierKey, point: &[F]) -> bool;
  fn batch_open(commitments: &[Self::Commitment], polys: &[P], keys: &[Self::CommitmentKey], point: &[F]) -> Self::BatchProof;
  fn batch_verify(commitments: &[Self::Commitment], proofs: &[Self::Proof], keys: &[Self::VerifierKey], point: &[F]) -> bool;
}

/// NaiveMLPolyCommit has perfect binding but no hiding.
/// It is used to test the MLPolyCommit trait.
/// It is not used in the actual implementation.
#[derive(Clone, Debug)]
pub struct NaiveMLPolyCommit;

#[derive(Clone, Debug)]
// NaiveCommitment is a placeholder and only performs null ops for arithmetic, use a real PCS instead
pub struct NaiveCommitment<F: CryptoField, P: MLPoly<F> + Clone>(P, #[allow(dead_code)] std::marker::PhantomData<F>);

// Operator trait implementations for NaiveCommitment
impl<F: CryptoField, P: MLPoly<F> + Clone> std::ops::Add for NaiveCommitment<F, P> {
  type Output = Self;

  fn add(self, _other: Self) -> Self {
    self
  }
}

impl<F: CryptoField, P: MLPoly<F> + Clone> std::ops::Sub for NaiveCommitment<F, P> {
  type Output = Self;

  fn sub(self, _other: Self) -> Self {
    self
  }
}

impl<F: CryptoField, P: MLPoly<F> + Clone> std::ops::Mul<F> for NaiveCommitment<F, P> {
  type Output = Self;

  fn mul(self, _scalar: F) -> Self {
    self
  }
}

impl<F: CryptoField, P: MLPoly<F> + Clone> Default for NaiveCommitment<F, P> {
  fn default() -> Self {
    panic!("NaiveCommitment::default() is not supported - use a real PCS for homomorphic operation")
  }
}

impl<F: CryptoField, P: MLPoly<F> + Clone> Commitment<F> for NaiveCommitment<F, P> {}

impl<F: CryptoField + 'static, P: MLPoly<F> + Clone> MLPolyCommit<F, P> for NaiveMLPolyCommit {
  type CommitmentKey = ();
  type VerifierKey = ();
  type Commitment = NaiveCommitment<F, P>;
  type Proof = F;
  type BatchProof = ();

  fn setup(_n: usize, _sf_log: usize, _offset: i128, _size: usize) -> Self::CommitmentKey {
    ()
  }

  fn commit(poly: &P, _key: &Self::CommitmentKey) -> Self::Commitment {
    NaiveCommitment(poly.clone(), std::marker::PhantomData)
  }

  fn open(commitment: &Self::Commitment, _poly: &P, _key: &Self::CommitmentKey, point: &[F]) -> Self::Proof {
    commitment.0.evaluate_at_point(point)
  }

  fn verify(commitment: &Self::Commitment, proof: &Self::Proof, _key: &Self::VerifierKey, point: &[F]) -> bool {
    commitment.0.evaluate_at_point(point) == *proof
  }

  fn batch_open(_commitments: &[Self::Commitment], _polys: &[P], _keys: &[Self::CommitmentKey], _point: &[F]) -> Self::BatchProof {
    todo!()
  }

  fn batch_verify(_commitments: &[Self::Commitment], _proofs: &[Self::Proof], _keys: &[Self::VerifierKey], _point: &[F]) -> bool {
    todo!()
  }
}
