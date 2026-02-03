use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;

use crate::util::serialization::{ark_de_vec, ark_se_vec};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(feature = "arkworks")]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

// Arkworks imports (only imported when arkworks feature is enabled)
#[cfg(feature = "arkworks")]
use ark_crypto_primitives::sponge::Absorb;
#[cfg(feature = "arkworks")]
use ark_ff::{PrimeField, UniformRand};

// Icicle imports (only imported when icicle feature is enabled and arkworks is not)
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
use icicle_core::field::Field;

use crate::SF_FLOAT;
/// Field with cryptographic traits
///
/// Provides field arithmetic operations and cryptographic traits required for proof systems:
/// - Basic arithmetic: zero, one, from_u32, from_bytes_le, to_bytes_le, invert
/// - Fiat-Shamir transcripts (Absorb/challenges)
/// - Random sampling (UniformRand/GenerateRandom)
/// - Serialization and other proof system requirements
///
/// This trait is defined differently based on which feature is enabled:
/// - feature="arkworks": Requires PrimeField + Absorb + UniformRand
/// - feature="icicle": Requires Field + GenerateRandom
///
/// Use this for all proof-related code (sumcheck, commitments, etc.)
#[cfg(feature = "arkworks")]
pub trait CryptoField:
  PrimeField
  + Absorb
  + UniformRand
  + Clone
  + Copy
  + std::fmt::Debug
  + PartialEq
  + 'static
  + std::ops::Add<Output = Self>
  + std::ops::Sub<Output = Self>
  + std::ops::Mul<Output = Self>
{
  fn zero() -> Self;
  fn one() -> Self;
  fn from_u32(n: u32) -> Self;
  fn from_bytes_le(bytes: &[u8]) -> Self;
  fn to_bytes_le(&self) -> Vec<u8>;
  fn invert(&self) -> Self;
}

// Arkworks implementation of CryptoField
#[cfg(feature = "arkworks")]
impl<F> CryptoField for F
where
  F: PrimeField + Absorb + UniformRand + std::ops::Add<Output = F> + std::ops::Sub<Output = F> + std::ops::Mul<Output = F>,
{
  fn zero() -> Self {
    F::ZERO
  }
  fn one() -> Self {
    F::ONE
  }
  fn from_u32(n: u32) -> Self {
    F::from(n as u64)
  }
  fn from_bytes_le(bytes: &[u8]) -> Self {
    // Pad to field size if needed
    let mut padded = vec![0u8; (F::MODULUS_BIT_SIZE as usize + 7) / 8];
    let copy_len = std::cmp::min(bytes.len(), padded.len());
    padded[..copy_len].copy_from_slice(&bytes[..copy_len]);
    F::from_le_bytes_mod_order(&padded)
  }
  fn to_bytes_le(&self) -> Vec<u8> {
    let mut bytes = vec![0u8; (F::MODULUS_BIT_SIZE as usize + 7) / 8];
    self.serialize_uncompressed(&mut bytes[..]).unwrap();
    bytes
  }
  fn invert(&self) -> Self {
    self.inverse().unwrap_or(F::ZERO)
  }
}

// Icicle path: CryptoField extends Field
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
pub trait CryptoField: Field + Clone + Copy + std::fmt::Debug + PartialEq + 'static {
  fn zero() -> Self;
  fn one() -> Self;
  fn from_u32(n: u32) -> Self;
  fn from_bytes_le(bytes: &[u8]) -> Self;
  fn to_bytes_le(&self) -> Vec<u8>;
  fn invert(&self) -> Self;
}

// Macro to implement CryptoField for icicle field types
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
macro_rules! impl_crypto_field_for_icicle {
  ($field_type:ty, $byte_size:expr) => {
    impl CryptoField for $field_type {
      fn zero() -> Self {
        // Use Default::default() which creates a zero value for icicle fields
        std::default::Default::default()
      }
      fn one() -> Self {
        use icicle_core::bignum::BigNum;
        <$field_type as BigNum>::from_bytes_le(&[1u8])
      }
      fn from_u32(n: u32) -> Self {
        // For zero, use zero() method
        if n == 0 {
          return <Self as CryptoField>::zero();
        }
        if n == 1 {
          return <Self as CryptoField>::one();
        }
        use icicle_core::bignum::BigNum;
        let bytes = (n as u64).to_le_bytes();
        <$field_type as BigNum>::from_bytes_le(&bytes)
      }
      fn from_bytes_le(bytes: &[u8]) -> Self {
        // Pad to field size for consistency with arkworks behavior
        use icicle_core::bignum::BigNum;
        const FIELD_BYTE_SIZE: usize = $byte_size;
        let mut padded = vec![0u8; FIELD_BYTE_SIZE];
        let copy_len = std::cmp::min(bytes.len(), FIELD_BYTE_SIZE);
        padded[..copy_len].copy_from_slice(&bytes[..copy_len]);
        <$field_type as BigNum>::from_bytes_le(&padded)
      }
      fn to_bytes_le(&self) -> Vec<u8> {
        use icicle_core::bignum::BigNum;
        let bytes = <$field_type as BigNum>::to_bytes_le(self);
        // Pad to fixed size to match arkworks behavior
        // This ensures transcript consistency between prover and verifier
        const FIELD_BYTE_SIZE: usize = $byte_size;
        let mut padded = vec![0u8; FIELD_BYTE_SIZE];
        let copy_len = std::cmp::min(bytes.len(), FIELD_BYTE_SIZE);
        padded[..copy_len].copy_from_slice(&bytes[..copy_len]);
        padded
      }
      fn invert(&self) -> Self {
        use icicle_core::traits::Invertible;
        if *self == <Self as CryptoField>::zero() {
          <Self as CryptoField>::zero()
        } else {
          self.inv()
        }
      }
    }
  };
}

// Implement CryptoField for icicle field types
// Map each field type to its byte size for consistent serialization
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl_crypto_field_for_icicle!(icicle_bls12_381::curve::ScalarField, 32);
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl_crypto_field_for_icicle!(icicle_bls12_381::curve::BaseField, 48);
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl_crypto_field_for_icicle!(icicle_bls12_381::curve::G2BaseField, 96);

// BN254 field types for icicle
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl_crypto_field_for_icicle!(icicle_bn254::curve::ScalarField, 32);
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl_crypto_field_for_icicle!(icicle_bn254::curve::BaseField, 32);
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl_crypto_field_for_icicle!(icicle_bn254::curve::G2BaseField, 64);

/// Pads a vector at the start with zeros to reach the target length
pub fn pad_at_start<F: CryptoField>(input: &[F], len: usize) -> Vec<F> {
  if input.len() >= len {
    return input.to_vec();
  }
  let mut result = vec![<F as CryptoField>::zero(); len - input.len()];
  result.extend_from_slice(input);
  result
}

/// Convert usize to little-endian Vec<u8> representation
pub fn usize_to_le_vec(value: usize) -> Vec<u8> {
  let bytes = value.to_le_bytes();
  // Remove trailing zeros to keep it minimal
  let mut end = bytes.len();
  while end > 1 && bytes[end - 1] == 0 {
    end -= 1;
  }
  bytes[..end].to_vec()
}

pub fn le_vec_to_usize(value: &[u8]) -> usize {
  let mut result = 0;
  for (i, &byte) in value.iter().enumerate() {
    result |= (byte as usize) << (i * 8);
  }
  result
}

/// Extract low 'bits' bits from a little-endian Vec<u8> number
/// Equivalent to: number & ((1 << bits) - 1)
pub fn extract_low_bits(number: &[u8], bits: usize) -> Vec<u8> {
  if bits == 0 {
    return vec![0];
  }

  let full_bytes = bits / 8;
  let remaining_bits = bits % 8;
  let mut result = vec![];

  // Extract full bytes
  for i in 0..full_bytes.min(number.len()).min(std::mem::size_of::<usize>()) {
    result.push(number[i]);
  }

  // Extract remaining bits from the next byte
  if remaining_bits > 0 && full_bytes < number.len() && full_bytes < std::mem::size_of::<usize>() {
    let mask = (1u8 << remaining_bits) - 1;
    let masked_byte = number[full_bytes] & mask;
    result.push(masked_byte);
  }

  result
}

/// Right shift a little-endian Vec<u8> number by 'bits' bits
/// Equivalent to: number >> bits
pub fn right_shift_le_vec(number: &[u8], bits: usize) -> Vec<u8> {
  if bits == 0 {
    return number.to_vec();
  }

  let byte_shift = bits / 8;
  let bit_shift = bits % 8;

  if byte_shift >= number.len() {
    return vec![0];
  }

  let mut result = Vec::new();

  if bit_shift == 0 {
    // Simple byte shift
    result.extend_from_slice(&number[byte_shift..]);
  } else {
    // Need to shift bits within bytes
    let mut carry = 0u8;
    for i in (byte_shift..number.len()).rev() {
      let current = number[i];
      let new_byte = (current >> bit_shift) | (carry << (8 - bit_shift));
      carry = current & ((1 << bit_shift) - 1);
      result.push(new_byte);
    }
    result.reverse();
  }

  // Remove leading zeros
  while result.len() > 1 && result.last() == Some(&0) {
    result.pop();
  }

  if result.is_empty() {
    vec![0]
  } else {
    result
  }
}

/// Fix multiple variables at arbitrary positions in a dense multilinear polynomial.
/// This allows fixing any subset of variables, not just from left or right.
pub fn fix_many<F: CryptoField>(
  dense_poly: &DenseMLPoly<F>,
  fixed: &[(usize, F)], // (var_index, value), where var_index in 0..n
) -> DenseMLPoly<F> {
  let n = dense_poly.n;
  assert!(fixed.len() <= n, "too many fixed variables");

  // Map each variable index -> Option<value>
  let mut value_by_var: Vec<Option<F>> = vec![None; n];
  for &(var, val) in fixed {
    assert!(var < n, "variable index {} out of range 0..{}", var, n - 1);
    assert!(value_by_var[var].is_none(), "duplicate assignment for variable {}", var);
    value_by_var[var] = Some(val);
  }

  // Split into unfixed and fixed variables (keeps original order).
  let mut unfixed_vars = Vec::new();
  let mut fixed_vars = Vec::new();
  for v in 0..n {
    if value_by_var[v].is_some() {
      fixed_vars.push(v);
    } else {
      unfixed_vars.push(v);
    }
  }

  let m = fixed_vars.len();
  if m == 0 {
    // nothing to do; you can also return a clone or just copy the slice
    return dense_poly.clone();
  }

  let total = 1usize << n;
  let src = &dense_poly.evaluations;

  // 1) Build permutation: old var index -> new var index.
  // New order: [unfixed_vars..., fixed_vars...].
  let mut pos_new = vec![0usize; n];
  for (i, &v) in unfixed_vars.iter().enumerate() {
    pos_new[v] = i;
  }
  for (j, &v) in fixed_vars.iter().enumerate() {
    pos_new[v] = unfixed_vars.len() + j;
  }

  // 2) Permute evaluation table according to this variable reordering.
  // Index bits: old_index bit v -> new_index bit pos_new[v].
  let mut evals = vec![src[0]; total];
  for idx_old in 0..total {
    let mut idx_new = 0usize;
    let mut mask = 1usize;
    for v in 0..n {
      if idx_old & mask != 0 {
        idx_new |= 1usize << pos_new[v];
      }
      mask <<= 1;
    }
    evals[idx_new] = src[idx_old];
  }

  // 3) Prepare suffix values for the now-rightmost m variables.
  let mut partial_suffix = Vec::with_capacity(m);
  for &v in &fixed_vars {
    partial_suffix.push(value_by_var[v].unwrap());
  }

  // 4) Collapse the rightmost m variables (same logic as old `fix_from_right`).
  let mut poly = evals;
  for i in 0..m {
    let r = partial_suffix[m - 1 - i]; // collapse last var first
    let half = 1usize << (n - 1 - i); // left half length
    for j in 0..half {
      let left = poly[j]; // bit = 0
      let right = poly[j + half]; // bit = 1
      poly[j] = left + r * (right - left);
    }
    // After this, valid region is poly[0 .. half)
  }

  DenseMLPoly::from_evaluations_slice(n - m, &poly[..(1 << (n - m))])
}

pub struct FixedBlock<'a, F> {
  pub start: usize,    // 0-based index of first variable in this block (0 -> x1)
  pub points: &'a [F], // values for x_{start+1}, ..., x_{start+values.len()}
}

pub fn fix_blocks<F: CryptoField>(dense_poly: &DenseMLPoly<F>, blocks: &[FixedBlock<'_, F>]) -> DenseMLPoly<F> {
  // Flatten blocks into (var_index, value) assignments.
  let mut assignments = Vec::new();
  for block in blocks {
    let start = block.start;
    for (offset, &val) in block.points.iter().enumerate() {
      assignments.push((start + offset, val));
    }
  }
  fix_many(dense_poly, &assignments)
}

/// Trait for multilinear polynomials that can be used in polynomial commitments
/// Now works with any field type that implements basic arithmetic operations
pub trait MLPoly<F>: std::fmt::Debug + std::any::Any + Send + Sync
where
  F: Clone + Copy + std::fmt::Debug + 'static,
{
  fn fix_variables(&self, partial_point: &[F]) -> Box<dyn MLPoly<F>>;
  fn n(&self) -> usize; // number of variables
  fn len(&self) -> usize; // number of evaluations (2^n)
  fn evaluate_at_point(&self, point: &[F]) -> F;
  fn evaluations(&self) -> Vec<F>;
  fn index(&self, index: usize) -> F;
  fn index_mut(&mut self, index: usize) -> &mut F;
  fn clone_box(&self) -> Box<dyn MLPoly<F>>;
  fn as_any(&self) -> &dyn std::any::Any;
  fn mul_by_scalar(&self, scalar: F) -> Box<dyn MLPoly<F>>;
  fn add(&self, other: &Box<dyn MLPoly<F>>) -> Box<dyn MLPoly<F>>;
}

use std::any::{Any, TypeId};
use std::sync::{OnceLock, RwLock};

/// Cached inverse constants for Lagrange interpolation
#[derive(Clone, Copy)]
pub struct CachedInverses<F: Copy> {
  pub two_inv: F,
  pub six_inv: F,
  pub neg_ln2_inv: F,
  pub neg_two_inv: F,
  pub neg_six_inv: F,
}

/// Global cache for inverse constants, keyed by field type
pub static LAGRANGE_INVERSE_CACHE: OnceLock<RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>> = OnceLock::new();

/// Get cached inverse constants for a field type, computing them if not already cached
pub fn get_cached_inverses<F: CryptoField + Send + Sync + 'static>() -> CachedInverses<F> {
  let cache = LAGRANGE_INVERSE_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
  let type_id = TypeId::of::<F>();

  // Try read first (fast path)
  {
    let read_guard = cache.read().unwrap();
    if let Some(cached) = read_guard.get(&type_id) {
      return *cached.downcast_ref::<CachedInverses<F>>().unwrap();
    }
  }

  // Compute inverses (only happens once per field type)
  let two_inv = <F as CryptoField>::from_u32(2).invert();
  let six_inv = <F as CryptoField>::from_u32(6).invert();
  let ln2 = -((2.0_f32.ln() * *SF_FLOAT).round() as i128);
  let neg_ln2_inv = F::from(ln2).invert();
  let inverses = CachedInverses {
    two_inv,
    six_inv,
    neg_ln2_inv,
    neg_two_inv: <F as CryptoField>::zero() - two_inv,
    neg_six_inv: <F as CryptoField>::zero() - six_inv,
  };

  // Insert into cache
  {
    let mut write_guard = cache.write().unwrap();
    write_guard.entry(type_id).or_insert_with(|| Box::new(inverses));
  }

  inverses
}

/// Evaluates a univariate polynomial in evaluation form at a given challenge point
/// Implementation is based on the Lagrange interpolation formula
/// Uses a global cache for inverse constants to avoid repeated field inversions
pub fn evaluate_univariate_polynomial<F: CryptoField + Send + Sync + 'static>(points: &Vec<F>, challenge: F) -> F {
  let n = points.len();

  // Special cases for small n (common in sumcheck: degree 2, 3, or 4)
  match n {
    0 => return <F as CryptoField>::zero(),
    1 => return points[0],
    2 => {
      // L_0(x) = (1 - x), L_1(x) = x
      // f(x) = points[0] + x * (points[1] - points[0])
      return points[0] + challenge * (points[1] - points[0]);
    }
    3 => {
      // L_0(x) = (x-1)(x-2)/2, L_1(x) = -x(x-2), L_2(x) = x(x-1)/2
      let cached = get_cached_inverses::<F>();
      let x = challenge;
      let x_minus_1 = x - <F as CryptoField>::one();
      let x_minus_2 = x - <F as CryptoField>::from_u32(2);

      let l0 = x_minus_1 * x_minus_2 * cached.two_inv;
      let l1 = <F as CryptoField>::zero() - x * x_minus_2;
      let l2 = x * x_minus_1 * cached.two_inv;

      return points[0] * l0 + points[1] * l1 + points[2] * l2;
    }
    4 => {
      // L_0(x) = (x-1)(x-2)(x-3)/(-6), L_1(x) = x(x-2)(x-3)/2
      // L_2(x) = x(x-1)(x-3)/(-2), L_3(x) = x(x-1)(x-2)/6
      let cached = get_cached_inverses::<F>();
      let x = challenge;
      let x_minus_1 = x - <F as CryptoField>::one();
      let x_minus_2 = x - <F as CryptoField>::from_u32(2);
      let x_minus_3 = x - <F as CryptoField>::from_u32(3);

      let l0 = x_minus_1 * x_minus_2 * x_minus_3 * cached.neg_six_inv;
      let l1 = x * x_minus_2 * x_minus_3 * cached.two_inv;
      let l2 = x * x_minus_1 * x_minus_3 * cached.neg_two_inv;
      let l3 = x * x_minus_1 * x_minus_2 * cached.six_inv;

      return points[0] * l0 + points[1] * l1 + points[2] * l2 + points[3] * l3;
    }
    _ => {}
  }

  // General case: use batch inversion (Montgomery's trick) to minimize field inversions
  // Total: only 2 field inversions instead of O(n²)

  // Step 1: Compute (challenge - i) for all i
  let diffs: Vec<F> = (0..n).map(|i| challenge - <F as CryptoField>::from_u32(i as u32)).collect();

  // Step 2: Batch inversion for diffs using Montgomery's trick
  let mut prefix_products = vec![<F as CryptoField>::one(); n];
  for i in 1..n {
    prefix_products[i] = prefix_products[i - 1] * diffs[i - 1];
  }

  let mut suffix_products = vec![<F as CryptoField>::one(); n];
  for i in (0..n - 1).rev() {
    suffix_products[i] = suffix_products[i + 1] * diffs[i + 1];
  }

  // Product of all (challenge - j)
  let all_product = prefix_products[n - 1] * diffs[n - 1];
  let all_product_inv = all_product.invert();

  // Compute inverses: inv[i] = prefix[i] * suffix[i] * all_product_inv
  let inv_diffs: Vec<F> = (0..n).map(|i| prefix_products[i] * suffix_products[i] * all_product_inv).collect();

  // Step 3: Precompute barycentric weights w_i = 1 / Π_{j≠i} (i - j)
  // For evaluation points 0, 1, ..., n-1:
  // w_i = (-1)^(n-1-i) / (i! * (n-1-i)!)
  let mut factorials = vec![<F as CryptoField>::one(); n];
  for i in 1..n {
    factorials[i] = factorials[i - 1] * <F as CryptoField>::from_u32(i as u32);
  }

  // Compute denominators for batch inversion
  let denoms: Vec<F> = (0..n).map(|i| factorials[i] * factorials[n - 1 - i]).collect();

  // Batch inversion for denominators
  let mut denom_prefix = vec![<F as CryptoField>::one(); n];
  for i in 1..n {
    denom_prefix[i] = denom_prefix[i - 1] * denoms[i - 1];
  }

  let mut denom_suffix = vec![<F as CryptoField>::one(); n];
  for i in (0..n - 1).rev() {
    denom_suffix[i] = denom_suffix[i + 1] * denoms[i + 1];
  }

  let all_denoms = denom_prefix[n - 1] * denoms[n - 1];
  let all_denoms_inv = all_denoms.invert();

  // Compute barycentric weights with proper signs
  let bary_weights: Vec<F> = (0..n)
    .map(|i| {
      let denom_inv = denom_prefix[i] * denom_suffix[i] * all_denoms_inv;
      let sign = if (n - 1 - i) % 2 == 0 {
        <F as CryptoField>::one()
      } else {
        <F as CryptoField>::zero() - <F as CryptoField>::one()
      };
      sign * denom_inv
    })
    .collect();

  // Step 4: Compute result using barycentric formula
  // f(x) = [Π_{j=0}^{n-1} (x - j)] * Σ_{i=0}^{n-1} [points[i] * w_i / (x - i)]
  let sum: F = (0..n).map(|i| points[i] * bary_weights[i] * inv_diffs[i]).fold(<F as CryptoField>::zero(), |acc, x| acc + x);

  all_product * sum
}

/// Efficient evaluation of multilinear Lagrange basis polynomials
///
/// Given a point r ∈ F^(log m), evaluates all Lagrange basis polynomials at r using only m field multiplications.
/// [VSBW13] Victor Vu, Srinath Setty, Andrew J. Blumberg, and Michael Walfish. A hybrid architecture for
/// verifiable computation. In Proceedings of the IEEE Symposium on Security and Privacy (S&P), 2013.
///
/// Also known as eq_polynomial_evals in some contexts - evaluates eq(i, r) for all i in the boolean hypercube.
/// Optimized to pre-allocate the final array size and perform in-place updates with parallelization.
pub fn evaluate_lagrange_basis<F: CryptoField>(r: &[F]) -> Vec<F> {
  let log_m = r.len();
  if log_m == 0 {
    return vec![<F as CryptoField>::one()];
  }

  let final_size = 1usize << log_m;
  // Pre-allocate the final array size upfront
  let mut a = vec![<F as CryptoField>::zero(); final_size];
  a[0] = <F as CryptoField>::one();

  for i in 0..log_m {
    let r_i = r[i];
    let current_size = 1usize << i;
    let offset = current_size;

    if current_size >= 1024 {
      // Parallel in-place update in two phases to avoid data races:
      // Phase 1: Write "high" values (read from [0..current_size], write to [current_size..2*current_size])
      // Phase 2: Update "low" values (read from both halves, write to [0..current_size])

      // Split the array into two non-overlapping halves
      let (left, right) = a[..current_size * 2].split_at_mut(current_size);

      // Phase 1: Compute high values in parallel (right[x] = r_i * left[x])
      left.par_iter().zip(right.par_iter_mut()).for_each(|(&left_val, right_val)| {
        *right_val = r_i * left_val;
      });

      // Phase 2: Update low values in parallel (left[x] = left[x] - right[x])
      left.par_iter_mut().zip(right.par_iter()).for_each(|(left_val, &right_val)| {
        *left_val = *left_val - right_val;
      });
    } else {
      // Sequential in-place update for smaller arrays (process in reverse order)
      for x in (0..current_size).rev() {
        let a_x = a[x];
        let mul_result = r_i * a_x;
        a[x + offset] = mul_result;
        a[x] = a_x - mul_result;
      }
    }
  }

  a
}

/// Computes the multilinear extension ũ(r) for vector u at point r
/// Uses exactly 2m field multiplications as per [VSBW13]
pub fn evaluate_multilinear_extension<F: CryptoField>(u: &[F], r: &[F]) -> F {
  let m = u.len();
  let log_m = r.len();

  // Verify that m = 2^(log_m)
  assert_eq!(m, 1 << log_m, "Vector length must be a power of 2 matching r dimension");

  // First m multiplications: evaluate Lagrange basis polynomials
  let lagrange_basis = evaluate_lagrange_basis(r); // uses m-1 multiplications

  // Additional m multiplications: compute ũ(r) = Σ u[i] * ẽq(i, r)
  // Parallelize the final sum when vector is large enough
  let result = if m >= 1024 {
    // Use parallel reduction for large vectors (1024+ elements)
    u.par_iter()
      .zip(lagrange_basis.par_iter())
      .map(|(&u_i, &basis_i)| u_i * basis_i)
      .reduce(|| <F as CryptoField>::zero(), |a, b| a + b)
  } else {
    // Use sequential iteration for smaller vectors
    let mut result = <F as CryptoField>::zero();
    for i in 0..m {
      result = result + u[i] * lagrange_basis[i]; // 1 multiplication per term, m total
    }
    result
  };
  // Total: (m-1) + m = 2m-1 multiplications
  // We need to account for 1 more multiplication somewhere to match Lemma 1's "2m" claim

  result
}

/// Precomputes the eq polynomial evaluations for a given partial point
/// Used in sparse polynomial fix_variables operation
/// Optimized to avoid intermediate allocations by using in-place updates with parallelization.
pub fn precompute_eq<F: CryptoField>(g: &[F]) -> Vec<F> {
  let dim = g.len();
  if dim == 0 {
    return vec![<F as CryptoField>::one()];
  }

  let mut dp = vec![<F as CryptoField>::zero(); 1 << dim];
  dp[0] = <F as CryptoField>::one() - g[0];
  dp[1] = g[0];

  for i in 1..dim {
    let num_elements = 1 << i;
    let g_i = g[i];

    if num_elements >= 1024 {
      // Parallel in-place update in two phases:
      // Phase 1: Write "high" values (read from [0..num_elements], write to [num_elements..2*num_elements])
      // Phase 2: Update "low" values

      // Split the array into two non-overlapping halves
      let (left, right) = dp[..num_elements * 2].split_at_mut(num_elements);

      // Phase 1: Compute high values in parallel (right[b] = g_i * left[b])
      left.par_iter().zip(right.par_iter_mut()).for_each(|(&left_val, right_val)| {
        *right_val = left_val * g_i;
      });

      // Phase 2: Update low values in parallel (left[b] = left[b] - right[b])
      left.par_iter_mut().zip(right.par_iter()).for_each(|(left_val, &right_val)| {
        *left_val = *left_val - right_val;
      });
    } else {
      // Sequential in-place update for smaller arrays (process in reverse order)
      for b in (0..num_elements).rev() {
        let prev = dp[b];
        let new_high = prev * g_i;
        dp[b + num_elements] = new_high;
        dp[b] = prev - new_high;
      }
    }
  }
  dp
}

/// Fix variables from the right (highest bits) in a multilinear polynomial.
/// This is useful when you need to fix the last N variables in LSB-first ordering.
/// For a polynomial with variables [v0, v1, ..., v_{n-1}] in LSB-first order,
/// this fixes [v_{n-k}, ..., v_{n-1}] where k = fixed_vars.len().
pub fn fix_variables_from_right<F: CryptoField>(poly: &DenseMLPoly<F>, fixed_vars: &[F]) -> DenseMLPoly<F> {
  assert!(fixed_vars.len() <= poly.n, "invalid size of partial point");
  if fixed_vars.is_empty() {
    return poly.clone();
  }

  let dim = fixed_vars.len();
  let remaining_vars = poly.n - dim;

  // Use eq polynomial evaluation to compute the partial evaluation
  let eq_evals = precompute_eq(fixed_vars);

  let result_evals = if poly.evaluations.len() >= (1 << 15) {
    // Use parallel iteration for large polynomials (2^15 evaluations or more)
    (0..poly.evaluations.len())
      .into_par_iter()
      .fold(
        || vec![<F as CryptoField>::zero(); 1 << remaining_vars],
        |mut local_result, idx| {
          let high_bits = idx >> remaining_vars;
          let low_bits = idx & ((1 << remaining_vars) - 1);
          if high_bits < eq_evals.len() {
            let eq_val = eq_evals[high_bits];
            local_result[low_bits] = local_result[low_bits] + eq_val * poly.evaluations[idx];
          }
          local_result
        },
      )
      .reduce(
        || vec![<F as CryptoField>::zero(); 1 << remaining_vars],
        |mut a, b| {
          for i in 0..a.len() {
            a[i] = a[i] + b[i];
          }
          a
        },
      )
  } else {
    // Use sequential iteration for smaller polynomials
    let mut result_evals = vec![<F as CryptoField>::zero(); 1 << remaining_vars];
    for idx in 0..poly.evaluations.len() {
      let high_bits = idx >> remaining_vars;
      let low_bits = idx & ((1 << remaining_vars) - 1);

      if high_bits < eq_evals.len() {
        let eq_val = eq_evals[high_bits];
        result_evals[low_bits] = result_evals[low_bits] + eq_val * poly.evaluations[idx];
      }
    }
    result_evals
  };

  DenseMLPoly::new(remaining_vars, result_evals)
}

/// Dense multilinear polynomial represented as a vector of evaluations at all points in the domain
/// Now supports both icicle's Field and arkworks' PrimeField
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "F: CanonicalSerialize + CanonicalDeserialize")]
pub struct DenseMLPoly<F> {
  pub n: usize, // number of variables
  #[serde(serialize_with = "ark_se_vec", deserialize_with = "ark_de_vec")]
  pub evaluations: Vec<F>,
}

impl<F: CryptoField + 'static> MLPoly<F> for DenseMLPoly<F> {
  fn len(&self) -> usize {
    self.evaluations.len()
  }

  fn index_mut(&mut self, index: usize) -> &mut F {
    &mut self.evaluations[index]
  }

  fn fix_variables(&self, partial_point: &[F]) -> Box<dyn MLPoly<F>> {
    Box::new(self.fix_variables(partial_point))
  }

  fn n(&self) -> usize {
    self.n
  }

  fn evaluate_at_point(&self, point: &[F]) -> F {
    assert!(point.len() == self.n, "invalid size of point");
    evaluate_multilinear_extension(&self.evaluations, &point)
  }

  fn evaluations(&self) -> Vec<F> {
    self.evaluations.clone()
  }

  fn index(&self, index: usize) -> F {
    self.evaluations[index]
  }

  fn clone_box(&self) -> Box<dyn MLPoly<F>> {
    Box::new(self.clone())
  }

  fn as_any(&self) -> &dyn std::any::Any {
    self
  }

  fn mul_by_scalar(&self, scalar: F) -> Box<dyn MLPoly<F>> {
    Box::new(self.mul_by_scalar(scalar))
  }

  fn add(&self, other: &Box<dyn MLPoly<F>>) -> Box<dyn MLPoly<F>> {
    Box::new(self.add(other.as_any().downcast_ref::<DenseMLPoly<F>>().unwrap()))
  }
}

impl<F: CryptoField> DenseMLPoly<F> {
  pub fn new(n: usize, evaluations: Vec<F>) -> Self {
    Self { n, evaluations }
  }

  pub fn len(&self) -> usize {
    self.evaluations.len()
  }

  pub fn is_empty(&self) -> bool {
    self.evaluations.is_empty()
  }

  pub fn from_evaluations(evaluations: Vec<F>) -> Self {
    let n = (evaluations.len() as f64).log2() as usize;
    assert_eq!(1 << n, evaluations.len(), "Vector length must be a power of 2");
    Self::new(n, evaluations)
  }

  pub fn from_evaluations_slice(n: usize, evaluations: &[F]) -> Self {
    Self::new(n, evaluations.to_vec())
  }

  /// Fix variables from the left side of the polynomial.
  /// Optimized to avoid intermediate allocations by using in-place updates with parallelization.
  pub fn fix_variables(&self, partial_point: &[F]) -> Self {
    assert!(partial_point.len() <= self.n, "invalid size of partial point");

    let mut poly = self.evaluations.to_vec();
    let nv = self.n;
    let dim = partial_point.len();

    // Evaluate single variable of partial point from left to right.
    // Each iteration halves the polynomial size.
    // In-place update: read from [2b, 2b+1], write to [b].
    for i in 1..dim + 1 {
      let r = partial_point[i - 1];
      let num_pairs = 1 << (nv - i);

      if num_pairs >= 1024 {
        // Parallel in-place update:
        // We read from pairs (poly[2b], poly[2b+1]) and write to poly[b].
        // Since output index b < input indices 2b and 2b+1, there's no conflict
        // between different values of b when processed in parallel.
        // Use chunks to process pairs and write results.
        let slice = &mut poly[..num_pairs * 2];
        // Process pairs: each pair at indices [2b, 2b+1] produces result at index b
        // We need to collect results first, then write back to avoid conflicts within rayon
        let results: Vec<F> = slice
          .par_chunks(2)
          .map(|pair| {
            let left = pair[0];
            let right = pair[1];
            left + r * (right - left)
          })
          .collect();

        // Write results back (safe because results.len() == num_pairs <= slice.len())
        poly[..num_pairs].copy_from_slice(&results);
      } else {
        // Sequential in-place update for smaller arrays
        for b in 0..num_pairs {
          let left = poly[b << 1];
          let right = poly[(b << 1) + 1];
          poly[b] = left + r * (right - left);
        }
      }
    }

    Self::from_evaluations_slice(nv - dim, &poly[..(1 << (nv - dim))])
  }

  pub fn mul_by_scalar(&self, scalar: F) -> Self
  where
    F: std::ops::Mul<Output = F>,
  {
    // Parallelize scalar multiplication when vector is large enough
    let new_evaluations = if self.evaluations.len() >= 1024 {
      self.evaluations.par_iter().map(|&x| x * scalar).collect()
    } else {
      self.evaluations.iter().map(|&x| x * scalar).collect()
    };
    Self::new(self.n, new_evaluations)
  }

  pub fn add(&self, other: &Self) -> Self {
    assert!(self.n == other.n, "mismatched number of variables");

    // Parallelize addition when vectors are large enough
    let new_evaluations = if self.evaluations.len() >= 1024 {
      self.evaluations.par_iter().zip(other.evaluations.par_iter()).map(|(&x, &y)| x + y).collect()
    } else {
      self.evaluations.iter().zip(other.evaluations.iter()).map(|(&x, &y)| x + y).collect()
    };

    Self::new(self.n, new_evaluations)
  }

  pub fn add_by_scalar(&self, scalar: F) -> Self
  where
    F: std::ops::Add<Output = F>,
  {
    // Parallelize scalar addition when vector is large enough
    let new_evaluations = if self.evaluations.len() >= 1024 {
      self.evaluations.par_iter().map(|&x| x + scalar).collect()
    } else {
      self.evaluations.iter().map(|&x| x + scalar).collect()
    };
    Self::new(self.n, new_evaluations)
  }

  /// Extend the polynomial to have more variables by padding with zeros.
  /// This increases the number of variables from `self.n` to `num_variables`.
  /// Takes self by value to avoid cloning when no extension is needed.
  pub fn extend_number_of_variables(mut self, num_variables: usize) -> Self {
    assert!(self.n <= num_variables, "Cannot extend to fewer variables");

    // Fast path: if already at the right size, return self without any cloning
    if self.n == num_variables {
      return self;
    }

    // Extend the evaluations vector in place
    let new_len = 1 << num_variables;
    self.evaluations.extend(vec![<F as CryptoField>::zero(); new_len - self.evaluations.len()]);
    self.n = num_variables;
    self
  }

  /// Get a slice of evaluations for a specific boolean input index.
  /// This extracts `n` consecutive evaluations starting at `n * index`.
  /// Returns a slice to avoid copying data.
  pub fn get_partial_evaluation_for_boolean_input(&self, index: usize, n: usize) -> &[F] {
    let start = n * index;
    let end = start + n;
    &self.evaluations[start..end]
  }

  /// Perform partial evaluation by fixing variables from the left.
  /// This is equivalent to `fix_variables` but returns a new polynomial.
  pub fn partial_evaluation(&self, fixed_vars: &[F]) -> Self {
    self.fix_variables(fixed_vars)
  }
}

pub fn add_broadcast<F: CryptoField>(a: &DenseMLPoly<F>, b: &DenseMLPoly<F>) -> DenseMLPoly<F> {
  let n_out = a.n.max(b.n);
  let out_len = 1usize << n_out;

  // We'll map each output index i (n_out bits) to indices in a and b
  // by dropping the highest (n_out - n_small) bits, i.e. keeping the lowest n_small bits.
  //
  // This assumes variable order is:
  //   x1 is the "lowest bit" (fastest-changing),
  //   x_n is the "highest bit" (slowest-changing).
  // TODO: Check if this is correct.
  let mask_a = if a.n == 0 { 1 } else { (1usize << a.n) - 1 };
  let mask_b = if b.n == 0 { 1 } else { (1usize << b.n) - 1 };

  // Parallelize broadcasting when output is large enough
  let out = if out_len >= 1024 {
    (0..out_len)
      .into_par_iter()
      .map(|i| {
        let ia = i & mask_a;
        let ib = i & mask_b;
        a.evaluations[ia] + b.evaluations[ib]
      })
      .collect()
  } else {
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
      let ia = i & mask_a;
      let ib = i & mask_b;
      out.push(a.evaluations[ia] + b.evaluations[ib]);
    }
    out
  };

  DenseMLPoly { n: n_out, evaluations: out }
}

impl<F: CryptoField> std::ops::Index<usize> for DenseMLPoly<F> {
  type Output = F;

  fn index(&self, index: usize) -> &Self::Output {
    &self.evaluations[index]
  }
}

impl<F: CryptoField> std::ops::IndexMut<usize> for DenseMLPoly<F> {
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    &mut self.evaluations[index]
  }
}

/// Selection polynomial representation
/// f(k1, k2, ..., kn, t1, t2, ..., tm) = f_1(k1, k2, ..., k_(n/2), t1, t2, ..., tm) * f_2(k_(n/2+1), ..., kn, t1, t2, ..., tm)
/// n = logK, m = logT
#[derive(Clone, Debug)]
pub struct SelectionPolynomial<F: CryptoField> {
  pub input_num_vars: usize,          // Number of variables for input (logT variables)
  pub table_num_vars: usize,          // Number of variables for table (logK variables)
  pub selection: Vec<(usize, usize)>, // (input_index, table_index) pairs
  _phantom: PhantomData<F>,
}

impl<F: CryptoField> SelectionPolynomial<F> {
  pub fn new(input_num_vars: usize, table_num_vars: usize, selection: Vec<(usize, usize)>) -> Self {
    Self {
      input_num_vars,
      table_num_vars,
      selection,
      _phantom: PhantomData,
    }
  }

  pub fn to_sparse(self) -> SparseMLPoly<F> {
    let n = self.input_num_vars + self.table_num_vars;
    let mut evaluations = HashMap::new();
    for (input_index, table_index) in self.selection.iter() {
      let index = input_index + table_index * (1 << self.input_num_vars);
      evaluations.insert(index, <F as CryptoField>::one());
    }
    let mut sparse = SparseMLPoly::new(n, evaluations, VecDeque::new(), self);
    sparse.build_sorted_indices();
    sparse
  }
}

/// Sparse multilinear polynomial represented as a map from point index to evaluation
/// and a queue of point indices for efficient iteration
#[derive(Clone, Debug)]
pub struct SparseMLPoly<F: CryptoField> {
  pub n: usize, // number of variables
  pub evaluations: HashMap<usize, F>,
  pub indices: VecDeque<usize>,
  pub selection: SelectionPolynomial<F>,
}

impl<F: CryptoField + 'static> MLPoly<F> for SparseMLPoly<F> {
  fn fix_variables(&self, partial_point: &[F]) -> Box<dyn MLPoly<F>> {
    Box::new(self.fix_variables(partial_point))
  }

  fn n(&self) -> usize {
    self.n
  }

  fn evaluate_at_point(&self, point: &[F]) -> F {
    assert!(point.len() == self.n, "invalid size of point");

    // Sparse evaluation: directly compute sum of eq(i, point) * poly[i] for non-zero entries
    // This is O(k) instead of O(k log k) from fix_variables (which sorts indices)
    if self.evaluations.is_empty() {
      return <F as CryptoField>::zero();
    }

    let eq_evals = precompute_eq(point);
    let mut result = <F as CryptoField>::zero();

    for (idx, &val) in self.evaluations.iter() {
      if *idx < eq_evals.len() {
        result = result + val * eq_evals[*idx];
      }
    }

    result
  }

  fn evaluations(&self) -> Vec<F> {
    let mut result = vec![<F as CryptoField>::zero(); 1 << self.n];
    for (idx, val) in self.evaluations.iter() {
      result[*idx] = *val;
    }
    result
  }

  fn len(&self) -> usize {
    1 << self.n
  }

  // index only works for indices that are less than 2^64 (usize)
  fn index(&self, index: usize) -> F {
    *self.evaluations.get(&index).unwrap_or(&<F as CryptoField>::zero())
  }

  fn index_mut(&mut self, index: usize) -> &mut F {
    self.evaluations.entry(index).or_insert(<F as CryptoField>::zero())
  }

  fn clone_box(&self) -> Box<dyn MLPoly<F>> {
    Box::new(self.clone())
  }

  fn as_any(&self) -> &dyn std::any::Any {
    self
  }

  fn mul_by_scalar(&self, scalar: F) -> Box<dyn MLPoly<F>> {
    Box::new(self.mul_by_scalar(scalar))
  }

  fn add(&self, other: &Box<dyn MLPoly<F>>) -> Box<dyn MLPoly<F>> {
    Box::new(self.add(other.as_any().downcast_ref::<SparseMLPoly<F>>().unwrap()))
  }
}

impl<F: CryptoField> SparseMLPoly<F> {
  pub fn new(n: usize, evaluations: HashMap<usize, F>, indices: VecDeque<usize>, selection: SelectionPolynomial<F>) -> Self {
    Self {
      n,
      evaluations,
      indices,
      selection,
    }
  }

  /// Build sorted indices from evaluations - call this before operations that need sorted indices (e.g., SD sumcheck)
  pub fn build_sorted_indices(&mut self) {
    let mut indices: Vec<usize> = self.evaluations.keys().cloned().collect();
    indices.sort_unstable_by(|a, b| a.cmp(b));
    self.indices = indices.into();
  }

  pub fn len(&self) -> usize {
    self.evaluations.len()
  }

  pub fn is_empty(&self) -> bool {
    self.evaluations.is_empty()
  }

  pub fn fix_variables(&self, partial_point: &[F]) -> Self {
    let dim = partial_point.len();
    assert!(dim <= self.n, "invalid partial point dimension");

    let mut window = (self.evaluations.len() as f64).log2() as usize;
    if window == 0 {
      window = 1;
    }

    let mut point = partial_point;
    let mut last = self.evaluations.clone();

    // batch evaluation
    while !point.is_empty() {
      let focus_length = if point.len() > window { window } else { point.len() };
      let focus = &point[..focus_length];
      point = &point[focus_length..];
      let pre = precompute_eq(focus);
      let dim = focus.len();
      let mut result: HashMap<usize, F> = HashMap::new();

      // Parallelize the sparse entry processing when there are enough entries
      if last.len() >= 128 {
        // Use parallel processing for larger sparse polynomials
        let partial_results: Vec<HashMap<usize, F>> = last
          .par_iter()
          .map(|src_entry| {
            let old_idx_bytes = src_entry.0;
            let low_bits = *old_idx_bytes & ((1 << dim) - 1);
            let gz = pre[low_bits];
            let new_idx_bytes = old_idx_bytes >> dim;
            let mut local_result = HashMap::new();
            local_result.insert(new_idx_bytes, gz * *src_entry.1);
            local_result
          })
          .collect();

        // Combine partial results
        for partial in partial_results {
          for (key, value) in partial {
            result.entry(key).and_modify(|e| *e = *e + value).or_insert(value);
          }
        }
      } else {
        // Use sequential processing for smaller sparse polynomials
        for src_entry in last.iter() {
          let old_idx_bytes = src_entry.0;
          let low_bits = *old_idx_bytes & ((1 << dim) - 1);
          let gz = pre[low_bits];
          let new_idx_bytes = old_idx_bytes >> dim;
          let dst_entry = result.entry(new_idx_bytes).or_insert(<F as CryptoField>::zero());
          *dst_entry = *dst_entry + gz * *src_entry.1;
        }
      }
      last = result;
    }

    // Return result without sorted indices - caller can call build_sorted_indices() if needed
    Self::new(self.n - dim, last, VecDeque::new(), self.selection.clone())
  }

  pub fn mul_by_scalar(&self, scalar: F) -> Self {
    // If scalar is zero, return empty sparse polynomial
    if scalar == <F as CryptoField>::zero() {
      return Self::new(self.n, HashMap::new(), VecDeque::new(), self.selection.clone());
    }

    // Parallelize scalar multiplication for large sparse polynomials
    let new_evaluations = if self.evaluations.len() >= 128 {
      // Use parallel processing for larger sparse polynomials
      let partial_results: Vec<(usize, F)> =
        self.evaluations.par_iter().map(|(k, v)| (*k, *v * scalar)).filter(|(_, result)| *result != <F as CryptoField>::zero()).collect();

      partial_results.into_iter().collect()
    } else {
      // Use sequential processing for smaller sparse polynomials
      let mut new_evaluations: HashMap<usize, F> = HashMap::new();
      for (k, v) in self.evaluations.iter() {
        let result = *v * scalar;
        if result != <F as CryptoField>::zero() {
          new_evaluations.insert(*k, result);
        }
      }
      new_evaluations
    };

    Self::new(self.n, new_evaluations, VecDeque::new(), self.selection.clone())
  }

  pub fn add(&self, other: &Self) -> Self {
    assert!(self.n == other.n, "mismatched number of variables");

    // Parallelize addition for large sparse polynomials
    let new_evaluations = if self.evaluations.len() + other.evaluations.len() >= 256 {
      // Use parallel processing for larger sparse polynomials
      let mut combined_keys: std::collections::HashSet<usize> = std::collections::HashSet::new();
      combined_keys.extend(self.evaluations.keys());
      combined_keys.extend(other.evaluations.keys());

      let partial_results: Vec<(usize, F)> = combined_keys
        .into_par_iter()
        .map(|k| {
          let self_val = self.evaluations.get(&k).copied().unwrap_or(<F as CryptoField>::zero());
          let other_val = other.evaluations.get(&k).copied().unwrap_or(<F as CryptoField>::zero());
          let sum = self_val + other_val;
          (k, sum)
        })
        .filter(|(_, v)| *v != <F as CryptoField>::zero())
        .collect();

      partial_results.into_iter().collect()
    } else {
      // Use sequential processing for smaller sparse polynomials
      let mut new_evaluations = self.evaluations.clone();

      // Add entries from other polynomial during iteration
      for (k, v) in other.evaluations.iter() {
        new_evaluations.entry(*k).and_modify(|e| *e = *e + *v).or_insert(*v);
      }

      // Filter out zeros that may result from cancellation (e.g., a + (-a) = 0)
      new_evaluations.retain(|_, v| *v != <F as CryptoField>::zero());

      new_evaluations
    };

    Self::new(self.n, new_evaluations, VecDeque::new(), self.selection.clone())
  }

  /// Materialize the sparse multilinear polynomial into a dense table of size 2^n.
  ///
  /// Indices are stored in little-endian Vec<u8> form; we convert them with
  /// `le_vec_to_usize`. This assumes `n` is small enough that 2^n fits in `usize`.
  pub fn to_dense(&self) -> DenseMLPoly<F> {
    // Sanity check: avoid shifting past usize width
    assert!(
      self.n < usize::BITS as usize,
      "too many variables ({}) to materialize into a dense table on this platform",
      self.n
    );

    let size = 1usize << self.n;
    let mut evaluations = vec![<F as CryptoField>::zero(); size];

    // Fill in non-zero entries from the sparse representation.
    // Missing entries stay as zero.
    for (idx, &val) in &self.evaluations {
      debug_assert!(*idx < size, "sparse index {} exceeds dense table size 2^{} = {}", idx, self.n, size);
      evaluations[*idx] = val;
    }

    DenseMLPoly::new(self.n, evaluations)
  }

  pub fn split_into_blocks(&self, block_size: usize) -> Vec<Self> {
    assert!(block_size > 0, "block_size must be > 0");

    let selection = &self.selection;

    // If there are no table vars, there is nothing to split.
    if selection.table_num_vars == 0 {
      return vec![self.clone()];
    }

    // ceil(table_num_vars / block_size)
    let num_blocks = (selection.table_num_vars + block_size - 1) / block_size;
    debug_assert!(num_blocks > 0);

    let mut blocks = Vec::with_capacity(num_blocks);

    // Constant mask range for a full block of size `block_size`.
    let full_block_mod = 1usize << block_size;

    for i in 0..num_blocks {
      let offset = i * block_size; // bit offset into table_index
      let start = 1usize << offset;

      // Number of table vars in this block (last block may be smaller)
      let remaining = selection.table_num_vars - offset;
      let this_block_vars = remaining.min(block_size);

      // Modulus matching the number of bits we actually keep in this block.
      let this_block_mod = 1usize << this_block_vars;

      let block_selection = selection
        .selection
        .iter()
        .map(|(input_index, table_index)| {
          // Extract bits [offset .. offset+this_block_vars)
          // (table_index >> offset) & ((1<<this_block_vars)-1)
          let v = (table_index / start) % full_block_mod;
          (*input_index, v % this_block_mod)
        })
        .collect::<Vec<(usize, usize)>>();

      let block_selection_polynomial = SelectionPolynomial::new(selection.input_num_vars, block_size, block_selection);
      let block_selection_polynomial_sparse = block_selection_polynomial.to_sparse();
      blocks.push(block_selection_polynomial_sparse);
    }

    blocks
  }
}

pub fn range_dense<F: CryptoField>(num_vars: usize) -> DenseMLPoly<F> {
  let vec_len = 1 << num_vars;
  let mut evaluations = Vec::with_capacity(vec_len);
  for i in 0..vec_len {
    evaluations.push(F::from(i as u32));
  }
  DenseMLPoly::new(num_vars, evaluations)
}

pub fn two_pow_dense<F: CryptoField>(num_vars: usize) -> DenseMLPoly<F> {
  assert!(num_vars == 4, "two_pow_dense only supports 4 variables");
  let vec_len = 1 << num_vars;
  let mut evaluations = Vec::with_capacity(vec_len);
  for i in 0..vec_len {
    evaluations.push(F::from((1 << (15 - i)) as u32));
  }
  DenseMLPoly::new(num_vars, evaluations)
}

pub fn fix_variables_zkgpt<F: CryptoField>(
  num_vars: usize,
  t_idx: &[i128], // n x m, entries in [0, 2^Q]
  r: &[F],        // len = log2(n)
  q_bits: usize,
) -> DenseMLPoly<F> {
  let n = 1usize << r.len();
  let logm = num_vars - r.len();
  let m = 1usize << logm;

  let logn = r.len();
  let left_bits = logn / 2;
  let right_bits = logn - left_bits;

  let y = &r[..right_bits];
  let x = &r[right_bits..];

  let eqy = precompute_eq::<F>(y);
  let eqx = precompute_eq::<F>(x);

  let max_t: usize = 1usize << q_bits;
  let fm_width = max_t + 1;
  let right_size = 1usize << right_bits;

  // 1) Flattened FM: fm[br * fm_width + t]
  let fm: Vec<F> = (0..right_size)
    .into_par_iter()
    .flat_map_iter(|br| {
      let mut row = vec![F::ZERO; fm_width];
      let step = eqy[br];
      for t in 1..=max_t {
        row[t] = row[t - 1] + step; // additions only
      }
      row
    })
    .collect();

  let right_mask = right_size - 1;

  // 2) Process columns in chunks (fewer Rayon tasks, better locality)
  let chunk = 64usize; // tune: 32/64/128 often good
  let mut out = vec![F::ZERO; m];

  out.par_chunks_mut(chunk).enumerate().for_each(|(chunk_id, out_chunk)| {
    let c0 = chunk_id * chunk;
    let c1 = (c0 + out_chunk.len()).min(m);

    // thread-local AC buffer reused across columns in this chunk
    let mut ac = vec![F::ZERO; 1usize << left_bits];

    for (j, c) in (c0..c1).enumerate() {
      // reset ac
      for v in ac.iter_mut() {
        *v = F::ZERO;
      }

      // IMPORTANT: this assumes your layout is b + c*n
      let col_base = c * n;

      for b in 0..n {
        let br = b & right_mask;
        let bl = b >> right_bits;

        let t = t_idx[col_base + b];
        if t > 0 {
          // ac[bl] += fm[br][t]
          ac[bl] += F::from(fm[br * fm_width + t as usize]);
        } else {
          ac[bl] -= F::from(fm[br * fm_width + (-t) as usize]);
        }
      }

      // outer dot with eqx
      let mut acc = F::ZERO;
      for (l, &ac_l) in ac.iter().enumerate() {
        acc += eqx[l] * ac_l; // only 2^{left_bits} multiplications per column
      }
      out_chunk[j] = acc;
    }
  });

  DenseMLPoly::new(logm, out)
}
