use ark_ff::fields::PrimeField;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::RwLock;

use crate::util::poly::CryptoField;
use crate::SIGN_BIT;
use ark_ff::BigInteger;

// from kzh_fold
pub trait Math {
  fn log_2(self) -> usize;
}

impl Math for usize {
  fn log_2(self) -> usize {
    assert_ne!(self, 0);

    if self.is_power_of_two() {
      (1usize.leading_zeros() - self.leading_zeros()) as usize
    } else {
      (0usize.leading_zeros() - self.leading_zeros()) as usize
    }
  }
}

// Global cache for commonly used field element conversions
static FIELD_CONVERSION_CACHE: LazyLock<RwLock<HashMap<u32, Vec<u8>>>> = LazyLock::new(|| RwLock::new(HashMap::new()));

/// Efficiently convert small integers to field elements with caching
pub fn cached_field_from_u32<F: CryptoField>(value: u32) -> F {
  // Fast path for very small values using byte conversion
  if value < 256 {
    let bytes = value.to_le_bytes();
    return <F as CryptoField>::from_bytes_le(&bytes);
  }

  // Check cache for larger values
  {
    let cache = FIELD_CONVERSION_CACHE.read().unwrap();
    if let Some(bytes) = cache.get(&value) {
      return <F as CryptoField>::from_bytes_le(bytes);
    }
  }

  // Compute and cache
  let bytes = value.to_le_bytes();
  let field_val = <F as CryptoField>::from_bytes_le(&bytes);

  {
    let mut cache = FIELD_CONVERSION_CACHE.write().unwrap();
    cache.insert(value, bytes.to_vec());
  }

  field_val
}

// Cached power-of-2 values to avoid repeated allocations
static POW_2_CACHE: LazyLock<std::sync::RwLock<std::collections::HashMap<usize, Vec<u8>>>> =
  LazyLock::new(|| std::sync::RwLock::new(std::collections::HashMap::new()));

pub fn pow_2<F: CryptoField>(i: usize) -> F {
  // Fast path for small powers using repeated doubling
  if i < 32 {
    let mut result = <F as CryptoField>::one();
    for _ in 0..i {
      result = result + result; // Same as result * 2
    }
    return result;
  }

  // Check cache first
  {
    let cache = POW_2_CACHE.read().unwrap();
    if let Some(bytes) = cache.get(&i) {
      return <F as CryptoField>::from_bytes_le(bytes);
    }
  }

  // Compute and cache for large powers
  let mut bits = vec![0; i + 1];
  bits[i] = 1;
  let bytes = bytes_le_from_bits_lsb_pad(&bits);
  let result = <F as CryptoField>::from_bytes_le(&bytes);

  // Cache the result
  {
    let mut cache = POW_2_CACHE.write().unwrap();
    cache.insert(i, bytes);
  }

  result
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FToI128Error {
  OutOfRange,
}

/// Signed decode of a prime-field element into i128 using the canonical symmetric representative:
///   t = x in [0, p-1]
///   if t <= (p-1)/2:  return  t
///   else:             return  t - p  (negative)
pub fn f_to_i128<F: PrimeField>(x: F) -> Result<i128, FToI128Error> {
  let t = x.into_bigint();

  // (p - 1) / 2 as a BigInt
  let half = F::MODULUS_MINUS_ONE_DIV_TWO;
  let modulus = F::MODULUS;

  if t <= half {
    // positive
    let u = bigint_to_u128(&t).ok_or(FToI128Error::OutOfRange)?;
    if u <= i128::MAX as u128 {
      Ok(u as i128)
    } else {
      Err(FToI128Error::OutOfRange)
    }
  } else {
    // negative: t - p = -(p - t)
    let mag = bigint_sub(&modulus, &t); // mag = p - t
    let u = bigint_to_u128(&mag).ok_or(FToI128Error::OutOfRange)?;

    // allow magnitude up to 2^127 (i128::MIN is -2^127)
    let max_mag: u128 = 1u128 << 127;
    if u < max_mag {
      Ok(-(u as i128))
    } else if u == max_mag {
      Ok(i128::MIN)
    } else {
      Err(FToI128Error::OutOfRange)
    }
  }
}

/// If you want the old signature, choose your overflow behavior.
/// Here: panic if the value doesn't fit in i128.
pub fn f_to_int<F: PrimeField>(x: F) -> i128 {
  f_to_i128::<F>(x).expect("field element does not fit in i128")
}

/// Convert BigInteger (little-endian u64 limbs) into u128 iff it fits.
fn bigint_to_u128<B: BigInteger>(b: &B) -> Option<u128> {
  let limbs = b.as_ref(); // &[u64]
  if limbs.is_empty() {
    return Some(0);
  }
  if limbs.len() >= 3 && limbs[2..].iter().any(|&x| x != 0) {
    return None; // needs more than 128 bits
  }
  let lo = limbs[0] as u128;
  let hi = if limbs.len() >= 2 { limbs[1] as u128 } else { 0 };
  Some(lo | (hi << 64))
}

/// Compute a - b for BigInteger values, assuming a >= b.
fn bigint_sub<B: BigInteger>(a: &B, b: &B) -> B {
  let mut out = *a;
  let out_limbs = out.as_mut(); // &mut [u64]
  let b_limbs = b.as_ref(); // &[u64]

  let mut borrow: u64 = 0;
  for i in 0..out_limbs.len() {
    let ai = out_limbs[i];
    let bi = if i < b_limbs.len() { b_limbs[i] } else { 0 };

    // subtract with borrow
    let (r1, under1) = ai.overflowing_sub(bi);
    let (r2, under2) = r1.overflowing_sub(borrow);
    out_limbs[i] = r2;

    borrow = (under1 as u64) | (under2 as u64);
  }

  debug_assert!(borrow == 0, "bigint_sub underflow: a < b");
  out
}

/// Batch append field elements to transcript for better performance
pub fn batch_append_to_transcript<F: CryptoField>(transcript: &mut merlin::Transcript, label: &'static [u8], values: &[F]) {
  // Serialize all values into a single byte buffer to reduce transcript calls
  let mut batch_bytes = Vec::with_capacity(values.len() * 32); // Assume ~32 bytes per field element

  for value in values {
    batch_bytes.extend_from_slice(&<F as CryptoField>::to_bytes_le(value));
  }

  transcript.append_message(label, &batch_bytes);
}

pub fn calc_pow<F: CryptoField>(alpha: F, n: usize) -> Vec<F> {
  if n == 0 {
    return Vec::new();
  }

  // Pre-allocate with exact capacity
  let mut pow: Vec<F> = Vec::with_capacity(n);
  pow.push(<F as CryptoField>::one());

  // Use previous value instead of recalculating from pow[0]
  let mut current = <F as CryptoField>::one();
  for _ in 1..n {
    current = current * alpha;
    pow.push(current);
  }
  pow
}

/// Numeric compare of little-endian byte slices (variable length).
/// Ignores MSB-side zeros (i.e., trailing zeros in LE).
pub fn cmp_numeric_le(a: &[u8], b: &[u8]) -> Ordering {
  // effective lengths (trim MSB zeros)
  let a_end = a.iter().rposition(|&x| x != 0).map_or(0, |i| i + 1);
  let b_end = b.iter().rposition(|&x| x != 0).map_or(0, |i| i + 1);

  // different effective byte-length => larger length is larger number
  if a_end != b_end {
    return a_end.cmp(&b_end);
  }

  // same effective length: compare MSB→LSB by reversing the slices
  a[..a_end].iter().rev().cmp(b[..b_end].iter().rev())
}

/// Multiply a little-endian byte vector by a usize scalar.
/// Returns a new little-endian byte vector. Trims MSB zeros.
/// (Intermediates use u128 to avoid overflow.)
pub fn mul_le_by_usize(mut bytes_le: Vec<u8>, k: usize) -> Vec<u8> {
  // Normalize trivial cases
  if k == 0 || bytes_le.is_empty() {
    return vec![0];
  }
  if k == 1 {
    // Trim MSB zeros
    while matches!(bytes_le.last(), Some(0)) {
      bytes_le.pop();
    }
    return bytes_le;
  }

  let kk = k as u128;
  let mut carry: u128 = 0;
  let mut out = Vec::with_capacity(bytes_le.len() + 2); // +2 is a small guess; will grow if needed
  for &b in &bytes_le {
    let t = (b as u128) * kk + carry;
    out.push((t & 0xFF) as u8);
    carry = t >> 8;
  }
  while carry != 0 {
    out.push((carry & 0xFF) as u8);
    carry >>= 8;
  }
  // Trim MSB zeros (last bytes in LE)
  while matches!(out.last(), Some(0)) {
    out.pop();
  }
  out
}

/// Add scalar k to a little-endian byte vector.
/// Returns minimal-length LE bytes (no trailing MSB zeros).
pub fn add_le_by_usize(mut bytes_le: Vec<u8>, k: usize) -> Vec<u8> {
  if k == 0 {
    while matches!(bytes_le.last(), Some(0)) {
      bytes_le.pop();
    }
    return bytes_le;
  }
  if bytes_le.is_empty() {
    // encode k directly
    let mut kk = k as u128;
    while kk != 0 {
      bytes_le.push((kk & 0xFF) as u8);
      kk >>= 8;
    }
    return bytes_le;
  }

  let mut carry = k as u128;
  for b in &mut bytes_le {
    let t = (*b as u128) + carry;
    *b = (t & 0xFF) as u8;
    carry = t >> 8;
    if carry == 0 {
      // no more carry; remaining higher bytes unchanged
      break;
    }
  }
  while carry != 0 {
    bytes_le.push((carry & 0xFF) as u8);
    carry >>= 8;
  }

  // trim any MSB zeros
  while matches!(bytes_le.last(), Some(0)) {
    bytes_le.pop();
  }
  bytes_le
}

/// Pads a vector with zeros to the given length
pub fn pad_zeros<F: CryptoField>(x: &[F], n: usize) -> Vec<F> {
  let mut padded = x.to_vec();
  padded.extend(vec![<F as CryptoField>::zero(); n - x.len()]);
  padded
}

/// Converts a Field element to a vector of bits in least significant order
pub fn to_bits_lsb<F: CryptoField>(x: &F) -> Vec<u8> {
  let bytes = <F as CryptoField>::to_bytes_le(x);
  let mut bits = Vec::with_capacity(bytes.len() * 8);
  for b in bytes {
    for i in 0..8 {
      bits.push((b >> i) & 1);
    }
  }
  bits
}

/// Converts a Field element to a vector of bits compatible with VirtualBitDecomp in least significant order
pub fn to_bits_lsb_virtual_bit_decomp<F: CryptoField>(x: &F, n: usize) -> Vec<u8> {
  let original_bits = to_bits_lsb(x);
  let negaive_bits = to_bits_lsb(&(<F as CryptoField>::zero() - *x));
  if original_bits[SIGN_BIT] == 1 {
    let mut bits = negaive_bits[..n - 1].to_vec();
    bits.push(1);
    bits
  } else {
    let mut bits = original_bits[..n - 1].to_vec();
    bits.push(0);
    bits
  }
}

/// Scales back a Field element by a factor of 2^sf_log
pub fn scale_back<F: CryptoField>(x: &F, bit_length: usize, sf_log: usize) -> F {
  let bits = to_bits_lsb_virtual_bit_decomp(x, bit_length);
  let sum = bits[sf_log..bits.len() - 1]
    .iter()
    .enumerate()
    .fold(<F as CryptoField>::zero(), |acc, (i, &x)| acc + F::from(x as u32) * pow_2::<F>(i));
  let sign = bits[bits.len() - 1];
  if sign == 1 {
    if sf_log == 0 {
      <F as CryptoField>::zero() - sum
    } else {
      <F as CryptoField>::zero() - sum - F::from(bits[sf_log - 1] as u32)
    }
  } else {
    if sf_log == 0 {
      sum
    } else {
      sum + F::from(bits[sf_log - 1] as u32)
    }
  }
}

/// Inverse of `to_bits_lsb`: packs LSB-first bits into LE bytes.
/// `bits.len()` must be a multiple of 8.
pub fn bytes_le_from_bits_lsb(bits: &[u8]) -> Vec<u8> {
  assert!(bits.len() % 8 == 0, "bits length must be a multiple of 8");
  let mut out = Vec::with_capacity(bits.len() / 8);
  for chunk in bits.chunks(8) {
    let mut byte = 0u8;
    for (i, &b) in chunk.iter().enumerate() {
      byte |= (b & 1) << i; // bit i is at position i (LSB-first)
    }
    out.push(byte);
  }
  out
}

/// Packs LSB-first bits to LE bytes, padding the last byte with zeros if needed.
pub fn bytes_le_from_bits_lsb_pad(bits: &[u8]) -> Vec<u8> {
  let mut out = Vec::with_capacity((bits.len() + 7) / 8);
  let mut cur = 0u8;
  for (i, &b) in bits.iter().enumerate() {
    cur |= (b & 1) << (i & 7);
    if (i & 7) == 7 {
      out.push(cur);
      cur = 0;
    }
  }
  if bits.len() & 7 != 0 {
    out.push(cur);
  }
  out
}

/// Returns the Field elements representing the bits of a given number in least significant order
pub fn bits_lsb_first<F: CryptoField>(mut x: usize) -> Vec<F> {
  let n_bits = n_bits(x);
  let mut bits = Vec::with_capacity(n_bits);
  for _ in 0..n_bits {
    bits.push(F::from((x & 1) as u32)); // take least significant bit
    x >>= 1; // shift right
  }
  bits
}

/// Returns the Field elements representing the bits of a given number represented in bytes in least significant order
pub fn bytes_lsb_first<F: CryptoField>(x: &Vec<u8>) -> Vec<F> {
  let mut bits = Vec::with_capacity(x.len() * 8);
  for b in x {
    for i in 0..8 {
      bits.push(F::from(((b >> i) & 1) as u32));
    }
  }
  // prune the trailing zeros
  while bits.len() > 1 && bits[bits.len() - 1] == <F as CryptoField>::zero() {
    bits.pop();
  }
  bits
}

/// Returns the number of bits required to represent a number
pub fn n_bits(x: usize) -> usize {
  if x == 0 {
    1
  } else {
    x.ilog2() as usize + 1
  }
}

/// Returns log2 of x rounded up to the nearest integer
pub fn log2_ceil(x: usize) -> u32 {
  if x <= 1 {
    0
  } else {
    usize::BITS - (x - 1).leading_zeros()
  }
}

/// Returns the next power of 2 greater than or equal to n
pub fn next_pow(n: u32) -> u32 {
  if n == 0 {
    return 1;
  }
  let mut v = n;
  v -= 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v += 1;
  v
}

/// Returns the number of variables needed for multilinear polynomial to represent the tensor with the given shape
pub fn get_n(shape: &[usize]) -> usize {
  shape.iter().map(|x| log2_ceil(*x) as usize).sum()
}
