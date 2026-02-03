/*
 * Serialization utilities for converting between serde and ark_serialize.
 * And other file I/O utilities.
 */
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter, Write};

// For serialization, ArrayD uses serde while G1Affine uses ark_serialize.
// In order to bridge between the two, the following code snippet is used:
// https://github.com/arkworks-rs/algebra/issues/178#issuecomment-1413219278
pub fn ark_se<S, A: CanonicalSerialize>(a: &A, s: S) -> Result<S::Ok, S::Error>
where
  S: serde::Serializer,
{
  let mut bytes = vec![];
  a.serialize_compressed(&mut bytes).map_err(serde::ser::Error::custom)?;
  s.serialize_bytes(&bytes)
}

pub fn ark_de<'de, D, A: CanonicalDeserialize>(data: D) -> Result<A, D::Error>
where
  D: serde::de::Deserializer<'de>,
{
  let s: Vec<u8> = serde::de::Deserialize::deserialize(data)?;
  let a = A::deserialize_compressed_unchecked(s.as_slice());
  a.map_err(serde::de::Error::custom)
}

pub fn measure_file_size(file_path: &str) -> u64 {
  let file = File::open(file_path).unwrap();
  let metadata = file.metadata().unwrap();
  let file_size_bytes = metadata.len();
  println!("{} size: {}", file_path, format_file_size(file_size_bytes));
  file_size_bytes
}

pub fn format_file_size(bytes: u64) -> String {
  const KB: f64 = 1024.0;
  const MB: f64 = KB * 1024.0;
  const GB: f64 = MB * 1024.0;

  if bytes as f64 >= GB {
    format!("{:.2} GB", bytes as f64 / GB)
  } else if bytes as f64 >= MB {
    format!("{:.2} MB", bytes as f64 / MB)
  } else if bytes as f64 >= KB {
    format!("{:.2} KB", bytes as f64 / KB)
  } else {
    format!("{} bytes", bytes)
  }
}

pub fn hash_str(s: &str) -> String {
  let mut hasher = DefaultHasher::new();
  s.hash(&mut hasher);
  let hash_value = hasher.finish();
  hash_value.to_string()
}

pub fn file_exists(path: &str) -> bool {
  fs::metadata(path).is_ok()
}

/// Serialize a vector of arkworks types to serde
pub fn ark_se_vec<S, A: CanonicalSerialize>(a: &Vec<A>, s: S) -> Result<S::Ok, S::Error>
where
  S: serde::Serializer,
{
  use serde::ser::SerializeSeq;
  let mut seq = s.serialize_seq(Some(a.len()))?;
  for elem in a {
    let mut bytes = vec![];
    elem.serialize_compressed(&mut bytes).map_err(serde::ser::Error::custom)?;
    seq.serialize_element(&bytes)?;
  }
  seq.end()
}

/// Deserialize a vector of arkworks types from serde
pub fn ark_de_vec<'de, D, A: CanonicalDeserialize>(data: D) -> Result<Vec<A>, D::Error>
where
  D: serde::de::Deserializer<'de>,
{
  let v: Vec<Vec<u8>> = serde::de::Deserialize::deserialize(data)?;
  v.into_iter().map(|bytes| A::deserialize_compressed_unchecked(bytes.as_slice()).map_err(serde::de::Error::custom)).collect()
}

/// Save a serializable proof to a file using bincode
pub fn serialize_proof<T: Serialize>(proof: &T, file_path: &str) -> std::io::Result<()> {
  let file = File::create(file_path)?;
  let mut writer = BufWriter::new(file);
  let encoded = bincode::serialize(proof).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
  writer.write_all(&encoded)?;
  writer.flush()?;
  Ok(())
}

/// Load a serializable proof from a file using bincode
pub fn deserialize_proof<T: for<'de> Deserialize<'de>>(file_path: &str) -> std::io::Result<T> {
  let file = File::open(file_path)?;
  let reader = BufReader::new(file);
  let decoded = bincode::deserialize_from(reader).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
  Ok(decoded)
}

/// Measure the serialized size of a proof in memory (without writing to file)
pub fn measure_proof_size<T: Serialize>(proof: &T, proof_name: &str) -> usize {
  let encoded = bincode::serialize(proof).expect("Failed to serialize proof for size measurement");
  let size = encoded.len();
  println!("{} size: {}", proof_name, format_file_size(size as u64));
  size
}

/// Measure the total proof size of multiple proofs
/// Returns the total size in bytes
pub fn measure_total_proof_size<S, O, R, T, U>(
  sumcheck_proofs: &S,
  opening_proofs: &O,
  range_proof: &R,
  two_pow_proof: &T,
  reducer_proofs: &U,
) -> String
where
  S: Serialize,
  O: Serialize,
  R: Serialize,
  T: Serialize,
  U: Serialize,
{
  println!("\n=== Proof Sizes ===");
  let sumcheck_size = measure_proof_size(sumcheck_proofs, "sumcheck_proofs");
  let opening_size = measure_proof_size(opening_proofs, "opening_proofs");
  let range_size = measure_proof_size(range_proof, "range_proof");
  let two_pow_size = measure_proof_size(two_pow_proof, "two_pow_proof");
  let reducer_size = measure_proof_size(reducer_proofs, "reducer_proofs");

  let total_size = sumcheck_size + opening_size + range_size + two_pow_size + reducer_size;
  println!("----------------------------------------");
  let formatted_total_size = format_file_size(total_size as u64);
  println!("Total proof size: {}", formatted_total_size);
  formatted_total_size
}
