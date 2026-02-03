use crate::util::poly::CryptoField;
use crate::util::serialization::{ark_de, ark_se};
use crate::util::transcript::Transcript;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: CanonicalSerialize + CanonicalDeserialize")]
pub struct SumcheckProof<F: Clone> {
  #[serde(serialize_with = "ark_se", deserialize_with = "ark_de")]
  pub final_eval: F,
  #[serde(serialize_with = "se_nested_vec", deserialize_with = "de_nested_vec")]
  pub round_messages: Vec<Vec<F>>,
}

/// Serialize nested Vec<Vec<F>> for arkworks types
fn se_nested_vec<S, A: CanonicalSerialize>(a: &Vec<Vec<A>>, s: S) -> Result<S::Ok, S::Error>
where
  S: serde::Serializer,
{
  use serde::ser::SerializeSeq;
  let mut seq = s.serialize_seq(Some(a.len()))?;
  for inner in a {
    let mut inner_bytes: Vec<Vec<u8>> = Vec::with_capacity(inner.len());
    for elem in inner {
      let mut bytes = vec![];
      elem.serialize_compressed(&mut bytes).map_err(serde::ser::Error::custom)?;
      inner_bytes.push(bytes);
    }
    seq.serialize_element(&inner_bytes)?;
  }
  seq.end()
}

/// Deserialize nested Vec<Vec<F>> for arkworks types
fn de_nested_vec<'de, D, A: CanonicalDeserialize>(data: D) -> Result<Vec<Vec<A>>, D::Error>
where
  D: serde::de::Deserializer<'de>,
{
  let v: Vec<Vec<Vec<u8>>> = serde::de::Deserialize::deserialize(data)?;
  v.into_iter()
    .map(|inner| inner.into_iter().map(|bytes| A::deserialize_compressed_unchecked(bytes.as_slice()).map_err(serde::de::Error::custom)).collect())
    .collect()
}

pub trait SumcheckProver<F: CryptoField> {
  type Instance;

  fn new(n: usize, num_polys: usize, transcript: &mut Transcript<F>) -> Self;

  fn prove(&mut self, instances: &Self::Instance, transcript: &mut Transcript<F>) -> SumcheckProof<F>;
}
