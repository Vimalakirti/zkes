use crate::util::poly::CryptoField;
use merlin::Transcript as MerlinTranscript;

/// A transcript that works with CryptoField using Merlin under the hood
#[derive(Clone)]
pub struct Transcript<F: CryptoField> {
  merlin: MerlinTranscript,
  _phantom: std::marker::PhantomData<F>,
}

impl<F: CryptoField> Transcript<F> {
  pub fn new(label: &'static [u8]) -> Self {
    Self {
      merlin: MerlinTranscript::new(label),
      _phantom: std::marker::PhantomData,
    }
  }

  pub fn append_message(&mut self, label: &'static [u8], message: &[u8]) {
    self.merlin.append_message(label, message);
  }

  pub fn append_u64(&mut self, label: &'static [u8], value: u64) {
    self.merlin.append_u64(label, value);
  }

  pub fn append_scalar(&mut self, label: &'static [u8], scalar: &F) {
    let bytes = <F as CryptoField>::to_bytes_le(scalar);
    self.merlin.append_message(label, &bytes);
  }

  pub fn append_scalars(&mut self, label: &'static [u8], scalars: &[F]) {
    for scalar in scalars {
      self.append_scalar(label, scalar);
    }
  }

  pub fn challenge_scalar(&mut self, label: &'static [u8]) -> F {
    let mut buf = vec![0u8; 64]; // Use 64 bytes for uniform randomness
    self.merlin.challenge_bytes(label, &mut buf);
    <F as CryptoField>::from_bytes_le(&buf)
  }

  pub fn challenge_vector(&mut self, label: &'static [u8], len: usize) -> Vec<F> {
    (0..len).map(|_| self.challenge_scalar(label)).collect()
  }
}
