use crate::util::poly::{evaluate_univariate_polynomial, CryptoField};
use crate::util::transcript::Transcript;

/// Verifier for sumcheck protocols with Fiat-Shamir
/// num_var: number of variables
/// num_poly: number of polynomials
pub struct SumcheckVerifier<F: CryptoField> {
  num_var: usize,
  num_poly: usize,
  _phantom: std::marker::PhantomData<F>,
}

impl<F: CryptoField + Send + Sync + 'static> SumcheckVerifier<F> {
  pub fn new(num_var: usize, num_poly: usize, transcript: &mut Transcript<F>) -> Self {
    // Commit to the same problem instance as prover
    transcript.append_u64(b"num_var", num_var as u64);
    transcript.append_u64(b"num_poly", num_poly as u64);

    Self {
      num_var,
      num_poly,
      _phantom: std::marker::PhantomData,
    }
  }

  /// Verify the sum-check proof
  pub fn verify(&mut self, transcript: &mut Transcript<F>, round_messages: Vec<Vec<F>>, claimed_sum: F) -> (Option<F>, Vec<F>) {
    if round_messages.len() != self.num_var {
      println!("Round messages length mismatch: {} != {}", round_messages.len(), self.num_var);
      return (None, vec![]);
    }

    let mut challenges = Vec::new();
    let mut running_sum = claimed_sum;

    // Verify each round
    for (round, round_message) in round_messages.iter().enumerate() {
      // Degree bound check: round polynomial must have exactly num_poly + 1 evaluations
      if round_message.len() != self.num_poly + 1 {
        println!(
          "Round {} degree mismatch: got {} evals, expected {}",
          round, round_message.len(), self.num_poly + 1
        );
        return (None, vec![]);
      }
      if round_message[0] + round_message[1] != running_sum {
        println!(
          "Round {} messages mismatch: {:?} != {:?}",
          round,
          round_message[0] + round_message[1],
          running_sum
        );
        return (None, vec![]);
      }

      // Append round message to transcript
      for (_i, &msg) in round_message.iter().enumerate() {
        transcript.append_scalar(b"round_message", &msg);
      }

      // Generate the same Fiat-Shamir challenge
      let challenge: F = transcript.challenge_scalar(b"challenge");
      // interpolate a univariate polynomial through the round messages (uses global cache)
      running_sum = evaluate_univariate_polynomial(round_message, challenge);

      challenges.push(challenge);
    }

    (Some(running_sum), challenges)
  }
}
