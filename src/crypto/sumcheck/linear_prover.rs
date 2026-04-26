use crate::crypto::sumcheck::{prover::SumcheckProof, SumcheckProver};
use crate::util::poly::{CryptoField, DenseMLPoly};
use crate::util::transcript::Transcript;
use rayon::prelude::*;

// TODO: make a unified sumcheck struct and include
// the linear sumcheck, sparse-dense sumcheck, and more provers as its sub-proving functions

/// Optimal linear sum-check protocol with Fiat-Shamir
///
/// Computes the sum: H = Σ_{x ∈ {0,1}^num_var} g(x)
/// where g(x) = Π_{i=1}^ℓ p_i(x) and each p_i is a dense multilinear polynomial
pub struct LinearSumcheckProver<F: CryptoField> {
  /// Number of variables num_var
  pub num_var: usize,
  /// Number of polynomials ℓ
  pub num_poly: usize,
  /// Arrays A_i storing evaluations of p_i(x)
  pub a_arrays: Vec<DenseMLPoly<F>>,
  /// Current round (0-indexed)
  pub current_round: usize,
  /// Random challenges from verifier (Fiat-Shamir if non-interactive)
  pub challenges: Vec<F>,
}

impl<F: CryptoField> SumcheckProver<F> for LinearSumcheckProver<F> {
  type Instance = Vec<DenseMLPoly<F>>;

  fn new(num_var: usize, num_polys: usize, transcript: &mut Transcript<F>) -> Self {
    // Commit to the problem instance
    transcript.append_u64(b"num_var", num_var as u64);
    transcript.append_u64(b"num_poly", num_polys as u64);

    Self {
      num_var,
      num_poly: num_polys,
      a_arrays: Vec::new(),
      current_round: 0,
      challenges: Vec::new(),
    }
  }

  fn prove(&mut self, instances: &Self::Instance, transcript: &mut Transcript<F>) -> SumcheckProof<F> {
    assert!(!instances.is_empty(), "Polynomials cannot be empty");
    let n_size = instances[0].len();
    assert_eq!(n_size, 1 << self.num_var, "Polynomials have incorrect size");
    assert_eq!(instances.len(), self.num_poly, "Number of polynomials mismatch");

    // Verify all polynomial arrays have correct size
    for (i, poly) in instances.iter().enumerate() {
      assert_eq!(poly.len(), n_size, "Polynomial {} has incorrect size", i);
    }

    // Store instances for proving
    self.a_arrays = instances.clone();
    let mut round_messages = Vec::new();

    // Execute num_var rounds
    for _round in 0..self.num_var {
      // Compute prover message for this round
      let round_message = self.compute_round_message();

      // Append round message to transcript
      for (_i, &msg) in round_message.iter().enumerate() {
        transcript.append_scalar(b"round_message", &msg);
      }

      // Generate Fiat-Shamir challenge
      let challenge = transcript.challenge_scalar(b"challenge");
      round_messages.push(round_message);
      self.receive_challenge(challenge);
    }

    let final_eval = self.final_evaluation();
    SumcheckProof { final_eval, round_messages }
  }
}

impl<F: CryptoField> LinearSumcheckProver<F> {
  /// Compute the prover's message for the current round
  /// Returns evaluations s_m(c') for c' ∈ {0, 2, ..., ℓ} (optimized version)
  fn compute_round_message(&self) -> Vec<F> {
    let m = self.current_round;
    let remaining_size = 1 << (self.num_var - m);

    // ℓ+1 evaluation points: {0, 1, 2, 3, ..., ℓ}
    let eval_points: Vec<usize> = (0..=self.num_poly).collect();

    // Parallelize over evaluation points
    let round_message: Vec<F> = eval_points
      .par_iter()
      .map(|eval_idx| {
        let c_prime_field = <F as CryptoField>::from_u32(*eval_idx as u32);

        // Parallelize the inner sum over y_bits
        (0..remaining_size >> 1)
          .into_par_iter()
          .map(|y_bits| self.compute_g_value(c_prime_field, y_bits))
          .reduce(|| <F as CryptoField>::zero(), |a, b| a + b)
      })
      .collect();

    round_message
  }

  /// Compute g(r_1, ..., r_{m-1}, c', y) = Π_{i=1}^ℓ p_i(r_1, ..., r_{m-1}, c', y)
  fn compute_g_value(&self, c_prime: F, y_bits: usize) -> F {
    let m = self.current_round;
    let remaining_vars = self.num_var - m;

    if remaining_vars == 0 {
      // All variables are bound, direct evaluation
      // Parallelize the product computation
      return (0..self.num_poly)
        .into_par_iter()
        .filter(|&i| !self.a_arrays[i].is_empty())
        .map(|i| self.a_arrays[i][0])
        .reduce(|| <F as CryptoField>::one(), |a, b| a * b);
    }

    // Parallelize the product computation across polynomials
    (0..self.num_poly)
      .into_par_iter()
      .map(|poly_idx| {
        // For each polynomial, evaluate at the current point
        // We need to interpolate the polynomial value at c_prime
        // Linear interpolation between adjacent points
        let val_0 = self.a_arrays[poly_idx][y_bits * 2];
        let val_1 = self.a_arrays[poly_idx][y_bits * 2 + 1];

        if c_prime == <F as CryptoField>::one() {
          val_1
        } else if c_prime == <F as CryptoField>::zero() {
          val_0
        } else {
          val_0 + c_prime * (val_1 - val_0)
        }
      })
      .reduce(|| <F as CryptoField>::one(), |a, b| a * b)
  }

  /// Process verifier's challenge and bind variables
  fn receive_challenge(&mut self, challenge: F) {
    self.challenges.push(challenge);
    self.bind_variable_to_challenge(challenge);
    self.current_round = self.current_round + 1;
  }

  /// Bind the current variable using the standard formula (Equation 20)
  /// Optimized to perform in-place updates with parallelization.
  fn bind_variable_to_challenge(&mut self, r_m: F) {
    let remaining_vars = self.num_var - self.current_round - 1;
    let new_size = 1 << remaining_vars;

    // Parallelize over polynomials
    self.a_arrays.par_iter_mut().for_each(|poly_array| {
      if new_size >= 1024 {
        // For large polynomials, use parallel processing with par_chunks
        let slice = &mut poly_array.evaluations[..new_size * 2];
        let results: Vec<F> = slice
          .par_chunks(2)
          .map(|pair| {
            let a_0_y = pair[0];
            let a_1_y = pair[1];
            a_0_y + r_m * (a_1_y - a_0_y)
          })
          .collect();
        poly_array.evaluations[..new_size].copy_from_slice(&results);
      } else {
        // Sequential in-place update for smaller polynomials
        for y in 0..new_size {
          let a_0_y = poly_array[y * 2];
          let a_1_y = poly_array[y * 2 + 1];
          poly_array[y] = a_0_y + r_m * (a_1_y - a_0_y);
        }
      }
    });
  }

  /// Get the final evaluation after all rounds complete
  fn final_evaluation(&self) -> F {
    assert_eq!(self.current_round, self.num_var, "Sum-check not complete");

    // Parallelize the final product computation
    (0..self.num_poly)
      .into_par_iter()
      .filter(|&poly_idx| !self.a_arrays[poly_idx].is_empty())
      .map(|poly_idx| self.a_arrays[poly_idx][0])
      .reduce(|| <F as CryptoField>::one(), |a, b| a * b)
  }
}

pub struct GeneralLinearSumcheckProver<F: CryptoField> {
  /// Number of variables num_var
  pub num_var: usize,
  /// Number of polynomials ℓ
  pub num_poly: usize,
  /// eq(r, x)
  pub eq: DenseMLPoly<F>,
  /// Scalars a_i
  pub a_scalars: Vec<F>,
  /// Arrays A_i storing evaluations of p_i(x)
  pub a_arrays: Vec<Vec<DenseMLPoly<F>>>,
  /// Current round (0-indexed)
  pub current_round: usize,
  /// Random challenges from verifier (Fiat-Shamir if non-interactive)
  pub challenges: Vec<F>,
}

impl<F: CryptoField> SumcheckProver<F> for GeneralLinearSumcheckProver<F> {
  type Instance = (Vec<F>, Vec<Vec<DenseMLPoly<F>>>, DenseMLPoly<F>);

  fn new(num_var: usize, num_polys: usize, transcript: &mut Transcript<F>) -> Self {
    // Commit to the problem instance
    transcript.append_u64(b"num_var", num_var as u64);
    transcript.append_u64(b"num_poly", num_polys as u64);

    Self {
      num_var,
      num_poly: num_polys,
      eq: DenseMLPoly::from_evaluations(vec![<F as CryptoField>::one()]),
      a_scalars: Vec::new(),
      a_arrays: Vec::new(),
      current_round: 0,
      challenges: Vec::new(),
    }
  }

  fn prove(&mut self, instances: &Self::Instance, transcript: &mut Transcript<F>) -> SumcheckProof<F> {
    assert!(!instances.1.is_empty() && !instances.0.is_empty(), "Polynomials cannot be empty");
    let n_size = instances.1[0][0].len();
    assert_eq!(n_size, 1 << self.num_var, "Polynomials have incorrect size");
    assert_eq!(instances.1[0].len(), self.num_poly - 1, "Number of polynomials mismatch");
    assert_eq!(instances.0.len(), instances.1.len(), "Number of scalars mismatch");

    // Verify all polynomial arrays have correct size
    for (i, poly_array) in instances.1.iter().enumerate() {
      for (j, poly) in poly_array.iter().enumerate() {
        assert_eq!(poly.len(), n_size, "Polynomial {} has incorrect size", i * self.num_poly + j);
      }
    }

    // Store instances for proving
    self.a_scalars = instances.0.clone();
    self.a_arrays = instances.1.clone();
    self.eq = instances.2.clone();
    let mut round_messages = Vec::new();

    // Execute num_var rounds
    for _round in 0..self.num_var {
      // Compute prover message for this round
      let round_message = self.compute_round_message();

      // Append round message to transcript
      for (_i, &msg) in round_message.iter().enumerate() {
        transcript.append_scalar(b"round_message", &msg);
      }

      // Generate Fiat-Shamir challenge
      let challenge = transcript.challenge_scalar(b"challenge");
      round_messages.push(round_message);
      self.receive_challenge(challenge);
    }

    let final_eval = self.final_evaluation();
    SumcheckProof { final_eval, round_messages }
  }
}

impl<F: CryptoField> GeneralLinearSumcheckProver<F> {
  /// Compute the prover's message for the current round
  /// Returns evaluations s_m(c') for c' ∈ {0, 2, ..., ℓ} (optimized version)
  fn compute_round_message(&self) -> Vec<F> {
    let m = self.current_round;
    let remaining_size = 1 << (self.num_var - m);

    // ℓ+1 evaluation points: {0, 1, 2, 3, ..., ℓ}
    let eval_points: Vec<usize> = (0..=self.num_poly).collect();

    // Parallelize over evaluation points
    let round_message: Vec<F> = eval_points
      .par_iter()
      .map(|eval_idx| {
        let c_prime_field = <F as CryptoField>::from_u32(*eval_idx as u32);

        // Parallelize the inner sum over y_bits
        (0..remaining_size >> 1)
          .into_par_iter()
          .map(|y_bits| self.compute_g_value(c_prime_field, y_bits))
          .reduce(|| <F as CryptoField>::zero(), |a, b| a + b)
      })
      .collect();

    round_message
  }

  /// Compute g(r_1, ..., r_{m-1}, c', y) = Π_{i=1}^ℓ p_i(r_1, ..., r_{m-1}, c', y)
  fn compute_g_value(&self, c_prime: F, y_bits: usize) -> F {
    let m = self.current_round;
    let remaining_vars = self.num_var - m;
    let num_terms = self.a_arrays.len();

    if remaining_vars == 0 {
      // All variables are bound, direct evaluation
      // Parallelize the product computation
      return (0..num_terms)
        .into_par_iter()
        .map(|j| {
          (0..self.num_poly - 1)
            .into_par_iter()
            .filter(|&i| !self.a_arrays[j][i].is_empty())
            .map(|i| self.a_arrays[j][i][0])
            .reduce(|| <F as CryptoField>::one(), |a, b| a * b)
            * self.a_scalars[j]
        })
        .reduce(|| <F as CryptoField>::zero(), |a, b| a + b)
        * self.eq[0];
    }

    let eq_val_0 = self.eq[y_bits * 2];
    let eq_val_1 = self.eq[y_bits * 2 + 1];

    let eq_val = if c_prime == <F as CryptoField>::one() {
      eq_val_1
    } else if c_prime == <F as CryptoField>::zero() {
      eq_val_0
    } else {
      eq_val_0 + c_prime * (eq_val_1 - eq_val_0)
    };

    // Parallelize the product computation across polynomials
    (0..num_terms)
      .into_par_iter()
      .map(|j| {
        (0..self.num_poly - 1)
          .into_par_iter()
          .map(|poly_idx| {
            // For each polynomial, evaluate at the current point
            // We need to interpolate the polynomial value at c_prime
            // Linear interpolation between adjacent points
            let val_0 = self.a_arrays[j][poly_idx][y_bits * 2];
            let val_1 = self.a_arrays[j][poly_idx][y_bits * 2 + 1];

            if c_prime == <F as CryptoField>::one() {
              val_1
            } else if c_prime == <F as CryptoField>::zero() {
              val_0
            } else {
              val_0 + c_prime * (val_1 - val_0)
            }
          })
          .reduce(|| <F as CryptoField>::one(), |a, b| a * b)
          * self.a_scalars[j]
      })
      .reduce(|| <F as CryptoField>::zero(), |a, b| a + b)
      * eq_val
  }

  /// Process verifier's challenge and bind variables
  fn receive_challenge(&mut self, challenge: F) {
    self.challenges.push(challenge);
    self.bind_variable_to_challenge(challenge);
    self.current_round = self.current_round + 1;
  }

  /// Bind the current variable using the standard formula (Equation 20)
  /// Optimized to perform in-place updates with parallelization.
  fn bind_variable_to_challenge(&mut self, r_m: F) {
    let remaining_vars = self.num_var - self.current_round - 1;
    let new_size = 1 << remaining_vars;

    // Parallelize over polynomial groups - with inner parallelization for large polynomials
    self.a_arrays.par_iter_mut().for_each(|poly_arrays| {
      poly_arrays.par_iter_mut().for_each(|poly_array| {
        if new_size >= 1024 {
          // For large polynomials, use parallel processing with par_chunks
          let slice = &mut poly_array.evaluations[..new_size * 2];
          let results: Vec<F> = slice
            .par_chunks(2)
            .map(|pair| {
              let a_0_y = pair[0];
              let a_1_y = pair[1];
              a_0_y + r_m * (a_1_y - a_0_y)
            })
            .collect();
          poly_array.evaluations[..new_size].copy_from_slice(&results);
        } else {
          // Sequential in-place update for smaller polynomials
          for y in 0..new_size {
            let a_0_y = poly_array[y * 2];
            let a_1_y = poly_array[y * 2 + 1];
            poly_array[y] = a_0_y + r_m * (a_1_y - a_0_y);
          }
        }
      });
    });

    // In-place update for eq polynomial (with parallelization for large sizes)
    if new_size >= 1024 {
      let slice = &mut self.eq.evaluations[..new_size * 2];
      let results: Vec<F> = slice
        .par_chunks(2)
        .map(|pair| {
          let eq_0_y = pair[0];
          let eq_1_y = pair[1];
          eq_0_y + r_m * (eq_1_y - eq_0_y)
        })
        .collect();
      self.eq.evaluations[..new_size].copy_from_slice(&results);
    } else {
      for y in 0..new_size {
        let eq_0_y = self.eq[y * 2];
        let eq_1_y = self.eq[y * 2 + 1];
        self.eq[y] = eq_0_y + r_m * (eq_1_y - eq_0_y);
      }
    }
  }

  /// Get the final evaluation after all rounds complete
  fn final_evaluation(&self) -> F {
    assert_eq!(self.current_round, self.num_var, "Sum-check not complete");

    let num_terms = self.a_arrays.len();
    // Parallelize the final product computation
    (0..num_terms)
      .into_par_iter()
      .map(|j| {
        (0..self.num_poly - 1)
          .into_par_iter()
          .filter(|&poly_idx| !self.a_arrays[j][poly_idx].is_empty())
          .map(|poly_idx| self.a_arrays[j][poly_idx][0])
          .reduce(|| <F as CryptoField>::one(), |a, b| a * b)
          * self.a_scalars[j]
      })
      .reduce(|| <F as CryptoField>::zero(), |a, b| a + b)
      * self.eq[0]
  }
}

/// Sparse boolean sumcheck prover for range proofs.
///
/// Proves Σ_x eq(r, x) · Σ_j w_j · aux_j(x) · (aux_j(x) - 1) = 0
/// where each aux_j is a selection polynomial (0/1 entries).
///
/// Instead of materialising 2^n dense arrays, this tracks only the O(K)
/// non-zero positions of each selection polynomial and evaluates the
/// round polynomials in O(K) per evaluation point per round.
/// Sparse boolean sumcheck prover for proving selection polynomials are binary.
///
/// Uses a dense eq array + sorted sparse entries per term to avoid both:
/// - Materializing full 2^n dense polynomials (memory)
/// - HashMap overhead (performance)
///
/// Computes: Σ_{x ∈ {0,1}^n} eq(r,x) · Σ_j w_j · s_j(x) · (s_j(x) - 1) = 0
pub struct SparseBoolSumcheckProver<F: CryptoField> {
  pub num_var: usize,
  pub current_round: usize,
  pub challenges: Vec<F>,
  /// Per-term: (weight = β², sorted entries as Vec<(position, bound_value)>).
  terms: Vec<(F, Vec<(usize, F)>)>,
  /// Dense eq(eq_challenge, ·) array. Size = 2^(num_var - current_round).
  eq_dense: Vec<F>,
}

impl<F: CryptoField> SparseBoolSumcheckProver<F> {
  /// Create a new prover. Writes the same transcript header as
  /// `GeneralLinearSumcheckProver::new(num_var, 3, transcript)`.
  pub fn new(num_var: usize, transcript: &mut Transcript<F>) -> Self {
    transcript.append_u64(b"num_var", num_var as u64);
    transcript.append_u64(b"num_poly", 3u64);
    Self {
      num_var,
      current_round: 0,
      challenges: Vec::new(),
      terms: Vec::new(),
      eq_dense: Vec::new(),
    }
  }

  /// Run the sumcheck protocol and return a proof.
  ///
  /// * `weights`  – per-term weight β² (same length as `positions`).
  /// * `positions` – per-term list of non-zero indices in the full 2^n
  ///   hypercube (selection polynomial entries; all values are implicitly 1).
  /// * `eq_challenge` – the random point r used to form eq(r, ·).
  pub fn prove(
    &mut self,
    weights: &[F],
    positions: &[Vec<usize>],
    eq_challenge: &[F],
    transcript: &mut Transcript<F>,
  ) -> SumcheckProof<F> {
    assert_eq!(weights.len(), positions.len());
    assert_eq!(eq_challenge.len(), self.num_var);

    // Initialize terms with sorted entries, each position has value = 1.
    self.terms = weights
      .iter()
      .zip(positions.iter())
      .map(|(&w, pos)| {
        let one = <F as CryptoField>::one();
        let mut entries: Vec<(usize, F)> = pos.iter().map(|&idx| (idx, one)).collect();
        entries.sort_unstable_by_key(|&(idx, _)| idx);
        (w, entries)
      })
      .collect();

    // Compute dense eq(eq_challenge, ·) over the full 2^n hypercube.
    self.eq_dense = crate::util::poly::evaluate_lagrange_basis(eq_challenge);

    let mut round_messages = Vec::new();

    for _round in 0..self.num_var {
      let msg = self.compute_round_message();
      for &m in &msg {
        transcript.append_scalar(b"round_message", &m);
      }
      let challenge = transcript.challenge_scalar(b"challenge");
      round_messages.push(msg);
      self.receive_challenge(challenge);
    }

    let final_eval = self.final_evaluation();
    SumcheckProof { final_eval, round_messages }
  }

  /// Compute round polynomial T(c) for c ∈ {0, 1, 2, 3} in a single pass
  /// over each term's sparse entries. Uses dense eq array for O(1) lookups.
  fn compute_round_message(&self) -> Vec<F> {
    let zero = <F as CryptoField>::zero();
    let one = <F as CryptoField>::one();
    let two = <F as CryptoField>::from_u32(2);
    let three = <F as CryptoField>::from_u32(3);

    // Parallel over terms: each computes its [T(0), T(1), T(2), T(3)] contribution
    let partial_sums: Vec<[F; 4]> = self.terms.par_iter().map(|(w, entries)| {
      let mut t = [zero; 4];
      let mut i = 0;
      while i < entries.len() {
        let pos = entries[i].0;
        let rest = pos >> 1;
        let bit = pos & 1;

        let mut v0 = zero;
        let mut v1 = zero;

        if bit == 0 {
          v0 = entries[i].1;
          i += 1;
          // Check if next entry is the paired odd position
          if i < entries.len() && entries[i].0 == 2 * rest + 1 {
            v1 = entries[i].1;
            i += 1;
          }
        } else {
          // Only odd position exists
          v1 = entries[i].1;
          i += 1;
        }

        let eq0 = self.eq_dense[2 * rest];
        let eq1 = self.eq_dense[2 * rest + 1];

        // c=0: aux=v0, eq=eq0
        t[0] = t[0] + *w * v0 * (v0 - one) * eq0;

        // c=1: aux=v1, eq=eq1
        t[1] = t[1] + *w * v1 * (v1 - one) * eq1;

        // c=2: aux = 2*v1 - v0, eq = 2*eq1 - eq0
        let aux2 = two * v1 - v0;
        let eq2 = two * eq1 - eq0;
        t[2] = t[2] + *w * aux2 * (aux2 - one) * eq2;

        // c=3: aux = 3*v1 - 2*v0, eq = 3*eq1 - 2*eq0
        let aux3 = three * v1 - two * v0;
        let eq3 = three * eq1 - two * eq0;
        t[3] = t[3] + *w * aux3 * (aux3 - one) * eq3;
      }
      t
    }).collect();

    // Reduce partial sums across threads
    let mut result = [zero; 4];
    for t in &partial_sums {
      for i in 0..4 {
        result[i] = result[i] + t[i];
      }
    }
    result.to_vec()
  }

  /// Bind the current variable to `challenge` and advance the round.
  fn receive_challenge(&mut self, challenge: F) {
    self.challenges.push(challenge);
    let zero = <F as CryptoField>::zero();

    // Fold eq_dense: eq_new[rest] = eq[2r] + challenge * (eq[2r+1] - eq[2r])
    let half = self.eq_dense.len() / 2;
    let new_eq: Vec<F> = (0..half)
      .into_par_iter()
      .map(|rest| {
        self.eq_dense[2 * rest] + challenge * (self.eq_dense[2 * rest + 1] - self.eq_dense[2 * rest])
      })
      .collect();
    self.eq_dense = new_eq;

    // Fold each term's sorted entries in parallel
    self.terms.par_iter_mut().for_each(|(_, entries)| {
      let mut new_entries = Vec::with_capacity(entries.len());
      let mut i = 0;
      while i < entries.len() {
        let pos = entries[i].0;
        let rest = pos >> 1;
        let bit = pos & 1;

        let mut v0 = zero;
        let mut v1 = zero;

        if bit == 0 {
          v0 = entries[i].1;
          i += 1;
          if i < entries.len() && entries[i].0 == 2 * rest + 1 {
            v1 = entries[i].1;
            i += 1;
          }
        } else {
          v1 = entries[i].1;
          i += 1;
        }

        let new_val = v0 + challenge * (v1 - v0);
        if new_val != zero {
          new_entries.push((rest, new_val));
        }
      }
      *entries = new_entries;
    });

    self.current_round += 1;
  }

  /// Final evaluation after all rounds.
  fn final_evaluation(&self) -> F {
    assert_eq!(self.current_round, self.num_var);
    let zero = <F as CryptoField>::zero();
    let one = <F as CryptoField>::one();
    let eq_val = if self.eq_dense.is_empty() { zero } else { self.eq_dense[0] };
    let mut inner = zero;
    for (w, entries) in &self.terms {
      let v = if entries.is_empty() || entries[0].0 != 0 { zero } else { entries[0].1 };
      inner = inner + *w * v * (v - one);
    }
    eq_val * inner
  }
}
