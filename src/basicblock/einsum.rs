use crate::basicblock::BasicBlock;
use crate::crypto::{LinearSumcheckProver, SumcheckProof, SumcheckProver, SumcheckVerifier};
use crate::dag::{Claim, DataType, Role, Witness};
use crate::util::arith::{get_n, log2_ceil};
use crate::util::poly::fix_variables_zkgpt;
use crate::util::poly::CryptoField;
use crate::util::poly::{evaluate_lagrange_basis, DenseMLPoly, MLPoly};
use crate::util::transcript::Transcript;
use crate::SF_LOG;
use ndarray::ArrayD;
use ndarray_einsum_beta::*;
use std::collections::{HashMap, HashSet};

pub fn broadcast_evals_by_doubling<F: Clone>(evals: &[F], add_dims: usize) -> Vec<F> {
  let mut out = evals.to_vec();
  for _ in 0..add_dims {
    out.extend_from_within(..); // duplicates the current content
  }
  out
}

pub fn einsum_output_shape(equation: &str, input_shapes: &[Vec<usize>]) -> Result<Vec<usize>, String> {
  // 1. Parse equation
  let (lhs, rhs) = if let Some((l, r)) = equation.split_once("->") {
    (l, Some(r))
  } else {
    (equation, None)
  };

  let input_terms: Vec<&str> = lhs.split(',').collect();
  if input_terms.len() != input_shapes.len() {
    return Err("Number of inputs does not match number of shapes".into());
  }

  // 2. Build index -> dimension map
  let mut dim_map: HashMap<char, usize> = HashMap::new();

  for (term, shape) in input_terms.iter().zip(input_shapes) {
    let indices: Vec<char> = term.chars().collect();
    if indices.len() != shape.len() {
      return Err(format!("Rank mismatch: term '{}' vs shape {:?}", term, shape));
    }

    for (&idx, &dim) in indices.iter().zip(shape) {
      if let Some(prev) = dim_map.get(&idx) {
        if *prev != dim {
          return Err(format!("Dimension conflict for index '{}': {} vs {}", idx, prev, dim));
        }
      } else {
        dim_map.insert(idx, dim);
      }
    }
  }

  // 3. Determine output indices
  let output_indices: Vec<char> = if let Some(rhs) = rhs {
    rhs.chars().collect()
  } else {
    // Implicit output: all indices that appear only once
    let mut counts: HashMap<char, usize> = HashMap::new();
    for term in &input_terms {
      for c in term.chars() {
        *counts.entry(c).or_insert(0) += 1;
      }
    }
    counts.into_iter().filter(|(_, v)| *v == 1).map(|(k, _)| k).collect()
  };

  // 4. Build output shape
  let mut out_shape = Vec::new();
  for idx in output_indices {
    match dim_map.get(&idx) {
      Some(&dim) => out_shape.push(dim),
      None => return Err(format!("Unknown output index '{}'", idx)),
    }
  }

  Ok(out_shape)
}

pub fn permute_evals_by_ranges<F: Clone>(evals: &[F], n: usize, ranges: &[(usize, usize)]) -> Vec<F> {
  assert_eq!(evals.len(), 1usize << n, "evals length must be 2^n");
  assert!(!ranges.is_empty(), "ranges must not be empty");

  // 1) Build the new variable order as a flat list of old variable indices.
  //    new_var_order[new_pos] = old_var_index
  let mut new_var_order = Vec::with_capacity(n);
  let mut seen = vec![false; n];

  for &(start, end) in ranges {
    assert!(start <= end && end <= n, "invalid range ({}, {})", start, end);
    for v in start..end {
      assert!(!seen[v], "variable {} appears in multiple ranges", v);
      seen[v] = true;
      new_var_order.push(v);
    }
  }

  assert!(new_var_order.len() == n, "ranges must cover all variables exactly once");

  // 2) Build mapping from old variable index -> new variable index:
  //
  //    For each old_var, pos_new[old_var] = new position in 0..n.
  let mut pos_new = vec![0usize; n];
  for (new_pos, &old_var) in new_var_order.iter().enumerate() {
    pos_new[old_var] = new_pos;
  }

  // 3) Permute the evaluation table according to this variable reordering.
  //
  //    For each old index idx_old, we construct a new index idx_new by
  //    moving bit `old_var` to bit `pos_new[old_var]`.
  let total = evals.len();
  if total == 0 {
    return Vec::new();
  }

  let mut out = vec![evals[0].clone(); total];

  for idx_old in 0..total {
    let mut idx_new = 0usize;
    let mut mask = 1usize;

    // Check each old variable bit.
    for old_var in 0..n {
      if idx_old & mask != 0 {
        let new_var = pos_new[old_var];
        idx_new |= 1usize << new_var;
      }
      mask <<= 1;
    }

    out[idx_new] = evals[idx_old].clone();
  }

  out
}

pub fn invert_points_by_ranges<F: Clone>(y: &[F], ranges: &[(usize, usize)]) -> Vec<F> {
  let n = y.len();

  // 1) Build forward permutation "new_order": list of old variable indices.
  let mut new_order = Vec::with_capacity(n);
  for &(start, end) in ranges {
    for v in start..end {
      new_order.push(v);
    }
  }
  assert!(new_order.len() == n, "ranges must cover 0..n exactly once");

  // 2) Build inverse permutation:
  //    old_var_to_newpos[old_var] = new_pos
  let mut old_var_to_newpos = vec![0usize; n];
  for (new_pos, &old_var) in new_order.iter().enumerate() {
    old_var_to_newpos[old_var] = new_pos;
  }

  // 3) Recover X: for each old variable index v,
  //    X[v] = Y[ old_var_to_newpos[v] ]
  let mut x = Vec::with_capacity(n);
  for old_var in 0..n {
    let new_pos = old_var_to_newpos[old_var];
    x.push(y[new_pos].clone());
  }

  x
}

/// Given evaluations of a multilinear g(x0,...,x_{nv-1}),
/// produce evaluations of
///   g'(x0,...,x_{keep_prefix-1}) =
///       sum_{(x_{keep_prefix},...,x_{nv-1}) in {0,1}^{nv-keep_prefix}} g(...)
///
/// - `evals` must have length 2^nv.
/// - Result has length 2^keep_prefix.
pub fn sum_over_suffix<F: CryptoField + Copy>(evals: &[F], nv: usize, keep_prefix: usize) -> Vec<F> {
  assert_eq!(evals.len(), 1usize << nv, "evals length must be 2^nv, got {} vs 2^{}", evals.len(), nv,);
  assert!(keep_prefix <= nv, "keep_prefix must be in 0..=nv, got {} vs {}", keep_prefix, nv,);

  // Work in-place on a copy.
  let mut poly = evals.to_vec();
  let suffix_vars = nv - keep_prefix;

  // Collapse from the rightmost variable (x_{nv-1}) down to x_{keep_prefix}.
  //
  // At step i = 0, we sum over x_{nv-1}; at i = 1, over x_{nv-2}; etc.
  // Each step halves the "active" length, pairing [j] and [j + half].
  for i in 0..suffix_vars {
    let half = 1usize << (nv - 1 - i);
    for j in 0..half {
      let left = poly[j]; // x_current = 0
      let right = poly[j + half]; // x_current = 1
      poly[j] = left + right; // sum over that variable
    }
    // After this, valid prefix is poly[0..half)
  }

  poly.truncate(1usize << keep_prefix);
  poly
}

/// Classification of einsum indices.
#[derive(Debug, Clone)]
pub struct EinsumIndexClassification {
  /// Free indices that appear exactly once across all inputs.
  pub free_once: Vec<char>,
  /// Free indices that appear more than once across all inputs.
  pub free_multi: Vec<char>,
  /// Summation indices: appear in inputs but not in the output.
  pub summation: Vec<char>,
}

pub fn char_to_range(symbol: &[char], shape: &[usize]) -> HashMap<char, (usize, usize)> {
  let mut char_to_range = HashMap::new();
  let mut start = 0;
  for (i, c) in symbol.iter().enumerate() {
    let log2_ceil = log2_ceil(shape[i]) as usize;
    char_to_range.insert(*c, (start, start + log2_ceil));
    start += log2_ceil;
  }
  char_to_range
}

pub fn einsum_helper<F: CryptoField>(
  equation: &str,
  shapes: &[Vec<usize>],
  challenge_point: &[F],
) -> (Vec<Vec<(usize, usize)>>, Vec<Vec<F>>, Vec<F>, usize) {
  let (einsum_classification, input_specs, out_indices) = classify_einsum_indices_from_shapes(equation);
  // concat free_once, free_multi, summation
  let mut all_indices = einsum_classification.free_once.clone();
  all_indices.extend(einsum_classification.free_multi.clone());
  all_indices.extend(einsum_classification.summation.clone());
  let input_num = shapes.len() - 1;
  assert_eq!(input_num, input_specs.len(), "number of input specs must equal number of input shapes");
  let output_c_to_r = char_to_range(&out_indices, &shapes[shapes.len() - 1]);

  let mut permute_vecs: Vec<Vec<(usize, usize)>> = Vec::with_capacity(input_num);
  let mut degree_one_challenges: Vec<Vec<F>> = Vec::with_capacity(input_num);
  let mut summation_set: HashSet<char> = HashSet::new();
  let mut summation_round: usize = 0;
  for i in 0..input_num {
    let shape = shapes[i].clone();
    let spec = input_specs[i].clone();
    let c_to_r = char_to_range(&spec, &shape);
    let mut permute_vec = Vec::with_capacity(spec.len());
    let mut partial_challenge = vec![];
    for index in all_indices.iter() {
      if c_to_r.contains_key(index) {
        permute_vec.push(*c_to_r.get(index).unwrap());
      }
    }
    for index in einsum_classification.free_once.iter() {
      if c_to_r.contains_key(index) {
        let output_range = *output_c_to_r.get(index).unwrap();
        let ch = challenge_point[output_range.0..output_range.1].to_vec();
        partial_challenge.extend(ch);
      }
    }
    for index in einsum_classification.summation.iter() {
      if c_to_r.contains_key(index) && !summation_set.contains(index) {
        summation_set.insert(*index);
        let range = *c_to_r.get(index).unwrap();
        summation_round += range.1 - range.0;
      }
    }
    permute_vecs.push(permute_vec);
    degree_one_challenges.push(partial_challenge);
  }

  let mut high_degree_challenge = vec![];
  for index in einsum_classification.free_multi.iter() {
    let output_range = output_c_to_r.get(index).unwrap();
    let ch = challenge_point[output_range.0..output_range.1].to_vec();
    high_degree_challenge.extend(ch);
  }

  (permute_vecs, degree_one_challenges, high_degree_challenge, summation_round)
}

/// Core function: classify indices given subscripts and input shapes.
pub fn classify_einsum_indices_from_shapes(subscripts: &str) -> (EinsumIndexClassification, Vec<Vec<char>>, Vec<char>) {
  // ---------- 1. Parse subscripts ----------
  let (lhs, rhs) = subscripts.split_once("->").expect("einsum string must contain '->'");

  let input_specs: Vec<Vec<char>> = lhs.split(',').map(|s| s.trim().chars().collect()).collect();

  let out_indices: Vec<char> = rhs.trim().chars().collect();
  let out_set: HashSet<char> = out_indices.iter().copied().collect();

  // ---------- 2. Count occurrences ----------
  // Count how many times each label appears across *inputs*.
  let mut input_count: HashMap<char, usize> = HashMap::new();

  for spec in input_specs.iter() {
    for &label in spec {
      *input_count.entry(label).or_insert(0) += 1;
    }
  }

  // ---------- 3. Build categories ----------
  // 1. Free indices (in output) that appear once in inputs.
  let mut free_once: Vec<char> = Vec::new();
  // 2. Free indices (in output) that appear > 1 in inputs.
  let mut free_multi: Vec<char> = Vec::new();
  // 3. Summation indices: appear in inputs but not in output.
  //    We'll track them in first-appearance order later.
  let mut summation_set: HashSet<char> = HashSet::new();

  // Free indices: use the *output order*.
  for &label in &out_indices {
    let count = input_count.get(&label).copied().unwrap_or(0);
    assert!(count > 0, "output index '{}' does not appear in any input", label);
    if count == 1 {
      free_once.push(label);
    } else {
      free_multi.push(label);
    }
  }

  // Summation indices: all labels that appear in inputs but NOT in output.
  // Preserve first-occurrence order over inputs.
  let mut summation: Vec<char> = Vec::new();
  for spec in &input_specs {
    for &label in spec {
      if !out_set.contains(&label) && !summation_set.contains(&label) {
        summation_set.insert(label);
        summation.push(label);
      }
    }
  }

  (
    EinsumIndexClassification {
      free_once,
      free_multi,
      summation,
    },
    input_specs,
    out_indices,
  )
}

#[derive(Debug, Clone)]
pub struct Einsum {
  pub equation: String,
}
impl<F: CryptoField> BasicBlock<F> for Einsum {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    let input_arrays = inputs.iter().map(|i| i.ndarray()).collect::<Vec<ArrayD<i128>>>();
    let input_arrays_refs: Vec<&dyn ndarray_einsum_beta::ArrayLike<i128>> =
      input_arrays.iter().map(|a| a as &dyn ndarray_einsum_beta::ArrayLike<i128>).collect();

    let output_array = einsum(&self.equation, &input_arrays_refs).unwrap();
    let col_major_output: Vec<_> = output_array.clone().view().reversed_axes().iter().cloned().collect();
    let output_shape = einsum_output_shape(
      &self.equation,
      &inputs.iter().map(|input| input.shape.clone()).collect::<Vec<Vec<usize>>>(),
    )
    .unwrap();
    let output_data = col_major_output
      .iter()
      .map(|f| {
        F::from(*f)
        //if *f > 0 {
        //  F::from(*f as u32)
        //} else {
        //  <F as CryptoField>::zero() - F::from(-*f as u32)
        //}
      })
      .collect::<Vec<F>>();
    vec![Witness::new(
      output_shape,
      output_data,
      DataType::Float,
      inputs.iter().fold(0, |acc, i| acc + i.sf),
      Role::Output,
    )]
  }

  fn prove(
    &self,
    witnesses: &[&Witness<F>],
    edge_ids: &[usize],
    out_claims: &[&Claim<F>],
    transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    assert!(out_claims.len() == 1, "Einsum expects 1 output claim for now");
    let challenge_point = out_claims[0].point.clone();
    let shapes = witnesses.iter().map(|i| i.shape.clone()).collect::<Vec<Vec<usize>>>();
    let (permute_vecs, degree_one_challenges, high_degree_challenge, summation_round) = einsum_helper(&self.equation, &shapes, &challenge_point);
    let mut sumcheck_input_polys = Vec::with_capacity(permute_vecs.len());
    for i in 0..permute_vecs.len() {
      let permute_vec = permute_vecs[i].clone();
      let witness = witnesses[i];
      let n = get_n(&witness.shape);

      let permuted_witness_poly = if n >= 18 && degree_one_challenges[i].len() > 0 {
        let witness_evaluations = witness.data_int.as_ref().unwrap();
        let permuted_witness_evaluations = permute_evals_by_ranges(witness_evaluations, n, &permute_vec);
        fix_variables_zkgpt(n, &permuted_witness_evaluations, &degree_one_challenges[i], *SF_LOG as usize + 2)
      } else {
        let witness_evaluations =
          &witness.data.as_ref().unwrap().as_any().downcast_ref::<DenseMLPoly<F>>().expect("Expected DenseMLPoly for witness").evaluations;
        let permuted_witness_evaluations = permute_evals_by_ranges(witness_evaluations, n, &permute_vec);
        DenseMLPoly::new(n, permuted_witness_evaluations).fix_variables(&degree_one_challenges[i])
      };

      sumcheck_input_polys.push(permuted_witness_poly);
    }
    let num_first_rounds = high_degree_challenge.len();
    let num_second_rounds = summation_round;

    let eq_poly_evals = evaluate_lagrange_basis(&high_degree_challenge);
    let eq_poly_evals = broadcast_evals_by_doubling(&eq_poly_evals, num_second_rounds);
    let eq_poly = DenseMLPoly::new(num_first_rounds + num_second_rounds, eq_poly_evals);
    sumcheck_input_polys.push(eq_poly);

    let mut sumcheck_prover = LinearSumcheckProver::new(num_first_rounds + num_second_rounds, sumcheck_input_polys.len(), transcript);
    let sumcheck_proof = sumcheck_prover.prove(&sumcheck_input_polys, transcript);

    let challenges = sumcheck_prover.challenges;
    let proofs = vec![sumcheck_proof];
    let mut claims = Vec::new();
    for i in 0..permute_vecs.len() {
      let point_i_perm = degree_one_challenges[i].iter().chain(challenges.iter()).cloned().collect::<Vec<F>>();
      let claim_i = Claim {
        edge_id: edge_ids[i],
        sparse_id: 0,
        point: invert_points_by_ranges(&point_i_perm, &permute_vecs[i]),
        eval: sumcheck_input_polys[i].evaluate_at_point(&challenges),
      };
      claims.push(claim_i);
    }

    (proofs, claims)
  }

  fn verify(&self, witnesses: &[&Witness<F>], claims: &[&Claim<F>], sumcheck_proofs: &[&SumcheckProof<F>], transcript: &mut Transcript<F>) -> bool {
    let shapes = witnesses.iter().map(|i| i.shape.clone()).collect::<Vec<Vec<usize>>>();
    let out_claim = claims[claims.len() - 1];
    let (_, _, high_degree_challenge, summation_round) = einsum_helper(&self.equation, &shapes, &out_claim.point);
    let num_first_rounds = high_degree_challenge.len();
    let num_second_rounds = summation_round;

    let mut verifier = SumcheckVerifier::new(num_first_rounds + num_second_rounds, claims.len(), transcript);
    let expected_sum = out_claim.eval;
    let (verification_result, _challenges) = verifier.verify(transcript, sumcheck_proofs[0].round_messages.clone(), expected_sum);
    if verification_result.is_none() {
      println!("verified einsum failed 1");
      return false;
    }
    true
  }
}
