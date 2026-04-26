use crate::dag::{DagBuilder, DataType, EdgeId, Role, Witness};
use crate::util::poly::CryptoField;

impl<F: CryptoField + 'static> DagBuilder<F> {
  /// Build a one-hot selector matrix.
  /// `S[i, indices[i]] = 1`, all others 0.
  /// Shape: `(len, table_size)`.
  /// Data is in MLE col-major order: element (i, v) at flat index `i + v * len_padded`.
  fn build_one_hot_selector(len: usize, table_size: usize, indices: &[usize]) -> Witness<F> {
    assert_eq!(indices.len(), len);
    let len_pad = len.next_power_of_two();
    let table_pad = table_size.next_power_of_two();
    let total = len_pad * table_pad;
    let mut data = vec![<F as CryptoField>::zero(); total];
    for (i, &idx) in indices.iter().enumerate() {
      assert!(idx < table_size, "Index {} out of range [0, {})", idx, table_size);
      // MLE col-major: element (i, v) is at flat index i + v * len_pad
      data[i + idx * len_pad] = <F as CryptoField>::one();
    }
    Witness::new(vec![len, table_size], data, DataType::Float, 0, Role::Constant)
  }

  /// Public accessor for building a one-hot selector (for single-pass argmax fixup).
  pub fn build_one_hot_selector_pub(len: usize, table_size: usize, indices: &[usize]) -> Witness<F> {
    Self::build_one_hot_selector(len, table_size, indices)
  }

  /// Embedding lookup: proves `H_0[i,:] = W_E[token_ids[i],:]`.
  ///
  /// `w_e`: committed embedding matrix edge, shape `(vocab_size, hidden_dim)`.
  /// `token_ids`: public transcript token IDs (length `seq_len`).
  ///
  /// Returns `(h0_edge, selector_edge)`:
  /// - `h0_edge`: output hidden states, shape `(seq_len, hidden_dim)`.
  /// - `selector_edge`: the one-hot selector param edge, exposed so the caller can
  ///   overwrite it during autoregressive generation (single-pass optimization).
  pub fn embedding_lookup(
    &mut self,
    w_e: EdgeId,
    seq_len: usize,
    vocab_size: usize,
    token_ids: &[usize],
  ) -> (EdgeId, EdgeId) {
    let s = Self::build_one_hot_selector(seq_len, vocab_size, token_ids);
    let s_id = self.param(s);
    // H_0 = einsum("sv,vd->sd", S, W_E) -> (seq_len, hidden_dim)
    // scale_back=false: S has sf=0, W_E has sf=SF_LOG, output should keep sf=SF_LOG.
    // scale_back=true would divide by 2^SF_LOG (since output_sf=first input's sf=0), destroying values.
    let h0 = self.einsum("sv,vd->sd".to_string(), vec![s_id, w_e], false)[0];
    (h0, s_id)
  }

  /// Add learned positional embeddings (GPT-2 APE).
  ///
  /// `h0`: edge for `H_0`, shape `(seq_len, hidden_dim)`.
  /// `pos_embed`: positional embedding witness, shape `(seq_len, hidden_dim)`.
  ///
  /// Returns edge for `H_pe = H_0 + P`, shape `(seq_len, hidden_dim)`.
  pub fn add_positional_encoding(&mut self, h0: EdgeId, pos_embed: Witness<F>) -> EdgeId {
    let p = self.param(pos_embed);
    self.add(h0, p)[0]
  }

  /// LM head projection using weight-tied `W_E`.
  ///
  /// `hidden`: transformer output edge, shape `(1, seq_len, hidden_dim)`.
  /// `w_e`: same committed embedding matrix edge, shape `(vocab_size, hidden_dim)`.
  ///
  /// Returns edge for logits with shape `(seq_len, vocab_size)`.
  pub fn lm_head_weight_tied(
    &mut self,
    hidden: EdgeId,
    w_e: EdgeId,
    seq_len: usize,
    vocab_size: usize,
  ) -> EdgeId {
    // logits = einsum("bsd,vd->bsv", hidden, W_E) -> (1, seq_len, vocab_size)
    let logits = self.einsum("bsd,vd->bsv".to_string(), vec![hidden, w_e], true)[0];
    self.change_shape(logits, vec![seq_len, vocab_size])
  }

  /// Sound argmax verification: proves `token_ids[i] = argmax(logits[i,:])` for each position.
  ///
  /// `logits`: edge with shape `(seq_len, vocab_size)`.
  /// `token_ids`: next-token IDs to verify (length `seq_len`).
  ///
  /// Returns the selector edge ID so callers can overwrite the one-hot matrix
  /// after a forward pass (single-pass optimization).
  pub fn argmax_check(
    &mut self,
    logits: EdgeId,
    seq_len: usize,
    vocab_size: usize,
    token_ids: &[usize],
  ) -> EdgeId {
    // 1. One-hot selector for claimed next tokens
    let s = Self::build_one_hot_selector(seq_len, vocab_size, token_ids);
    let s_id = self.param(s);

    // 2. Extract selected logits: selected[i] = logits[i, token_ids[i]]
    // scale_back=false: logits has sf=SF_LOG, S has sf=0 → output sf = SF_LOG (no scaling needed)
    let selected = self.einsum("sv,sv->s".to_string(), vec![logits, s_id], false)[0];

    // 3. Broadcast subtract: diffs[i,j] = selected[i] - logits[i,j]
    let selected_broad = self.change_shape(selected, vec![seq_len, 1]);
    let diffs = self.sub(selected_broad, logits)[0];

    // 4. Range check: all diffs >= 0 => selected[i] >= logits[i,j] for all j
    self.add_nonneg_node(diffs);

    s_id
  }
}
