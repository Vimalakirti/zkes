use crate::dag::{DagBuilder, EdgeId, Witness};
use crate::util::poly::CryptoField;

pub fn dense_add_relu<F: CryptoField + 'static>(w: Witness<F>, b: Witness<F>) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This dense layer expects 1 input");
    let x = x[0];
    let w_e = g.param(w);
    let h = g.einsum("i,ij->j".to_string(), vec![x, w_e], true)[0];
    let b_e = g.param(b);
    let h = g.add(h, b_e)[0];
    // TODO: fix this
    // let h = g.add_lookup_node(vec![h], NonlinearType::Relu, *SF_LOG as usize, *SF_LOG as usize)[0];
    vec![h]
  }
}
