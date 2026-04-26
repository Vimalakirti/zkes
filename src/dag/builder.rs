use crate::basicblock::add::{Add, Sub};
use crate::basicblock::clamp::{ClampLower, ZeroCheck};
use crate::basicblock::einsum::Einsum;
use crate::basicblock::exp::{ExpHelper, NonStructuredExp, TwoPow};
use crate::basicblock::llama::SigmoidConst;
use crate::basicblock::range::NonNegative;
use crate::basicblock::scale::{ScaleDown, ScaleUp};
use crate::basicblock::shape::ChangeShape;
use crate::basicblock::BasicBlock;
use crate::basicblock::BasicBlockType;
use crate::dag::Dag;
use crate::dag::DataType;
use crate::dag::{AliasId, EdgeId, Node, NodeId, Role, Witness};
use crate::util::arith::next_pow;
use crate::util::poly::CryptoField;
use crate::util::shape::{broadcast_shape, pad_to_pow_of_two};
use crate::SF_LOG;
use crate::TABLE_SIZE_LOG;
use ndarray::ArrayD;
use std::collections::HashMap;

fn letters(a: usize) -> String {
  (0..a).map(|i| (b'a' + i as u8) as char).collect()
}

/* =========================
Builder + DSL
========================= */
pub struct DagBuilder<F: CryptoField + 'static> {
  pub nodes: Vec<Node>,
  pub num_edges: usize, // monotonically increasing physical edge IDs
  pub init_values: Vec<Option<Witness<F>>>,
  // Lookups
  pub range: Vec<NodeId>,      // currently only support non-negative range
  pub two_pow: Vec<NodeId>,    // nodes that compute 2^(-k)
  pub zero_check: Vec<NodeId>, // nodes that assert their input MLE is identically zero
}

impl<F: CryptoField + 'static> DagBuilder<F> {
  pub fn new() -> Self {
    Self {
      nodes: Vec::new(),
      num_edges: 0,
      init_values: Vec::new(),
      range: Vec::new(),
      two_pow: Vec::new(),
      zero_check: Vec::new(),
    }
  }

  /// Create a graph input edge (no known value).
  pub fn input(&mut self, shape: Vec<usize>, data_type: DataType) -> EdgeId {
    let witness = Witness::new_wo_data(
      shape,
      data_type,
      if data_type == DataType::Float { *SF_LOG as usize } else { 0 },
      Role::Input,
    );
    let e = self.num_edges;
    self.num_edges += 1;
    self.init_values.push(Some(witness));
    e
  }

  /// Create a **parameter/constant** edge with a known value.
  pub fn param(&mut self, t: Witness<F>) -> EdgeId {
    let e = self.num_edges;
    self.num_edges += 1;
    assert_eq!(t.role, Role::Constant, "Parameters must be constants");
    self.init_values.push(Some(t));
    e
  }

  pub fn add_gkr_node(&mut self, inps: Vec<EdgeId>, basicblock: BasicBlockType) -> Vec<EdgeId> {
    let nid = self.nodes.len();
    let eid = self.num_edges;
    let outs: Vec<EdgeId> = (eid..eid + basicblock.out_arity()).collect();
    self.nodes.push(Node {
      id: nid,
      kind: basicblock,
      inputs: inps,
      outputs: outs.clone(),
    });
    self.num_edges += outs.len();
    outs
  }

  pub fn add_nonneg_node(&mut self, a: EdgeId) {
    let nid = self.nodes.len();
    let nonneg_basicblock = BasicBlockType::NonNegative(NonNegative {
      table_size_log: *TABLE_SIZE_LOG,
    });
    let _ = self.add_gkr_node(vec![a], nonneg_basicblock);
    self.init_values.push(Some(Witness::new_wo_data(vec![1], DataType::Float, 0, Role::Auxiliary)));
    self.range.push(nid);
  }

  /// Wire a ZeroCheck node: asserts MLE of edge `a` is identically zero.
  /// The input edge is exposed as a sink; `Dag::prove`/`verify` draw a fresh
  /// challenge for it via the `output_ports` mechanism and the verifier
  /// rejects unless the claim's eval equals zero.
  pub fn add_zero_check_node(&mut self, a: EdgeId) {
    let nid = self.nodes.len();
    let zc_basicblock = BasicBlockType::ZeroCheck(ZeroCheck);
    let _ = self.add_gkr_node(vec![a], zc_basicblock);
    self.zero_check.push(nid);
  }

  /// One-sided clamp: y = max(x, -C), enforced by the gadget
  ///   NonNeg(y - x)  ∧  NonNeg(y + C)  ∧  ZeroCheck((y - x)(y + C))
  /// Used to cap exp inputs within [-C, ∞) so ExpHelper's K_BITS range is
  /// always sufficient (masked scores of roughly -100·SF get clamped to -C).
  pub fn clamp_lower(&mut self, x: EdgeId) -> EdgeId {
    let x_w = self.init_values[x].as_ref().unwrap();
    let shape = x_w.shape.clone();
    let sf = x_w.sf;
    let data_type = x_w.data_type;
    let n_flat: usize = shape.iter().map(|s| next_pow(*s as u32) as usize).product();
    let flat_shape = vec![n_flat];

    // Zero out padding of x so delta/u are exactly 0 in padding (upstream ops
    // can leave garbage there, which would fail the NonNeg range checks).
    let x = self.mask(x, shape.clone());

    // Pick C = 14 * round(ln2 * SF): one k-step below the K_BITS=4 boundary
    // (max k = 15), so after clamping every exp input is safely within range.
    let sf_val: f64 = (1u64 << sf) as f64;
    let ln2_sf = (2.0_f64.ln() * sf_val).round() as u64;
    let c_val: u64 = 14 * ln2_sf;

    // y = ClampLower(x)
    let clamp_bb = BasicBlockType::ClampLower(ClampLower { c: c_val });
    let y = self.add_gkr_node(vec![x], clamp_bb)[0];
    self.init_values.push(Some(Witness::new_wo_data(shape.clone(), data_type, sf, Role::Output)));

    // delta = y - x, range-check delta ≥ 0
    let delta = self.sub(y, x)[0];
    let delta_flat = self.change_shape(delta, flat_shape.clone());
    self.add_nonneg_node(delta_flat);

    // u = y + C, range-check u ≥ 0
    let n_real: usize = shape.iter().product();
    let c_vals: Vec<F> = (0..n_real).map(|_| F::from(c_val as u32)).collect();
    let c_arr = ArrayD::from_shape_vec(shape.clone(), c_vals).unwrap();
    let c_pad = pad_to_pow_of_two(&c_arr, &<F as CryptoField>::zero());
    let c_col_major: Vec<_> = c_pad.clone().view().reversed_axes().iter().cloned().collect();
    let c_witness = Witness::new(shape.clone(), c_col_major, data_type, sf, Role::Constant);
    let c_edge = self.param(c_witness);
    let u = self.add(y, c_edge)[0];
    let u_flat = self.change_shape(u, flat_shape.clone());
    self.add_nonneg_node(u_flat);

    // z = delta * u (elementwise), then ZeroCheck(z)
    let z = self.einsum("a,a->a".to_string(), vec![delta_flat, u_flat], false)[0];
    self.add_zero_check_node(z);

    y
  }

  pub fn change_shape(&mut self, a: EdgeId, shape: Vec<usize>) -> EdgeId {
    // WARNING: This function does not check if the shape is valid. Use with caution.
    let change_shape_basicblock = BasicBlockType::ChangeShape(ChangeShape { new_shape: shape.clone() });
    let outs = self.add_gkr_node(vec![a], change_shape_basicblock);
    self.init_values.push(Some(Witness::new_wo_data(
      shape,
      self.init_values[a].as_ref().unwrap().data_type,
      self.init_values[a].as_ref().unwrap().sf,
      Role::Output,
    )));
    outs[0]
  }

  // ---- DSL ----
  pub fn reshape(&mut self, a: EdgeId, shape: Vec<usize>) -> Vec<EdgeId> {
    let witness = self.init_values[a].as_ref().unwrap().clone();
    let original_shape = witness.shape.clone();

    // currently only support reshape between (batch_size, seq_len, head_num, head_dim) and (batch_size, seq_len, head_num * head_dim)
    let out = if shape.len() > original_shape.len() {
      assert!(
        shape[shape.len() - 1] * shape[shape.len() - 2] == original_shape[original_shape.len() - 1],
        "Invalid shape"
      );
      let a = self.change_shape(
        a,
        vec![original_shape[0], original_shape[1], shape[shape.len() - 1], shape[shape.len() - 2]],
      );
      self.einsum("bsdh->bshd".to_string(), vec![a], false)
    } else if shape.len() < original_shape.len() {
      assert!(
        shape[shape.len() - 1] == original_shape[original_shape.len() - 1] * original_shape[original_shape.len() - 2],
        "Invalid shape"
      );
      let o = self.einsum("bshd->bsdh".to_string(), vec![a], false);
      let o = self.change_shape(o[0], shape);
      vec![o]
    } else {
      panic!("Not supported yet");
    };

    out
  }

  pub fn mask(&mut self, a: EdgeId, raw_mask_shape: Vec<usize>) -> EdgeId {
    let s = letters(raw_mask_shape.len());
    let val_num = &raw_mask_shape.iter().fold(1, |acc, x| acc * x);
    let vals = (0..*val_num).map(|_| F::from(1)).collect();
    let val_arr = ArrayD::from_shape_vec(raw_mask_shape.clone(), vals).unwrap();
    let pad_val_arr = pad_to_pow_of_two(&val_arr, &<F as CryptoField>::zero());
    let col_major_output: Vec<_> = pad_val_arr.clone().view().reversed_axes().iter().cloned().collect();
    let mask = Witness::new(raw_mask_shape, col_major_output, DataType::Float, 0, Role::Constant);
    let e = self.param(mask);
    let out = self.einsum(format!("{},{}->{}", s, s, s), vec![a, e], false);
    out[0]
  }

  // this broadcast add is not correct when the broadcast shape dim is not 2^n, remember to add a mask to ensure the output is correct
  pub fn add(&mut self, a: EdgeId, b: EdgeId) -> Vec<EdgeId> {
    let add_basicblock = BasicBlockType::Add(Add);

    assert!(
      self.init_values[a].is_some() && self.init_values[b].is_some(),
      "Inputs must be initialized"
    );
    let inps_values = vec![self.init_values[a].as_ref().unwrap(), self.init_values[b].as_ref().unwrap()];
    let out_value = if self.init_values[a].as_ref().unwrap().data.is_none() || self.init_values[b].as_ref().unwrap().data.is_none() {
      let shape = broadcast_shape(&self.init_values[a].as_ref().unwrap().shape, &self.init_values[b].as_ref().unwrap().shape).unwrap();
      let sf = self.init_values[a].as_ref().unwrap().sf;
      let data_type = self.init_values[a].as_ref().unwrap().data_type;
      Witness::new_wo_data(shape, data_type, sf, Role::Output)
    } else {
      let mut out = add_basicblock.run(inps_values.as_slice()).first().unwrap().to_owned();
      out.role = Role::Constant;
      out
    };
    self.init_values.push(Some(out_value));

    self.add_gkr_node(vec![a, b], add_basicblock)
  }

  // this broadcast sub is not correct when the broadcast shape dim is not 2^n, remember to add a mask to ensure the output is correct
  pub fn sub(&mut self, a: EdgeId, b: EdgeId) -> Vec<EdgeId> {
    let sub_basicblock = BasicBlockType::Sub(Sub);

    assert!(
      self.init_values[a].is_some() && self.init_values[b].is_some(),
      "Inputs must be initialized"
    );
    let inps_values = vec![self.init_values[a].as_ref().unwrap(), self.init_values[b].as_ref().unwrap()];
    let out_value = if self.init_values[a].as_ref().unwrap().data.is_none() || self.init_values[b].as_ref().unwrap().data.is_none() {
      let shape = broadcast_shape(&self.init_values[a].as_ref().unwrap().shape, &self.init_values[b].as_ref().unwrap().shape).unwrap();
      let sf = self.init_values[a].as_ref().unwrap().sf;
      let data_type = self.init_values[a].as_ref().unwrap().data_type;
      Witness::new_wo_data(shape, data_type, sf, Role::Output)
    } else {
      let mut out = sub_basicblock.run(inps_values.as_slice()).first().unwrap().to_owned();
      out.role = Role::Constant;
      out
    };
    self.init_values.push(Some(out_value));

    self.add_gkr_node(vec![a, b], sub_basicblock)
  }

  pub fn einsum(&mut self, equation: String, inputs: Vec<EdgeId>, scale_back: bool) -> Vec<EdgeId> {
    let einsum_basicblock = BasicBlockType::Einsum(Einsum { equation: equation.clone() });
    let input_shapes = inputs.iter().map(|&i| self.init_values[i].as_ref().unwrap().shape.clone()).collect::<Vec<Vec<usize>>>();
    let mut shape_map = HashMap::new();
    let input_symbols = equation.split("->").nth(0).unwrap().split(",").map(|s| s.trim()).collect::<Vec<&str>>();
    for (i, symbols) in input_symbols.iter().enumerate() {
      for (j, c) in symbols.chars().enumerate() {
        shape_map.insert(c.to_string(), input_shapes[i][j]);
      }
    }
    let output_shape =
      equation.split("->").nth(1).unwrap().to_string().chars().map(|c| *shape_map.get(&c.to_string()).unwrap()).collect::<Vec<usize>>();
    let output_data_type = self.init_values[inputs[0]].as_ref().unwrap().data_type;
    let input_sf = inputs.iter().map(|&i| self.init_values[i].as_ref().unwrap().sf).sum::<usize>();
    let output_sf = self.init_values[inputs[0]].as_ref().unwrap().sf;
    let mut outs = self.add_gkr_node(inputs.clone(), einsum_basicblock);
    self.init_values.push(Some(Witness::new_wo_data(output_shape.clone(), output_data_type, input_sf, Role::Output)));
    if scale_back {
      outs = self.scale(outs[0], input_sf, output_sf);
    }
    outs
  }

  pub fn sigmoid_const(&mut self, a: EdgeId) -> Vec<EdgeId> {
    let sigmoid_const_basicblock = BasicBlockType::SigmoidConst(SigmoidConst);
    assert!(self.init_values[a].is_some(), "Input must be initialized");
    let inp_value = self.init_values[a].as_ref().unwrap();
    let shape = inp_value.shape.clone();
    let sf = inp_value.sf;
    let data_type = inp_value.data_type;
    let out_value = Witness::new_wo_data(shape, data_type, sf, Role::Output);
    self.init_values.push(Some(out_value));
    self.add_gkr_node(vec![a], sigmoid_const_basicblock)
  }

  pub fn sigmoid(&mut self, a: EdgeId) -> Vec<EdgeId> {
    let sigmoid_c = self.sigmoid_const(a)[0];
    let scores = self.add(a, sigmoid_c)[0];
    let scores = self.exp(scores)[0];
    vec![scores]
  }

  pub fn scale(&mut self, a: EdgeId, input_sf: usize, output_sf: usize) -> Vec<EdgeId> {
    let nid = self.nodes.len();
    let shape = self.init_values[a].as_ref().unwrap().shape.clone();
    let data_type = self.init_values[a].as_ref().unwrap().data_type;
    let scale_basicblock = if input_sf > output_sf {
      BasicBlockType::ScaleDown(ScaleDown { input_sf, output_sf })
    } else {
      BasicBlockType::ScaleUp(ScaleUp { input_sf, output_sf })
    };
    self.init_values.push(Some(Witness::new_wo_data(shape.clone(), data_type, output_sf, Role::Output)));
    self.init_values.push(Some(Witness::new_wo_data(vec![1], data_type, 0, Role::Auxiliary)));
    self.range.push(nid);
    self.add_gkr_node(vec![a], scale_basicblock)
  }

  pub fn nonstructured_exp(&mut self, a: EdgeId, t: EdgeId) -> Vec<EdgeId> {
    let shape = self.init_values[a].as_ref().unwrap().shape.clone();
    let data_type = self.init_values[a].as_ref().unwrap().data_type;

    let exp_basicblock = BasicBlockType::NonStructuredExp(NonStructuredExp);
    self.init_values.push(Some(Witness::new_wo_data(shape.clone(), data_type, *SF_LOG, Role::Output)));
    self.init_values.push(Some(Witness::new_wo_data(vec![1], data_type, 0, Role::Auxiliary)));
    self.add_gkr_node(vec![a, t], exp_basicblock)
  }

  pub fn exp(&mut self, a: EdgeId) -> Vec<EdgeId> {
    // Clamp input so ExpHelper's K_BITS=4 decomposition is always valid
    // (masked-attention scores ≈ -100·SF would otherwise overflow k).
    let a = self.clamp_lower(a);
    let nid = self.nodes.len();
    let shape = self.init_values[a].as_ref().unwrap().shape.clone();
    let flat_shape = vec![shape.iter().map(|s| next_pow(*s as u32) as usize).product()];
    let data_type = self.init_values[a].as_ref().unwrap().data_type;

    let exp_basicblock = BasicBlockType::ExpHelper(ExpHelper);
    self.init_values.push(Some(Witness::new_wo_data(shape.clone(), data_type, *SF_LOG, Role::Output)));
    self.init_values.push(Some(Witness::new_wo_data(vec![1], data_type, 0, Role::Auxiliary)));
    self.range.push(nid);
    let outs = self.add_gkr_node(vec![a], exp_basicblock); // x --> k * (-ln(2)*sf) + r
    let mut r = outs[0]; // dense poly
    let k = outs[1]; // sparse poly

    r = self.scale(r, *SF_LOG as usize, 15)[0];

    // A. compute 2^(-k)
    let nid = self.nodes.len();
    self.two_pow.push(nid);
    let two_pow_basicblock = BasicBlockType::TwoPow(TwoPow);
    let mut two_pow_out = self.add_gkr_node(vec![k], two_pow_basicblock)[0];
    self.init_values.push(Some(Witness::new_wo_data(shape.clone(), data_type, 15, Role::Output)));
    two_pow_out = self.change_shape(two_pow_out, flat_shape.clone());

    // B. compute exp(r)
    let val_num = &shape.iter().fold(1, |acc, x| acc * x);

    // B1. compute 1/6
    let vals_one_sixth = (0..*val_num).map(|_| F::from(5461)).collect(); // 2^15 / 6
    let vals_one_sixth = ArrayD::from_shape_vec(shape.clone(), vals_one_sixth).unwrap();
    let pad_vals_one_sixth = pad_to_pow_of_two(&vals_one_sixth, &<F as CryptoField>::zero());
    let col_major_one_sixth: Vec<_> = pad_vals_one_sixth.clone().view().reversed_axes().iter().cloned().collect();
    let one_sixth = Witness::new(flat_shape.clone(), col_major_one_sixth, DataType::Float, 15, Role::Constant);
    let one_sixth = self.param(one_sixth);

    // B2. compute 1/2
    let vals_half = (0..*val_num).map(|_| F::from(16384)).collect(); // 2^15 / 2
    let vals_half = ArrayD::from_shape_vec(shape.clone(), vals_half).unwrap();
    let pad_vals_half = pad_to_pow_of_two(&vals_half, &<F as CryptoField>::zero());
    let col_major_half: Vec<_> = pad_vals_half.clone().view().reversed_axes().iter().cloned().collect();
    let half = Witness::new(flat_shape.clone(), col_major_half, DataType::Float, 15, Role::Constant);
    let half = self.param(half);

    // B3. compute 1 (at sf=15, 1.0 is represented as 2^15 = 32768)
    let vals_one = (0..*val_num).map(|_| F::from(1u32 << 15)).collect();
    let vals_one = ArrayD::from_shape_vec(shape.clone(), vals_one).unwrap();
    let pad_vals_one = pad_to_pow_of_two(&vals_one, &<F as CryptoField>::zero());
    let col_major_one: Vec<_> = pad_vals_one.clone().view().reversed_axes().iter().cloned().collect();
    let one = Witness::new(flat_shape.clone(), col_major_one, DataType::Float, 15, Role::Constant);
    let one = self.param(one);

    // B4. compute exp(r) by Taylor series
    r = self.change_shape(r, flat_shape);
    let r_square = self.einsum("a,a->a".to_string(), vec![r, r], true);
    let r_one_sixth = self.einsum("a,a->a".to_string(), vec![r, one_sixth], true);
    let r_one_sixth_plus_half = self.add(r_one_sixth[0], half)[0];
    let deg_two_plus_deg_three = self.einsum("a,a->a".to_string(), vec![r_one_sixth_plus_half, r_square[0]], true);
    let deg_one_plus_deg_two_plus_deg_three = self.add(deg_two_plus_deg_three[0], r);
    let exp_r = self.add(deg_one_plus_deg_two_plus_deg_three[0], one)[0];

    // C. compute 2^(-k) * exp(r)
    let exp_x = self.einsum("a,a->a".to_string(), vec![two_pow_out, exp_r], false);
    let exp_x = self.scale(exp_x[0], 30, *SF_LOG as usize)[0];
    let exp = self.change_shape(exp_x, shape);
    vec![exp]
  }

  /// Compile once: build consumers/producers, ports, and topological order.
  /// Returns (Dag, init_edge_values) where init_edge_values[e] is Some(tensor)
  /// for params/constants created via `param()`.
  pub fn compile(self) -> (Dag, Vec<Vec<Witness<F>>>) {
    let DagBuilder {
      nodes,
      num_edges,
      init_values,
      range,
      two_pow,
      zero_check,
    } = self;

    // edge -> consumers
    let mut consumers: Vec<Vec<NodeId>> = vec![Vec::new(); num_edges];
    for n in &nodes {
      for &e in &n.inputs {
        consumers[e].push(n.id);
      }
    }

    // produced edges + producers map
    let mut produced = vec![false; num_edges];
    let mut producers = vec![None; num_edges];
    for n in &nodes {
      for &e in &n.outputs {
        produced[e] = true;
        producers[e] = Some(n.id);
      }
    }

    let input_ports: Vec<EdgeId> = (0..num_edges).filter(|&e| !produced[e] && init_values[e].as_ref().unwrap().role == Role::Input).collect();
    let mut output_ports: Vec<EdgeId> = (0..num_edges).filter(|&e| consumers[e].is_empty()).collect();
    output_ports.extend(range.iter().filter(|&n| matches!(nodes[*n].kind, BasicBlockType::NonNegative(_))).map(|n| nodes[*n].inputs[0]));
    output_ports.extend(zero_check.iter().map(|n| nodes[*n].inputs[0]));

    // in-degree = #inputs that come from produced edges (ignore graph inputs/params)
    let mut indeg = vec![0usize; nodes.len()];
    for n in &nodes {
      indeg[n.id] = n.inputs.iter().filter(|&&e| produced[e]).count();
    }

    // adjacency: node -> downstream nodes via outputs' consumers
    let mut outgoing: Vec<Vec<NodeId>> = vec![Vec::new(); nodes.len()];
    for n in &nodes {
      for &e in &n.outputs {
        for &v in &consumers[e] {
          outgoing[n.id].push(v);
        }
      }
    }

    // Kahn topo
    let mut q: Vec<NodeId> = indeg.iter().enumerate().filter_map(|(i, &d)| (d == 0).then_some(i)).collect();
    let mut topo = Vec::with_capacity(nodes.len());
    while let Some(u) = q.pop() {
      topo.push(u);
      for &v in &outgoing[u] {
        indeg[v] -= 1;
        if indeg[v] == 0 {
          q.push(v);
        }
      }
    }
    assert_eq!(topo.len(), nodes.len(), "graph has a cycle or disconnected inputs");

    // --------- Build alias view ----------
    let mut alias_to_edge: Vec<EdgeId> = Vec::new();
    let mut alias_to_consumer: Vec<NodeId> = Vec::new();
    let mut alias_input_slot: Vec<usize> = Vec::new();
    let mut edge_aliases: Vec<Vec<AliasId>> = vec![Vec::new(); num_edges];

    for (nid, node) in nodes.iter().enumerate() {
      for (slot, &e) in node.inputs.iter().enumerate() {
        let aid = AliasId(alias_to_edge.len());
        alias_to_edge.push(e);
        alias_to_consumer.push(nid);
        alias_input_slot.push(slot);
        edge_aliases[e].push(aid);
      }
    }

    let dag = Dag {
      nodes,
      num_edges,
      topo,
      range,
      two_pow,
      zero_check,
      consumers,
      producers,
      input_ports,
      output_ports,
      edge_aliases,
      alias_to_edge,
      alias_to_consumer,
      alias_input_slot,
    };

    let init_values = init_values.iter().map(|value| vec![value.as_ref().unwrap().clone()]).collect::<Vec<Vec<Witness<F>>>>();

    (dag, init_values)
  }

  /// Compose via a recipe: f(&mut DagBuilder, EdgeId) -> EdgeId
  pub fn pipe<Fn>(&mut self, inlet: &[EdgeId], f: Fn) -> Vec<EdgeId>
  where
    Fn: FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId>,
  {
    f(self, inlet)
  }
}
