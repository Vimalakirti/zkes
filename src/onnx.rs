use crate::dag::builder::DagBuilder;
use crate::dag::{DataType, Role, Witness};
use crate::util::poly::CryptoField;
use crate::util::shape::pad_to_pow_of_two;
use crate::{SF_FLOAT, SF_LOG};
use icicle_core::field::Field;
use ndarray::ArrayD;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;
use tract_onnx::pb;
use tract_onnx::pb::AttributeProto;
use tract_onnx::pb::{tensor_proto::DataType as OnnxDataType, type_proto::Tensor};
use tract_onnx::prelude::{DatumType, Framework};
use tract_onnx::tensor::load_tensor;

// This function is used for generating fake inputs for onnx models
// Fake inputs are random field (i.e., Fr) elements whose shapes and types match those described in the input tensors of an ONNX model.
// Generating these when loading an ONNX file saves us from creating different input tensors ourselves when testing new ONNX.
// It is only for testing purposes
pub fn generate_fake_inputs_for_onnx<F: CryptoField + 'static>(filename: &str) -> Vec<ArrayD<F>> {
  let onnx = tract_onnx::onnx();
  let onnx_graph = onnx.proto_model_for_path(filename).unwrap().graph.unwrap();

  let mut inputs = vec![];

  for onnx_input in onnx_graph.input.iter() {
    let tract_onnx::pb::type_proto::Value::TensorType(t) = onnx_input.r#type.as_ref().unwrap().value.as_ref().unwrap();
    let shape = get_shape_from_onnx_tensor(t);

    let input = generate_fake_tensor(t.elem_type(), shape);
    let input = pad_to_pow_of_two(&input, &<F as CryptoField>::zero());
    inputs.push(input);
  }
  inputs
}

pub fn generate_fake_tensor<F: CryptoField + 'static>(dtype: OnnxDataType, shape: Vec<usize>) -> ArrayD<F> {
  eprintln!("\x1b[93mWARNING\x1b[0m: Generating fake tensor for ONNX model. This is only for testing purposes.");
  let mut rng = StdRng::from_entropy();
  let val_num = shape.iter().fold(1, |acc, x| acc * x);
  let input = match dtype {
    OnnxDataType::Float | OnnxDataType::Float16 | OnnxDataType::Double => (0..val_num).map(|_| F::from(rng.gen_range(0..2))).collect(),
    OnnxDataType::Int8 | OnnxDataType::Int16 | OnnxDataType::Int32 | OnnxDataType::Int64 => (0..val_num).map(|_| F::from(1)).collect(),
    OnnxDataType::Uint8 | OnnxDataType::Uint16 | OnnxDataType::Uint32 | OnnxDataType::Uint64 => (0..val_num).map(|_| F::from(1)).collect(),
    OnnxDataType::Bool => (0..val_num).map(|_| F::from(rng.gen_range(0..2))).collect(),
    _ => panic!("Unsupported constant type: {:?}", dtype),
  };
  ArrayD::from_shape_vec(shape, input).unwrap()
}

// This function is used for getting the shape of an ONNX input tensor
pub fn get_shape_from_onnx_tensor(tensor: &Tensor) -> Vec<usize> {
  tensor
    .shape
    .as_ref()
    .unwrap()
    .dim
    .iter()
    .map(|x| {
      if let tract_onnx::pb::tensor_shape_proto::dimension::Value::DimValue(x) = x.value.as_ref().unwrap() {
        *x as usize
      } else {
        1 //panic!("Unknown dimension")
      }
    })
    .collect::<Vec<_>>()
}

// Converts ints for the ONNX DataType enum into Our DataType enum
// https://docs.rs/tract-onnx/latest/tract_onnx/pb/tensor_proto/enum.DataType.html
pub fn onnx_datatype_to_datatype(t: i32) -> DataType {
  match t {
    2 | 3 | 4 | 5 | 6 | 7 | 12 | 13 => DataType::Int,
    1 | 10 | 11 => DataType::Float,
    9 => DataType::Bool,
    _ => panic!("DatumType {:?} not supported", t),
  }
}

// This function is used for parsing the inputs of onnx models
fn parse_onnx_inputs<F: CryptoField + 'static>(onnx_graph: &pb::GraphProto, builder: &mut DagBuilder<F>, hashmap: &mut HashMap<String, usize>) {
  for i in onnx_graph.input.iter() {
    let tract_onnx::pb::type_proto::Value::TensorType(t) = i.r#type.as_ref().unwrap().value.as_ref().unwrap();
    let edge_id = builder.input(get_shape_from_onnx_tensor(t), onnx_datatype_to_datatype(t.elem_type));
    hashmap.insert(i.name.clone(), edge_id);
  }
}

// This function is used for parsing the constants of onnx models
fn parse_onnx_constants<'a, F: CryptoField + 'static>(
  onnx_graph: &'a pb::GraphProto,
  builder: &mut DagBuilder<F>,
  hashmap: &mut HashMap<String, usize>,
) {
  let onnx = tract_onnx::onnx();
  let constants = onnx_graph.initializer.iter().map(|tensor| (tensor.name.clone(), tensor)).chain(
    onnx_graph
      .node
      .iter()
      .filter(|node| node.op_type == "Constant")
      .map(|node| (node.output[0].clone(), node.attribute[0].t.as_ref().unwrap())),
  );

  for (name, tensor) in constants {
    let (tensor, data_type) = if tensor.data_location.is_none() {
      let tensor = load_tensor(&*onnx.provider, tensor, None).unwrap();
      let data_type = match tensor.datum_type() {
        DatumType::F32 => DataType::Float,
        DatumType::I64 => DataType::Int,
        DatumType::Bool => DataType::Bool,
        _ => panic!("Unsupported constant type: {:?}", tensor.datum_type()),
      };
      let tensor = match tensor.datum_type() {
        DatumType::F32 => {
          let tensor = tensor.into_array::<f32>().unwrap();
          Ok(tensor.map(|x| {
            // handle the case where the constant is very close to zero (i.e., epsilon to prevent division by zero)
            if *x < 1e-10 && *x > 0.0 {
              // the reason we use 1 here is because it is the smallest positive value that can be represented in the field
              return F::from(1);
            }
            let mut y = (*x * *SF_FLOAT).round();
            y = y.clamp(-(1 << 15) as f32, (1 << 15) as f32);
            let value = if y < 0.0 {
              <F as CryptoField>::zero() - F::from((-y) as u32)
            } else {
              F::from(y as u32)
            };
            value
          }))
        }
        DatumType::I64 => {
          let tensor = tensor.into_array::<i64>().unwrap();
          Ok(tensor.map(|x| {
            if *x < 0 {
              <F as CryptoField>::zero() - F::from((-x) as u32)
            } else {
              F::from(*x as u32)
            }
          }))
        }
        DatumType::Bool => {
          let tensor = tensor.into_array::<bool>().unwrap();
          Ok(tensor.map(|x| F::from(*x as u32)))
        }
        _ => Err(format!("Unsupported constant type: {:?}", tensor.datum_type())),
      }
      .unwrap();
      (tensor, data_type)
    } else {
      // if the data_location is not None, we generate fake weights for now.
      // TODO: we can add support for loading weights from file later
      let shape: Vec<usize> = tensor.dims.iter().map(|&i| i as usize).collect();
      let dtype = OnnxDataType::from_i32(tensor.data_type).unwrap().into();
      let data_type = onnx_datatype_to_datatype(tensor.data_type);
      let tensor = generate_fake_tensor(dtype, shape);
      (tensor, data_type)
    };
    let pad_tensor = pad_to_pow_of_two(&tensor, &<F as CryptoField>::zero());
    let edge_id = builder.param(Witness::new(
      tensor.shape().to_vec(),
      pad_tensor.reversed_axes().into_raw_vec(), // reverse axes to store the tensor in column-major order
      data_type,
      if data_type == DataType::Float { *SF_LOG as usize } else { 0 },
      Role::Constant,
    ));
    hashmap.insert(name.clone(), edge_id);
  }
}

// This function is used for processing a layer of onnx models.
fn process_onnx_node<F: CryptoField + 'static>(node: &pb::NodeProto, graph: &mut DagBuilder<F>, hashmap: &mut HashMap<String, usize>) {
  let node_input = node.input.iter().map(|x| hashmap.get(x).unwrap()).collect::<Vec<_>>();
  let node_attributes: Vec<&AttributeProto> = node.attribute.iter().map(|x| x).collect();
  let dag_outputs = match node.op_type.as_str() {
    "Add" => graph.add(*node_input[0], *node_input[1]),
    _ => panic!("Unsupported onnx operation: {}", node.op_type),
  };
  node.output.iter().zip(dag_outputs).for_each(|(output, dag_output)| {
    hashmap.insert(output.clone(), dag_output);
  });
}

// This function is used for loading onnx models and returning the DAG builder
pub fn load_onnx<F: CryptoField + 'static>(filename: &str) -> DagBuilder<F> {
  let onnx = tract_onnx::onnx();
  let onnx_graph = onnx.proto_model_for_path(filename).unwrap().graph.unwrap();

  let mut onnx_edges_to_dag_edges = HashMap::new();
  let mut g = DagBuilder::new();

  parse_onnx_inputs(&onnx_graph, &mut g, &mut onnx_edges_to_dag_edges);
  parse_onnx_constants(&onnx_graph, &mut g, &mut onnx_edges_to_dag_edges);

  println!("onnx_edges_to_dag_edges: {:?}", onnx_edges_to_dag_edges);

  for node in onnx_graph.node.iter().filter(|node| node.op_type.as_str() != "Constant") {
    process_onnx_node(node, &mut g, &mut onnx_edges_to_dag_edges);
  }

  g
}
