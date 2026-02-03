/*
 * Shape utilities:
 * The functions are used for shape-related operations, such as
 * slicing and padding arrays.
 */
use crate::util::poly::CryptoField;
use ndarray::{ArrayD, Axis, Dimension, IxDyn, Slice, SliceInfo};

// slice the arr with the given indices. But this function is not used in the codebase currently.
#[allow(dead_code)]
pub fn slice_nd_array<F: CryptoField + 'static>(arr: ArrayD<F>, indices: &[usize]) -> ArrayD<F> {
  // Create slices from the indices
  let slices: Vec<_> = indices.iter().map(|&i| (0..i).into()).collect();

  // Convert slices into a SliceInfo instance
  let slice_info = unsafe { SliceInfo::<_, IxDyn, IxDyn>::new(slices).unwrap() };

  // Slice the array
  arr.slice_move(slice_info)
}

pub fn flatten_last_dimension<F: CryptoField + 'static>(arr: &ArrayD<F>) -> ArrayD<Vec<F>> {
  let shape = arr.shape().to_vec();
  let new_shape = IxDyn(&shape[..shape.len() - 1]);

  ArrayD::from_shape_fn(new_shape, |idx| {
    let mut full_idx = idx.as_array_view().to_vec();
    full_idx.push(0);
    let slice = arr.slice_each_axis(|ax| {
      if ax.axis.index() < full_idx.len() - 1 {
        ndarray::Slice::from(full_idx[ax.axis.index()]..=full_idx[ax.axis.index()])
      } else {
        ndarray::Slice::from(..)
      }
    });
    slice.to_owned().into_raw_vec()
  })
}

// Pads each dimension of input by the corresponding amount in padding on both ends.
pub fn pad<G: Clone>(input: &ArrayD<G>, padding: &Vec<[usize; 2]>, pad_val: &G) -> ArrayD<G> {
  let tmp = input.into_iter().collect();
  let input = ArrayD::from_shape_vec(input.raw_dim(), tmp).unwrap();
  assert_eq!(input.ndim(), padding.len());
  let mut padded_shape = input.raw_dim();
  for (ax, (&ax_len, &[pad_lo, pad_hi])) in input.shape().iter().zip(padding).enumerate() {
    padded_shape[ax] = ax_len + pad_lo + pad_hi;
  }

  let mut padded = ArrayD::from_elem(padded_shape, pad_val);
  let padded_dim = padded.raw_dim();
  {
    // Select portion of padded array that needs to be copied from the
    // original array.
    let mut orig_portion = padded.view_mut();
    for (ax, &[pad_lo, pad_hi]) in padding.iter().enumerate() {
      orig_portion.slice_axis_inplace(Axis(ax), Slice::from(pad_lo as isize..padded_dim[ax] as isize - (pad_hi as isize)));
    }
    // Copy the data from the original array.
    orig_portion.assign(&input);
  }

  let dim = padded.raw_dim();
  let tmp = padded.into_iter().map(|x| x.clone()).collect();
  let padded = ArrayD::from_shape_vec(dim, tmp).unwrap();

  padded
}

pub fn pad_to_pow_of_two<G: Clone>(input: &ArrayD<G>, pad_val: &G) -> ArrayD<G> {
  let padding: Vec<_> = input.shape().iter().map(|x| [0, x.next_power_of_two() - x]).collect();
  pad(&input, &padding, &pad_val)
}

pub fn broadcastDims(dims: &Vec<&Vec<usize>>, N: usize) -> Vec<usize> {
  let len = dims.iter().map(|x| x.len()).max().unwrap();
  (0..len - N)
    .map(|i| dims.iter().map(|dim| if dim.len() >= len - i { dim[i + dim.len() - len] } else { 1 }).max().unwrap())
    .collect()
}

/// Infer the broadcasted shape of two input shapes (NumPy-style).
/// - Shapes are compared from the rightmost axis to the left.
/// - Two axes are compatible if they are equal or one of them is 1.
/// - Zero-length axes:
///     * (0, x) is compatible with (0, x) -> 0
///     * (0, x) is compatible with (1, x) -> 0
///     * (0, x) is NOT compatible with (n>1, x)
/// Returns `Ok(result_shape)` or `Err` if not broadcastable.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
  let na = a.len();
  let nb = b.len();
  let n = na.max(nb);

  let mut out_rev = Vec::with_capacity(n);

  for i in 0..n {
    let da = if i < na { a[na - 1 - i] } else { 1 };
    let db = if i < nb { b[nb - 1 - i] } else { 1 };

    // Handle zero-length axes explicitly (NumPy-like behavior)
    let d = if da == 0 || db == 0 {
      match (da, db) {
        (0, 0) => 0,
        (0, 1) | (1, 0) => 0,
        // 0 with >1 is incompatible
        (0, n) | (n, 0) if n > 1 => return Err(format!("Incompatible dims at axis -{} (from right): {} vs {}", i + 1, da, db)),
        _ => unreachable!(),
      }
    } else {
      // Standard broadcasting
      if da == db || da == 1 || db == 1 {
        da.max(db)
      } else {
        return Err(format!("Incompatible dims at axis -{} (from right): {} vs {}", i + 1, da, db));
      }
    };

    out_rev.push(d);
  }

  out_rev.reverse();
  Ok(out_rev)
}

/// Return output axes that `x` is matched to (not broadcasted, not zero axes),
/// under NumPy-style broadcasting with your zero-dim rules.
/// `y` is the other operand (only used to determine the broadcasted output shape).
pub fn matched_axes(x: &[usize], out: &[usize]) -> Result<Vec<usize>, String> {
  let nx = x.len();
  let no = out.len();

  if nx > no {
    return Err("x has more dimensions than broadcasted output".into());
  }

  let mut matched = Vec::new();

  // right-align, compare from rightmost axis
  for i in 0..no {
    let out_axis = no - 1 - i;
    let d_out = out[out_axis];
    let d_x = if i < nx { x[nx - 1 - i] } else { 1 };

    // matched means exact equality, excluding zero-length output axes
    if d_out != 0 && d_x == d_out && matched.len() < nx {
      matched.push(out_axis);
    }
  }

  matched.reverse();
  Ok(matched)
}
