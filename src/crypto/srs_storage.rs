use crate::crypto::polycommit::{kzh3::KZH3SRS, PairingTrait};
use std::fs::{create_dir_all, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

#[cfg(feature = "arkworks")]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

#[cfg(feature = "icicle")]
use bincode;

/// Store KZH3 SRS to a file
/// File path format: {maximum_degree}.srs where maximum_degree is the parameter
#[cfg(feature = "arkworks")]
pub fn store_kzh3_srs<P: PairingTrait>(srs: &KZH3SRS<P>, maximum_degree: usize) -> Result<(), Box<dyn std::error::Error>>
where
  P::G1Affine: CanonicalSerialize,
  P::G2Affine: CanonicalSerialize,
{
  let file_path = format!("{}.srs", maximum_degree);

  // Create directory if it doesn't exist
  if let Some(parent) = Path::new(&file_path).parent() {
    create_dir_all(parent)?;
  }

  let file = File::create(&file_path)?;
  let mut writer = BufWriter::new(file);

  srs.serialize_compressed(&mut writer)?;
  writer.flush()?;

  println!("KZH3 SRS stored to: {}", file_path);
  Ok(())
}

/// Load KZH3 SRS from a file
/// File path format: {maximum_degree}.srs where maximum_degree is the parameter
#[cfg(feature = "arkworks")]
pub fn load_kzh3_srs<P: PairingTrait>(maximum_degree: usize) -> Result<KZH3SRS<P>, Box<dyn std::error::Error>>
where
  P::G1Affine: CanonicalDeserialize,
  P::G2Affine: CanonicalDeserialize,
{
  let file_path = format!("{}.srs", maximum_degree);

  if !Path::new(&file_path).exists() {
    return Err(format!("SRS file not found: {}", file_path).into());
  }

  let file = File::open(&file_path)?;
  let mut reader = BufReader::new(file);

  let srs = KZH3SRS::<P>::deserialize_compressed(&mut reader)?;

  println!("KZH3 SRS loaded from: {}", file_path);
  Ok(srs)
}

/// Store KZH3 SRS to a file (ICICLE version using proper serialization)
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
pub fn store_kzh3_srs<P: PairingTrait>(srs: &KZH3SRS<P>, maximum_degree: usize) -> Result<(), Box<dyn std::error::Error>>
where
  P::ScalarField: crate::util::poly::CryptoField,
  P::G1Affine: icicle_core::affine::Affine,
  P::G2Affine: icicle_core::affine::Affine,
  <P::G1Affine as icicle_core::affine::Affine>::BaseField: icicle_core::bignum::BigNum,
  <P::G2Affine as icicle_core::affine::Affine>::BaseField: icicle_core::bignum::BigNum,
{
  use icicle_core::bignum::BigNum;

  let file_path = format!("{}.srs", maximum_degree);

  // Create directory if it doesn't exist
  if let Some(parent) = Path::new(&file_path).parent() {
    create_dir_all(parent)?;
  }

  let file = File::create(&file_path)?;
  let mut writer = BufWriter::new(file);

  // Write basic parameters
  bincode::serialize_into(&mut writer, &srs.degree_x)?;
  bincode::serialize_into(&mut writer, &srs.degree_y)?;
  bincode::serialize_into(&mut writer, &srs.degree_z)?;

  // Serialize G1Affine points using BigNum trait
  bincode::serialize_into(&mut writer, &srs.h_xyz.len())?;
  for point in &srs.h_xyz {
    use icicle_core::affine::Affine;
    let x = Affine::x(point);
    let y = Affine::y(point);
    let x_bytes = <P::G1Affine as Affine>::BaseField::to_bytes_le(&x);
    let y_bytes = <P::G1Affine as Affine>::BaseField::to_bytes_le(&y);
    writer.write_all(&x_bytes)?;
    writer.write_all(&y_bytes)?;
  }

  bincode::serialize_into(&mut writer, &srs.h_yz.len())?;
  for point in &srs.h_yz {
    use icicle_core::affine::Affine;
    let x = Affine::x(point);
    let y = Affine::y(point);
    let x_bytes = <P::G1Affine as Affine>::BaseField::to_bytes_le(&x);
    let y_bytes = <P::G1Affine as Affine>::BaseField::to_bytes_le(&y);
    writer.write_all(&x_bytes)?;
    writer.write_all(&y_bytes)?;
  }

  bincode::serialize_into(&mut writer, &srs.h_z.len())?;
  for point in &srs.h_z {
    use icicle_core::affine::Affine;
    let x = Affine::x(point);
    let y = Affine::y(point);
    let x_bytes = <P::G1Affine as Affine>::BaseField::to_bytes_le(&x);
    let y_bytes = <P::G1Affine as Affine>::BaseField::to_bytes_le(&y);
    writer.write_all(&x_bytes)?;
    writer.write_all(&y_bytes)?;
  }

  // Serialize G2Affine points using BigNum trait
  bincode::serialize_into(&mut writer, &srs.v_x.len())?;
  for point in &srs.v_x {
    use icicle_core::affine::Affine;
    let x = Affine::x(point);
    let y = Affine::y(point);
    let x_bytes = <P::G2Affine as Affine>::BaseField::to_bytes_le(&x);
    let y_bytes = <P::G2Affine as Affine>::BaseField::to_bytes_le(&y);
    writer.write_all(&x_bytes)?;
    writer.write_all(&y_bytes)?;
  }

  bincode::serialize_into(&mut writer, &srs.v_y.len())?;
  for point in &srs.v_y {
    use icicle_core::affine::Affine;
    let x = Affine::x(point);
    let y = Affine::y(point);
    let x_bytes = <P::G2Affine as Affine>::BaseField::to_bytes_le(&x);
    let y_bytes = <P::G2Affine as Affine>::BaseField::to_bytes_le(&y);
    writer.write_all(&x_bytes)?;
    writer.write_all(&y_bytes)?;
  }

  bincode::serialize_into(&mut writer, &srs.v_z.len())?;
  for point in &srs.v_z {
    use icicle_core::affine::Affine;
    let x = Affine::x(point);
    let y = Affine::y(point);
    let x_bytes = <P::G2Affine as Affine>::BaseField::to_bytes_le(&x);
    let y_bytes = <P::G2Affine as Affine>::BaseField::to_bytes_le(&y);
    writer.write_all(&x_bytes)?;
    writer.write_all(&y_bytes)?;
  }

  // Serialize the single G2Affine point
  use icicle_core::affine::Affine;
  let x = Affine::x(&srs.v);
  let y = Affine::y(&srs.v);
  let x_bytes = <P::G2Affine as Affine>::BaseField::to_bytes_le(&x);
  let y_bytes = <P::G2Affine as Affine>::BaseField::to_bytes_le(&y);
  writer.write_all(&x_bytes)?;
  writer.write_all(&y_bytes)?;

  writer.flush()?;

  println!("KZH3 SRS stored to: {}", file_path);
  Ok(())
}

/// Load KZH3 SRS from a file (ICICLE version using proper deserialization)
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
pub fn load_kzh3_srs<P: PairingTrait>(maximum_degree: usize) -> Result<KZH3SRS<P>, Box<dyn std::error::Error>>
where
  P::ScalarField: crate::util::poly::CryptoField,
  P::G1Affine: icicle_core::affine::Affine,
  P::G2Affine: icicle_core::affine::Affine,
  <P::G1Affine as icicle_core::affine::Affine>::BaseField: icicle_core::bignum::BigNum,
  <P::G2Affine as icicle_core::affine::Affine>::BaseField: icicle_core::bignum::BigNum,
{
  use icicle_core::bignum::BigNum;
  use std::io::Read;

  let file_path = format!("{}.srs", maximum_degree);

  if !Path::new(&file_path).exists() {
    return Err(format!("SRS file not found: {}", file_path).into());
  }

  let file = File::open(&file_path)?;
  let mut reader = BufReader::new(file);

  // Read basic parameters
  let degree_x: usize = bincode::deserialize_from(&mut reader)?;
  let degree_y: usize = bincode::deserialize_from(&mut reader)?;
  let degree_z: usize = bincode::deserialize_from(&mut reader)?;

  // Helper function to read a G1Affine point from bytes
  let read_g1_affine = |reader: &mut BufReader<File>| -> Result<P::G1Affine, Box<dyn std::error::Error>> {
    let field_byte_size = 48; // BLS12-381 base field size (Fp has 48 bytes)
    let mut x_bytes = vec![0u8; field_byte_size];
    let mut y_bytes = vec![0u8; field_byte_size];
    reader.read_exact(&mut x_bytes)?;
    reader.read_exact(&mut y_bytes)?;

    // Reconstruct the field elements from bytes
    let x = <P::G1Affine as icicle_core::affine::Affine>::BaseField::from_bytes_le(&x_bytes);
    let y = <P::G1Affine as icicle_core::affine::Affine>::BaseField::from_bytes_le(&y_bytes);

    // Reconstruct the affine point from coordinates
    use icicle_core::affine::Affine;
    let point = <P::G1Affine as Affine>::from_xy(x, y);
    Ok(point)
  };

  // Helper function to read a G2Affine point from bytes
  let read_g2_affine = |reader: &mut BufReader<File>| -> Result<P::G2Affine, Box<dyn std::error::Error>> {
    let field_byte_size = 96; // BLS12-381 G2 base field size (Fp2 has 96 bytes)
    let mut x_bytes = vec![0u8; field_byte_size];
    let mut y_bytes = vec![0u8; field_byte_size];
    reader.read_exact(&mut x_bytes)?;
    reader.read_exact(&mut y_bytes)?;

    // Reconstruct the field elements from bytes
    let x = <P::G2Affine as icicle_core::affine::Affine>::BaseField::from_bytes_le(&x_bytes);
    let y = <P::G2Affine as icicle_core::affine::Affine>::BaseField::from_bytes_le(&y_bytes);

    // Reconstruct the affine point from coordinates
    use icicle_core::affine::Affine;
    let point = <P::G2Affine as Affine>::from_xy(x, y);
    Ok(point)
  };

  // Read G1Affine points
  let h_xyz_len: usize = bincode::deserialize_from(&mut reader)?;
  let mut h_xyz = Vec::with_capacity(h_xyz_len);
  for _ in 0..h_xyz_len {
    h_xyz.push(read_g1_affine(&mut reader)?);
  }

  let h_yz_len: usize = bincode::deserialize_from(&mut reader)?;
  let mut h_yz = Vec::with_capacity(h_yz_len);
  for _ in 0..h_yz_len {
    h_yz.push(read_g1_affine(&mut reader)?);
  }

  let h_z_len: usize = bincode::deserialize_from(&mut reader)?;
  let mut h_z = Vec::with_capacity(h_z_len);
  for _ in 0..h_z_len {
    h_z.push(read_g1_affine(&mut reader)?);
  }

  // Read G2Affine points
  let v_x_len: usize = bincode::deserialize_from(&mut reader)?;
  let mut v_x = Vec::with_capacity(v_x_len);
  for _ in 0..v_x_len {
    v_x.push(read_g2_affine(&mut reader)?);
  }

  let v_y_len: usize = bincode::deserialize_from(&mut reader)?;
  let mut v_y = Vec::with_capacity(v_y_len);
  for _ in 0..v_y_len {
    v_y.push(read_g2_affine(&mut reader)?);
  }

  let v_z_len: usize = bincode::deserialize_from(&mut reader)?;
  let mut v_z = Vec::with_capacity(v_z_len);
  for _ in 0..v_z_len {
    v_z.push(read_g2_affine(&mut reader)?);
  }

  // Read the single G2Affine point
  let v = read_g2_affine(&mut reader)?;

  let srs = KZH3SRS {
    degree_x,
    degree_y,
    degree_z,
    h_xyz,
    h_yz,
    h_z,
    v_x,
    v_y,
    v_z,
    v,
  };

  println!("KZH3 SRS loaded from: {}", file_path);
  Ok(srs)
}
