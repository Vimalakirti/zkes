use std::env;
use std::time::Instant;
use zktransformer::crypto::polycommit::{kzh3::setup_kzh3_srs, PairingTrait};
use zktransformer::crypto::srs_storage::{load_kzh3_srs, store_kzh3_srs};

#[cfg(feature = "icicle")]
use icicle_core::traits::GenerateRandom;

#[cfg(feature = "icicle")]
type TestPairing = zktransformer::crypto::polycommit::IcicleBn254;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let args: Vec<String> = env::args().collect();

  if args.len() < 3 {
    println!("Usage: {} <command> <maximum_degree>", args[0]);
    println!("Commands:");
    println!("  generate - Generate and store SRS for maximum_degree");
    println!("  load     - Load SRS for maximum_degree");
    println!("  test     - Generate, store, and load SRS for maximum_degree");
    println!();
    println!("Arguments:");
    println!("  maximum_degree   - Maximum degree for SRS generation (used for filename: maximum_degree.srs)");
    return Ok(());
  }

  let command = &args[1];
  let maximum_degree: usize = args[2].parse().expect("Invalid maximum_degree");

  match command.as_str() {
    "generate" => {
      generate_and_store_srs(maximum_degree)?;
    }
    "load" => {
      load_and_verify_srs(maximum_degree)?;
    }
    "test" => {
      test_full_cycle(maximum_degree)?;
    }
    _ => {
      eprintln!("Unknown command: {}", command);
      eprintln!("Valid commands: generate, load, test");
      return Err("Invalid command".into());
    }
  }

  Ok(())
}

#[cfg(feature = "icicle")]
fn generate_and_store_srs(maximum_degree: usize) -> Result<(), Box<dyn std::error::Error>> {
  println!("Generating KZH3 SRS with maximum_degree={}", maximum_degree);

  let start_time = Instant::now();

  let mut dummy_rng = ();
  let srs = setup_kzh3_srs::<TestPairing, _>(maximum_degree, &mut dummy_rng);

  let generation_time = start_time.elapsed();
  println!("SRS generation completed in {:?}", generation_time);

  println!("SRS properties:");
  println!("  degree_x: {}", srs.degree_x);
  println!("  degree_y: {}", srs.degree_y);
  println!("  degree_z: {}", srs.degree_z);
  println!("  h_xyz length: {}", srs.h_xyz.len());
  println!("  h_yz length: {}", srs.h_yz.len());
  println!("  h_z length: {}", srs.h_z.len());
  println!("  v_x length: {}", srs.v_x.len());
  println!("  v_y length: {}", srs.v_y.len());
  println!("  v_z length: {}", srs.v_z.len());

  let store_start = Instant::now();
  store_kzh3_srs(&srs, maximum_degree)?;
  let store_time = store_start.elapsed();
  println!("SRS stored in {:?}", store_time);

  Ok(())
}

#[cfg(feature = "icicle")]
fn load_and_verify_srs(maximum_degree: usize) -> Result<(), Box<dyn std::error::Error>> {
  println!("Loading KZH3 SRS for maximum_degree={}", maximum_degree);

  let load_start = Instant::now();
  let srs: zktransformer::crypto::polycommit::kzh3::KZH3SRS<TestPairing> = load_kzh3_srs(maximum_degree)?;
  let load_time = load_start.elapsed();

  println!("SRS loaded in {:?}", load_time);

  println!("Loaded SRS properties:");
  println!("  degree_x: {}", srs.degree_x);
  println!("  degree_y: {}", srs.degree_y);
  println!("  degree_z: {}", srs.degree_z);
  println!("  h_xyz length: {}", srs.h_xyz.len());
  println!("  h_yz length: {}", srs.h_yz.len());
  println!("  h_z length: {}", srs.h_z.len());
  println!("  v_x length: {}", srs.v_x.len());
  println!("  v_y length: {}", srs.v_y.len());
  println!("  v_z length: {}", srs.v_z.len());

  Ok(())
}

#[cfg(feature = "icicle")]
fn test_full_cycle(maximum_degree: usize) -> Result<(), Box<dyn std::error::Error>> {
  println!("Testing full cycle: generate -> store -> load for maximum_degree={}", maximum_degree);

  // Generate SRS
  println!("\n=== Generation Phase ===");
  let mut dummy_rng = ();
  let original_srs = setup_kzh3_srs::<TestPairing, _>(maximum_degree, &mut dummy_rng);
  println!("Original SRS generated");

  // Store SRS
  println!("\n=== Storage Phase ===");
  store_kzh3_srs(&original_srs, maximum_degree)?;

  // Load SRS
  println!("\n=== Loading Phase ===");
  let loaded_srs: zktransformer::crypto::polycommit::kzh3::KZH3SRS<TestPairing> = load_kzh3_srs(maximum_degree)?;

  // Verify they match
  println!("\n=== Verification Phase ===");
  let matches = original_srs.degree_x == loaded_srs.degree_x
    && original_srs.degree_y == loaded_srs.degree_y
    && original_srs.degree_z == loaded_srs.degree_z
    && original_srs.h_xyz.len() == loaded_srs.h_xyz.len()
    && original_srs.h_yz.len() == loaded_srs.h_yz.len()
    && original_srs.h_z.len() == loaded_srs.h_z.len()
    && original_srs.v_x.len() == loaded_srs.v_x.len()
    && original_srs.v_y.len() == loaded_srs.v_y.len()
    && original_srs.v_z.len() == loaded_srs.v_z.len();

  if matches {
    println!("✅ Success! Original and loaded SRS match in structure");
    println!("📁 SRS file: {}.srs", maximum_degree);

    // Get file size
    let file_path = format!("{}.srs", maximum_degree);
    if let Ok(metadata) = std::fs::metadata(&file_path) {
      let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
      println!("📊 File size: {:.2} MB", size_mb);
    }
  } else {
    println!("❌ Error: Original and loaded SRS do not match!");
    return Err("SRS mismatch".into());
  }

  Ok(())
}

#[cfg(not(feature = "icicle"))]
fn generate_and_store_srs(_maximum_degree: usize) -> Result<(), Box<dyn std::error::Error>> {
  eprintln!("Error: This binary requires the 'icicle' feature to be enabled");
  Err("icicle feature not enabled".into())
}

#[cfg(not(feature = "icicle"))]
fn load_and_verify_srs(_maximum_degree: usize) -> Result<(), Box<dyn std::error::Error>> {
  eprintln!("Error: This binary requires the 'icicle' feature to be enabled");
  Err("icicle feature not enabled".into())
}

#[cfg(not(feature = "icicle"))]
fn test_full_cycle(_maximum_degree: usize) -> Result<(), Box<dyn std::error::Error>> {
  eprintln!("Error: This binary requires the 'icicle' feature to be enabled");
  Err("icicle feature not enabled".into())
}
