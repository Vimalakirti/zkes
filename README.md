# zktransformer

A zero-knowledge proof system for verifiable inference of transformer models. Generate succinct proofs that a neural network inference was computed correctly without revealing the model weights or inputs.

## Features

- **Supported Models**: GPT-2, GPT-J-6B, BERT-Large, LLaMA-2-7B
- **Polynomial Commitments**: KZH2 and KZH3 schemes with sparse polynomial support
- **Sumcheck Protocol**: Linear and sparse-dense sumcheck provers for efficient verification
- **Multiple Backends**:
  - [arkworks](https://github.com/arkworks-rs) - Pure Rust implementation
  - [icicle](https://github.com/ingonyama-zk/icicle) - GPU-accelerated operations
- **Curve Support**: BN254, BLS12-381, Goldilocks

## Architecture

```
src/
├── basicblock/     # Neural network layer implementations (Add, Einsum, Permute, etc.)
├── crypto/
│   ├── polycommit/ # KZH2/KZH3 polynomial commitment schemes
│   └── sumcheck/   # Sumcheck protocol (prover & verifier)
├── dag/            # Computation graph builder for transformer architectures
└── util/           # Polynomials, transcripts, serialization
```

## Quick Start

### Prerequisites

- Rust 1.70+
- For GPU acceleration: CUDA toolkit (optional, for icicle backend)

### Build

```bash
# Default build (BN254 curve, arkworks backend)
cargo build --release

# With icicle GPU acceleration
cargo build --release --features icicle
```

### Run

```bash
# Run with default settings (BN254)
cargo run --release -- config.yaml

# Run with BLS12-381 curve
cargo run --release --no-default-features --features bls12_381,arkworks -- config.yaml

# Run with Goldilocks field
cargo run --release --no-default-features --features goldilocks,icicle -- config.yaml
```

### Run Specific Models

```bash
# GPT-2
cargo run --release --bin gpt2

# GPT-2 with real weights
cargo run --release --bin gpt2_real

# BERT-Large
cargo run --release --bin bert

# GPT-J-6B
cargo run --release --bin gptj

# LLaMA-2-7B
cargo run --release --bin llama
```

### Generate SRS

```bash
# Generate Structured Reference String for polynomial size 2^20
cargo run --release --bin setup -- generate 20

# Load and verify existing SRS
cargo run --release --bin setup -- load 20
```

## Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_kzh3

# Run with logging
RUST_LOG=debug cargo test
```

## Benchmarks

```bash
# Run sumcheck prover benchmark
cargo bench --bench sumcheck_prover

# Run permutation proof benchmark
cargo bench --bench permute_prove

# Run KZH opening benchmark
cargo bench --bench kzh_openings_fast
```

## Configuration

See `config.yaml` for configuration options:

```yaml
model: gpt2
input_path: input.bin
output_path: output.bin
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `bn254` | Use BN254 curve (default) |
| `bls12_381` | Use BLS12-381 curve |
| `goldilocks` | Use Goldilocks field |
| `arkworks` | Use arkworks backend (default) |
| `icicle` | Use icicle GPU-accelerated backend |

## Logging

Control log verbosity with `RUST_LOG`:

```bash
RUST_LOG=debug cargo run --release -- config.yaml 2>&1 | tee output.log
```

## License

Apache-2.0
