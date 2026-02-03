# zktransformer

## Run
Use the default prime field `bls12_381` to run the program.
```bash
cargo run --release -- <config file>
```

Use the prime field `goldilocks` to run the program.
```bash
cargo run --release --no-default-features --features goldilocks -- <config file>
```

## Logging
Use the `RUST_LOG` environment variable to set the logging level.
```bash
RUST_LOG=debug cargo run --release -- <config file> > output.log 2>&1
```

## Test
```bash
cargo test
```

## Config
See `config.yaml` for the config file.
