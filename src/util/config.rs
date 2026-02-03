use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Config {
  pub task: String,
  pub sf: ScaleFactorConfig,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ScaleFactorConfig {
  pub scale_factor_log: usize,
  pub table_size_log: usize,
  pub table_commit_log: usize,
}
