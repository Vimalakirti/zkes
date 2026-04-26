#!/usr/bin/env python3
"""
Quantized GPT-2 perplexity on WikiText-2 using Q15 fixed-point arithmetic.
All values are scaled by 2^15 = 32768.
"""

import argparse
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config

# Scale factor (override via SF_LOG env var for quick sweeps).
SF_LOG = int(os.environ.get("SF_LOG", 15))
SF = 1 << SF_LOG

# Oracle toggles for bisecting the 1.5 PPL gap. Comma-separated subset of:
#   gelu, softmax, ln, lmhead, matmul_noround
ORACLE = os.environ.get("ORACLE", "")


def quantize(x: torch.Tensor) -> torch.Tensor:
    """Scale float tensor to Q15 fixed-point (as float for easier computation)."""
    if "fp32weights" in os.environ.get("ORACLE", ""):
        return x * SF  # skip rounding — fractional Q15 weights
    return torch.round(x * SF)


def dequantize(x: torch.Tensor) -> torch.Tensor:
    """Convert Q15 fixed-point back to float."""
    return x / SF


def rescale(x: torch.Tensor) -> torch.Tensor:
    """Rescale after multiplication: divide by SF to maintain Q scale."""
    return torch.round(x / SF)


def rescale_2sf(x: torch.Tensor) -> torch.Tensor:
    """
    Fused rescale for three-factor products (e.g. x*r*w in LayerNorm/RMSNorm
    output): single rounding step dividing by SF^2. fp64 intermediate because
    Q values up to SF^3 ~ 2^45 overflow fp32 mantissa precision.
    """
    return torch.round(x.to(torch.float64) / (SF * SF)).to(x.dtype)


def rescale_3sf(x: torch.Tensor) -> torch.Tensor:
    """
    Rescale by SF^3 (used when norm r advice is Q30 instead of Q15).
    Product x*r*w reaches ~SF^4 ~ 2^60, well beyond fp32; fp64 keeps it
    precise enough for integer rounding.
    """
    SF3 = float(SF) ** 3
    return torch.round(x.to(torch.float64) / SF3)


def q_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Quantized matrix multiplication with rescale.
    fp64 accumulation: Q15 × Q15 sums reach 2^36–2^40 in MLP/attention matmuls,
    which exceed fp32's 24-bit mantissa. Circuit does exact integer arithmetic,
    so fp64 here is the sim's closest match.
    """
    out64 = torch.matmul(a.to(torch.float64), b.to(torch.float64))
    if "matmul_noround" in ORACLE:
        return (out64 / SF).to(a.dtype)  # keep fractional Q15 (no rounding)
    return torch.round(out64 / SF).to(a.dtype)


def q_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Quantized element-wise multiplication with rescale."""
    return rescale(a * b)


K_BITS = 5
K_MAX = (1 << K_BITS) - 1  # 31
CLAMP_LOWER = -K_MAX * math.log(2) * SF  # circuit's clamp_lower threshold


def q_exp(x: torch.Tensor) -> torch.Tensor:
    """
    Circuit-faithful quantized exp (mirrors ExpHelper + Taylor in zk-torch-2
    src/dag/builder.rs:332-400 and src/basicblock/clamp.rs + exp.rs).

    Input x (Q15): clamped to [-K_MAX*ln(2)*SF, 0]. Decomposition:
    x = k*(-ln2*SF) + r with k in [0, K_MAX], |r| <= ln2/2 * SF.

    K_BITS=5 and quartic Taylor extend the effective exp input range and
    reduce polynomial truncation error (vs circuit's current K_BITS=4 cubic).
    """
    x = torch.clamp(x, min=CLAMP_LOWER, max=0.0)

    ln2_sf = math.log(2) * SF
    k = torch.round(-x / ln2_sf)
    k = torch.clamp(k, min=0, max=K_MAX)
    r = x + k * ln2_sf

    t = r / SF
    f_r = 1.0 + t * (1.0 + t * (0.5 + t * (1.0 / 6.0 + t / 24.0)))

    result = SF * torch.pow(0.5, k) * f_r
    return torch.round(torch.clamp(result, min=0.0))


def q_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Advice-based softmax matching SoftmaxConst (src/basicblock/llama.rs:225-316).
    softmax(x)_j = q_exp(x + c) with c derived from log-sum-exp shift.
    """
    if "softmax" in ORACLE:
        x_real = x / SF
        out_real = F.softmax(x_real, dim=dim)
        return torch.round(out_real * SF)
    x_max = x.max(dim=dim, keepdim=True).values
    shift = x - x_max
    exp_shift = q_exp(shift)
    sum_shift = exp_shift.sum(dim=dim, keepdim=True)
    ratio = torch.clamp(sum_shift / SF, min=1e-12)
    lse_shift = torch.round(SF * torch.log(ratio))
    c = -(x_max + lse_shift)
    return q_exp(x + c)


def q_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Advice-based sigmoid matching SigmoidConst (src/basicblock/llama.rs:318-390).
    sigmoid(x) = q_exp(x + c), c = -x - SF * softplus(-x/SF).
    """
    t = -x / SF
    softplus = torch.nn.functional.softplus(t)
    c = -x - torch.round(SF * softplus)
    return q_exp(x + c)


def q_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Quantized LayerNorm using the advice-based approach from quant.md.
    Input x: Q15 scaled
    weight, bias: Q15 scaled
    Output: Q15 scaled

    From quant.md (LayerNorm variant):
    Step 1: X_centered = X - mean(X)
    Step 2: r = round(2^15 / rms(X_centered)) as advice
    Step 3: Verify and apply as in RMSNorm
    """
    if "ln" in ORACLE:
        x_real = x / SF
        out_real = F.layer_norm(x_real, (x_real.shape[-1],),
                                 weight=weight / SF, bias=bias / SF, eps=eps)
        return torch.round(out_real * SF)
    # Step 1: Compute mean and subtract (per quant.md for LayerNorm)
    mean = x.mean(dim=-1, keepdim=True)
    x_centered = x - torch.round(mean)

    # Compute sum of squares with rescale
    x_sq = rescale(x_centered * x_centered)
    mean_sq = x_sq.mean(dim=-1, keepdim=True)

    # Step 2: Prover supplies r at Q30 scale (r ≈ SF²/rms) instead of Q15.
    # Circuit: prover still produces r via one Newton iteration from the Q15
    # coarse r, but stores the result at Q30 so the downstream grid is
    # SF x finer. Verifier constraint becomes mean_sq · r² ≈ SF⁴.
    variance = mean_sq / SF + eps
    rms = torch.sqrt(torch.clamp(variance, min=eps))
    SF2 = float(SF) * float(SF)
    r = torch.round(torch.clamp(SF2 / rms.to(torch.float64), max=SF2 * 100.0))

    # Step 3: Output = round(x_centered · r · weight / SF³) + bias.
    prod = x_centered.to(torch.float64) * r * weight.to(torch.float64)
    out = rescale_3sf(prod).to(x.dtype) + bias
    return out


def q_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    GELU tanh-approx (HuggingFace GPT-2 default):
      GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Using tanh(u) = 2*sigmoid(2u) - 1, this collapses to:
      GELU(x) = x * sigmoid(2*sqrt(2/pi) * (x + 0.044715 * x^3))
    So we only need one sigmoid, matching the circuit's advice-exp form.
    """
    if "gelu" in ORACLE:
        x_real = x / SF
        out_real = F.gelu(x_real, approximate='tanh')
        return torch.round(out_real * SF)
    a = torch.round(torch.tensor(2.0 * math.sqrt(2.0 / math.pi) * SF, device=x.device))  # ~1.5958 * SF
    b = torch.round(torch.tensor(0.044715 * SF, device=x.device))
    x2 = rescale(x * x)
    x3 = rescale(x2 * x)
    u = x + rescale(b * x3)
    z = rescale(a * u)
    return rescale(x * q_sigmoid(z))


class QuantizedGPT2Attention(nn.Module):
    """Quantized GPT-2 attention in Q15 fixed-point."""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = math.sqrt(self.head_dim)

        # Weights will be loaded and quantized
        self.c_attn_weight = None  # [3 * embed_dim, embed_dim]
        self.c_attn_bias = None
        self.c_proj_weight = None  # [embed_dim, embed_dim]
        self.c_proj_bias = None

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection: hidden_states @ c_attn_weight + c_attn_bias
        qkv = q_matmul(hidden_states, self.c_attn_weight) + self.c_attn_bias

        # Split into Q, K, V
        q, k, v = qkv.split(self.embed_dim, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: Q @ K^T / sqrt(d_k)
        # In Q15: (Q @ K^T) / SF / sqrt(d_k) * SF = Q @ K^T / sqrt(d_k)
        attn_scores = q_matmul(q, k.transpose(-2, -1))

        # Scale by 1/sqrt(d_k) in Q15
        scale_factor = torch.round(torch.tensor(SF / self.scale, device=hidden_states.device))
        attn_scores = rescale(attn_scores * scale_factor)

        # Apply causal mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax (Q15)
        attn_probs = q_softmax(attn_scores, dim=-1)

        # Attention output: attn_probs @ V
        attn_output = q_matmul(attn_probs, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Output projection
        attn_output = q_matmul(attn_output, self.c_proj_weight) + self.c_proj_bias

        return attn_output


class QuantizedGPT2MLP(nn.Module):
    """Quantized GPT-2 MLP in Q15 fixed-point."""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.n_embd
        self.intermediate_size = config.n_inner if config.n_inner is not None else 4 * config.n_embd

        self.c_fc_weight = None
        self.c_fc_bias = None
        self.c_proj_weight = None
        self.c_proj_bias = None

    def forward(self, hidden_states):
        # First linear
        hidden_states = q_matmul(hidden_states, self.c_fc_weight) + self.c_fc_bias

        # GELU activation
        hidden_states = q_gelu(hidden_states)

        # Second linear
        hidden_states = q_matmul(hidden_states, self.c_proj_weight) + self.c_proj_bias

        return hidden_states


class QuantizedGPT2Block(nn.Module):
    """Quantized GPT-2 transformer block."""

    def __init__(self, config):
        super().__init__()
        self.ln_1_weight = None
        self.ln_1_bias = None
        self.attn = QuantizedGPT2Attention(config)
        self.ln_2_weight = None
        self.ln_2_bias = None
        self.mlp = QuantizedGPT2MLP(config)

    def forward(self, hidden_states, attention_mask=None):
        # Pre-norm architecture
        residual = hidden_states
        hidden_states = q_layer_norm(hidden_states, self.ln_1_weight, self.ln_1_bias)
        hidden_states = self.attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = q_layer_norm(hidden_states, self.ln_2_weight, self.ln_2_bias)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class QuantizedGPT2Model(nn.Module):
    """Quantized GPT-2 model in Q15 fixed-point."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.n_embd

        self.wte = None  # Token embeddings [vocab_size, embed_dim]
        self.wpe = None  # Position embeddings [max_position, embed_dim]

        self.blocks = nn.ModuleList([QuantizedGPT2Block(config) for _ in range(config.n_layer)])

        self.ln_f_weight = None
        self.ln_f_bias = None

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get embeddings (already quantized)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        token_embeds = F.embedding(input_ids, self.wte)
        position_embeds = F.embedding(position_ids, self.wpe)

        hidden_states = token_embeds + position_embeds

        # Create causal mask. Must be large enough to dominate the maximum
        # raw attention score. GPT-2 block 4 head 11 is an outlier attention
        # head with raw scores reaching ~223 after the /sqrt(d) scaling, so
        # the previous -100*SF choice failed to suppress masked tokens there
        # and cost ~1.5 PPL. -1000*SF fully saturates the clamp in q_exp.
        if attention_mask is None:
            big_neg = float(-1000 * SF)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), big_neg, device=device),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)

        # Forward through blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, causal_mask)

        # Final layer norm
        hidden_states = q_layer_norm(hidden_states, self.ln_f_weight, self.ln_f_bias)

        return hidden_states


class QuantizedGPT2LMHead(nn.Module):
    """Quantized GPT-2 with LM head for perplexity computation."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = QuantizedGPT2Model(config)
        self.lm_head_weight = None  # Tied with wte

    def forward(self, input_ids, labels=None):
        hidden_states = self.transformer(input_ids)

        # LM head: Q-domain einsum with rescale (mirrors circuit final einsum),
        # dequantize once for cross-entropy loss.
        if "lmhead" in ORACLE:
            # No final rescale: logits at Q30 scale, dequantize by SF² for fp32 CE.
            logits30 = torch.matmul(hidden_states.to(torch.float64),
                                     self.lm_head_weight.t().to(torch.float64))
            logits = (logits30 / (float(SF) * float(SF))).to(torch.float32)
        else:
            logits_q = q_matmul(hidden_states, self.lm_head_weight.t())
            logits = dequantize(logits_q)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return type('Output', (), {'loss': loss, 'logits': logits})()

    @classmethod
    def from_pretrained(cls, model_id, device='cpu'):
        """Load and quantize weights from pretrained GPT-2."""
        print(f"Loading pretrained model: {model_id}")
        fp_model = GPT2LMHeadModel.from_pretrained(model_id)
        config = fp_model.config

        model = cls(config)

        # Quantize and load embeddings
        print("Quantizing embeddings...")
        model.transformer.wte = quantize(fp_model.transformer.wte.weight.data).to(device)
        model.transformer.wpe = quantize(fp_model.transformer.wpe.weight.data).to(device)

        # LM head (tied weights)
        model.lm_head_weight = model.transformer.wte

        # Load and quantize each block
        print("Quantizing transformer blocks...")
        for i, (q_block, fp_block) in enumerate(zip(model.transformer.blocks, fp_model.transformer.h)):
            # Layer norm 1
            q_block.ln_1_weight = quantize(fp_block.ln_1.weight.data).to(device)
            q_block.ln_1_bias = quantize(fp_block.ln_1.bias.data).to(device)

            # Attention
            # GPT-2 uses Conv1D which stores weights as [in_features, out_features]
            # c_attn: [embed_dim, 3*embed_dim] - no transpose needed for x @ W
            q_block.attn.c_attn_weight = quantize(fp_block.attn.c_attn.weight.data).to(device)
            q_block.attn.c_attn_bias = quantize(fp_block.attn.c_attn.bias.data).to(device)
            q_block.attn.c_proj_weight = quantize(fp_block.attn.c_proj.weight.data).to(device)
            q_block.attn.c_proj_bias = quantize(fp_block.attn.c_proj.bias.data).to(device)

            # Layer norm 2
            q_block.ln_2_weight = quantize(fp_block.ln_2.weight.data).to(device)
            q_block.ln_2_bias = quantize(fp_block.ln_2.bias.data).to(device)

            # MLP
            # GPT-2 Conv1D: [in_features, out_features] - no transpose needed
            q_block.mlp.c_fc_weight = quantize(fp_block.mlp.c_fc.weight.data).to(device)
            q_block.mlp.c_fc_bias = quantize(fp_block.mlp.c_fc.bias.data).to(device)
            q_block.mlp.c_proj_weight = quantize(fp_block.mlp.c_proj.weight.data).to(device)
            q_block.mlp.c_proj_bias = quantize(fp_block.mlp.c_proj.bias.data).to(device)

        # Final layer norm
        print("Quantizing final layer norm...")
        model.transformer.ln_f_weight = quantize(fp_model.transformer.ln_f.weight.data).to(device)
        model.transformer.ln_f_bias = quantize(fp_model.transformer.ln_f.bias.data).to(device)

        # Clean up
        del fp_model

        return model.to(device)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def compute_ppl(model, encodings, device, stride: int):
    """Compute perplexity with strided sliding window."""
    model.eval()

    max_length = model.config.n_positions
    seq_len = encodings.input_ids.size(1)

    nll_sum = torch.tensor(0.0, device=device)
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride), desc="Computing PPL"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

        num_valid_tokens = (target_ids != -100).sum().item()
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size

        if num_loss_tokens > 0:
            nll_sum += neg_log_likelihood * num_loss_tokens
            n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = (nll_sum / n_tokens).item()
    ppl = math.exp(avg_nll)
    return ppl, avg_nll, n_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="openai-community/gpt2-large",
                        help="Pretrained model to quantize")
    parser.add_argument("--stride", type=int, default=512)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Scale factor: 2^{SF_LOG} = {SF}")

    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_id)

    # Load and quantize model
    model = QuantizedGPT2LMHead.from_pretrained(args.model_id, device=device)

    # Load WikiText-2 test set
    print("Loading WikiText-2 dataset...")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    print(f"Total tokens: {encodings.input_ids.size(1)}")

    # Compute perplexity
    ppl, avg_nll, n_tokens = compute_ppl(model, encodings, device, stride=args.stride)

    print("\n=== Results (WikiText-2, test) - Quantized Q15 ===")
    print(f"Model: {args.model_id} (Q15 quantized)")
    print(f"Scale factor: 2^{SF_LOG} = {SF}")
    print(f"Stride: {args.stride}")
    print(f"Tokens contributing to loss: {n_tokens}")
    print(f"Average NLL per token:      {avg_nll:.6f}")
    print(f"Perplexity:                {ppl:.4f}")


if __name__ == "__main__":
    main()
