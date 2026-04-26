#!/usr/bin/env python3
"""
Quantized LLaMA-2-7B perplexity on WikiText-2 using Q15 fixed-point arithmetic.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Scale factor (override via SF_LOG env var for quick sweeps).
SF_LOG = int(os.environ.get("SF_LOG", 15))
SF = 1 << SF_LOG


def quantize(x: torch.Tensor) -> torch.Tensor:
    """Scale float tensor to Q15 fixed-point (as float for easier computation)."""
    return torch.round(x * SF)


def dequantize(x: torch.Tensor) -> torch.Tensor:
    """Convert Q15 fixed-point back to float."""
    return x / SF


def rescale(x: torch.Tensor) -> torch.Tensor:
    """Rescale after multiplication: divide by SF to maintain Q scale."""
    return torch.round(x / SF)


def rescale_2sf(x: torch.Tensor) -> torch.Tensor:
    """Fused rescale for three-factor products; see gpt2_quant.py for details."""
    return torch.round(x.to(torch.float64) / (SF * SF)).to(x.dtype)


def rescale_3sf(x: torch.Tensor) -> torch.Tensor:
    """Rescale by SF^3; used when RMSNorm r advice is Q30 instead of Q15."""
    SF3 = float(SF) ** 3
    return torch.round(x.to(torch.float64) / SF3)


def q_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Quantized matrix multiplication with rescale."""
    return rescale(torch.matmul(a, b))


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

    Input x (Q15): clamped to [-K_MAX*ln(2)*SF, 0]. The circuit applies
    clamp_lower before exp and assumes x <= 0 so that k in the decomposition
    x = k * (-ln2*SF) + r is in [0, K_MAX] (K_BITS=5 here, vs circuit's 4).

    Output: round(SF * e^(x/SF)).

    Taylor term t^4/24 included for lower softmax/sigmoid error.
    """
    x = torch.clamp(x, min=CLAMP_LOWER, max=0.0)

    ln2_sf = math.log(2) * SF

    k = torch.round(-x / ln2_sf)
    k = torch.clamp(k, min=0, max=K_MAX)
    r = x + k * ln2_sf  # residual (Q15), |r| <= ln2/2 * SF

    # Quartic Taylor: 1 + t + t^2/2 + t^3/6 + t^4/24, t = r/SF.
    t = r / SF
    f_r = 1.0 + t * (1.0 + t * (0.5 + t * (1.0 / 6.0 + t / 24.0)))

    result = SF * torch.pow(0.5, k) * f_r
    return torch.round(torch.clamp(result, min=0.0))


def q_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Advice-based softmax matching SoftmaxConst (src/basicblock/llama.rs:225-316).
    Computes c so that softmax(x)_j = q_exp(x + c) with no divide.
    """
    x_max = x.max(dim=dim, keepdim=True).values
    shift = x - x_max  # <= 0
    exp_shift = q_exp(shift)  # Q15 of exp((x - max)/SF)
    sum_shift = exp_shift.sum(dim=dim, keepdim=True)  # == SF * real_sum

    # c = -max - SF * ln(real_sum); real_sum = sum_shift / SF.
    ratio = torch.clamp(sum_shift / SF, min=1e-12)
    lse_shift = torch.round(SF * torch.log(ratio))
    c = -(x_max + lse_shift)
    return q_exp(x + c)


def q_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Quantized RMSNorm using the advice-based approach from quant.md.
    Input x: Q15 scaled (X_i = round(2^15 * x_i))
    weight: Q15 scaled
    Output: Q15 scaled

    From quant.md:
    Step 1: Prover computes r = round(2^15 / rms(x)) as advice
    Step 2: Verifier checks mean_sq * r² ≈ (2^15)² (after rescale ≈ 2^15)
    Step 3: Apply Y = round(X * r / 2^15) * W (with rescale)
    """
    # Step 1: Compute sum of squares with rescale
    # sum_sq = Σ(X_i * X_i) with rescale
    x_sq = rescale(x * x)  # X_i² / SF -> Q15 representation of x_i²
    mean_sq = x_sq.mean(dim=-1, keepdim=True)  # Q15 representation of mean(x²)

    # Prover supplies r at Q30 scale (r ≈ SF²/rms). Circuit equivalent: one
    # Newton step from a Q15 coarse r, stored at Q30 for SF× finer grid.
    # Verifier constraint: mean_sq · r² ≈ SF⁴.
    variance = mean_sq / SF + eps
    rms = torch.sqrt(torch.clamp(variance, min=eps))
    SF2 = float(SF) * float(SF)
    r = torch.round(torch.clamp(SF2 / rms.to(torch.float64), max=SF2 * 100.0))

    # Step 2: Verification constraint (implicit — sim computes r exactly).
    # Step 3: Output = round(x · r · weight / SF³), single rounding.
    prod = x.to(torch.float64) * r * weight.to(torch.float64)
    out = rescale_3sf(prod).to(x.dtype)
    return out


def q_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Advice-based sigmoid matching SigmoidConst (src/basicblock/llama.rs:318-390
    and builder.rs:288-305): sigmoid(x) = q_exp(x + c) where
    c = -x - SF * softplus(-x/SF). No divide.
    """
    t = -x / SF  # float; softplus is numerically stable
    softplus = torch.nn.functional.softplus(t)
    c = -x - torch.round(SF * softplus)
    return q_exp(x + c)


def q_silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Swish): x * sigmoid(x). Circuit uses this in llama_mlp."""
    return rescale(x * q_sigmoid(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input (HuggingFace style)."""
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    """
    Apply rotary position embeddings to Q and K tensors (HuggingFace style).
    q, k: Q15 scaled tensors of shape [batch, heads, seq, head_dim]
    cos, sin: Q15 scaled tensors of shape [1, 1, seq, head_dim]

    Formula: q_embed = q * cos + rotate_half(q) * sin
    """
    # Apply rotation: q * cos + rotate_half(q) * sin
    q_rotated = rescale(q * cos) + rescale(rotate_half(q) * sin)
    k_rotated = rescale(k * cos) + rescale(rotate_half(k) * sin)

    return q_rotated, k_rotated


def precompute_rope_cache(dim: int, max_seq_len: int, base: float = 10000.0, device='cpu'):
    """Precompute rotary position embedding cache (in Q15 format)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(max_seq_len, device=device).float()

    # angles: [seq_len, dim/2]
    angles = torch.outer(positions, inv_freq)

    # Duplicate for full dimension
    angles = torch.cat([angles, angles], dim=-1)

    # Compute cos and sin in Q15
    cos_cache = quantize(torch.cos(angles))
    sin_cache = quantize(torch.sin(angles))

    return cos_cache, sin_cache


class QuantizedLlamaAttention(nn.Module):
    """Quantized LLaMA attention in Q15 fixed-point."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = math.sqrt(self.head_dim)

        # Weights will be loaded and quantized
        self.q_proj_weight = None  # [hidden_size, hidden_size]
        self.k_proj_weight = None  # [hidden_size, num_kv_heads * head_dim]
        self.v_proj_weight = None  # [hidden_size, num_kv_heads * head_dim]
        self.o_proj_weight = None  # [hidden_size, hidden_size]

    def forward(self, hidden_states, cos, sin, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projections
        q = q_matmul(hidden_states, self.q_proj_weight)
        k = q_matmul(hidden_states, self.k_proj_weight)
        v = q_matmul(hidden_states, self.v_proj_weight)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim]
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand K and V for grouped-query attention
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Attention scores: Q @ K^T / sqrt(d_k)
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
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Output projection
        attn_output = q_matmul(attn_output, self.o_proj_weight)

        return attn_output


class QuantizedLlamaMLP(nn.Module):
    """Quantized LLaMA MLP with SwiGLU activation in Q15 fixed-point."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj_weight = None  # [hidden_size, intermediate_size]
        self.up_proj_weight = None    # [hidden_size, intermediate_size]
        self.down_proj_weight = None  # [intermediate_size, hidden_size]

    def forward(self, hidden_states):
        # Gate and up projections
        gate = q_matmul(hidden_states, self.gate_proj_weight)
        up = q_matmul(hidden_states, self.up_proj_weight)

        # SwiGLU: SiLU(gate) * up
        gate_activated = q_silu(gate)
        hidden_states = q_mul(gate_activated, up)

        # Down projection
        hidden_states = q_matmul(hidden_states, self.down_proj_weight)

        return hidden_states


class QuantizedLlamaBlock(nn.Module):
    """Quantized LLaMA transformer block."""

    def __init__(self, config):
        super().__init__()
        self.input_layernorm_weight = None
        self.self_attn = QuantizedLlamaAttention(config)
        self.post_attention_layernorm_weight = None
        self.mlp = QuantizedLlamaMLP(config)
        self.config = config

    def forward(self, hidden_states, cos, sin, attention_mask=None):
        # Pre-norm attention
        residual = hidden_states
        hidden_states = q_rms_norm(hidden_states, self.input_layernorm_weight,
                                   eps=self.config.rms_norm_eps)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = residual + hidden_states

        # Pre-norm MLP
        residual = hidden_states
        hidden_states = q_rms_norm(hidden_states, self.post_attention_layernorm_weight,
                                   eps=self.config.rms_norm_eps)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class QuantizedLlamaModel(nn.Module):
    """Quantized LLaMA model in Q15 fixed-point."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.embed_tokens = None  # Token embeddings [vocab_size, hidden_size]

        self.layers = nn.ModuleList([QuantizedLlamaBlock(config) for _ in range(config.num_hidden_layers)])

        self.norm_weight = None  # Final RMSNorm weight

        # RoPE cache will be set during loading
        self.cos_cache = None
        self.sin_cache = None

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get embeddings (already quantized)
        hidden_states = F.embedding(input_ids, self.embed_tokens)

        # Create causal mask. -1000*SF is needed to dominate worst-case raw
        # attention scores; see gpt2_quant.py notes and ppl.md 2026-04-22 entry.
        if attention_mask is None:
            big_neg = float(-1000 * SF)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), big_neg, device=device),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)

        # Forward through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, self.cos_cache, self.sin_cache, causal_mask)

        # Final layer norm
        hidden_states = q_rms_norm(hidden_states, self.norm_weight, eps=self.config.rms_norm_eps)

        return hidden_states


class QuantizedLlamaForCausalLM(nn.Module):
    """Quantized LLaMA with LM head for perplexity computation."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = QuantizedLlamaModel(config)
        self.lm_head_weight = None

    def forward(self, input_ids, labels=None):
        hidden_states = self.model(input_ids)

        # LM head: Q-domain einsum with rescale (matches circuit final einsum),
        # then dequantize once for cross-entropy loss.
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
    def from_pretrained(cls, model_id, device='cpu', cache_dir=None):
        """Load and quantize weights from pretrained LLaMA."""
        print(f"Loading pretrained model: {model_id}")
        fp_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
            cache_dir=cache_dir
        )
        config = fp_model.config

        model = cls(config)

        # Quantize and load embeddings
        print("Quantizing embeddings...")
        model.model.embed_tokens = quantize(fp_model.model.embed_tokens.weight.data).to(device)

        # LM head
        if hasattr(fp_model, 'lm_head') and fp_model.lm_head.weight is not None:
            model.lm_head_weight = quantize(fp_model.lm_head.weight.data).to(device)
        else:
            # Tied embeddings
            model.lm_head_weight = model.model.embed_tokens

        # Precompute RoPE cache
        print("Computing RoPE cache...")
        head_dim = config.hidden_size // config.num_attention_heads
        rope_base = getattr(config, 'rope_theta', 10000.0)
        max_seq_len = getattr(config, 'max_position_embeddings', 4096)
        cos_cache, sin_cache = precompute_rope_cache(head_dim, max_seq_len, rope_base, device)
        model.model.cos_cache = cos_cache.to(device)
        model.model.sin_cache = sin_cache.to(device)

        # Load and quantize each layer
        print(f"Quantizing {config.num_hidden_layers} transformer layers...")
        for i, (q_layer, fp_layer) in enumerate(tqdm(
            zip(model.model.layers, fp_model.model.layers),
            desc="Quantizing layers",
            total=config.num_hidden_layers
        )):
            # Input LayerNorm
            q_layer.input_layernorm_weight = quantize(fp_layer.input_layernorm.weight.data).to(device)

            # Self attention
            q_layer.self_attn.q_proj_weight = quantize(fp_layer.self_attn.q_proj.weight.data.t()).to(device)
            q_layer.self_attn.k_proj_weight = quantize(fp_layer.self_attn.k_proj.weight.data.t()).to(device)
            q_layer.self_attn.v_proj_weight = quantize(fp_layer.self_attn.v_proj.weight.data.t()).to(device)
            q_layer.self_attn.o_proj_weight = quantize(fp_layer.self_attn.o_proj.weight.data.t()).to(device)

            # Post-attention LayerNorm
            q_layer.post_attention_layernorm_weight = quantize(fp_layer.post_attention_layernorm.weight.data).to(device)

            # MLP
            q_layer.mlp.gate_proj_weight = quantize(fp_layer.mlp.gate_proj.weight.data.t()).to(device)
            q_layer.mlp.up_proj_weight = quantize(fp_layer.mlp.up_proj.weight.data.t()).to(device)
            q_layer.mlp.down_proj_weight = quantize(fp_layer.mlp.down_proj.weight.data.t()).to(device)

        # Final layer norm
        print("Quantizing final layer norm...")
        model.model.norm_weight = quantize(fp_model.model.norm.weight.data).to(device)

        # Clean up
        del fp_model
        torch.cuda.empty_cache()

        return model.to(device)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def compute_ppl(model, encodings, device, stride: int, max_length: int):
    """Compute perplexity with strided sliding window."""
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
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Pretrained model to quantize")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Max context length for evaluation")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "c4"],
                        help="Eval corpus. wikitext = wikitext-2-raw-v1 test, c4 = allenai/c4 en validation stream")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of C4 samples to use (ignored for wikitext)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Custom cache directory for model weights")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Scale factor: 2^{SF_LOG} = {SF}")

    # Load eval corpus FIRST (uses default cache)
    if args.dataset == "wikitext":
        print("Loading WikiText-2 dataset...")
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        full_text = "\n\n".join(test["text"])
    else:
        print(f"Loading C4 dataset ({args.num_samples} samples, streaming)...")
        stream = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for i, sample in enumerate(stream):
            if i >= args.num_samples:
                break
            texts.append(sample["text"])
        full_text = "\n\n".join(texts)
        print(f"Collected {len(texts)} C4 samples")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)

    # Tokenize dataset
    encodings = tokenizer(full_text, return_tensors="pt")
    print(f"Total tokens: {encodings.input_ids.size(1)}")

    # Load and quantize model
    print(f"Loading and quantizing model: {args.model_id}")
    model = QuantizedLlamaForCausalLM.from_pretrained(args.model_id, device=device, cache_dir=args.cache_dir)

    # Get max length
    model_max_length = getattr(model.config, 'max_position_embeddings', args.max_length)
    max_length = min(args.max_length, model_max_length)
    print(f"Using max_length: {max_length}")

    # Compute perplexity
    ppl, avg_nll, n_tokens = compute_ppl(model, encodings, device, stride=args.stride, max_length=max_length)

    corpus_tag = "WikiText-2, test" if args.dataset == "wikitext" else f"C4 validation, {args.num_samples} samples"
    print(f"\n=== Results ({corpus_tag}) - Quantized Q15 ===")
    print(f"Model: {args.model_id} (Q15 quantized)")
    print(f"Scale factor: 2^{SF_LOG} = {SF}")
    print(f"Stride: {args.stride}")
    print(f"Max length: {max_length}")
    print(f"Tokens contributing to loss: {n_tokens}")
    print(f"Average NLL per token:      {avg_nll:.6f}")
    print(f"Perplexity:                {ppl:.4f}")


if __name__ == "__main__":
    main()
