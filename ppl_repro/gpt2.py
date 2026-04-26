#!/usr/bin/env python3
"""
GPT-2 perplexity on WikiText-2 (wikitext-2-raw-v1) using the exact
strided sliding-window method from the Hugging Face Transformers docs:
https://huggingface.co/docs/transformers/en/perplexity
"""

import argparse
import math
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

try:
    from accelerate import Accelerator
    _HAS_ACCELERATE = True
except Exception:
    _HAS_ACCELERATE = False


def get_device():
    # The HF doc uses Accelerator().device
    if _HAS_ACCELERATE:
        return Accelerator().device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def compute_ppl(model, encodings, device, stride: int):
    """
    Compute perplexity with strided sliding window, as in the HF doc.

    Returns: (ppl: float, avg_nll: float, n_tokens: int)
    """
    model.eval()

    max_length = model.config.n_positions  # GPT-2 context length (usually 1024)
    seq_len = encodings.input_ids.size(1)

    nll_sum = torch.tensor(0.0, device=device)
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride), desc="Computing PPL"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may differ from stride on last step

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # mask context tokens (ignored in loss)

        outputs = model(input_ids, labels=target_ids)
        # loss is average NLL over valid labels
        # N.B. model computes loss over trg_len - 1 labels due to internal shift
        neg_log_likelihood = outputs.loss

        # Count loss tokens (doc subtracts batch_size due to internal label shift)
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
    parser.add_argument("--model_id", type=str, default="openai-community/gpt2",
                        help="GPT-2 Small (124M). Use gpt2-medium, gpt2-large, or gpt2-xl for larger models.")
    parser.add_argument("--stride", type=int, default=512,
                        help="Doc example uses stride=512; stride=1024 is non-overlap baseline.")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    args = parser.parse_args()

    device = get_device()

    # Match doc: GPT2LMHeadModel + GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_id)

    torch_dtype = None
    if args.dtype == "fp32":
        torch_dtype = torch.float32
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16

    model = GPT2LMHeadModel.from_pretrained(args.model_id, torch_dtype=torch_dtype).to(device)

    # Match doc: load WikiText-2 raw test split, join with "\n\n"
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    ppl, avg_nll, n_tokens = compute_ppl(model, encodings, device, stride=args.stride)

    print("\n=== Results (WikiText-2, test) ===")
    print(f"Model: {args.model_id}")
    print(f"Stride: {args.stride}")
    print(f"Tokens contributing to loss: {n_tokens}")
    print(f"Average NLL per token:      {avg_nll:.6f}")
    print(f"Perplexity:                {ppl:.4f}")


if __name__ == "__main__":
    main()

