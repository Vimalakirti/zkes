#!/usr/bin/env python3
"""
LLaMA-2-7B perplexity on WikiText-2 (wikitext-2-raw-v1) using the exact
strided sliding-window method from the Hugging Face Transformers docs:
https://huggingface.co/docs/transformers/en/perplexity
"""

import argparse
import math
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

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
def compute_ppl(model, encodings, device, stride: int, max_length: int):
    """
    Compute perplexity with strided sliding window, as in the HF doc.

    Returns: (ppl: float, avg_nll: float, n_tokens: int)
    """
    model.eval()

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
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Model ID for LLaMA-2. Default: meta-llama/Llama-2-7b-hf")
    parser.add_argument("--stride", type=int, default=512,
                        help="Stride for sliding window. Default: 512")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Max context length. LLaMA-2 supports up to 4096. Default: 4096")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "c4"],
                        help="Eval corpus. wikitext = wikitext-2-raw-v1 test, c4 = allenai/c4 en validation stream")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of C4 samples to use (ignored for wikitext)")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Trust remote code for model loading")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Custom cache directory for model weights")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir
    )

    # Determine torch dtype
    torch_dtype = None
    if args.dtype == "fp32":
        torch_dtype = torch.float32
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16

    # Load eval corpus FIRST (before changing cache for model)
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

    # Load model
    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto",  # Automatically distribute across available GPUs
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir
    )
    encodings = tokenizer(full_text, return_tensors="pt")

    print(f"Total tokens: {encodings.input_ids.size(1)}")

    # Use the smaller of model's max length and user-specified max length
    model_max_length = getattr(model.config, 'max_position_embeddings', args.max_length)
    max_length = min(args.max_length, model_max_length)
    print(f"Using max_length: {max_length}")

    ppl, avg_nll, n_tokens = compute_ppl(model, encodings, device, stride=args.stride, max_length=max_length)

    corpus_tag = "WikiText-2, test" if args.dataset == "wikitext" else f"C4 validation, {args.num_samples} samples"
    print(f"\n=== Results ({corpus_tag}) ===")
    print(f"Model: {args.model_id}")
    print(f"Stride: {args.stride}")
    print(f"Max length: {max_length}")
    print(f"Tokens contributing to loss: {n_tokens}")
    print(f"Average NLL per token:      {avg_nll:.6f}")
    print(f"Perplexity:                {ppl:.4f}")


if __name__ == "__main__":
    main()
