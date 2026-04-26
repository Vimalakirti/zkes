# PPL reproduction scripts

FP32 and Q15 perplexity scripts for GPT-2 and LLaMA-2 on WikiText-2 and C4,
copied from `/scratch/bjchen4_icgpu/ppl/`.

GPT-2 scripts run on WikiText-2 only. LLaMA scripts support
`--dataset {wikitext,c4}`: WikiText-2 uses the `wikitext-2-raw-v1`
test split; C4 uses `allenai/c4` `en` validation (streaming, default
1000 samples — controlled by `--num_samples`). All runs use the
HuggingFace strided sliding-window PPL method (stride=512).

## Scripts

- `gpt2.py` — GPT-2 FP32 via HuggingFace `GPT2LMHeadModel`
- `gpt2_quant.py` — GPT-2 Q15 (circuit-faithful: K=4 cubic Taylor,
  sigmoid-GELU, `-1000·SF` causal mask, two-rescale LayerNorm)
- `llama.py` — LLaMA-2 FP32 via HuggingFace `AutoModelForCausalLM`
- `llama_quant.py` — LLaMA-2 Q15 (circuit-faithful: cubic Taylor exp,
  Q15 reciprocal-RMS advice, `-1000·SF` causal mask)

Q15 uses scale factor `SF = 2^15 = 32768`.

## Reproducing the four corners

### GPT-2 Small (124M), WikiText-2

```
python gpt2.py      --model_id openai-community/gpt2 --dtype fp32
python gpt2_quant.py --model_id openai-community/gpt2
```

### LLaMA-2-7B, WikiText-2

```
python llama.py      --max_length 1024 --dtype fp32
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python llama_quant.py --max_length 1024
```

### LLaMA-2-7B, C4

```
python llama.py      --max_length 1024 --dtype fp32 --dataset c4
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python llama_quant.py --max_length 1024 --dataset c4
```

`CUDA_VISIBLE_DEVICES=<gpu>` picks the GPU. LLaMA-2-7B FP32 weights
need ~28 GB of GPU memory.

## Reference numbers (2026-04-23, stride=512, max_length=1024)

| Model | Dataset | FP32 PPL | Q15 PPL | Δ |
|---|---|---:|---:|---:|
| GPT-2 Small | WikiText-2 | 25.1704 | 25.1713 | +0.0009 (+0.004%) |
| GPT-2 Large | WikiText-2 | 16.4443 | 16.4442 | −0.0001 |
| LLaMA-2-7B  | WikiText-2 | 5.1848  | 6.2132  | +1.028 (+19.8%) |
| LLaMA-2-7B  | C4         | 6.9975  | 7.3823  | +0.385 (+5.5%) |

(C4 numbers are for LLaMA-2-7B only — GPT-2 is evaluated on
WikiText-2 since that matches the standard HuggingFace PPL baseline.)

C4 is the corpus used in zkLLM, so use `--dataset c4` for
apples-to-apples comparison with zkLLM's published numbers.
