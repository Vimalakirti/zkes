#!/usr/bin/env python3
"""
Extract all constants from an ONNX model, quantize float tensors with a fixed-point scale,
and save each tensor as:
  - <name>.bin   (raw little-endian bytes, row-major)
  - <name>.json  (metadata: shape, dtype, scale_factor, etc.)
Also writes a top-level manifest.json summarizing everything.

Quantization rule (for float32/float64 by default):
  q = round(x * 2^sf) stored as int64

Example:
  ./extract_constants_binjson.py model.onnx out_dir --sf 16
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper


INVALID_CHARS = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be a valid filename."""
    result = name
    for char in INVALID_CHARS:
        result = result.replace(char, "_")
    result = result.strip()
    # Avoid empty filenames
    return result if result else "unnamed"


def stable_short_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def ensure_unique_stem(stem: str, used: set, original_name: str) -> str:
    """Avoid collisions after sanitization by appending a short hash if needed."""
    if stem not in used:
        used.add(stem)
        return stem
    # Collision: add hash of original name
    stem2 = f"{stem}__{stable_short_hash(original_name)}"
    if stem2 not in used:
        used.add(stem2)
        return stem2
    # Extremely unlikely, but just in case
    i = 2
    while True:
        stem3 = f"{stem2}__{i}"
        if stem3 not in used:
            used.add(stem3)
            return stem3
        i += 1


def dtype_to_str(dt: np.dtype) -> str:
    # Normalize dtype strings a bit
    dt = np.dtype(dt)
    return dt.name  # e.g., "float32", "int64", "uint8"


def to_little_endian_contiguous(arr: np.ndarray) -> np.ndarray:
    """Ensure C-contiguous and little-endian for stable binary files."""
    arr = np.ascontiguousarray(arr)
    dt = arr.dtype
    # If dtype is big-endian, convert to little-endian
    if dt.byteorder == ">" or (dt.byteorder == "=" and np.dtype(dt).byteorder == ">"):
        arr = arr.byteswap().newbyteorder("<")
    elif dt.byteorder == "=":
        # Native endian; we assume little-endian on typical platforms, but enforce anyway:
        arr = arr.astype(dt.newbyteorder("<"), copy=False)
    else:
        # already little-endian ("<") or not-applicable ("|")
        pass
    return arr


def quantize_float_tensor(
    x: np.ndarray,
    sf: int,
    store_int_dtype: np.dtype = np.int64,
    rounding: str = "round",
) -> np.ndarray:
    """
    Quantize float tensor x into integer tensor:
      q = round(x * 2^sf) as int64 (default)

    rounding: "round" uses bankers rounding (np.round).
              If you want round-half-away-from-zero, change this function.
    """
    scale = 1 << sf

    # Use float64 accumulator for safety even if input is float32
    xf = x.astype(np.float64, copy=False) * float(scale)

    if rounding == "round":
        q = np.round(xf)
    else:
        raise ValueError(f"Unsupported rounding mode: {rounding}")

    # Cast to integer
    q = q.astype(store_int_dtype)
    return q


def save_tensor_bin_json(
    tensor: np.ndarray,
    out_dir: Path,
    file_stem: str,
    meta: Dict[str, Any],
) -> Tuple[Path, Path, int]:
    """
    Save tensor to <file_stem>.bin (raw bytes) and <file_stem>.json (metadata).
    Returns (bin_path, json_path, num_bytes).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    tensor_le = to_little_endian_contiguous(tensor)
    bin_path = out_dir / f"{file_stem}.bin"
    json_path = out_dir / f"{file_stem}.json"

    data = tensor_le.tobytes(order="C")
    with open(bin_path, "wb") as f:
        f.write(data)

    meta_out = dict(meta)
    meta_out.update(
        {
            "shape": list(tensor.shape),
            "stored_dtype": dtype_to_str(tensor_le.dtype),
            "num_elements": int(tensor.size),
            "num_bytes": int(len(data)),
            "layout": "row-major",
            "endianness": "little",
            "bin_file": bin_path.name,
        }
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, sort_keys=True)

    return bin_path, json_path, len(data)


def extract_constants_binjson(
    onnx_path: str,
    output_dir: str,
    sf: int,
    quantize_dtypes: Tuple[np.dtype, ...] = (np.float32, np.float64),
    store_non_float: bool = True,
    store_int_dtype: np.dtype = np.int64,
) -> None:
    """
    Extract constants from initializers and Constant nodes. For float32/float64 constants,
    store round(x * 2^sf) as int64. Save each tensor as .bin + .json plus a manifest.json.
    """
    onnx_path_p = Path(onnx_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ONNX model from: {onnx_path_p}")
    model = onnx.load(str(onnx_path_p))

    used_stems: set = set()
    manifest: Dict[str, Any] = {
        "onnx_path": str(onnx_path_p),
        "output_dir": str(out_dir),
        "scale_factor": int(sf),
        "scale": int(1 << sf),
        "quantized_float_dtypes": [np.dtype(d).name for d in quantize_dtypes],
        "stored_int_dtype": np.dtype(store_int_dtype).name,
        "tensors": [],  # filled below
    }

    tensors_dir = out_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    def handle_tensor(
        name: str,
        tensor: np.ndarray,
        source: str,
        original_dtype: str,
    ) -> None:
        safe = sanitize_filename(name)
        stem = ensure_unique_stem(safe, used_stems, original_name=name)

        is_quantized = False
        meta: Dict[str, Any] = {
            "name": name,
            "source": source,  # "initializer" or "constant_node"
            "original_dtype": original_dtype,
        }

        if tensor.dtype in quantize_dtypes:
            # Quantize
            q = quantize_float_tensor(tensor, sf=sf, store_int_dtype=store_int_dtype)
            is_quantized = True
            meta.update(
                {
                    "quantized": True,
                    "quantization": "round(x * 2^sf)",
                    "scale_factor": int(sf),
                    "scale": int(1 << sf),
                }
            )
            saved = q
        else:
            if not store_non_float:
                return
            meta.update({"quantized": False})
            saved = tensor
        
        if "attn.c_attn.weight" in stem:
            saved_q = saved[:, :768]
            saved_k = saved[:, 768:1536]
            saved_v = saved[:, 1536:]

            bin_path_q, json_path_q, num_bytes_q = save_tensor_bin_json(
                saved_q, tensors_dir, stem + "_q", meta
            )
            bin_path_k, json_path_k, num_bytes_k = save_tensor_bin_json(
                saved_k, tensors_dir, stem + "_k", meta
            )
            bin_path_v, json_path_v, num_bytes_v = save_tensor_bin_json(
                saved_v, tensors_dir, stem + "_v", meta
            )
            manifest["tensors"].append(
                {
                    "name": name + "_q",
                    "file_stem": stem + "_q",
                    "bin": str(Path("tensors") / bin_path_q.name),
                    "json": str(Path("tensors") / json_path_q.name),
                    "source": source,
                    "original_dtype": original_dtype,
                    "stored_dtype": dtype_to_str(saved_q.dtype),
                    "shape": list(saved_q.shape),
                    "num_bytes": int(num_bytes_q),
                    "quantized": bool(is_quantized),
                }
            )
        
            manifest["tensors"].append(
                {
                    "name": name + "_k",
                    "file_stem": stem + "_k",
                    "bin": str(Path("tensors") / bin_path_k.name),
                    "json": str(Path("tensors") / json_path_k.name),
                    "source": source,
                    "original_dtype": original_dtype,
                    "stored_dtype": dtype_to_str(saved_k.dtype),
                    "shape": list(saved_k.shape),
                    "num_bytes": int(num_bytes_k),
                    "quantized": bool(is_quantized),
                }
            )
        
            manifest["tensors"].append(
                {
                    "name": name + "_v",
                    "file_stem": stem + "_v",
                    "bin": str(Path("tensors") / bin_path_v.name),
                    "json": str(Path("tensors") / json_path_v.name),
                    "source": source,
                    "original_dtype": original_dtype,
                    "stored_dtype": dtype_to_str(saved_v.dtype),
                    "shape": list(saved_v.shape),
                    "num_bytes": int(num_bytes_v),
                    "quantized": bool(is_quantized),
                }
            )

        elif "attn.c_attn.bias" in stem:
            saved_q = saved[:768]
            saved_k = saved[768:1536]
            saved_v = saved[1536:]

            bin_path_q, json_path_q, num_bytes_q = save_tensor_bin_json(
                saved_q, tensors_dir, stem + "_q", meta
            )
            bin_path_k, json_path_k, num_bytes_k = save_tensor_bin_json(
                saved_k, tensors_dir, stem + "_k", meta
            )
            bin_path_v, json_path_v, num_bytes_v = save_tensor_bin_json(
                saved_v, tensors_dir, stem + "_v", meta
            )

            manifest["tensors"].append(
                {
                    "name": name + "_q",
                    "file_stem": stem + "_q",
                    "bin": str(Path("tensors") / bin_path_q.name),
                    "json": str(Path("tensors") / json_path_q.name),
                    "source": source,
                    "original_dtype": original_dtype,
                    "stored_dtype": dtype_to_str(saved_q.dtype),
                    "shape": list(saved_q.shape),
                    "num_bytes": int(num_bytes_q),
                    "quantized": bool(is_quantized),
                }
            )
        
            manifest["tensors"].append(
                {
                    "name": name + "_k",
                    "file_stem": stem + "_k",
                    "bin": str(Path("tensors") / bin_path_k.name),
                    "json": str(Path("tensors") / json_path_k.name),
                    "source": source,
                    "original_dtype": original_dtype,
                    "stored_dtype": dtype_to_str(saved_k.dtype),
                    "shape": list(saved_k.shape),
                    "num_bytes": int(num_bytes_k),
                    "quantized": bool(is_quantized),
                }
            )
        
            manifest["tensors"].append(
                {
                    "name": name + "_v",
                    "file_stem": stem + "_v",
                    "bin": str(Path("tensors") / bin_path_v.name),
                    "json": str(Path("tensors") / json_path_v.name),
                    "source": source,
                    "original_dtype": original_dtype,
                    "stored_dtype": dtype_to_str(saved_v.dtype),
                    "shape": list(saved_v.shape),
                    "num_bytes": int(num_bytes_v),
                    "quantized": bool(is_quantized),
                }
            )
        else:
            bin_path, json_path, num_bytes = save_tensor_bin_json(
                saved, tensors_dir, stem, meta
            )

            manifest["tensors"].append(
                {
                    "name": name,
                    "file_stem": stem,
                    "bin": str(Path("tensors") / bin_path.name),
                    "json": str(Path("tensors") / json_path.name),
                    "source": source,
                    "original_dtype": original_dtype,
                    "stored_dtype": dtype_to_str(saved.dtype),
                    "shape": list(saved.shape),
                    "num_bytes": int(num_bytes),
                    "quantized": bool(is_quantized),
                }
            )

            print(
                f"  Saved: {name} -> {bin_path} (shape: {saved.shape}, "
                f"stored dtype: {saved.dtype}, original dtype: {original_dtype}, "
                f"{'QUANTIZED' if is_quantized else 'raw'})"
            )

    # 1) Initializers
    print(f"Found {len(model.graph.initializer)} initializers")
    for initializer in model.graph.initializer:
        name = initializer.name
        arr = numpy_helper.to_array(initializer)
        handle_tensor(
            name=name,
            tensor=arr,
            source="initializer",
            original_dtype=dtype_to_str(arr.dtype),
        )

    # 2) Constant nodes
    constant_nodes = 0
    saved_constants = 0
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        constant_nodes += 1
        # Common case: attribute "value" with TensorProto in attr.t
        for attr in node.attribute:
            if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                arr = numpy_helper.to_array(attr.t)
                name = node.output[0] if node.output else f"constant_{constant_nodes}"
                handle_tensor(
                    name=name,
                    tensor=arr,
                    source="constant_node",
                    original_dtype=dtype_to_str(arr.dtype),
                )
                saved_constants += 1

    print(f"Found {constant_nodes} Constant nodes (saved {saved_constants})")

    # Write top-level manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"\nWrote manifest: {manifest_path}")
    print(f"All tensors saved under: {tensors_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract ONNX constants, quantize float constants, and write .bin + .json."
    )
    parser.add_argument("onnx_path", type=str, help="Path to the ONNX model file")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--sf", type=int, required=True, help="Scale factor sf (uses 2^sf)")
    parser.add_argument(
        "--quantize",
        type=str,
        default="float32,float64",
        help="Comma-separated dtypes to quantize (default: float32,float64). "
             "Supported: float16,float32,float64",
    )
    parser.add_argument(
        "--store-non-float",
        action="store_true",
        help="Also store non-quantized tensors (ints/bools/etc). Default: store them.",
    )
    parser.add_argument(
        "--skip-non-float",
        action="store_true",
        help="Skip storing non-quantized tensors (ints/bools/etc).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.onnx_path):
        print(f"Error: ONNX file not found: {args.onnx_path}")
        return 1

    # dtype parsing
    allowed = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
    }
    q = []
    for part in args.quantize.split(","):
        part = part.strip()
        if not part:
            continue
        if part not in allowed:
            raise ValueError(f"Unsupported dtype in --quantize: {part}")
        q.append(allowed[part])
    quantize_dtypes = tuple(q) if q else (np.float32, np.float64)

    store_non_float = True
    if args.skip_non_float:
        store_non_float = False
    elif args.store_non_float:
        store_non_float = True

    extract_constants_binjson(
        args.onnx_path,
        args.output_dir,
        sf=args.sf,
        quantize_dtypes=quantize_dtypes,
        store_non_float=store_non_float,
        store_int_dtype=np.int64,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
