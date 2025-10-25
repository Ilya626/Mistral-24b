#!/usr/bin/env python3
"""Utility to merge a LoRA adapter into a base model and save the result."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        required=True,
        help="Path to the base Hugging Face model directory.",
    )
    parser.add_argument(
        "--lora-adapter",
        required=True,
        help="Path to the trained LoRA adapter directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write the merged model.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype to load the base model with (default: auto).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Device map to use when loading the model. "
            "Use 'auto' to leverage available GPUs."
        ),
    )
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def main() -> None:
    args = parse_args()

    base_model_path = Path(args.base_model).expanduser().resolve()
    lora_path = Path(args.lora_adapter).expanduser().resolve()
    output_path = Path(args.output_dir).expanduser().resolve()

    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model path not found: {base_model_path}")
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA adapter path not found: {lora_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    dtype = resolve_dtype(args.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=args.device,
    )
    model = PeftModel.from_pretrained(model, lora_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}")


if __name__ == "__main__":
    main()
