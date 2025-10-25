#!/usr/bin/env python3
"""Quick evaluation script for merged models using a prompt dataset.

This helper loads a (sub)set of prompts from the grandmaster2 dataset (or a
user-provided file/dataset) and generates responses with the supplied model.
The outputs are saved under ``evaluation_logs`` for later comparison.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quick qualitative checks on a model")
    parser.add_argument(
        "--model",
        type=str,
        default="/workspace/final_model",
        help="Path or identifier of the model to evaluate.",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Optional path to a JSON/JSONL/TXT file containing prompts.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="grandmaster2",
        help="Dataset name or path (Hugging Face Datasets) used when prompts-file is not provided.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train[:16]",
        help="Dataset split or slice to load (HF datasets syntax).",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
        help="Column name holding prompts in the dataset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=16,
        help="Maximum number of prompts to evaluate.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_logs",
        help="Directory to store JSON logs with generations.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code flag to transformers loaders.",
    )
    parser.add_argument(
        "--use-bfloat16",
        action="store_true",
        help="Load model weights in bfloat16 when supported.",
    )
    return parser.parse_args()


def load_prompts_from_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompts file '{path}' does not exist")

    if path.suffix.lower() in {".json", ".jsonl"}:
        prompts: List[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    if "prompt" in payload:
                        prompts.append(str(payload["prompt"]))
                    elif "instruction" in payload:
                        prompts.append(str(payload["instruction"]))
                elif isinstance(payload, str):
                    prompts.append(payload)
        if not prompts:
            raise ValueError(
                f"No prompts found in JSON(L) file '{path}'. Please ensure it contains a 'prompt' or 'instruction' field."
            )
        return prompts

    # Treat everything else as plain text (one prompt per line)
    with path.open("r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in text file '{path}'.")
    return prompts


def load_prompts_from_dataset(
    dataset_name: str,
    split: str,
    prompt_column: str,
    max_samples: int,
) -> List[str]:
    ds: Dataset = load_dataset(dataset_name, split=split)  # type: ignore[arg-type]
    if prompt_column not in ds.column_names:
        raise ValueError(
            f"Column '{prompt_column}' is not present in dataset '{dataset_name}'. Available columns: {ds.column_names}"
        )

    num_samples = min(len(ds), max_samples) if max_samples > 0 else len(ds)
    prompts = [str(example[prompt_column]) for example in ds.select(range(num_samples))]
    return prompts


def prepare_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompts_file:
        return load_prompts_from_file(Path(args.prompts_file))[: args.max_samples]
    return load_prompts_from_dataset(args.dataset, args.dataset_split, args.prompt_column, args.max_samples)


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()

    prompts = prepare_prompts(args)
    if not prompts:
        raise RuntimeError("No prompts available for evaluation.")

    dtype = torch.bfloat16 if args.use_bfloat16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    generations = []
    log_lines = []

    with torch.inference_mode():
        for idx, prompt in enumerate(prompts, start=1):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )[0]
            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            generations.append(
                {
                    "prompt": prompt,
                    "generation": generated_text,
                }
            )
            log_lines.append(f"[{idx:02d}] Prompt: {prompt}\n---\n{generated_text}\n")
            print(f"[{idx:02d}] Prompt:\n{prompt}\n")
            print(f"[{idx:02d}] Generation:\n{generated_text}\n{'=' * 60}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = ensure_output_dir(Path(args.output_dir))
    output_path = output_dir / f"evaluation_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump({"model": args.model, "prompts": generations}, fp, ensure_ascii=False, indent=2)

    print(f"\n>>> Оценка завершена. Лог сохранён: {output_path}")


if __name__ == "__main__":
    main()
