#!/usr/bin/env python3
"""Activation-informed profile builder for gradient SLERP merges.

This script analyses both source models on the grandmaster2 dataset, collects
per-layer activations for self-attention and MLP blocks, and exports a detailed
SLERP profile (40 layers) together with raw statistics for manual inspection.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

LLAMA3_BOS = "<|begin_of_text|>"
LLAMA3_SOT = "<|start_header_id|>"
LLAMA3_EOT = "<|eot_id|>"
LLAMA3_EOH = "<|end_header_id|>"

BUCKETS = ("all", "confident", "uncertain")
KIND_KEYS = ("self_attn", "mlp")
EPS = 1e-8


@dataclass
class BucketStats:
    sums: torch.Tensor
    counts: torch.Tensor

    def mean(self) -> torch.Tensor:
        return torch.where(self.counts > 0, self.sums / self.counts, torch.zeros_like(self.sums))


@dataclass
class LayerStats:
    buckets: Dict[str, BucketStats]

    @classmethod
    def create(cls, num_layers: int) -> "LayerStats":
        return cls({bucket: BucketStats(torch.zeros(num_layers), torch.zeros(num_layers)) for bucket in BUCKETS})


@dataclass
class ModelStatistics:
    layer_stats: Dict[str, LayerStats]
    nll_sums: Dict[str, float]
    nll_counts: Dict[str, int]
    hooks: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list)
    current_buffers: Dict[str, List[Optional[torch.Tensor]]] = field(default_factory=dict)

    @classmethod
    def create(cls, num_layers: int) -> "ModelStatistics":
        layer_stats = {kind: LayerStats.create(num_layers) for kind in KIND_KEYS}
        nll_sums = {bucket: 0.0 for bucket in BUCKETS}
        nll_counts = {bucket: 0 for bucket in BUCKETS}
        current_buffers = {kind: [None] * num_layers for kind in KIND_KEYS}
        return cls(layer_stats, nll_sums, nll_counts, [], current_buffers)

    def start_step(self) -> None:
        for kind in KIND_KEYS:
            buffers = self.current_buffers[kind]
            for idx in range(len(buffers)):
                buffers[idx] = None

    def register_hook(self, layer_module: torch.nn.Module, kind: str, layer_idx: int) -> None:
        def hook(_module: torch.nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor | Tuple[torch.Tensor, ...]):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Convert to float32 and move to CPU to keep memory pressure low.
            norms = torch.linalg.vector_norm(hidden.detach().to(torch.float32), dim=-1)
            # Drop the first token later during commit to align with shifted labels.
            self.current_buffers[kind][layer_idx] = norms.cpu()

        handle = layer_module.register_forward_hook(hook)
        self.hooks.append(handle)

    def commit(self, mask_all: torch.Tensor, mask_conf: torch.Tensor, mask_uncertain: torch.Tensor, token_logprobs: torch.Tensor) -> None:
        mask_all_cpu = mask_all.cpu()
        mask_conf_cpu = mask_conf.cpu()
        mask_uncertain_cpu = mask_uncertain.cpu()
        logprobs_cpu = token_logprobs.detach().to(torch.float32).cpu()

        for kind in KIND_KEYS:
            buffers = self.current_buffers[kind]
            for layer_idx, norms in enumerate(buffers):
                if norms is None:
                    raise RuntimeError(f"Missing activations for {kind} layer {layer_idx}. Hooks might not be registered correctly.")
                # Align with shifted labels (ignore the BOS token).
                trimmed = norms[:, 1:]
                if trimmed.shape != mask_all_cpu.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {kind} layer {layer_idx}: norms {trimmed.shape}, mask {mask_all_cpu.shape}"
                    )
                for bucket, mask in zip(BUCKETS, (mask_all_cpu, mask_conf_cpu, mask_uncertain_cpu)):
                    bucket_stats = self.layer_stats[kind].buckets[bucket]
                    masked_values = trimmed.masked_select(mask)
                    bucket_stats.sums[layer_idx] += masked_values.sum()
                    bucket_stats.counts[layer_idx] += mask.sum().item()

        for bucket, mask in zip(BUCKETS, (mask_all_cpu, mask_conf_cpu, mask_uncertain_cpu)):
            selected = logprobs_cpu.masked_select(mask)
            if selected.numel() == 0:
                continue
            self.nll_sums[bucket] += float(-selected.sum().item())
            self.nll_counts[bucket] += int(selected.numel())

    def finalize(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "layers": {},
            "nll": {},
        }
        for kind in KIND_KEYS:
            data["layers"][kind] = {}
            for bucket, stats in self.layer_stats[kind].buckets.items():
                data["layers"][kind][bucket] = {
                    "sum": stats.sums.tolist(),
                    "count": stats.counts.tolist(),
                    "mean": stats.mean().tolist(),
                }
        for bucket in BUCKETS:
            count = self.nll_counts[bucket]
            mean = (self.nll_sums[bucket] / count) if count > 0 else None
            data["nll"][bucket] = {
                "sum": self.nll_sums[bucket],
                "count": count,
                "mean": mean,
            }
        return data

    def remove_hooks(self) -> None:
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an activation-informed SLERP profile from grandmaster2 prompts")
    parser.add_argument("--vistral-model", type=str, default=None, help="Path or identifier of the Vistral model")
    parser.add_argument("--cydonia-model", type=str, default=None, help="Path or identifier of the Cydonia model")
    parser.add_argument(
        "--single-model",
        type=str,
        choices=("vistral", "cydonia"),
        default=None,
        help="Restrict profiling to a single model (skips loading the other one)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Model identifier to use as base_model in the generated YAML (defaults to the first profiled model)",
    )
    parser.add_argument("--dataset-split", type=str, default="train[:256]", help="grandmaster2 split to use")
    parser.add_argument("--max-prompts", type=int, default=256, help="Maximum prompts to analyse (<= dataset slice)")
    parser.add_argument(
        "--sample-prompts",
        type=int,
        default=0,
        help="Randomly sample this many prompts from the loaded slice (0 keeps all prompts)",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Seed for the prompt sampler (defaults to an OS-provided seed)",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Probability threshold that marks a token as confident",
    )
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=0.1,
        help="Probability threshold that marks a token as uncertain",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt used in the shared Llama 3 template",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="profiles/grandmaster2",
        help="Directory where the profile and raw metrics will be stored",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Torch dtype for model weights",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Forward trust_remote_code to transformers loaders")
    return parser.parse_args()


def load_prompts(split: str, max_prompts: int, sample_prompts: int, sample_seed: Optional[int]) -> List[str]:
    ds: Dataset = load_dataset("grandmaster2", split=split)  # type: ignore[arg-type]
    prompts = [str(example["prompt"]) for example in ds]
    if max_prompts > 0:
        prompts = prompts[:max_prompts]
    if sample_prompts > 0:
        if sample_prompts > len(prompts):
            raise ValueError(
                f"Requested {sample_prompts} prompts but only {len(prompts)} are available after max_prompts filtering"
            )
        rng = random.Random(sample_seed)
        prompts = rng.sample(prompts, sample_prompts)
    if not prompts:
        raise RuntimeError("grandmaster2 slice produced an empty prompt list")
    return prompts


def apply_llama3_template(system_prompt: str, user_prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append(("system", system_prompt))
    messages.append(("user", user_prompt))
    parts = [LLAMA3_BOS]
    for role, content in messages:
        parts.append(f"{LLAMA3_SOT}{role}{LLAMA3_EOH}\n{content}{LLAMA3_EOT}")
    parts.append(f"{LLAMA3_SOT}assistant{LLAMA3_EOH}\n")
    return "".join(parts)


def prepare_template_prompts(prompts: Iterable[str], system_prompt: str) -> List[str]:
    return [apply_llama3_template(system_prompt, prompt) for prompt in prompts]


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def build_profiler(model: AutoModelForCausalLM) -> ModelStatistics:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError("Unsupported model architecture: expected .model.layers with decoder blocks")
    layers = model.model.layers  # type: ignore[attr-defined]
    num_layers = len(layers)
    stats = ModelStatistics.create(num_layers)
    for idx, layer in enumerate(layers):
        if hasattr(layer, "self_attn"):
            stats.register_hook(layer.self_attn, "self_attn", idx)
        else:
            raise AttributeError(f"Layer {idx} is missing self_attn module")
        if hasattr(layer, "mlp"):
            stats.register_hook(layer.mlp, "mlp", idx)
        else:
            raise AttributeError(f"Layer {idx} is missing mlp module")
    return stats


def resolve_model_device(model: AutoModelForCausalLM) -> torch.device:
    if hasattr(model, "device"):
        return model.device  # type: ignore[return-value]
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_inputs_to_device(inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: tensor.to(device) for key, tensor in inputs.items()}


def compute_masks_and_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask_all = shift_labels != -100
    if not mask_all.any():
        empty = torch.zeros_like(shift_labels, dtype=torch.float32)
        return mask_all, empty, empty
    shift_labels_clamped = shift_labels.clone()
    shift_labels_clamped[~mask_all] = 0
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    gathered = torch.gather(log_probs, 2, shift_labels_clamped.unsqueeze(-1)).squeeze(-1)
    probs = torch.exp(gathered)
    return mask_all, probs, gathered


def derive_bucket_masks(mask_all: torch.Tensor, probs: torch.Tensor, confidence_threshold: float, uncertainty_threshold: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask_conf = mask_all & (probs >= confidence_threshold)
    mask_uncertain = mask_all & (probs <= uncertainty_threshold)
    return mask_all, mask_conf, mask_uncertain


def summarize_weights(vistral_values: torch.Tensor, cydonia_values: torch.Tensor) -> List[float]:
    combined = vistral_values + cydonia_values + EPS
    weights = torch.where(combined > 0, vistral_values / combined, torch.zeros_like(vistral_values))
    return weights.tolist()


def compute_default_weight(vistral_nll: Optional[float], cydonia_nll: Optional[float]) -> float:
    if vistral_nll is None and cydonia_nll is None:
        return 0.5
    if vistral_nll is None:
        return 0.0
    if cydonia_nll is None:
        return 1.0
    vistral_score = 1.0 / max(vistral_nll, EPS)
    cydonia_score = 1.0 / max(cydonia_nll, EPS)
    return float(vistral_score / (vistral_score + cydonia_score))


def export_yaml_profile(path: Path, base_model: str, vistral_model: str, cydonia_model: str, attn_weights: List[float], mlp_weights: List[float], default_weight: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "merge_method": "slerp",
        "base_model": base_model,
        "dtype": "bfloat16",
        "models": [
            {"model": vistral_model},
            {"model": cydonia_model},
        ],
        "parameters": {
            "t": [
                {"filter": "self_attn", "value": attn_weights},
                {"filter": "mlp", "value": mlp_weights},
                {"value": default_weight},
            ],
            "embed_slerp": True,
        },
        "tokenizer_source": "union",
    }
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Auto-generated AIM profile. Edit with care.\n")
        handle.write("# Generated at: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        handle.write("# Source models: Vistral='" + vistral_model + "', Cydonia='" + cydonia_model + "'\n")
        handle.write(_json_to_yaml(content))


def _json_to_yaml(node, indent: int = 0) -> str:
    space = " " * indent
    if isinstance(node, dict):
        lines = []
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{space}{key}:")
                lines.append(_json_to_yaml(value, indent + 2))
            else:
                lines.append(f"{space}{key}: {value}")
        return "\n".join(lines) + "\n"
    if isinstance(node, list):
        lines = []
        for item in node:
            if isinstance(item, (dict, list)):
                lines.append(f"{space}-")
                lines.append(_json_to_yaml(item, indent + 2))
            else:
                lines.append(f"{space}- {item}")
        return "\n".join(lines) + "\n"
    return f"{space}{node}\n"


def export_metrics_json(path: Path, metadata: Dict[str, object], model_stats: Dict[str, Dict[str, object]]) -> None:
    payload = {
        "metadata": metadata,
        "models": model_stats,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def export_metrics_csv(path: Path, model_stats: Dict[str, Dict[str, object]]) -> None:
    rows: List[Tuple[str, str, str, int, float, float]] = []
    for model_name, stats in model_stats.items():
        layer_section = stats["layers"]
        for kind in KIND_KEYS:
            for bucket in BUCKETS:
                means = layer_section[kind][bucket]["mean"]
                counts = layer_section[kind][bucket]["count"]
                for layer_idx, (mean_value, count_value) in enumerate(zip(means, counts)):
                    rows.append(
                        (
                            model_name,
                            kind,
                            bucket,
                            layer_idx,
                            float(mean_value),
                            float(count_value),
                        )
                    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "kind", "bucket", "layer", "mean_norm", "token_count"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    dtype = resolve_dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = "auto" if torch.cuda.is_available() else None

    if args.vistral_model is None and args.cydonia_model is None:
        raise RuntimeError("At least one model must be provided for profiling")

    if args.single_model == "vistral" and args.vistral_model is None:
        raise RuntimeError("--single-model vistral requires --vistral-model to be provided")
    if args.single_model == "cydonia" and args.cydonia_model is None:
        raise RuntimeError("--single-model cydonia requires --cydonia-model to be provided")

    run_vistral = args.vistral_model is not None and (args.single_model in (None, "vistral"))
    run_cydonia = args.cydonia_model is not None and (args.single_model in (None, "cydonia"))

    if not run_vistral and not run_cydonia:
        raise RuntimeError("Nothing to do: the selected --single-model does not have a corresponding model id")

    prompts = load_prompts(args.dataset_split, args.max_prompts, args.sample_prompts, args.sample_seed)
    templated_prompts = prepare_template_prompts(prompts, args.system_prompt)

    def load_model_and_tokenizer(model_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer, ModelStatistics]:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
        ensure_pad_token(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=args.trust_remote_code,
        )
        if device_map is None:
            model.to(device)
        model.eval()
        profiler = build_profiler(model)
        return model, tokenizer, profiler

    vistral_model: Optional[AutoModelForCausalLM] = None
    vistral_tokenizer: Optional[AutoTokenizer] = None
    vistral_stats: Optional[ModelStatistics] = None
    cydonia_model: Optional[AutoModelForCausalLM] = None
    cydonia_tokenizer: Optional[AutoTokenizer] = None
    cydonia_stats: Optional[ModelStatistics] = None

    if run_vistral:
        print("Loading Vistral model...", flush=True)
        vistral_model, vistral_tokenizer, vistral_stats = load_model_and_tokenizer(args.vistral_model)  # type: ignore[arg-type]

    if run_cydonia:
        print("Loading Cydonia model...", flush=True)
        cydonia_model, cydonia_tokenizer, cydonia_stats = load_model_and_tokenizer(args.cydonia_model)  # type: ignore[arg-type]

    if run_vistral and run_cydonia and vistral_model is not None and cydonia_model is not None:
        num_layers = len(vistral_model.model.layers)  # type: ignore[attr-defined]
        if num_layers != len(cydonia_model.model.layers):  # type: ignore[attr-defined]
            raise RuntimeError("Models must have the same number of layers for SLERP profiling")

    batches = [
        templated_prompts[i : i + args.batch_size]
        for i in range(0, len(templated_prompts), args.batch_size)
    ]

    def run_batches(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        stats: ModelStatistics,
        description: str,
    ) -> None:
        model_device = resolve_model_device(model)
        for batch_prompts in tqdm(batches, desc=description, unit="batch"):
            inputs = tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            ensure_pad_token(tokenizer)
            inputs = move_inputs_to_device(inputs, model_device)
            labels = inputs["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            stats.start_step()
            with torch.inference_mode():
                outputs = model(**inputs, labels=labels)
            logits = outputs.logits.detach()
            mask_all, probs, token_logprobs = compute_masks_and_logprobs(logits, labels)
            mask_all, mask_conf, mask_uncertain = derive_bucket_masks(
                mask_all, probs, args.confidence_threshold, args.uncertainty_threshold
            )
            stats.commit(mask_all, mask_conf, mask_uncertain, token_logprobs)

    model_profiles: Dict[str, Dict[str, object]] = {}

    if run_vistral and vistral_model is not None and vistral_tokenizer is not None and vistral_stats is not None:
        print("Profiling Vistral activations...", flush=True)
        run_batches(vistral_model, vistral_tokenizer, vistral_stats, "Vistral batches")
        model_profiles["vistral"] = vistral_stats.finalize()

    if run_cydonia and cydonia_model is not None and cydonia_tokenizer is not None and cydonia_stats is not None:
        print("Profiling Cydonia activations...", flush=True)
        run_batches(cydonia_model, cydonia_tokenizer, cydonia_stats, "Cydonia batches")
        model_profiles["cydonia"] = cydonia_stats.finalize()

    if not model_profiles:
        raise RuntimeError("No profiles were generated. Check model loading configuration.")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_dir = Path(args.output_dir) / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    generated_profile_path: Optional[Path] = None

    if "vistral" in model_profiles and "cydonia" in model_profiles:
        vistral_profile = model_profiles["vistral"]
        cydonia_profile = model_profiles["cydonia"]

        vistral_attn = torch.tensor(vistral_profile["layers"]["self_attn"]["all"]["mean"], dtype=torch.float32)
        cydonia_attn = torch.tensor(cydonia_profile["layers"]["self_attn"]["all"]["mean"], dtype=torch.float32)
        vistral_mlp = torch.tensor(vistral_profile["layers"]["mlp"]["all"]["mean"], dtype=torch.float32)
        cydonia_mlp = torch.tensor(cydonia_profile["layers"]["mlp"]["all"]["mean"], dtype=torch.float32)

        attn_weights = summarize_weights(vistral_attn, cydonia_attn)
        mlp_weights = summarize_weights(vistral_mlp, cydonia_mlp)

        default_weight = compute_default_weight(
            vistral_profile["nll"]["all"]["mean"],
            cydonia_profile["nll"]["all"]["mean"],
        )

        profile_path = base_dir / "aim_profile.yml"
        export_yaml_profile(
            profile_path,
            args.base_model
            or args.vistral_model
            or args.cydonia_model
            or "",
            args.vistral_model or "",
            args.cydonia_model or "",
            attn_weights,
            mlp_weights,
            default_weight,
        )
        generated_profile_path = profile_path

    metadata = {
        "dataset": "grandmaster2",
        "dataset_split": args.dataset_split,
        "max_prompts": args.max_prompts,
        "sample_prompts": args.sample_prompts,
        "sample_seed": args.sample_seed,
        "batch_size": args.batch_size,
        "confidence_threshold": args.confidence_threshold,
        "uncertainty_threshold": args.uncertainty_threshold,
        "dtype": args.dtype,
        "base_model": args.base_model or args.vistral_model or args.cydonia_model,
        "timestamp": timestamp,
    }

    export_metrics_json(base_dir / "metrics.json", metadata, model_profiles)
    export_metrics_csv(base_dir / "metrics.csv", model_profiles)

    if generated_profile_path is not None:
        print("Profile saved to", generated_profile_path)
    elif args.single_model is not None:
        print("Single-model run: SLERP profile generation skipped.")
    else:
        print("Only one model was profiled; SLERP profile generation skipped.")

    print("Raw metrics saved to", base_dir)

    if vistral_stats is not None:
        vistral_stats.remove_hooks()
    if cydonia_stats is not None:
        cydonia_stats.remove_hooks()


if __name__ == "__main__":
    main()
