from __future__ import annotations

import os
import re
from types import ModuleType
from typing import Any

from huggingface_hub import hf_hub_download, list_repo_files

_torch: ModuleType | None = None
_safetensors_available: bool = False


def _get_torch() -> ModuleType:
    global _torch
    if _torch is None:
        try:
            import torch

            _torch = torch
        except ImportError as e:
            raise ImportError(
                "torch is required for weight analysis. "
                "Install with: pip install dr-hf[weights]"
            ) from e
    return _torch


def _check_safetensors() -> bool:
    global _safetensors_available
    try:
        from safetensors import safe_open  # noqa: F401

        _safetensors_available = True
    except ImportError:
        _safetensors_available = False
    return _safetensors_available


def discover_model_weight_files(repo_id: str, branch: str = "main") -> list[str]:
    try:
        all_files = list_repo_files(repo_id=repo_id, revision=branch)

        weight_patterns = [
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model-00001-of-00001.bin",
        ]

        weight_files = []
        for file in all_files:
            if any(pattern in file for pattern in weight_patterns):
                weight_files.append(file)
            elif ("pytorch_model-" in file and file.endswith(".bin")) or file.endswith(
                ".safetensors"
            ):
                weight_files.append(file)

        return sorted(weight_files)

    except Exception:
        return []


def download_model_weights(
    repo_id: str, branch: str, filename: str, local_dir: str | None = None
) -> tuple[str | None, bool, str]:
    try:
        file_path = hf_hub_download(
            repo_id=repo_id, filename=filename, revision=branch, local_dir=local_dir
        )
        return file_path, True, ""
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "Entry Not Found" in error_msg:
            return None, False, f"Weight file {filename} not available"
        return None, False, error_msg


def calculate_weight_statistics(weight_path: str) -> dict[str, Any]:
    torch = _get_torch()

    try:
        if weight_path.endswith(".safetensors"):
            if not _check_safetensors():
                return {"error": "safetensors library not available"}
            from safetensors import safe_open

            weights: dict[str, Any] = {}
            with safe_open(weight_path, framework="pt") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
        else:
            weights = torch.load(weight_path, map_location="cpu")

        if not isinstance(weights, dict):
            return {"error": "Weights file is not a dictionary"}

        stats: dict[str, Any] = {
            "file_path": weight_path,
            "file_size_mb": round(os.path.getsize(weight_path) / (1024 * 1024), 2),
            "num_tensors": len(weights),
            "tensor_info": {},
            "parameter_stats": {},
            "layer_analysis": analyze_layer_structure(weights),
        }

        total_params = 0
        tensor_stats = []

        for name, tensor in weights.items():
            if torch.is_tensor(tensor):
                tensor_params = tensor.numel()
                total_params += tensor_params

                tensor_info = {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "parameters": tensor_params,
                    "size_mb": round(
                        tensor.numel() * tensor.element_size() / (1024 * 1024), 3
                    ),
                    "statistics": calculate_tensor_stats(tensor),
                }
                tensor_stats.append(tensor_info)

        stats["tensor_info"] = tensor_stats
        stats["parameter_stats"] = {
            "total_parameters": total_params,
            "total_parameters_millions": round(total_params / 1_000_000, 2),
            "total_parameters_billions": round(total_params / 1_000_000_000, 3),
        }

        stats["global_statistics"] = calculate_global_weight_stats(weights)

        return stats

    except Exception as e:
        return {"error": f"Failed to analyze weights: {str(e)}"}


def calculate_tensor_stats(tensor: Any) -> dict[str, Any]:
    torch = _get_torch()

    try:
        flat_tensor = tensor.flatten().float()

        stats: dict[str, Any] = {
            "mean": float(torch.mean(flat_tensor)),
            "std": float(torch.std(flat_tensor)),
            "min": float(torch.min(flat_tensor)),
            "max": float(torch.max(flat_tensor)),
            "median": float(torch.median(flat_tensor)),
            "abs_mean": float(torch.mean(torch.abs(flat_tensor))),
            "zero_fraction": float(torch.sum(flat_tensor == 0.0) / flat_tensor.numel()),
        }

        percentiles = [25, 75, 90, 95, 99]
        for p in percentiles:
            stats[f"percentile_{p}"] = float(torch.quantile(flat_tensor, p / 100.0))

        return stats
    except Exception as e:
        return {"error": f"Failed to calculate tensor stats: {str(e)}"}


def analyze_layer_structure(weights: dict[str, Any]) -> dict[str, Any]:
    torch = _get_torch()

    layer_info: dict[str, list[str]] = {
        "embedding_layers": [],
        "transformer_layers": [],
        "output_layers": [],
        "layer_norm_layers": [],
        "attention_layers": [],
        "feedforward_layers": [],
        "other_layers": [],
    }

    layer_counts: dict[str, int] = {
        "total_layers": 0,
        "attention_heads": 0,
        "feedforward_layers": 0,
        "layer_norms": 0,
    }

    for name, tensor in weights.items():
        if not torch.is_tensor(tensor):
            continue

        name_lower = name.lower()

        if "embed" in name_lower:
            layer_info["embedding_layers"].append(name)
        elif (
            "ln" in name_lower
            or "layer_norm" in name_lower
            or "layernorm" in name_lower
        ):
            layer_info["layer_norm_layers"].append(name)
            layer_counts["layer_norms"] += 1
        elif any(
            attn_key in name_lower
            for attn_key in ["attn", "attention", "self_attention"]
        ):
            layer_info["attention_layers"].append(name)
        elif any(
            ff_key in name_lower
            for ff_key in ["mlp", "ffn", "feed_forward", "intermediate"]
        ):
            layer_info["feedforward_layers"].append(name)
            layer_counts["feedforward_layers"] += 1
        elif "lm_head" in name_lower or "output" in name_lower:
            layer_info["output_layers"].append(name)
        elif "transformer" in name_lower or "layer" in name_lower:
            layer_info["transformer_layers"].append(name)
        else:
            layer_info["other_layers"].append(name)

    transformer_layers: set[int] = set()
    for name in weights.keys():
        layer_matches = re.findall(r"(?:layer|h)\.(\d+)\.", name.lower())
        if layer_matches:
            transformer_layers.update(int(match) for match in layer_matches)

    layer_counts["estimated_transformer_layers"] = len(transformer_layers)
    layer_counts["total_layers"] = len(weights)

    return {
        "layer_categorization": layer_info,
        "layer_counts": layer_counts,
        "transformer_layer_indices": sorted(list(transformer_layers))
        if transformer_layers
        else [],
    }


def calculate_global_weight_stats(weights: dict[str, Any]) -> dict[str, Any]:
    torch = _get_torch()

    try:
        all_weights = []
        for tensor in weights.values():
            if torch.is_tensor(tensor):
                all_weights.append(tensor.flatten().float())

        if not all_weights:
            return {"error": "No tensors found"}

        global_tensor = torch.cat(all_weights)

        global_stats: dict[str, Any] = {
            "global_mean": float(torch.mean(global_tensor)),
            "global_std": float(torch.std(global_tensor)),
            "global_min": float(torch.min(global_tensor)),
            "global_max": float(torch.max(global_tensor)),
            "global_abs_mean": float(torch.mean(torch.abs(global_tensor))),
            "global_zero_fraction": float(
                torch.sum(global_tensor == 0.0) / global_tensor.numel()
            ),
        }

        percentiles = [1, 5, 25, 50, 75, 95, 99]
        for p in percentiles:
            global_stats[f"global_percentile_{p}"] = float(
                torch.quantile(global_tensor, p / 100.0)
            )

        return global_stats
    except Exception as e:
        return {"error": f"Failed to calculate global stats: {str(e)}"}


def analyze_model_weights(
    repo_id: str,
    branch: str,
    weight_files: list[str] | None = None,
    delete_after_analysis: bool = False,
) -> dict[str, Any]:
    if weight_files is None:
        weight_files = discover_model_weight_files(repo_id, branch)

    if not weight_files:
        return {
            "weights_available": False,
            "error": "No weight files found",
            "discovered_files": [],
        }

    analysis: dict[str, Any] = {
        "weights_available": True,
        "discovered_files": weight_files,
        "file_analyses": {},
        "summary": {},
    }

    total_params = 0
    total_size_mb = 0.0
    downloaded_files: list[str] = []

    try:
        for weight_file in weight_files:
            file_path, success, error_msg = download_model_weights(
                repo_id, branch, weight_file
            )

            if success and file_path:
                downloaded_files.append(file_path)

                file_stats = calculate_weight_statistics(file_path)
                analysis["file_analyses"][weight_file] = file_stats

                if "parameter_stats" in file_stats:
                    total_params += file_stats["parameter_stats"].get(
                        "total_parameters", 0
                    )
                if "file_size_mb" in file_stats:
                    total_size_mb += file_stats["file_size_mb"]
            else:
                analysis["file_analyses"][weight_file] = {
                    "error": error_msg,
                    "downloaded": False,
                }

        analysis["summary"] = {
            "total_files_analyzed": len(
                [f for f in analysis["file_analyses"].values() if "error" not in f]
            ),
            "total_parameters": total_params,
            "total_parameters_millions": round(total_params / 1_000_000, 2),
            "total_parameters_billions": round(total_params / 1_000_000_000, 3),
            "total_size_mb": round(total_size_mb, 2),
            "total_size_gb": round(total_size_mb / 1024, 3),
        }

    finally:
        if delete_after_analysis and downloaded_files:
            for file_path in downloaded_files:
                try:
                    os.remove(file_path)
                except Exception:
                    pass

    return analysis
