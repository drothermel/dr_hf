from __future__ import annotations

import os
import re
from typing import Any

from huggingface_hub import hf_hub_download, list_repo_files

from ._torch import get_torch
from .models import (
    GlobalWeightStats,
    LayerAnalysis,
    LayerCategorization,
    LayerCounts,
    ParameterStats,
    TensorInfo,
    TensorStats,
    WeightFileStatistics,
    WeightsAnalysis,
    WeightsSummary,
)

_safetensors_available: bool = False


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


def calculate_weight_statistics(weight_path: str) -> WeightFileStatistics:
    torch = get_torch()

    try:
        if weight_path.endswith(".safetensors"):
            if not _check_safetensors():
                return WeightFileStatistics(
                    file_path=weight_path,
                    file_size_mb=0.0,
                    num_tensors=0,
                    error="safetensors library not available",
                )
            from safetensors import safe_open

            weights: dict[str, Any] = {}
            with safe_open(weight_path, framework="pt") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
        else:
            weights = torch.load(weight_path, map_location="cpu")

        if not isinstance(weights, dict):
            return WeightFileStatistics(
                file_path=weight_path,
                file_size_mb=0.0,
                num_tensors=0,
                error="Weights file is not a dictionary",
            )

        total_params = 0
        tensor_infos: list[TensorInfo] = []

        for name, tensor in weights.items():
            if torch.is_tensor(tensor):
                tensor_params = tensor.numel()
                total_params += tensor_params

                tensor_infos.append(
                    TensorInfo(
                        name=name,
                        shape=list(tensor.shape),
                        dtype=str(tensor.dtype),
                        parameters=tensor_params,
                        size_mb=round(
                            tensor.numel() * tensor.element_size() / (1024 * 1024), 3
                        ),
                        statistics=calculate_tensor_stats(tensor),
                    )
                )

        return WeightFileStatistics(
            file_path=weight_path,
            file_size_mb=round(os.path.getsize(weight_path) / (1024 * 1024), 2),
            num_tensors=len(weights),
            tensor_info=tensor_infos,
            parameter_stats=ParameterStats(total_parameters=total_params),
            layer_analysis=analyze_layer_structure(weights),
            global_statistics=calculate_global_weight_stats(weights),
        )

    except Exception as e:
        return WeightFileStatistics(
            file_path=weight_path,
            file_size_mb=0.0,
            num_tensors=0,
            error=f"Failed to analyze weights: {str(e)}",
        )


def calculate_tensor_stats(tensor: Any) -> TensorStats | None:
    torch = get_torch()

    try:
        flat_tensor = tensor.flatten().float()

        stats = TensorStats(
            mean=float(torch.mean(flat_tensor)),
            std=float(torch.std(flat_tensor)),
            min=float(torch.min(flat_tensor)),
            max=float(torch.max(flat_tensor)),
            median=float(torch.median(flat_tensor)),
            abs_mean=float(torch.mean(torch.abs(flat_tensor))),
            zero_fraction=float(torch.sum(flat_tensor == 0.0) / flat_tensor.numel()),
            percentile_25=float(torch.quantile(flat_tensor, 0.25)),
            percentile_75=float(torch.quantile(flat_tensor, 0.75)),
            percentile_90=float(torch.quantile(flat_tensor, 0.90)),
            percentile_95=float(torch.quantile(flat_tensor, 0.95)),
            percentile_99=float(torch.quantile(flat_tensor, 0.99)),
        )

        return stats
    except Exception:
        return None


def analyze_layer_structure(weights: dict[str, Any]) -> LayerAnalysis:
    torch = get_torch()

    categorization = LayerCategorization()
    counts = LayerCounts()

    for name, tensor in weights.items():
        if not torch.is_tensor(tensor):
            continue

        name_lower = name.lower()

        if "embed" in name_lower:
            categorization.embedding_layers.append(name)
        elif (
            "ln" in name_lower
            or "layer_norm" in name_lower
            or "layernorm" in name_lower
        ):
            categorization.layer_norm_layers.append(name)
            counts.layer_norms += 1
        elif any(
            attn_key in name_lower
            for attn_key in ["attn", "attention", "self_attention"]
        ):
            categorization.attention_layers.append(name)
        elif any(
            ff_key in name_lower
            for ff_key in ["mlp", "ffn", "feed_forward", "intermediate"]
        ):
            categorization.feedforward_layers.append(name)
            counts.feedforward_layers += 1
        elif "lm_head" in name_lower or "output" in name_lower:
            categorization.output_layers.append(name)
        elif "transformer" in name_lower or "layer" in name_lower:
            categorization.transformer_layers.append(name)
        else:
            categorization.other_layers.append(name)

    transformer_layers: set[int] = set()
    for name in weights.keys():
        layer_matches = re.findall(r"(?:layer|h)\.(\d+)\.", name.lower())
        if layer_matches:
            transformer_layers.update(int(match) for match in layer_matches)

    counts.estimated_transformer_layers = len(transformer_layers)
    counts.total_layers = len(weights)

    return LayerAnalysis(
        layer_categorization=categorization,
        layer_counts=counts,
        transformer_layer_indices=sorted(list(transformer_layers))
        if transformer_layers
        else [],
    )


def calculate_global_weight_stats(weights: dict[str, Any]) -> GlobalWeightStats | None:
    torch = get_torch()

    try:
        all_weights = []
        for tensor in weights.values():
            if torch.is_tensor(tensor):
                all_weights.append(tensor.flatten().float())

        if not all_weights:
            return None

        global_tensor = torch.cat(all_weights)

        return GlobalWeightStats(
            global_mean=float(torch.mean(global_tensor)),
            global_std=float(torch.std(global_tensor)),
            global_min=float(torch.min(global_tensor)),
            global_max=float(torch.max(global_tensor)),
            global_abs_mean=float(torch.mean(torch.abs(global_tensor))),
            global_zero_fraction=float(
                torch.sum(global_tensor == 0.0) / global_tensor.numel()
            ),
            global_percentile_1=float(torch.quantile(global_tensor, 0.01)),
            global_percentile_5=float(torch.quantile(global_tensor, 0.05)),
            global_percentile_25=float(torch.quantile(global_tensor, 0.25)),
            global_percentile_50=float(torch.quantile(global_tensor, 0.50)),
            global_percentile_75=float(torch.quantile(global_tensor, 0.75)),
            global_percentile_95=float(torch.quantile(global_tensor, 0.95)),
            global_percentile_99=float(torch.quantile(global_tensor, 0.99)),
        )
    except Exception:
        return None


def analyze_model_weights(
    repo_id: str,
    branch: str,
    weight_files: list[str] | None = None,
    delete_after_analysis: bool = False,
) -> WeightsAnalysis:
    if weight_files is None:
        weight_files = discover_model_weight_files(repo_id, branch)

    if not weight_files:
        return WeightsAnalysis(
            available=False,
            error="No weight files found",
            discovered_files=[],
        )

    file_analyses: dict[str, WeightFileStatistics] = {}
    total_params = 0
    total_size_mb = 0.0
    downloaded_files: list[str] = []
    successful_analyses = 0

    try:
        for weight_file in weight_files:
            file_path, success, error_msg = download_model_weights(
                repo_id, branch, weight_file
            )

            if success and file_path:
                downloaded_files.append(file_path)

                file_stats = calculate_weight_statistics(file_path)
                file_analyses[weight_file] = file_stats

                if file_stats.error is None:
                    successful_analyses += 1
                    if file_stats.parameter_stats:
                        total_params += file_stats.parameter_stats.total_parameters
                    total_size_mb += file_stats.file_size_mb
            else:
                file_analyses[weight_file] = WeightFileStatistics(
                    file_path="",
                    file_size_mb=0.0,
                    num_tensors=0,
                    error=error_msg,
                )

        return WeightsAnalysis(
            available=True,
            discovered_files=weight_files,
            file_analyses=file_analyses,
            summary=WeightsSummary(
                total_files_analyzed=successful_analyses,
                total_parameters=total_params,
                total_size_mb=round(total_size_mb, 2),
            ),
        )

    finally:
        if delete_after_analysis and downloaded_files:
            for file_path in downloaded_files:
                try:
                    os.remove(file_path)
                except Exception:
                    pass
