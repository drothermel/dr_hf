from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from .branches import extract_step_from_branch, get_checkpoint_branches
from .configs import analyze_model_config, download_config_file
from .models import (
    ArchitectureInfo,
    CheckpointAnalysis,
    CheckpointComponents,
    CheckpointSummaryRow,
    ConfigAnalysis,
    LearningRateInfo,
    OptimizerAnalysis,
    OptimizerComponentInfo,
    ParamGroupInfo,
    ParameterEstimate,
    WeightsAnalysis,
    WeightsSummary,
)
from .weights import analyze_model_weights

_torch: ModuleType | None = None


def _get_torch() -> ModuleType:
    global _torch
    if _torch is None:
        try:
            import torch

            _torch = torch
        except ImportError as e:
            raise ImportError(
                "torch is required for checkpoint analysis. "
                "Install with: uv add dr-hf[weights]"
            ) from e
    return _torch


def download_optimizer_checkpoint(
    repo_id: str, branch: str = "main", local_dir: str | None = None
) -> tuple[str | None, bool, str]:
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename="training/optim.pt",
            revision=branch,
            local_dir=local_dir,
        )
        return file_path, True, ""
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "Entry Not Found" in error_msg:
            return None, False, "Optimizer checkpoint not available"
        return None, False, error_msg


def _parse_optimizer_component(
    component: Any, component_name: str
) -> OptimizerComponentInfo:
    info = OptimizerComponentInfo(
        type=type(component).__name__,
        size=len(component) if hasattr(component, "__len__") else "N/A",
    )

    if component_name == "param_groups" and isinstance(component, list):
        info.num_param_groups = len(component)
        if component:
            first_group = component[0]
            if isinstance(first_group, dict):
                info.param_group_keys = list(first_group.keys())
                if "lr" in first_group:
                    info.learning_rate = first_group["lr"]
                if "initial_lr" in first_group:
                    info.initial_learning_rate = first_group["initial_lr"]

    elif component_name == "state" and isinstance(component, dict):
        info.num_parameters = len(component)
        if component:
            first_param_state = next(iter(component.values()))
            if isinstance(first_param_state, dict):
                info.state_keys = list(first_param_state.keys())

    return info


def _parse_learning_rate_info(checkpoint: dict[str, Any]) -> LearningRateInfo:
    lr_info = LearningRateInfo()

    if "param_groups" in checkpoint and isinstance(checkpoint["param_groups"], list):
        for i, group in enumerate(checkpoint["param_groups"]):
            if isinstance(group, dict):
                group_info = ParamGroupInfo(
                    group_index=i,
                    current_lr=group.get("lr"),
                    initial_lr=group.get("initial_lr"),
                    weight_decay=group.get("weight_decay"),
                    momentum=group.get("momentum"),
                )
                lr_info.param_group_lrs.append(group_info)

    if "lr_scheduler" in checkpoint:
        lr_info.lr_scheduler_present = True
        scheduler = checkpoint["lr_scheduler"]
        if isinstance(scheduler, dict):
            lr_info.lr_scheduler_keys = list(scheduler.keys())
            if "last_epoch" in scheduler:
                lr_info.scheduler_last_epoch = scheduler["last_epoch"]

    return lr_info


def analyze_optimizer_checkpoint(checkpoint_path: str) -> OptimizerAnalysis:
    torch = _get_torch()

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        analysis = OptimizerAnalysis(
            available=True,
            checkpoint_keys=list(checkpoint.keys())
            if isinstance(checkpoint, dict)
            else ["non_dict_checkpoint"],
            checkpoint_type=type(checkpoint).__name__,
        )

        if isinstance(checkpoint, dict):
            for key in ["state", "param_groups", "optimizer", "lr_scheduler"]:
                if key in checkpoint:
                    analysis.optimizer_info[key] = _parse_optimizer_component(
                        checkpoint[key], key
                    )
            analysis.learning_rate_info = _parse_learning_rate_info(checkpoint)

        return analysis

    except Exception as e:
        return OptimizerAnalysis(available=False, error=str(e))


def _parse_config_analysis(config_result: dict[str, Any]) -> ConfigAnalysis:
    if "error" in config_result:
        return ConfigAnalysis(available=False, error=config_result["error"])

    arch_info = None
    if "architecture_info" in config_result:
        raw_arch = config_result["architecture_info"]
        est_params = None
        if (
            "estimated_parameters" in raw_arch
            and "error" not in raw_arch["estimated_parameters"]
        ):
            est_params = ParameterEstimate(**raw_arch["estimated_parameters"])

        arch_info = ArchitectureInfo(
            hidden_size=raw_arch.get("hidden_size"),
            num_layers=raw_arch.get("num_layers"),
            num_attention_heads=raw_arch.get("num_attention_heads"),
            intermediate_size=raw_arch.get("intermediate_size"),
            vocab_size=raw_arch.get("vocab_size"),
            max_position_embeddings=raw_arch.get("max_position_embeddings"),
            sequence_length=raw_arch.get("sequence_length"),
            model_type=raw_arch.get("model_type"),
            activation_function=raw_arch.get("activation_function"),
            layer_norm_eps=raw_arch.get("layer_norm_eps"),
            dropout=raw_arch.get("dropout"),
            pad_token_id=raw_arch.get("pad_token_id"),
            eos_token_id=raw_arch.get("eos_token_id"),
            bos_token_id=raw_arch.get("bos_token_id"),
            torch_dtype=raw_arch.get("torch_dtype"),
            use_cache=raw_arch.get("use_cache"),
            tie_word_embeddings=raw_arch.get("tie_word_embeddings"),
            rope_scaling=raw_arch.get("rope_scaling"),
            estimated_parameters=est_params,
        )

    return ConfigAnalysis(
        available=True,
        raw_config=config_result.get("raw_config"),
        architecture_info=arch_info,
        config_keys=config_result.get("config_keys", []),
        config_type=config_result.get("config_type"),
    )


def _parse_weights_analysis(weights_result: dict[str, Any]) -> WeightsAnalysis:
    if not weights_result.get("weights_available", False):
        return WeightsAnalysis(
            available=False,
            error=weights_result.get("error"),
            discovered_files=weights_result.get("discovered_files", []),
        )

    summary = None
    if "summary" in weights_result:
        raw_summary = weights_result["summary"]
        summary = WeightsSummary(
            total_files_analyzed=raw_summary.get("total_files_analyzed", 0),
            total_parameters=raw_summary.get("total_parameters", 0),
            total_parameters_millions=raw_summary.get("total_parameters_millions", 0.0),
            total_parameters_billions=raw_summary.get("total_parameters_billions", 0.0),
            total_size_mb=raw_summary.get("total_size_mb", 0.0),
            total_size_gb=raw_summary.get("total_size_gb", 0.0),
        )

    return WeightsAnalysis(
        available=True,
        discovered_files=weights_result.get("discovered_files", []),
        file_analyses=weights_result.get("file_analyses", {}),
        summary=summary,
    )


def analyze_complete_checkpoint(
    repo_id: str,
    branch: str,
    include_weights: bool = False,
    weight_files: list[str] | None = None,
    delete_weights_after: bool = False,
) -> CheckpointAnalysis:
    components = CheckpointComponents()

    optimizer_path, optimizer_success, optimizer_error = download_optimizer_checkpoint(
        repo_id, branch
    )

    if optimizer_success and optimizer_path:
        components.optimizer = analyze_optimizer_checkpoint(optimizer_path)
    else:
        components.optimizer = OptimizerAnalysis(available=False, error=optimizer_error)

    config_path, config_success, config_error = download_config_file(repo_id, branch)

    if config_success and config_path:
        config_result = analyze_model_config(config_path)
        components.config = _parse_config_analysis(config_result)
    else:
        components.config = ConfigAnalysis(available=False, error=config_error)

    if include_weights:
        weights_result = analyze_model_weights(
            repo_id, branch, weight_files, delete_weights_after
        )
        components.weights = _parse_weights_analysis(weights_result)

    return CheckpointAnalysis(
        branch=branch,
        step=extract_step_from_branch(branch),
        components=components,
    )


def process_single_checkpoint(
    repo_id: str,
    branch: str,
    include_weights: bool = False,
    delete_weights_after: bool = False,
) -> tuple[str, CheckpointAnalysis]:
    try:
        analysis = analyze_complete_checkpoint(
            repo_id, branch, include_weights, delete_weights_after=delete_weights_after
        )
        return branch, analysis

    except Exception as e:
        error_analysis = CheckpointAnalysis(
            branch=branch,
            step=extract_step_from_branch(branch),
            error=str(e),
            components=CheckpointComponents(
                optimizer=OptimizerAnalysis(available=False, error=str(e)),
                config=ConfigAnalysis(available=False, error=str(e)),
                weights=WeightsAnalysis(available=False, error=str(e)),
            ),
        )
        return branch, error_analysis


def process_all_checkpoints(
    repo_id: str,
    max_workers: int = 4,
    include_weights: bool = False,
    delete_weights_after: bool = False,
) -> dict[str, CheckpointAnalysis]:
    branches = get_checkpoint_branches(repo_id)

    if not branches:
        return {}

    results: dict[str, CheckpointAnalysis] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_branch = {
            executor.submit(
                process_single_checkpoint,
                repo_id,
                branch,
                include_weights,
                delete_weights_after,
            ): branch
            for branch in branches
        }

        for future in as_completed(future_to_branch):
            branch, analysis = future.result()
            results[branch] = analysis

    return results


def create_comprehensive_summary(
    all_analyses: dict[str, CheckpointAnalysis],
) -> pd.DataFrame:
    summary_data = []

    for analysis in all_analyses.values():
        if analysis.error:
            continue
        row = CheckpointSummaryRow.from_analysis(analysis)
        summary_data.append(row.model_dump())

    df = pd.DataFrame(summary_data)
    return df.sort_values("step") if not df.empty else df


def create_learning_rate_summary(
    all_analyses: dict[str, CheckpointAnalysis],
) -> pd.DataFrame:
    comprehensive_df = create_comprehensive_summary(all_analyses)

    lr_columns = [
        "branch",
        "step",
        "optimizer_available",
        "current_lr",
        "weight_decay",
        "momentum",
        "num_param_groups",
        "optimizer_error",
    ]

    return (
        comprehensive_df[lr_columns] if not comprehensive_df.empty else comprehensive_df
    )


def save_checkpoint_analysis(
    analysis: CheckpointAnalysis, branch: str, output_dir: str | None = None
) -> str:
    sanitized = branch.replace("/", "_").replace("-", "_")
    filename = f"optimizer_analysis_{sanitized}.json"

    if output_dir:
        filepath = Path(output_dir) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
    else:
        filepath = Path(filename)

    with open(filepath, "w") as f:
        json.dump(analysis.model_dump(), f, indent=2, default=str)

    return str(filepath)


def save_all_analyses_outputs(
    all_analyses: dict[str, CheckpointAnalysis], output_dir: str | None = None
) -> tuple[str, str, str]:
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        comprehensive_csv_path = output_path / "comprehensive_checkpoint_summary.csv"
        lr_csv_path = output_path / "learning_rate_summary.csv"
        json_path = output_path / "all_checkpoint_analyses.json"
    else:
        comprehensive_csv_path = Path("comprehensive_checkpoint_summary.csv")
        lr_csv_path = Path("learning_rate_summary.csv")
        json_path = Path("all_checkpoint_analyses.json")

    comprehensive_summary = create_comprehensive_summary(all_analyses)
    if not comprehensive_summary.empty:
        comprehensive_summary.to_csv(comprehensive_csv_path, index=False)

    lr_summary = create_learning_rate_summary(all_analyses)
    if not lr_summary.empty:
        lr_summary.to_csv(lr_csv_path, index=False)

    serializable = {k: v.model_dump() for k, v in all_analyses.items()}
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    return str(comprehensive_csv_path), str(lr_csv_path), str(json_path)
