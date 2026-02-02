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
                "Install with: pip install dr-hf[weights]"
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


def analyze_optimizer_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    torch = _get_torch()

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        analysis: dict[str, Any] = {
            "checkpoint_keys": list(checkpoint.keys())
            if isinstance(checkpoint, dict)
            else ["non_dict_checkpoint"],
            "checkpoint_type": type(checkpoint).__name__,
            "optimizer_info": {},
            "learning_rate_info": {},
        }

        if isinstance(checkpoint, dict):
            for key in ["state", "param_groups", "optimizer", "lr_scheduler"]:
                if key in checkpoint:
                    analysis["optimizer_info"][key] = analyze_optimizer_component(
                        checkpoint[key], key
                    )

            analysis["learning_rate_info"] = extract_learning_rate_info(checkpoint)

        return analysis

    except Exception as e:
        return {"error": str(e)}


def analyze_optimizer_component(component: Any, component_name: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "type": type(component).__name__,
        "size": len(component) if hasattr(component, "__len__") else "N/A",
    }

    if component_name == "param_groups" and isinstance(component, list):
        info["num_param_groups"] = len(component)
        if component:
            first_group = component[0]
            if isinstance(first_group, dict):
                info["param_group_keys"] = list(first_group.keys())
                if "lr" in first_group:
                    info["learning_rate"] = first_group["lr"]
                if "initial_lr" in first_group:
                    info["initial_learning_rate"] = first_group["initial_lr"]

    elif component_name == "state" and isinstance(component, dict):
        info["num_parameters"] = len(component)
        if component:
            first_param_state = next(iter(component.values()))
            if isinstance(first_param_state, dict):
                info["state_keys"] = list(first_param_state.keys())

    return info


def extract_learning_rate_info(checkpoint: dict[str, Any]) -> dict[str, Any]:
    lr_info: dict[str, Any] = {}

    if "param_groups" in checkpoint and isinstance(checkpoint["param_groups"], list):
        lr_info["param_group_lrs"] = []
        for i, group in enumerate(checkpoint["param_groups"]):
            if isinstance(group, dict):
                group_lr_info: dict[str, Any] = {"group_index": i}
                if "lr" in group:
                    group_lr_info["current_lr"] = group["lr"]
                if "initial_lr" in group:
                    group_lr_info["initial_lr"] = group["initial_lr"]
                if "weight_decay" in group:
                    group_lr_info["weight_decay"] = group["weight_decay"]
                if "momentum" in group:
                    group_lr_info["momentum"] = group["momentum"]
                lr_info["param_group_lrs"].append(group_lr_info)

    if "lr_scheduler" in checkpoint:
        lr_info["lr_scheduler_present"] = True
        scheduler = checkpoint["lr_scheduler"]
        if isinstance(scheduler, dict):
            lr_info["lr_scheduler_keys"] = list(scheduler.keys())
            if "last_epoch" in scheduler:
                lr_info["scheduler_last_epoch"] = scheduler["last_epoch"]

    return lr_info


def analyze_complete_checkpoint(
    repo_id: str,
    branch: str,
    include_weights: bool = False,
    weight_files: list[str] | None = None,
    delete_weights_after: bool = False,
) -> dict[str, Any]:
    analysis: dict[str, Any] = {
        "branch": branch,
        "step": extract_step_from_branch(branch),
        "components": {
            "optimizer": {"available": False},
            "config": {"available": False},
            "weights": {"available": False},
        },
    }

    optimizer_path, optimizer_success, optimizer_error = download_optimizer_checkpoint(
        repo_id, branch
    )
    analysis["components"]["optimizer"]["available"] = optimizer_success

    if optimizer_success and optimizer_path:
        optimizer_analysis = analyze_optimizer_checkpoint(optimizer_path)
        analysis["components"]["optimizer"].update(optimizer_analysis)
    else:
        analysis["components"]["optimizer"]["error"] = optimizer_error

    config_path, config_success, config_error = download_config_file(repo_id, branch)
    analysis["components"]["config"]["available"] = config_success

    if config_success and config_path:
        config_analysis = analyze_model_config(config_path)
        analysis["components"]["config"].update(config_analysis)
    else:
        analysis["components"]["config"]["error"] = config_error

    if include_weights:
        weights_analysis = analyze_model_weights(
            repo_id, branch, weight_files, delete_weights_after
        )
        analysis["components"]["weights"] = weights_analysis

    return analysis


def process_single_checkpoint(
    repo_id: str,
    branch: str,
    include_weights: bool = False,
    delete_weights_after: bool = False,
) -> tuple[str, dict[str, Any]]:
    try:
        analysis = analyze_complete_checkpoint(
            repo_id, branch, include_weights, delete_weights_after=delete_weights_after
        )
        return branch, analysis

    except Exception as e:
        error_analysis: dict[str, Any] = {
            "branch": branch,
            "step": extract_step_from_branch(branch),
            "error": str(e),
            "components": {
                "optimizer": {"available": False, "error": str(e)},
                "config": {"available": False, "error": str(e)},
                "weights": {"available": False, "error": str(e)},
            },
        }
        return branch, error_analysis


def process_all_checkpoints(
    repo_id: str,
    max_workers: int = 4,
    include_weights: bool = False,
    delete_weights_after: bool = False,
) -> dict[str, dict[str, Any]]:
    branches = get_checkpoint_branches(repo_id)

    if not branches:
        return {}

    results: dict[str, dict[str, Any]] = {}

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
    all_analyses: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    summary_data = []

    for branch, analysis in all_analyses.items():
        if "error" in analysis:
            continue

        row: dict[str, Any] = {
            "branch": branch,
            "step": analysis.get("step", 0),
            "optimizer_available": False,
            "config_available": False,
            "weights_available": False,
            "current_lr": None,
            "weight_decay": None,
            "momentum": None,
            "num_param_groups": None,
            "model_type": None,
            "hidden_size": None,
            "num_layers": None,
            "vocab_size": None,
            "estimated_params_millions": None,
            "total_weight_params_millions": None,
            "optimizer_error": "",
            "config_error": "",
            "weights_error": "",
        }

        components = analysis.get("components", {})

        optimizer_comp = components.get("optimizer", {})
        row["optimizer_available"] = optimizer_comp.get("available", False)
        if row["optimizer_available"]:
            lr_info = optimizer_comp.get("learning_rate_info", {})
            if "param_group_lrs" in lr_info and lr_info["param_group_lrs"]:
                first_group = lr_info["param_group_lrs"][0]
                row["current_lr"] = first_group.get("current_lr")
                row["weight_decay"] = first_group.get("weight_decay")
                row["momentum"] = first_group.get("momentum")
                row["num_param_groups"] = len(lr_info["param_group_lrs"])
        else:
            row["optimizer_error"] = optimizer_comp.get("error", "")

        config_comp = components.get("config", {})
        row["config_available"] = config_comp.get("available", False)
        if row["config_available"]:
            arch_info = config_comp.get("architecture_info", {})
            row["model_type"] = arch_info.get("model_type")
            row["hidden_size"] = arch_info.get("hidden_size")
            row["num_layers"] = arch_info.get("num_layers")
            row["vocab_size"] = arch_info.get("vocab_size")
            if "estimated_parameters" in arch_info:
                est_params = arch_info["estimated_parameters"]
                if "estimated_total_millions" in est_params:
                    row["estimated_params_millions"] = est_params[
                        "estimated_total_millions"
                    ]
        else:
            row["config_error"] = config_comp.get("error", "")

        weights_comp = components.get("weights", {})
        row["weights_available"] = weights_comp.get("weights_available", False)
        if row["weights_available"]:
            summary = weights_comp.get("summary", {})
            row["total_weight_params_millions"] = summary.get(
                "total_parameters_millions"
            )
        else:
            row["weights_error"] = weights_comp.get("error", "")

        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    return df.sort_values("step") if not df.empty else df


def create_learning_rate_summary(
    all_analyses: dict[str, dict[str, Any]],
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
    analysis: dict[str, Any], branch: str, output_dir: str | None = None
) -> str:
    sanitized = branch.replace("/", "_").replace("-", "_")
    filename = f"optimizer_analysis_{sanitized}.json"

    if output_dir:
        filepath = Path(output_dir) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
    else:
        filepath = Path(filename)

    with open(filepath, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    return str(filepath)


def save_all_analyses_outputs(
    all_analyses: dict[str, dict[str, Any]], output_dir: str | None = None
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

    with open(json_path, "w") as f:
        json.dump(all_analyses, f, indent=2, default=str)

    return str(comprehensive_csv_path), str(lr_csv_path), str(json_path)
