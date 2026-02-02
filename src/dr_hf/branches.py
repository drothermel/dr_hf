from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from huggingface_hub import list_repo_refs


def get_all_repo_branches(repo_id: str) -> list[str]:
    refs = list_repo_refs(repo_id)
    return [ref.name for ref in refs.branches]


def is_checkpoint_branch(branch: str) -> bool:
    pattern = re.compile(r"^step\d+-seed-.+$")
    return bool(pattern.match(branch))


def get_checkpoint_branches(repo_id: str) -> list[str]:
    all_branches = get_all_repo_branches(repo_id)
    checkpoint_branches = [b for b in all_branches if is_checkpoint_branch(b)]
    return sort_branches_by_step(checkpoint_branches)


def parse_branch_name(branch: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "branch": branch,
        "valid": False,
        "step": None,
        "seed": None,
    }

    if not is_checkpoint_branch(branch):
        return result

    step_match = re.search(r"step(\d+)-", branch)
    if step_match:
        result["step"] = int(step_match.group(1))

    seed_match = re.search(r"seed-(.+)$", branch)
    if seed_match:
        result["seed"] = seed_match.group(1)

    result["valid"] = result["step"] is not None and result["seed"] is not None
    return result


def extract_step_from_branch(branch: str) -> int:
    parsed = parse_branch_name(branch)
    return parsed["step"] or 0


def extract_seed_from_branch(branch: str) -> str:
    parsed = parse_branch_name(branch)
    return parsed["seed"] or "unknown"


def sort_branches_by_step(branches: list[str]) -> list[str]:
    return sorted(branches, key=extract_step_from_branch)


def group_branches_by_seed(branches: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for branch in branches:
        seed = extract_seed_from_branch(branch)
        if seed not in groups:
            groups[seed] = []
        groups[seed].append(branch)

    for seed in groups:
        groups[seed] = sort_branches_by_step(groups[seed])

    return groups


def get_step_range_for_seed(branches: list[str]) -> tuple[int, int]:
    if not branches:
        return (0, 0)

    steps = [extract_step_from_branch(b) for b in branches]
    return (min(steps), max(steps))


def create_branch_metadata(repo_id: str) -> dict[str, Any]:
    all_branches = get_all_repo_branches(repo_id)
    checkpoint_branches = [b for b in all_branches if is_checkpoint_branch(b)]
    other_branches = [b for b in all_branches if not is_checkpoint_branch(b)]

    seed_groups = group_branches_by_seed(checkpoint_branches)

    seed_configurations: dict[str, Any] = {}
    for seed, branches in seed_groups.items():
        step_range = get_step_range_for_seed(branches)
        seed_configurations[seed] = {
            "count": len(branches),
            "step_range": list(step_range),
            "branches": [
                {"step": extract_step_from_branch(b), "branch": b} for b in branches
            ],
        }

    return {
        "repo_id": repo_id,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_branches": len(all_branches),
        "checkpoint_branches": len(checkpoint_branches),
        "seed_configurations": seed_configurations,
        "other_branches": sorted(other_branches),
        "all_checkpoint_branches": sorted(
            checkpoint_branches, key=extract_step_from_branch
        ),
    }
