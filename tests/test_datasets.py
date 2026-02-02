from __future__ import annotations

from dr_hf.datasets import sanitize_repo_name


def test_sanitize_repo_name() -> None:
    assert sanitize_repo_name("allenai/dataset") == "allenai--dataset"
    assert sanitize_repo_name("org/my dataset") == "org--my-dataset"
    assert sanitize_repo_name("simple") == "simple"
    # Edge cases
    assert sanitize_repo_name("") == ""
    assert sanitize_repo_name("org//dataset") == "org----dataset"
    assert sanitize_repo_name(" org/dataset ") == "-org--dataset-"
    assert sanitize_repo_name("/org/dataset/") == "--org--dataset--"
