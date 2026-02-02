from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset


def load_or_download_dataset(
    path: Path, repo_id: str, split: str = "train"
) -> pd.DataFrame:
    download_dataset(path, repo_id, split, force_reload=False)
    return pd.read_parquet(path)


def download_dataset(
    path: Path, repo_id: str, split: str = "train", force_reload: bool = False
) -> None:
    if force_reload or not path.exists():
        raw_ds = load_dataset(repo_id, split=split)
        raw_ds.to_parquet(path)


def sanitize_repo_name(repo_id: str) -> str:
    return repo_id.replace("/", "--").replace(" ", "-")
