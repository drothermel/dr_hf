from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


def load_or_download_dataset(
    path: Path, repo_id: str, split: str = "train"
) -> pd.DataFrame:
    download_dataset(path, repo_id, split, force_reload=False)
    return pd.read_parquet(path)


def download_dataset(
    path: Path, repo_id: str, split: str = "train", force_reload: bool = False
) -> None:
    if force_reload or not path.exists():
        try:
            raw_ds: Dataset = load_dataset(repo_id, split=split)
            raw_ds.to_parquet(path)
        except Exception as e:
            logger.error(
                f"Failed to download dataset: repo_id={repo_id}, split={split}, path={path}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to download dataset '{repo_id}' (split='{split}') to {path}"
            ) from e


def sanitize_repo_name(repo_id: str) -> str:
    return repo_id.replace("/", "--").replace(" ", "-")
