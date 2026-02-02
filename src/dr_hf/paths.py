from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get_data_dir() -> Path:
    data_dir = os.getenv("DATA_DIR")
    if data_dir is None:
        raise RuntimeError(
            "DATA_DIR environment variable not set. "
            "Please set DATA_DIR to point to your data directory."
        )
    return Path(data_dir)


def get_repo_dir() -> Path:
    repo_dir = os.getenv("REPO_DIR")
    if repo_dir is None:
        raise RuntimeError(
            "REPO_DIR environment variable not set. "
            "Please set REPO_DIR to point to your repository directory."
        )
    return Path(repo_dir)
