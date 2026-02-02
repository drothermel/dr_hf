# datasets

Dataset loading, downloading, and caching using the HuggingFace datasets library.

## Functions

### load_or_download_dataset
```python
def load_or_download_dataset(
    path: Path,
    repo_id: str,
    split: str = "train"
) -> pd.DataFrame
```
Load a dataset from cache or download if not present. Downloads the dataset and saves to parquet format at the specified path. Returns a pandas DataFrame.

### download_dataset
```python
def download_dataset(
    path: Path,
    repo_id: str,
    split: str = "train",
    force_reload: bool = False
) -> None
```
Download a HuggingFace dataset and save to parquet format at the specified path. If `force_reload` is False and the file already exists, the download is skipped.

### sanitize_repo_name
```python
def sanitize_repo_name(repo_id: str) -> str
```
Convert a repository ID to a filesystem-safe string. Replaces `/` with `--` and spaces with `-`.

## Usage

```python
from pathlib import Path
from dr_hf import (
    load_or_download_dataset,
    download_dataset,
    sanitize_repo_name,
)

# Load dataset (downloads if not cached)
data_path = Path("./data/squad_train.parquet")
df = load_or_download_dataset(data_path, repo_id="squad", split="train")
print(f"Loaded {len(df)} examples")

# Download to local parquet (only if not exists)
download_path = Path("./data/squad_dev.parquet")
download_dataset(download_path, repo_id="squad", split="validation")
print(f"Saved to: {download_path}")

# Force re-download
download_dataset(download_path, repo_id="squad", split="validation", force_reload=True)

# Sanitize repo names for filesystem
safe_name = sanitize_repo_name("allenai/c4")
print(safe_name)  # "allenai--c4"
```
