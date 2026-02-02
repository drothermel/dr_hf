# datasets

Dataset loading, downloading, and caching using the HuggingFace datasets library.

## Functions

### load_or_download_dataset
```python
def load_or_download_dataset(
    dataset_name: str,
    output_dir: str | None = None,
    split: str | None = None,
    subset: str | None = None
) -> Dataset | DatasetDict
```
Load a dataset from cache or download if not present. Uses HuggingFace datasets library.

### download_dataset
```python
def download_dataset(
    dataset_name: str,
    output_dir: str | None = None,
    split: str | None = None,
    subset: str | None = None
) -> Path
```
Download a HuggingFace dataset and save to parquet format. Returns the path to the output directory.

### sanitize_repo_name
```python
def sanitize_repo_name(repo_name: str) -> str
```
Convert a repository name to a filesystem-safe string. Replaces `/` with `__`.

## Usage

```python
from dr_hf import (
    load_or_download_dataset,
    download_dataset,
    sanitize_repo_name,
)

# Load dataset (uses cache if available)
dataset = load_or_download_dataset("squad", split="train")
print(f"Loaded {len(dataset)} examples")

# Download to local parquet
output_path = download_dataset(
    "glue",
    subset="mrpc",
    split="train",
    output_dir="./data"
)
print(f"Saved to: {output_path}")

# Sanitize repo names for filesystem
safe_name = sanitize_repo_name("allenai/c4")
print(safe_name)  # "allenai__c4"
```
