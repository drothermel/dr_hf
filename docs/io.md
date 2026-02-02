# io

HfApi upload/download operations for files and parquet datasets.

## Functions

### upload_file_to_hf
```python
def upload_file_to_hf(
    file_path: Path,
    repo_id: str,
    path_in_repo: str | None = None,
    repo_type: str = "dataset",
    hf_token: str | None = None
) -> str
```
Upload a file to a HuggingFace repository. Returns the URL of the uploaded file.

### cached_download_tables_from_hf
```python
def cached_download_tables_from_hf(
    repo_id: str,
    local_dir: Path | None = None,
    hf_token: str | None = None,
    filepaths: list[str] | None = None,
    allow_patterns: list[str] | None = None
) -> list[pd.DataFrame]
```
Download parquet files from HuggingFace with local caching. Returns a list of DataFrames.

### get_tables_from_cache
```python
def get_tables_from_cache(
    repo_id: str,
    local_dir: Path,
    filepaths: list[str] | None = None,
    allow_patterns: list[str] | None = None
) -> list[pd.DataFrame]
```
Read parquet files from local cache directory.

### read_local_parquet_paths
```python
def read_local_parquet_paths(
    local_dir: Path,
    filepaths: list[str] | None = None,
    allow_patterns: list[str] | None = None
) -> list[str] | list[Path]
```
List parquet files in a local directory, optionally filtered by paths or patterns.

### query_hf_with_duckdb (requires `[duckdb]`)
```python
def query_hf_with_duckdb(
    repo_id: str,
    query: str,
    hf_token: str | None = None
) -> pd.DataFrame
```
Query a HuggingFace dataset directly using DuckDB SQL. Requires the `[duckdb]` optional dependency.

## Usage

```python
from pathlib import Path
from dr_hf import (
    upload_file_to_hf,
    cached_download_tables_from_hf,
    query_hf_with_duckdb,
)

# Upload a file
url = upload_file_to_hf(
    Path("results.parquet"),
    repo_id="username/my-dataset",
    path_in_repo="data/results.parquet",
    hf_token="hf_..."
)
print(f"Uploaded to: {url}")

# Download parquet tables with caching
dfs = cached_download_tables_from_hf(
    "allenai/c4",
    local_dir=Path("./cache"),
    allow_patterns=["data/train-00000-*.parquet"]
)
combined = pd.concat(dfs)

# Query with DuckDB (requires [duckdb])
df = query_hf_with_duckdb(
    "squad",
    "SELECT * FROM 'hf://datasets/squad/squad/train/*.parquet' LIMIT 100"
)
```
