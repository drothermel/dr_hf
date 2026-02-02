# io

HfApi upload/download operations for files and parquet datasets.

## Functions

### upload_file_to_hf
```python
def upload_file_to_hf(
    local_path: str | Path,
    hf_loc: HFLocation,
    *,
    hf_token: str | None = None
) -> str
```
Upload a file to a HuggingFace repository. The `HFLocation` must have exactly one filepath specified. Returns the URL of the uploaded file.

### cached_download_tables_from_hf
```python
def cached_download_tables_from_hf(
    hf_loc: HFLocation,
    *,
    cache_dir: str | Path,
    hf_token: str | None = None,
    force_download: bool = False,
    verbose: bool = True
) -> dict[str, str | Path]
```
Download parquet files from HuggingFace with local caching. Returns a dictionary mapping remote filepaths to local file paths. Files are only downloaded if they don't exist locally (unless `force_download=True`).

### get_tables_from_cache
```python
def get_tables_from_cache(
    hf_loc: HFLocation,
    cache_dir: str | Path
) -> dict[str, pd.DataFrame]
```
Read parquet files from local cache directory. Returns a dictionary mapping file stems to DataFrames.

### read_local_parquet_paths
```python
def read_local_parquet_paths(
    local_paths: list[str] | list[Path]
) -> dict[str, pd.DataFrame]
```
Read parquet files from a list of local paths. Returns a dictionary mapping file stems to DataFrames.

### query_hf_with_duckdb (requires `[duckdb]`)
```python
def query_hf_with_duckdb(
    hf_loc: HFLocation,
    connection: duckdb.DuckDBPyConnection
) -> dict[str, pd.DataFrame]
```
Query a HuggingFace dataset directly using DuckDB. Requires a DuckDB connection object. Returns a dictionary mapping file stems to DataFrames.

## Usage

```python
from pathlib import Path
from dr_hf import (
    upload_file_to_hf,
    cached_download_tables_from_hf,
    query_hf_with_duckdb,
    HFLocation,
)

# Upload a file
loc = HFLocation(
    org="username",
    repo_name="my-dataset",
    filepaths=["data/results.parquet"]
)
url = upload_file_to_hf(
    Path("results.parquet"),
    loc,
    hf_token="hf_..."
)
print(f"Uploaded to: {url}")

# Download parquet tables with caching
loc = HFLocation(
    org="allenai",
    repo_name="c4",
    filepaths=["en/train-00000-of-01024.parquet"]
)
tables = cached_download_tables_from_hf(
    loc,
    cache_dir=Path("./cache"),
    hf_token="hf_..."
)
# tables is a dict: {"en/train-00000-of-01024": Path("./cache/...")}

# Read cached tables as DataFrames
from dr_hf import get_tables_from_cache
dfs = get_tables_from_cache(loc, cache_dir=Path("./cache"))
# dfs is a dict: {"en/train-00000-of-01024": pd.DataFrame}

# Query with DuckDB (requires [duckdb] optional dependency)
import duckdb  # Optional: install with pip install dr-hf[duckdb]
conn = duckdb.connect()
loc = HFLocation.from_uri("hf://datasets/squad/squad")
results = query_hf_with_duckdb(loc, conn)
```
