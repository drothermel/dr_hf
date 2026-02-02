# paths

Environment path management using dotenv.

## Functions

### get_data_dir
```python
def get_data_dir() -> str
```
Get the DATA_DIR environment variable. Loads from `.env` file if present.

### get_repo_dir
```python
def get_repo_dir() -> str
```
Get the REPO_DIR environment variable. Loads from `.env` file if present.

## Environment Variables

Both functions expect these environment variables to be set (either in the shell or in a `.env` file):

- `DATA_DIR` - Base directory for data storage
- `REPO_DIR` - Base directory for repository operations

## Usage

```python
from pathlib import Path
from dr_hf import get_data_dir, get_repo_dir

# Get paths from environment
data_dir = Path(get_data_dir())
repo_dir = Path(get_repo_dir())

# Use for cache locations
cache_path = data_dir / "hf_cache" / "datasets"
output_path = repo_dir / "outputs" / "analysis"
```

## .env File

Create a `.env` file in your project root:

```
DATA_DIR=/path/to/data
REPO_DIR=/path/to/repos
```
