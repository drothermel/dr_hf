# location

Pydantic model for HuggingFace resource URIs and paths.

## Types

```python
HFRepoID = str   # "org/repo-name"
HFResource = str # "hf://datasets/org/repo"
```

## HFLocation Model

```python
class HFLocation(BaseModel):
    org: str
    repo_name: str
    repo_type: str = "datasets"
    filepaths: list[str] = []
```

### Computed Fields

- `repo_id` - Full repository ID: `"org/repo-name"`
- `repo_uri` - HuggingFace URI: `"hf://datasets/org/repo-name"`
- `repo_link` - HTTPS URL to HuggingFace page
- `rest_api_repo_url` - REST API endpoint URL
- `hf_hub_repo_type` - Hub library repo type (e.g., "dataset")

### Class Methods

#### from_uri
```python
@classmethod
def from_uri(cls, uri: str) -> HFLocation
```
Parse a HuggingFace URI into an HFLocation.

#### norm_posix
```python
@staticmethod
def norm_posix(path: str | Path) -> str
```
Normalize a path to POSIX format.

### Instance Methods

#### build_local_dir
```python
def build_local_dir(self, data_dir: Path) -> Path
```
Build a local directory path for caching.

#### get_path_uri
```python
def get_path_uri(self, filepath: str) -> str
```
Get a full URI for a specific file path within the repo.

## Usage

```python
from pathlib import Path
from dr_hf import HFLocation

# Create from components
loc = HFLocation(org="allenai", repo_name="c4")
print(loc.repo_id)    # "allenai/c4"
print(loc.repo_uri)   # "hf://datasets/allenai/c4"
print(loc.repo_link)  # HttpUrl("https://huggingface.co/datasets/allenai/c4")

# Parse from URI
loc = HFLocation.from_uri("hf://datasets/squad/squad")
print(loc.org)        # "squad"
print(loc.repo_name)  # "squad"

# Build local cache path
cache_dir = loc.build_local_dir(Path("./data"))
print(cache_dir)      # Path("./data/datasets/squad/squad")

# Get URI for specific file
file_uri = loc.get_path_uri("train/data.parquet")
print(file_uri)       # "hf://datasets/squad/squad/train/data.parquet"

# With filepaths
loc = HFLocation(
    org="allenai",
    repo_name="c4",
    filepaths=["en/train-00000.parquet", "en/train-00001.parquet"]
)
```
