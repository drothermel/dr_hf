# dr-hf

HuggingFace utilities for repository management, dataset operations, and model analysis.

## Installation

```bash
uv add dr-hf
```

For model weight analysis (requires PyTorch):
```bash
uv add dr-hf[weights]
```

For DuckDB query support:
```bash
uv add dr-hf[duckdb]
```

## Features

- **Branch Management**: Discover and parse checkpoint branches from HF repos
- **Dataset Operations**: Load, download, and cache HF datasets
- **Model Analysis**: Analyze model configs, weights, and checkpoints
- **Location Management**: Pydantic models for HF resource URIs

## Quick Start

```python
from dr_hf import get_all_repo_branches, HFLocation

# List branches in a repo
branches = get_all_repo_branches("username/repo-name")

# Create a location reference
loc = HFLocation(org="allenai", repo_name="my-dataset")
print(loc.repo_uri)  # hf://datasets/allenai/my-dataset
```

## License

MIT
