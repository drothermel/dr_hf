# API Reference

Auto-generated API documentation is available via pdoc.

## Generating Docs

### Interactive Server

```bash
uv run pdoc dr_hf
```

Opens a local server at http://localhost:8080 with interactive documentation.

### Static HTML

```bash
uv run pdoc dr_hf -o docs/api_html
```

Generates static HTML documentation in `docs/api_html/`.

## Module Index

- `dr_hf.branches` - Branch discovery and parsing
- `dr_hf.configs` - Model configuration analysis
- `dr_hf.weights` - Model weight analysis (requires `[weights]`)
- `dr_hf.checkpoints` - Checkpoint orchestration (requires `[weights]`)
- `dr_hf.datasets` - Dataset loading and caching
- `dr_hf.io` - HfApi upload/download operations
- `dr_hf.location` - HFLocation Pydantic model
- `dr_hf.paths` - Environment path management
- `dr_hf.models` - All Pydantic data models

## Public API

All public functions and models are exported from the top-level `dr_hf` package:

```python
from dr_hf import (
    # Functions
    get_checkpoint_branches,
    parse_branch_name,
    analyze_model_config,
    analyze_model_weights,
    # ...

    # Models
    BranchInfo,
    ConfigAnalysis,
    WeightsAnalysis,
    # ...
)
```

See the [README](../README.md) for the complete list of exports.
