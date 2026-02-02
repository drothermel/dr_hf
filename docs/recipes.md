# Recipes & Patterns

Common usage patterns for dr_hf.

## Track Learning Rate Over Training

```python
from dr_hf import process_all_checkpoints

analyses = process_all_checkpoints("org/model-checkpoints", max_workers=8)

for branch, analysis in sorted(analyses.items(), key=lambda x: x[1].step):
    opt = analysis.components.optimizer
    if opt.available and opt.learning_rate_info:
        groups = opt.learning_rate_info.param_group_lrs
        if groups:
            lr = groups[0].current_lr
            print(f"Step {analysis.step:6d}: LR = {lr:.2e}")
```

## Compare Checkpoints Across Seeds

```python
from dr_hf import (
    get_checkpoint_branches,
    group_branches_by_seed,
    process_single_checkpoint,
)

branches = get_checkpoint_branches("org/model-checkpoints")
by_seed = group_branches_by_seed(branches)

for seed, seed_branches in by_seed.items():
    print(f"\n=== Seed: {seed} ===")

    # Analyze first and last checkpoint
    first = seed_branches[0]
    last = seed_branches[-1]

    _, first_analysis = process_single_checkpoint("org/model-checkpoints", first)
    _, last_analysis = process_single_checkpoint("org/model-checkpoints", last)

    print(f"  Steps: {first_analysis.step} -> {last_analysis.step}")
```

## Export Checkpoint Summary to CSV

```python
from dr_hf import (
    process_all_checkpoints,
    create_comprehensive_summary,
)

analyses = process_all_checkpoints("org/model-checkpoints")
df = create_comprehensive_summary(analyses)

# Filter and export
df_filtered = df[["branch", "step", "current_lr", "model_type", "hidden_size"]]
df_filtered.to_csv("checkpoint_summary.csv", index=False)
```

## Download Dataset with Caching

```python
from pathlib import Path
from dr_hf import cached_download_tables_from_hf, get_tables_from_cache, HFLocation
import pandas as pd

# Create location with specific filepaths
loc = HFLocation(
    org="allenai",
    repo_name="c4",
    filepaths=["en/train-00000-of-01024.parquet"]
)
cache_dir = Path("./data/cache")

# Download specific files (returns dict of paths)
tables = cached_download_tables_from_hf(
    loc,
    cache_dir=cache_dir,
)

# Read cached tables as DataFrames
dfs = get_tables_from_cache(loc, cache_dir=cache_dir)

# Combine into single DataFrame
combined = pd.concat(list(dfs.values()), ignore_index=True)
print(f"Loaded {len(combined)} rows")
```

## Analyze Model Architecture Without Weights

```python
from dr_hf import download_config_file, analyze_model_config

# Quick architecture check (no PyTorch needed)
path, success, _ = download_config_file("meta-llama/Llama-2-7b-hf")
if success:
    analysis = analyze_model_config(path)
    arch = analysis.architecture_info

    print(f"Model: {arch.model_type}")
    if arch.estimated_parameters is not None:
        print(f"Parameters: ~{arch.estimated_parameters.estimated_total_millions}M")
    else:
        print(f"Parameters: N/A")
    print(f"Hidden: {arch.hidden_size}, Layers: {arch.num_layers}")
    print(f"Vocab: {arch.vocab_size}")
```

## Query HuggingFace Dataset with DuckDB

```python
import duckdb
from dr_hf import query_hf_with_duckdb, HFLocation

# Requires: uv add dr-hf[duckdb]
conn = duckdb.connect()
loc = HFLocation.from_uri("hf://datasets/squad/squad")

# Query returns dict of DataFrames (one per file)
results = query_hf_with_duckdb(loc, conn)

# Access the first result
if results:
    df = list(results.values())[0]
    print(df.head())
```

## Build HFLocation from Different Sources

```python
from dr_hf import HFLocation

# From components
loc1 = HFLocation(org="allenai", repo_name="c4")

# From URI
loc2 = HFLocation.from_uri("hf://datasets/squad/squad")

# With specific files
loc3 = HFLocation(
    org="allenai",
    repo_name="dolma",
    filepaths=["data/train-00000.parquet", "data/train-00001.parquet"]
)

# Get file URIs
for fp in loc3.filepaths:
    print(loc3.get_path_uri(fp))
```

## Batch Process Multiple Repositories

```python
from dr_hf import create_branch_metadata

repos = [
    "org/model-v1-checkpoints",
    "org/model-v2-checkpoints",
    "org/model-v3-checkpoints",
]

for repo in repos:
    metadata = create_branch_metadata(repo)
    print(f"\n{repo}:")
    print(f"  Total checkpoints: {metadata.checkpoint_branches}")
    for seed, config in metadata.seed_configurations.items():
        steps = config.step_range
        print(f"  {seed}: {config.count} checkpoints (steps {steps[0]}-{steps[1]})")
```
