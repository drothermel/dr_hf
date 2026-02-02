# weights

Model weight statistics and layer analysis. Requires the `[weights]` optional dependency.

```bash
uv add dr-hf[weights]
```

## Functions

### discover_model_weight_files
```python
def discover_model_weight_files(repo_id: str, branch: str = "main") -> list[str]
```
Find weight files in a repository (pytorch_model.bin, model.safetensors, sharded files).

### download_model_weights
```python
def download_model_weights(
    repo_id: str,
    branch: str,
    filename: str,
    local_dir: str | None = None
) -> tuple[str | None, bool, str]
```
Download a specific weight file. Returns `(file_path, success, error_message)`.

### calculate_weight_statistics
```python
def calculate_weight_statistics(weight_path: str) -> WeightFileStatistics
```
Analyze a weight file, computing tensor statistics and layer categorization.

### calculate_tensor_stats
```python
def calculate_tensor_stats(tensor: Any) -> TensorStats | None
```
Calculate statistics for a single tensor (mean, std, min, max, percentiles).

### analyze_layer_structure
```python
def analyze_layer_structure(weights: dict[str, Any]) -> LayerAnalysis
```
Categorize layers by type (embedding, attention, feedforward, layer norm, etc.).

### calculate_global_weight_stats
```python
def calculate_global_weight_stats(weights: dict[str, Any]) -> GlobalWeightStats | None
```
Calculate statistics across all weights combined.

### analyze_model_weights
```python
def analyze_model_weights(
    repo_id: str,
    branch: str,
    weight_files: list[str] | None = None,
    delete_after_analysis: bool = False
) -> WeightsAnalysis
```
Complete workflow: discover, download, and analyze all weight files.

## Models

- `WeightsAnalysis` - Full analysis with `available`, `discovered_files`, `file_analyses`, `summary`
- `WeightsSummary` - Aggregated stats with computed fields (total_parameters_millions, etc.)
- `WeightFileStatistics` - Per-file stats with tensor info and layer analysis
- `TensorInfo` - Per-tensor metadata (name, shape, dtype, parameters, size_mb, statistics)
- `TensorStats` - Statistics (mean, std, min, max, percentiles)
- `LayerAnalysis` - Layer categorization and counts
- `LayerCategorization` - Lists of layers by type
- `LayerCounts` - Count summary
- `GlobalWeightStats` - Global statistics across all weights
- `ParameterStats` - Parameter counts with computed millions/billions

## Usage

```python
from dr_hf import (
    discover_model_weight_files,
    analyze_model_weights,
    calculate_weight_statistics,
)

# Discover weight files
files = discover_model_weight_files("gpt2", branch="main")
print(f"Found: {files}")

# Full analysis workflow
analysis = analyze_model_weights("gpt2", branch="main")
if analysis.available:
    summary = analysis.summary
    print(f"Total parameters: {summary.total_parameters_millions}M")
    print(f"Total size: {summary.total_size_gb}GB")

    # Per-file details
    for filename, stats in analysis.file_analyses.items():
        print(f"\n{filename}:")
        print(f"  Tensors: {stats.num_tensors}")

        # Layer breakdown
        if stats.layer_analysis:
            counts = stats.layer_analysis.layer_counts
            print(f"  Transformer layers: {counts.estimated_transformer_layers}")
            print(f"  Layer norms: {counts.layer_norms}")
```
