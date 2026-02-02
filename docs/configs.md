# configs

Model config.json analysis and architecture extraction.

## Functions

### download_config_file
```python
def download_config_file(
    repo_id: str,
    branch: str = "main",
    local_dir: str | None = None
) -> tuple[str | None, bool, str]
```
Download config.json from a HuggingFace repository. Returns `(file_path, success, error_message)`.

### analyze_model_config
```python
def analyze_model_config(config_path: str) -> ConfigAnalysis
```
Load and analyze a model config file. Returns a `ConfigAnalysis` model with architecture info.

### extract_model_architecture_info
```python
def extract_model_architecture_info(config: dict[str, Any]) -> ArchitectureInfo
```
Extract standardized architecture information from a config dict. Handles different naming conventions (hidden_size vs d_model, etc.).

### estimate_parameter_count
```python
def estimate_parameter_count(
    hidden_size: int,
    num_layers: int,
    vocab_size: int,
    intermediate_size: int | None = None
) -> ParameterEstimate | None
```
Estimate total parameter count from architecture dimensions. Returns `None` if required fields are missing.

## Models

- `ConfigAnalysis` - Analysis result with `available`, `raw_config`, `architecture_info`, `config_keys`
- `ArchitectureInfo` - Standardized architecture fields (hidden_size, num_layers, vocab_size, etc.)
- `ParameterEstimate` - Estimated parameter counts with computed fields for millions

## Usage

```python
from dr_hf import (
    download_config_file,
    analyze_model_config,
    estimate_parameter_count,
)

# Download and analyze config
path, success, error = download_config_file("gpt2", branch="main")
if success:
    analysis = analyze_model_config(path)
    arch = analysis.architecture_info

    print(f"Model type: {arch.model_type}")
    print(f"Hidden size: {arch.hidden_size}")
    print(f"Layers: {arch.num_layers}")
    print(f"Vocab size: {arch.vocab_size}")

    if arch.estimated_parameters:
        print(f"Estimated params: {arch.estimated_parameters.estimated_total_millions}M")

# Manual parameter estimation
estimate = estimate_parameter_count(
    hidden_size=768,
    num_layers=12,
    vocab_size=50257,
    intermediate_size=3072,
)
print(f"Estimated: {estimate.estimated_total_millions}M parameters")
```
