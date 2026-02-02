# checkpoints

Checkpoint orchestration combining configs, weights, and optimizer analysis. Requires the `[weights]` optional dependency.

```bash
uv add dr-hf[weights]
```

## Functions

### download_optimizer_checkpoint
```python
def download_optimizer_checkpoint(
    repo_id: str,
    branch: str = "main",
    local_dir: str | None = None
) -> tuple[str | None, bool, str]
```
Download optimizer checkpoint (training/optim.pt). Returns `(file_path, success, error_message)`.

### analyze_optimizer_checkpoint
```python
def analyze_optimizer_checkpoint(checkpoint_path: str) -> OptimizerAnalysis
```
Analyze an optimizer checkpoint file, extracting learning rates and state info.

### analyze_complete_checkpoint
```python
def analyze_complete_checkpoint(
    repo_id: str,
    branch: str,
    include_weights: bool = False,
    weight_files: list[str] | None = None,
    delete_weights_after: bool = False
) -> CheckpointAnalysis
```
Analyze a complete checkpoint (optimizer + config + optionally weights).

### process_single_checkpoint
```python
def process_single_checkpoint(
    repo_id: str,
    branch: str,
    include_weights: bool = False,
    delete_weights_after: bool = False
) -> tuple[str, CheckpointAnalysis]
```
Process a single checkpoint branch. Returns `(branch, analysis)`.

### process_all_checkpoints
```python
def process_all_checkpoints(
    repo_id: str,
    max_workers: int = 4,
    include_weights: bool = False,
    delete_weights_after: bool = False
) -> dict[str, CheckpointAnalysis]
```
Process all checkpoint branches in parallel.

### create_comprehensive_summary
```python
def create_comprehensive_summary(
    all_analyses: dict[str, CheckpointAnalysis]
) -> pd.DataFrame
```
Create a DataFrame summary of all checkpoint analyses.

### create_learning_rate_summary
```python
def create_learning_rate_summary(
    all_analyses: dict[str, CheckpointAnalysis]
) -> pd.DataFrame
```
Create a focused summary of learning rate progression.

### save_checkpoint_analysis
```python
def save_checkpoint_analysis(
    analysis: CheckpointAnalysis,
    branch: str,
    output_dir: str | None = None
) -> str
```
Save a single analysis to JSON.

### save_all_analyses_outputs
```python
def save_all_analyses_outputs(
    all_analyses: dict[str, CheckpointAnalysis],
    output_dir: str | None = None
) -> tuple[str, str, str]
```
Save comprehensive CSV, LR CSV, and JSON. Returns paths to all three files.

## Models

- `CheckpointAnalysis` - Full checkpoint analysis with `branch`, `step`, `components`
- `CheckpointComponents` - Container for `optimizer`, `config`, `weights` analyses
- `CheckpointSummaryRow` - DataFrame row model with `from_analysis()` class method
- `OptimizerAnalysis` - Optimizer state with `checkpoint_keys`, `optimizer_info`, `learning_rate_info`
- `OptimizerComponentInfo` - Component details (param_groups, state)
- `LearningRateInfo` - Learning rate extraction with param group details
- `ParamGroupInfo` - Per-group LR, weight decay, momentum

## Usage

```python
from dr_hf import (
    process_all_checkpoints,
    create_comprehensive_summary,
    save_all_analyses_outputs,
)

# Process all checkpoints in a repo
analyses = process_all_checkpoints(
    "org/model-checkpoints",
    max_workers=8,
    include_weights=False,  # Skip weights for speed
)

# Create summary DataFrame
summary_df = create_comprehensive_summary(analyses)
print(summary_df[["branch", "step", "current_lr", "model_type"]])

# Track learning rate over training
for branch, analysis in sorted(analyses.items(), key=lambda x: x[1].step):
    opt = analysis.components.optimizer
    if opt.available and opt.learning_rate_info:
        lr = opt.learning_rate_info.param_group_lrs[0].current_lr
        print(f"Step {analysis.step}: LR = {lr}")

# Save all outputs
csv_path, lr_path, json_path = save_all_analyses_outputs(analyses, "output/")
```
