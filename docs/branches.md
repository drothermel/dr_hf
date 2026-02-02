# branches

Branch discovery, parsing, and metadata extraction for HuggingFace repositories.

## Functions

### get_all_repo_branches
```python
def get_all_repo_branches(repo_id: str) -> list[str]
```
Get all branch names from a HuggingFace repository.

### is_checkpoint_branch
```python
def is_checkpoint_branch(branch: str) -> bool
```
Check if a branch name matches the checkpoint pattern `stepN-seed-*`.

### get_checkpoint_branches
```python
def get_checkpoint_branches(repo_id: str) -> list[str]
```
Get checkpoint branches from a repo, filtered and sorted by step.

### parse_branch_name
```python
def parse_branch_name(branch: str) -> BranchInfo
```
Parse a branch name into its components. Returns a `BranchInfo` model with `step`, `seed`, `valid`, and `branch` fields.

### extract_step_from_branch
```python
def extract_step_from_branch(branch: str) -> int
```
Extract the step number from a branch name. Returns 0 if not a valid checkpoint branch.

### extract_seed_from_branch
```python
def extract_seed_from_branch(branch: str) -> str
```
Extract the seed string from a branch name. Returns "unknown" if not a valid checkpoint branch.

### sort_branches_by_step
```python
def sort_branches_by_step(branches: list[str]) -> list[str]
```
Sort branch names by their step number.

### group_branches_by_seed
```python
def group_branches_by_seed(branches: list[str]) -> dict[str, list[str]]
```
Group branches by seed, with each group sorted by step.

### get_step_range_for_seed
```python
def get_step_range_for_seed(branches: list[str]) -> tuple[int, int]
```
Get the minimum and maximum step numbers for a list of branches.

### create_branch_metadata
```python
def create_branch_metadata(repo_id: str) -> BranchMetadata
```
Create comprehensive metadata for all branches in a repository.

## Models

- `BranchInfo` - Parsed branch with `branch`, `valid`, `step`, `seed`
- `SeedBranchInfo` - Step and branch name pair
- `SeedConfiguration` - Seed group with count, step range, branches
- `BranchMetadata` - Full repo metadata with all configurations

## Usage

```python
from dr_hf import (
    get_checkpoint_branches,
    parse_branch_name,
    group_branches_by_seed,
    create_branch_metadata,
)

# Get and parse checkpoint branches
branches = get_checkpoint_branches("org/model-checkpoints")
for branch in branches:
    info = parse_branch_name(branch)
    if info.valid:
        print(f"Step {info.step}, Seed: {info.seed}")

# Group by seed for comparison
groups = group_branches_by_seed(branches)
for seed, seed_branches in groups.items():
    print(f"Seed {seed}: {len(seed_branches)} checkpoints")

# Get full metadata
metadata = create_branch_metadata("org/model-checkpoints")
print(f"Total checkpoints: {metadata.checkpoint_branches}")
for seed, config in metadata.seed_configurations.items():
    print(f"  {seed}: steps {config.step_range[0]}-{config.step_range[1]}")
```
