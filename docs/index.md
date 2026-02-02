# dr_hf Documentation

HuggingFace utilities for repository management, dataset operations, and model analysis.

## Design Philosophy

- **Type-safe**: Comprehensive Pydantic models for all data structures
- **Optional dependencies**: Core features work without PyTorch; weight analysis optional
- **Fail fast**: Uses assertions for validation, not silent failures
- **Lazy loading**: Heavy dependencies loaded only when needed

## Modules

### Repository Operations
- [branches](branches.md) - Branch discovery, parsing, and metadata extraction
- [location](location.md) - Pydantic model for HF resource URIs

### Model Analysis
- [configs](configs.md) - Model config.json analysis and architecture extraction
- [weights](weights.md) - Model weight statistics and layer analysis (requires `[weights]`)
- [checkpoints](checkpoints.md) - Checkpoint orchestration combining configs, weights, optimizer (requires `[weights]`)

### Data Operations
- [datasets](datasets.md) - Dataset loading, downloading, and caching
- [io](io.md) - HfApi upload/download operations

### Utilities
- [paths](paths.md) - Environment path management

### Data Models
- [models](models.md) - All Pydantic models for type-safe data handling

## Common Patterns

See [Recipes & Patterns](recipes.md) for common usage patterns.

## API Reference

See [Full API Reference](api.md) for auto-generated documentation.
