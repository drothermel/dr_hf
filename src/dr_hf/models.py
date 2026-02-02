from __future__ import annotations

from pydantic import BaseModel, computed_field


class ParamGroupInfo(BaseModel):
    group_index: int
    current_lr: float | None = None
    initial_lr: float | None = None
    weight_decay: float | None = None
    momentum: float | None = None


class LearningRateInfo(BaseModel):
    param_group_lrs: list[ParamGroupInfo] = []
    lr_scheduler_present: bool = False
    lr_scheduler_keys: list[str] = []
    scheduler_last_epoch: int | None = None


class OptimizerComponentInfo(BaseModel):
    type: str
    size: int | str = "N/A"
    num_param_groups: int | None = None
    param_group_keys: list[str] = []
    learning_rate: float | None = None
    initial_learning_rate: float | None = None
    num_parameters: int | None = None
    state_keys: list[str] = []


class OptimizerAnalysis(BaseModel):
    available: bool = False
    error: str | None = None
    checkpoint_keys: list[str] = []
    checkpoint_type: str | None = None
    optimizer_info: dict[str, OptimizerComponentInfo] = {}
    learning_rate_info: LearningRateInfo | None = None


class ParameterEstimate(BaseModel):
    embedding_parameters: int
    per_layer_parameters: int
    total_layer_parameters: int
    output_head_parameters: int
    estimated_total_parameters: int
    estimated_total_millions: float


class ArchitectureInfo(BaseModel):
    hidden_size: int | None = None
    num_layers: int | None = None
    num_attention_heads: int | None = None
    intermediate_size: int | None = None
    vocab_size: int | None = None
    max_position_embeddings: int | None = None
    sequence_length: int | None = None
    model_type: str | None = None
    activation_function: str | None = None
    layer_norm_eps: float | None = None
    dropout: float | None = None
    pad_token_id: int | None = None
    eos_token_id: int | None = None
    bos_token_id: int | None = None
    torch_dtype: str | None = None
    use_cache: bool | None = None
    tie_word_embeddings: bool | None = None
    rope_scaling: dict | None = None
    estimated_parameters: ParameterEstimate | None = None


class ConfigAnalysis(BaseModel):
    available: bool = False
    error: str | None = None
    raw_config: dict | None = None
    architecture_info: ArchitectureInfo | None = None
    config_keys: list[str] = []
    config_type: str | None = None


class WeightsSummary(BaseModel):
    total_files_analyzed: int = 0
    total_parameters: int = 0
    total_parameters_millions: float = 0.0
    total_parameters_billions: float = 0.0
    total_size_mb: float = 0.0
    total_size_gb: float = 0.0


class WeightsAnalysis(BaseModel):
    available: bool = False
    error: str | None = None
    discovered_files: list[str] = []
    file_analyses: dict = {}
    summary: WeightsSummary | None = None

    @computed_field
    @property
    def weights_available(self) -> bool:
        return self.available


class CheckpointComponents(BaseModel):
    optimizer: OptimizerAnalysis = OptimizerAnalysis()
    config: ConfigAnalysis = ConfigAnalysis()
    weights: WeightsAnalysis = WeightsAnalysis()


class CheckpointAnalysis(BaseModel):
    branch: str
    step: int
    error: str | None = None
    components: CheckpointComponents = CheckpointComponents()


class CheckpointSummaryRow(BaseModel):
    branch: str
    step: int
    optimizer_available: bool = False
    config_available: bool = False
    weights_available: bool = False
    current_lr: float | None = None
    weight_decay: float | None = None
    momentum: float | None = None
    num_param_groups: int | None = None
    model_type: str | None = None
    hidden_size: int | None = None
    num_layers: int | None = None
    vocab_size: int | None = None
    estimated_params_millions: float | None = None
    total_weight_params_millions: float | None = None
    optimizer_error: str = ""
    config_error: str = ""
    weights_error: str = ""

    @classmethod
    def from_analysis(cls, analysis: CheckpointAnalysis) -> CheckpointSummaryRow:
        row = cls(branch=analysis.branch, step=analysis.step)

        opt = analysis.components.optimizer
        row.optimizer_available = opt.available
        if opt.available and opt.learning_rate_info:
            lr_info = opt.learning_rate_info
            if lr_info.param_group_lrs:
                first = lr_info.param_group_lrs[0]
                row.current_lr = first.current_lr
                row.weight_decay = first.weight_decay
                row.momentum = first.momentum
                row.num_param_groups = len(lr_info.param_group_lrs)
        elif opt.error:
            row.optimizer_error = opt.error

        cfg = analysis.components.config
        row.config_available = cfg.available
        if cfg.available and cfg.architecture_info:
            arch = cfg.architecture_info
            row.model_type = arch.model_type
            row.hidden_size = arch.hidden_size
            row.num_layers = arch.num_layers
            row.vocab_size = arch.vocab_size
            if arch.estimated_parameters:
                row.estimated_params_millions = (
                    arch.estimated_parameters.estimated_total_millions
                )
        elif cfg.error:
            row.config_error = cfg.error

        wgt = analysis.components.weights
        row.weights_available = wgt.available
        if wgt.available and wgt.summary:
            row.total_weight_params_millions = wgt.summary.total_parameters_millions
        elif wgt.error:
            row.weights_error = wgt.error

        return row
