from __future__ import annotations

from dr_hf.models import (
    ArchitectureInfo,
    CheckpointAnalysis,
    CheckpointComponents,
    CheckpointSummaryRow,
    ConfigAnalysis,
    LearningRateInfo,
    OptimizerAnalysis,
    ParamGroupInfo,
    ParameterEstimate,
    WeightsAnalysis,
    WeightsSummary,
)


def test_param_group_info() -> None:
    info = ParamGroupInfo(group_index=0, current_lr=0.001, weight_decay=0.01)
    assert info.group_index == 0
    assert info.current_lr == 0.001
    assert info.weight_decay == 0.01
    assert info.momentum is None


def test_learning_rate_info() -> None:
    lr_info = LearningRateInfo(
        param_group_lrs=[ParamGroupInfo(group_index=0, current_lr=0.001)],
        lr_scheduler_present=True,
    )
    assert len(lr_info.param_group_lrs) == 1
    assert lr_info.lr_scheduler_present


def test_optimizer_analysis() -> None:
    analysis = OptimizerAnalysis(
        available=True,
        checkpoint_keys=["state", "param_groups"],
        checkpoint_type="dict",
    )
    assert analysis.available
    assert "state" in analysis.checkpoint_keys


def test_config_analysis() -> None:
    est = ParameterEstimate(
        embedding_parameters=1000,
        per_layer_parameters=500,
        total_layer_parameters=2000,
        output_head_parameters=1000,
        estimated_total_parameters=4000,
        estimated_total_millions=0.004,
    )
    arch = ArchitectureInfo(
        hidden_size=768,
        num_layers=12,
        vocab_size=50000,
        estimated_parameters=est,
    )
    cfg = ConfigAnalysis(available=True, architecture_info=arch)
    assert cfg.available
    assert cfg.architecture_info.hidden_size == 768


def test_weights_analysis() -> None:
    summary = WeightsSummary(
        total_files_analyzed=1,
        total_parameters=1000000,
        total_parameters_millions=1.0,
    )
    wgt = WeightsAnalysis(available=True, summary=summary)
    assert wgt.available
    assert wgt.weights_available
    assert wgt.summary.total_parameters_millions == 1.0


def test_checkpoint_analysis() -> None:
    analysis = CheckpointAnalysis(
        branch="step1000-seed-default",
        step=1000,
        components=CheckpointComponents(),
    )
    assert analysis.branch == "step1000-seed-default"
    assert analysis.step == 1000
    assert not analysis.components.optimizer.available


def test_checkpoint_summary_row_from_analysis() -> None:
    lr_info = LearningRateInfo(
        param_group_lrs=[
            ParamGroupInfo(group_index=0, current_lr=0.001, weight_decay=0.01)
        ]
    )
    opt = OptimizerAnalysis(available=True, learning_rate_info=lr_info)
    arch = ArchitectureInfo(hidden_size=768, num_layers=12, model_type="gpt2")
    cfg = ConfigAnalysis(available=True, architecture_info=arch)

    analysis = CheckpointAnalysis(
        branch="step1000-seed-default",
        step=1000,
        components=CheckpointComponents(optimizer=opt, config=cfg),
    )

    row = CheckpointSummaryRow.from_analysis(analysis)
    assert row.branch == "step1000-seed-default"
    assert row.step == 1000
    assert row.optimizer_available
    assert row.current_lr == 0.001
    assert row.weight_decay == 0.01
    assert row.config_available
    assert row.hidden_size == 768
    assert row.model_type == "gpt2"


def test_model_dump() -> None:
    analysis = CheckpointAnalysis(
        branch="step0-seed-test",
        step=0,
    )
    data = analysis.model_dump()
    assert data["branch"] == "step0-seed-test"
    assert data["step"] == 0
    assert "components" in data
