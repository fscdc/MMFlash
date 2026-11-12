# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DraftTrainStrategy: per-draft-model required features + forward/loss + projection.

A strategy is the only place that knows how a draft model (DFlash / MMFlash)
turns a normalized ``TrainBatch`` into a loss; ``TrainerCore``
stays branch-free and the strategy owns the target projection. Imports model
code, so it is imported by training entry points, not at package load.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from specforge.runtime.contracts import TrainBatch


@dataclass(frozen=True)
class StepOutput:
    """Per-step result: loss + strategy-specific metrics, kept generic so
    per-position (TTT) and single-scalar strategies share one trainer loop."""

    loss: torch.Tensor
    metrics: Dict[str, Any]
    ratio_metrics: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class StepContext:
    """Training-schedule state passed into ``forward_loss`` for objectives that
    depend on where in training we are; the built-in strategies ignore it."""

    global_step: int = 0
    total_steps: Optional[int] = None


class DraftTrainStrategy(abc.ABC):
    name: str
    required_features: set

    @abc.abstractmethod
    def trainable_module(self) -> nn.Module:
        """The module whose parameters the optimizer/backend owns."""

    def validate_batch(self, batch: TrainBatch) -> None:
        missing = {f for f in self.required_features if f not in batch.tensors}
        if missing:
            raise ValueError(
                f"{self.name} batch missing required features {sorted(missing)}; "
                f"present={sorted(batch.tensors)}"
            )

    @abc.abstractmethod
    def forward_loss(
        self, batch: TrainBatch, ctx: Optional["StepContext"] = None
    ) -> StepOutput: ...

    def checkpoint_state_filter(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Select the keys this strategy persists as draft weights."""
        return state_dict


class DFlashTrainStrategy(DraftTrainStrategy):
    """DFlash block-parallel strategy wrapping ``OnlineDFlashModel`` (the baseline MMFlash extends).

    One block-wise forward pass per step with a scalar loss on hard real-token labels
    (no target distribution, no vocab map).
    """

    name = "dflash"
    required_features = {"input_ids", "hidden_states", "loss_mask"}

    def __init__(self, dflash_model: nn.Module) -> None:
        self.dflash_model = dflash_model

    def trainable_module(self) -> nn.Module:
        return self.dflash_model

    def _device(self) -> torch.device:
        return next(self.dflash_model.parameters()).device

    def forward_loss(
        self, batch: TrainBatch, ctx: Optional[StepContext] = None
    ) -> StepOutput:
        self.validate_batch(batch)
        t = batch.tensors
        device = self._device()
        loss, accuracy, model_metrics = self.dflash_model(
            input_ids=t["input_ids"].to(device),
            hidden_states=t["hidden_states"].to(device),
            loss_mask=t["loss_mask"].to(device),
        )
        metrics = {"accuracy": accuracy.detach()}
        if "accuracy_denom" in model_metrics:
            metrics["accuracy_denom"] = model_metrics["accuracy_denom"]
        return StepOutput(loss=loss, metrics=metrics)

    def checkpoint_state_filter(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        # Everything trainable lives under draft_model.; the target
        # embedding/head are a separate module, not persisted as draft weights.
        return {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k
        }


class MMFlashTrainStrategy(DraftTrainStrategy):
    """MMFlash block-parallel strategy wrapping ``OnlineMMFlashModel``.

    Same step as ``DFlashTrainStrategy`` plus the per-token ``visual_score`` channel of image captures, which the
    MMFlash objective uses to weight the loss toward visually dependent tokens.
    """

    name = "mmflash"
    required_features = {"input_ids", "hidden_states", "loss_mask"}

    def __init__(self, mmflash_model: nn.Module) -> None:
        self.mmflash_model = mmflash_model

    def trainable_module(self) -> nn.Module:
        return self.mmflash_model

    def _device(self) -> torch.device:
        return next(self.mmflash_model.parameters()).device

    def forward_loss(
        self, batch: TrainBatch, ctx: Optional[StepContext] = None
    ) -> StepOutput:
        self.validate_batch(batch)
        t = batch.tensors
        device = self._device()
        # visual_score is optional at the batch level: offline files, text
        # capture and evaluation batches do not carry it, and the model then
        # treats every row as text-only (training.loss_type for all rows).
        visual_score = t.get("visual_score")
        loss, accuracy, model_metrics = self.mmflash_model(
            input_ids=t["input_ids"].to(device),
            hidden_states=t["hidden_states"].to(device),
            loss_mask=t["loss_mask"].to(device),
            visual_score=None if visual_score is None else visual_score.to(device),
        )
        metrics = {"accuracy": accuracy.detach()}
        if "accuracy_denom" in model_metrics:
            metrics["accuracy_denom"] = model_metrics["accuracy_denom"]
        return StepOutput(
            loss=loss,
            metrics=metrics,
            ratio_metrics=model_metrics.get("ratio_metrics", {}),
        )

    def checkpoint_state_filter(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        # Everything trainable lives under draft_model.; the target
        # embedding/head are a separate module, not persisted as draft weights.
        return {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k
        }


__all__ = [
    "DraftTrainStrategy",
    "DFlashTrainStrategy",
    "MMFlashTrainStrategy",
    "StepOutput",
    "StepContext",
]
