# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Algorithm-owned model and configuration providers.

The public assembler owns topology and lifecycle.  This module owns the facts
that vary by draft method: how to create its config and model, which target
features it captures, and which dataset policies it needs.  All heavy imports
remain inside callables so resolving the algorithm registry stays import-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from specforge.config import Config
    from specforge.modeling.draft.dflash_kernels import DFlashKernels


@dataclass
class AlgorithmModelParts:
    """Algorithm-specific pieces returned to the application assembler."""

    model: Any
    target_head: Any = None
    capture_layers: Optional[List[int]] = None


def dflash_needs_input_tools(cfg: Config, draft_model: Any) -> bool:
    method_config = getattr(draft_model.config, "dflash_config", None) or {}
    needs_mask_fallback = (
        cfg.model.mask_token_id is None and method_config.get("mask_token_id") is None
    )
    return cfg.mode == "online" or needs_mask_fallback


def _torch_dtype(cfg: Config):
    import torch

    return getattr(torch, cfg.model.torch_dtype)


def _device():
    from specforge.utils import get_local_device

    return get_local_device()


def _warm_start(
    cfg: Config,
    draft_model: Any,
    draft_config: Any,
    *,
    allow_missing_embedding: bool = False,
) -> None:
    if not cfg.model.draft_checkpoint_path:
        return
    from specforge.training.model_loading import warm_start_draft_model

    warm_start_draft_model(
        draft_model,
        cfg.model.draft_checkpoint_path,
        draft_config=draft_config,
        strategy=cfg.training.strategy,
        allow_missing_embedding=allow_missing_embedding,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )


def _finish_registered_draft(
    cfg: Config,
    draft_config: PretrainedConfig,
    draft_model: Any,
):
    _warm_start(cfg, draft_model, draft_config)
    return draft_model.to(device=_device(), dtype=_torch_dtype(cfg))


def build_registered_draft(cfg: Config, draft_config: PretrainedConfig):
    from specforge.modeling.auto import AutoDraftModel

    draft_config._attn_implementation = cfg.training.attention_backend
    draft_model = AutoDraftModel.from_config(
        draft_config,
        torch_dtype=_torch_dtype(cfg),
    )
    return _finish_registered_draft(cfg, draft_config, draft_model)


def build_dflash_draft(
    cfg: Config,
    draft_config: PretrainedConfig,
    kernels: Optional[DFlashKernels],
):
    from specforge.modeling.auto import AutoDraftModel

    draft_config._attn_implementation = cfg.training.attention_backend
    draft_model = AutoDraftModel.from_config(
        draft_config,
        torch_dtype=_torch_dtype(cfg),
        dflash_kernels=kernels,
    )
    return _finish_registered_draft(cfg, draft_config, draft_model)


def resolve_dflash_capture_layers(
    _cfg: Config, draft_config: Any, _target_config: Any
) -> List[int]:
    method_config = (
        draft_config.get("dflash_config", {})
        if isinstance(draft_config, dict)
        else getattr(draft_config, "dflash_config", {})
    ) or {}
    layers = list(method_config.get("target_layer_ids", []))
    if not layers:
        raise ValueError("draft config does not define target capture layer ids")
    return layers


def _resolve_mask_token_id(cfg: Config, draft_model: Any, tokenizer: Any) -> int:
    from specforge.training.model_utils import resolve_mask_token_id

    return resolve_mask_token_id(
        explicit=cfg.model.mask_token_id,
        # An image run's input tools are a processor, which carries no token ids
        # of its own; its tokenizer does.
        tokenizer=getattr(tokenizer, "tokenizer", tokenizer),
        draft_model=draft_model,
        embedding_vocab_size=int(draft_model.config.vocab_size),
    )


def _build_dflash_family_model(
    cfg: Config,
    draft_model: Any,
    tokenizer: Any,
    model_factory: Callable[[Dict[str, Any]], Any],
) -> AlgorithmModelParts:
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

    mask_token_id = _resolve_mask_token_id(cfg, draft_model, tokenizer)
    draft_model.mask_token_id = mask_token_id
    method_config = getattr(draft_model.config, "dflash_config", None)
    if method_config is None:
        draft_model.config.dflash_config = {}
        method_config = draft_model.config.dflash_config
    method_config["mask_token_id"] = mask_token_id
    method_config["target_layer_ids"] = list(draft_model.target_layer_ids)

    # SGLang serves the draft's noise and anchor tokens through the target's
    # own embedding module, which for Gemma 3/3n/4 scales its lookups; the
    # embedding built here applies the same factor (lm_head stays raw).
    target_parts = TargetEmbeddingsAndHead.from_pretrained(
        cfg.model.target_model_path,
        embed_key=cfg.model.embedding_key,
        lm_head_key=cfg.model.lm_head_key,
        cache_dir=cfg.model.cache_dir,
        device=_device().type,
        dtype=_torch_dtype(cfg),
        trust_remote_code=cfg.model.trust_remote_code,
        embed_scale=cfg.model.target_embedding_scale,
    )
    common = {
        "draft_model": draft_model,
        "target_lm_head": target_parts.lm_head,
        "target_embed_tokens": target_parts.embed_tokens,
        "mask_token_id": mask_token_id,
        "block_size": int(draft_model.block_size),
        "attention_backend": cfg.training.attention_backend,
        "num_anchors": cfg.training.num_anchors,
        "loss_decay_gamma": cfg.training.loss_decay_gamma,
        "objective_chunk_blocks": cfg.training.objective_chunk_blocks,
    }
    model = model_factory(common).to(device=_device(), dtype=_torch_dtype(cfg))
    return AlgorithmModelParts(
        model=model,
        capture_layers=list(draft_model.target_layer_ids),
    )


def build_dflash_model(
    cfg: Config,
    draft_model: Any,
    _draft_config: Any,
    _target_config: Any,
    tokenizer: Any,
) -> AlgorithmModelParts:
    from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel

    return _build_dflash_family_model(
        cfg,
        draft_model,
        tokenizer,
        lambda common: OnlineDFlashModel(
            **common,
            loss_type=cfg.training.loss_type,
            dpace_alpha=cfg.training.dpace_alpha,
        ),
    )


# --- MMFlash: clones of the DFlash builders so the strategy can diverge -------
# ``dflash_config`` stays: it is the draft-config JSON key the family and SGLang
# share. ``_build_dflash_family_model`` stays: it is the common assembler.

def mmflash_needs_input_tools(cfg: Config, draft_model: Any) -> bool:
    method_config = getattr(draft_model.config, "dflash_config", None) or {}
    needs_mask_fallback = (
        cfg.model.mask_token_id is None and method_config.get("mask_token_id") is None
    )
    return cfg.mode == "online" or needs_mask_fallback


def build_mmflash_draft(
    cfg: Config,
    draft_config: PretrainedConfig,
    kernels: Optional[DFlashKernels],
):
    from specforge.modeling.auto import AutoDraftModel

    draft_config._attn_implementation = cfg.training.attention_backend
    draft_model = AutoDraftModel.from_config(
        draft_config,
        torch_dtype=_torch_dtype(cfg),
        mmflash_kernels=kernels,
    )
    return _finish_registered_draft(cfg, draft_config, draft_model)


def resolve_mmflash_capture_layers(
    _cfg: Config, draft_config: Any, _target_config: Any
) -> List[int]:
    method_config = (
        draft_config.get("dflash_config", {})
        if isinstance(draft_config, dict)
        else getattr(draft_config, "dflash_config", {})
    ) or {}
    layers = list(method_config.get("target_layer_ids", []))
    if not layers:
        raise ValueError("draft config does not define target capture layer ids")
    return layers


def build_mmflash_model(
    cfg: Config,
    draft_model: Any,
    _draft_config: Any,
    _target_config: Any,
    tokenizer: Any,
) -> AlgorithmModelParts:
    from specforge.algorithms.common.mmflash_model import OnlineMMFlashModel

    if cfg.training.draft_sparse is not None:
        return _build_sparse_mmflash_model(cfg, draft_model, _target_config, tokenizer)

    return _build_dflash_family_model(
        cfg,
        draft_model,
        tokenizer,
        lambda common: OnlineMMFlashModel(
            **common,
            loss_type=cfg.training.loss_type,
            dpace_alpha=cfg.training.dpace_alpha,
            visual_alpha=cfg.training.visual_alpha,
            mmflash_smoothing=cfg.training.mmflash_smoothing,
        ),
    )


def _build_sparse_mmflash_model(
    cfg: Config,
    draft_model: Any,
    target_config: Any,
    tokenizer: Any,
) -> AlgorithmModelParts:
    """``training.draft_sparse`` set: the same model, trained on the sparse context.

    Kept apart from ``build_mmflash_model`` so a run without the block builds
    exactly what it always did. The visual spans are the runs of the target's
    image/video pad tokens, read from the full (vision-language) target config
    -- the same ids SGLang builds the multimodal offsets of the served request
    from.
    """
    from specforge.algorithms.common.mmflash_sparse import (
        DraftSparseContext,
        OnlineSparseMMFlashModel,
        visual_token_ids_of,
    )

    draft_sparse = DraftSparseContext.from_config(cfg.training.draft_sparse)
    visual_token_ids = visual_token_ids_of(target_config)
    if not visual_token_ids:
        raise ValueError(
            "training.draft_sparse needs the target's image/video pad token ids "
            f"(image_token_id / video_token_id), but {cfg.model.target_model_path!r} "
            "exposes none in its config"
        )
    return _build_dflash_family_model(
        cfg,
        draft_model,
        tokenizer,
        lambda common: OnlineSparseMMFlashModel(
            **common,
            loss_type=cfg.training.loss_type,
            dpace_alpha=cfg.training.dpace_alpha,
            visual_alpha=cfg.training.visual_alpha,
            mmflash_smoothing=cfg.training.mmflash_smoothing,
            draft_sparse=draft_sparse,
            visual_token_ids=visual_token_ids,
        ),
    )


def mmflash_min_loss_tokens(_cfg: Config, draft_config: Any) -> int:
    block_size = getattr(draft_config, "block_size", None)
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size < 1
    ):
        raise ValueError(
            "MMFlash draft config must define a positive integer block_size"
        )
    return 2 * block_size


def populate_mmflash_generated_config(
    payload: Dict[str, Any], target_config: Any, _cfg: Config
) -> None:
    from specforge.modeling.draft.mmflash import build_target_layer_ids

    target_layers = getattr(target_config, "num_hidden_layers", None)
    if not isinstance(target_layers, int) or target_layers < 1:
        raise ValueError(
            "MMFlash auto-generation requires target num_hidden_layers, got "
            f"{target_layers!r}"
        )
    payload["num_target_layers"] = target_layers
    payload["block_size"] = 16
    payload["dflash_config"] = {
        "target_layer_ids": build_target_layer_ids(target_layers, 1)
    }


def apply_mmflash_overrides(cfg: Config, draft_config: Any) -> None:
    if cfg.model.draft_num_hidden_layers is None:
        return
    from specforge.modeling.draft.mmflash import build_target_layer_ids

    target_layers = int(draft_config.num_target_layers)
    method_config = dict(getattr(draft_config, "dflash_config", None) or {})
    method_config["target_layer_ids"] = build_target_layer_ids(
        target_layers, cfg.model.draft_num_hidden_layers
    )
    draft_config.dflash_config = method_config


def dflash_min_loss_tokens(_cfg: Config, draft_config: Any) -> int:
    block_size = getattr(draft_config, "block_size", None)
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size < 1
    ):
        raise ValueError(
            "DFlash-family draft config must define a positive integer block_size"
        )
    return 2 * block_size


def populate_dflash_generated_config(
    payload: Dict[str, Any], target_config: Any, _cfg: Config
) -> None:
    from specforge.modeling.draft.dflash import build_target_layer_ids

    target_layers = getattr(target_config, "num_hidden_layers", None)
    if not isinstance(target_layers, int) or target_layers < 1:
        raise ValueError(
            "DFlash auto-generation requires target num_hidden_layers, got "
            f"{target_layers!r}"
        )
    payload["num_target_layers"] = target_layers
    payload["block_size"] = 16
    payload["dflash_config"] = {
        "target_layer_ids": build_target_layer_ids(target_layers, 1)
    }


def apply_dflash_overrides(cfg: Config, draft_config: Any) -> None:
    if cfg.model.draft_num_hidden_layers is None:
        return
    from specforge.modeling.draft.dflash import build_target_layer_ids

    target_layers = int(draft_config.num_target_layers)
    method_config = dict(getattr(draft_config, "dflash_config", None) or {})
    method_config["target_layer_ids"] = build_target_layer_ids(
        target_layers, cfg.model.draft_num_hidden_layers
    )
    draft_config.dflash_config = method_config


__all__ = [
    "AlgorithmModelParts",
    "apply_dflash_overrides",
    "apply_mmflash_overrides",
    "build_mmflash_draft",
    "build_mmflash_model",
    "mmflash_min_loss_tokens",
    "mmflash_needs_input_tools",
    "populate_mmflash_generated_config",
    "resolve_mmflash_capture_layers",
    "build_dflash_model",
    "build_registered_draft",
    "dflash_min_loss_tokens",
    "dflash_needs_input_tools",
    "populate_dflash_generated_config",
    "resolve_dflash_capture_layers",
]
