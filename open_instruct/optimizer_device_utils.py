# Copyright 2024 AllenAI. All rights reserved.
#
# Helpers for aligning PyTorch fused AdamW optimizer state with CUDA parameters after
# DeepSpeed checkpoint hydrate (ZeRO loaders can leave `step`/buffers on CPU while params
# are on GPU, which breaks fused Adam kernels).

from __future__ import annotations

from typing import Any

import torch

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def unwrap_deepspeed_torch_optimizer(ds_opt: Any) -> torch.optim.Optimizer | None:
    """Walk nested ``optimizer`` wrappers (e.g. DeepSpeed FP16 / ZeRO) to the leaf ``torch.optim.Optimizer``."""

    cur: Any = ds_opt
    last_torch: torch.optim.Optimizer | None = None
    seen_ids: set[int] = set()
    while cur is not None:
        cid = id(cur)
        if cid in seen_ids:
            break
        seen_ids.add(cid)
        if isinstance(cur, torch.optim.Optimizer):
            last_torch = cur
        inner = getattr(cur, "optimizer", None)
        if inner is None or inner is cur:
            break
        cur = inner
    return last_torch


def reconcile_fused_adam_optimizer_state_devices(
    engine_optimizer: Any,
    *,
    fused: bool,
    rank: int = 0,
) -> None:
    """Move fused-Adam CUDA optimizer-state tensors onto each parameter's device (and fix ``step`` dtype).

    Intended to run immediately after a successful DeepSpeed ``load_checkpoint`` before training steps.

    Args:
        engine_optimizer: DeepSpeed-managed optimizer passed to actors (often wraps a Torch AdamW).
        fused: Whether training uses ``torch.optim.AdamW(..., fused=True)``.
        rank: Distributed rank; avoids duplicate spam from every process.
    """
    if not fused or not torch.cuda.is_available():
        return

    torch_opt = unwrap_deepspeed_torch_optimizer(engine_optimizer)
    if torch_opt is None:
        if rank == 0:
            logger.debug("Skipping fused Adam state reconcile: no leaf torch.optim.Optimizer found.")
        return

    migrated = 0
    for gi, group in enumerate(torch_opt.param_groups):
        group_fused = bool(group.get("fused", torch_opt.defaults.get("fused", False)))
        if not group_fused:
            continue
        for p in group["params"]:
            if p not in torch_opt.state or not p.is_cuda:
                continue
            state = torch_opt.state[p]
            device = p.device

            # PyTorch fused Adam expects scalar step as float32 on the parameter device (see fused `load_state_dict`).
            step_key = "step"
            if step_key in state:
                step_t = state[step_key]
                if isinstance(step_t, torch.Tensor):
                    moved = step_t.to(device=device, non_blocking=True)
                    if moved.dtype != torch.float32:
                        moved = moved.float()
                    if moved.data_ptr() != step_t.data_ptr() or moved.dtype != step_t.dtype or moved.device != step_t.device:
                        migrated += 1
                    state[step_key] = moved

            for k, v in list(state.items()):
                if k == step_key or not isinstance(v, torch.Tensor):
                    continue
                if v.device != device:
                    nv = v.to(device=device, dtype=v.dtype, non_blocking=True)
                    state[k] = nv
                    migrated += 1

    if migrated == 0:
        return
    if rank == 0:
        logger.warning(
            "Optimizer state tensors mismatched fused-Adam CUDA devices after checkpoint hydration; "
            "moving state entries onto each CUDA parameter."
        )
        logger.info(f"Re-aligned {migrated} optimizer-state tensor entr(ies) for fused CUDA Adam continuity.")
