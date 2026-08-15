# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
    _fp32x2_to_fp4x2,
)
from vllm.triton_utils import tl, triton

NVFP4_MLA_HEAD_DIM = 512
NVFP4_MLA_BLOCK_SIZE = 16
NVFP4_MLA_DATA_BYTES = NVFP4_MLA_HEAD_DIM // 2
NVFP4_MLA_SCALE_BYTES = NVFP4_MLA_HEAD_DIM // NVFP4_MLA_BLOCK_SIZE
NVFP4_MLA_TOKEN_BYTES = NVFP4_MLA_DATA_BYTES + NVFP4_MLA_SCALE_BYTES


@triton.jit
def _store_nvfp4_mla_cache_kernel(
    kv_ptr,
    kv_stride,
    slot_mapping_ptr,
    cache_ptr,
    cache_block_stride,
    cache_block_size,
    HEAD_DIM: tl.constexpr,
    DATA_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot < 0:
        return

    offsets = tl.arange(0, HEAD_DIM)
    values = tl.load(kv_ptr + token_idx * kv_stride + offsets).to(tl.float32)
    values = values.to(tl.bfloat16).to(tl.float32)
    paired = tl.reshape(values, (NUM_BLOCKS, HALF_BLOCK, 2))
    even, odd = tl.split(paired)

    amax = tl.maximum(
        tl.max(tl.abs(even), axis=1),
        tl.max(tl.abs(odd), axis=1),
    )
    scale = tl.maximum(amax * (1.0 / 6.0), 2**-9)
    scale_fp8 = scale.to(tl.float8e4nv)
    inv_scale = 1.0 / scale_fp8.to(tl.float32)
    inv_scale = tl.reshape(inv_scale, (NUM_BLOCKS, 1))
    packed = _fp32x2_to_fp4x2(even * inv_scale, odd * inv_scale)

    block_idx = slot // cache_block_size
    block_offset = slot % cache_block_size
    block_base = cache_ptr + block_idx.to(tl.int64) * cache_block_stride
    data_ptr = block_base + block_offset * DATA_BYTES
    scale_ptr = block_base + cache_block_size * DATA_BYTES + block_offset * SCALE_BYTES
    tl.store(
        data_ptr + tl.arange(0, DATA_BYTES),
        tl.reshape(packed, (DATA_BYTES,)),
    )
    tl.store(
        scale_ptr + tl.arange(0, SCALE_BYTES),
        scale_fp8.to(tl.uint8, bitcast=True),
    )


def store_nvfp4_mla_cache(
    kv: torch.Tensor,
    slot_mapping: torch.Tensor,
    cache: torch.Tensor,
) -> None:
    if cache.ndim != 3 or cache.shape[-1] != NVFP4_MLA_TOKEN_BYTES:
        raise ValueError(
            "NVFP4 MLA cache must have shape [blocks, block_size, 288], "
            f"got {tuple(cache.shape)}"
        )
    _store_nvfp4_mla_cache_kernel[(slot_mapping.numel(),)](
        kv,
        kv.stride(0),
        slot_mapping,
        cache,
        cache.stride(0),
        cache.shape[1],
        HEAD_DIM=NVFP4_MLA_HEAD_DIM,
        DATA_BYTES=NVFP4_MLA_DATA_BYTES,
        SCALE_BYTES=NVFP4_MLA_SCALE_BYTES,
        NUM_BLOCKS=NVFP4_MLA_SCALE_BYTES,
        HALF_BLOCK=NVFP4_MLA_BLOCK_SIZE // 2,
        num_warps=4,
    )


def prepare_q_and_store_nvfp4_mla_cache(
    q: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    q = F.rms_norm(q, (q.shape[-1],), eps=eps)
    q, rotated_kv = rotary_emb.forward_native(positions, q, kv.unsqueeze(1))
    assert rotated_kv is not None
    rotated_kv = rotated_kv.squeeze(1)
    store_nvfp4_mla_cache(rotated_kv, slot_mapping, cache)
    return q


def _gather_dequant_nvfp4_rows(
    cache: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    from flashinfer import nvfp4_kv_dequantize

    indices = indices.reshape(indices.shape[0], -1)
    block_size = cache.shape[1]
    safe_indices = indices.clamp(0, cache.shape[0] * block_size - 1).to(torch.int64)
    block_indices = torch.div(safe_indices, block_size, rounding_mode="floor")
    block_offsets = safe_indices % block_size

    data_offsets = block_offsets * NVFP4_MLA_DATA_BYTES
    data_offsets = data_offsets[..., None] + torch.arange(
        NVFP4_MLA_DATA_BYTES, device=cache.device
    )
    scale_offsets = (
        block_size * NVFP4_MLA_DATA_BYTES + block_offsets * NVFP4_MLA_SCALE_BYTES
    )
    scale_offsets = scale_offsets[..., None] + torch.arange(
        NVFP4_MLA_SCALE_BYTES, device=cache.device
    )

    packed = cache[
        block_indices[..., None],
        torch.div(data_offsets, NVFP4_MLA_TOKEN_BYTES, rounding_mode="floor"),
        data_offsets % NVFP4_MLA_TOKEN_BYTES,
    ].reshape(-1, NVFP4_MLA_DATA_BYTES)
    scales = cache[
        block_indices[..., None],
        torch.div(scale_offsets, NVFP4_MLA_TOKEN_BYTES, rounding_mode="floor"),
        scale_offsets % NVFP4_MLA_TOKEN_BYTES,
    ].reshape(-1, NVFP4_MLA_SCALE_BYTES)
    global_scale = torch.ones(1, dtype=torch.float32, device=cache.device)
    dequant = nvfp4_kv_dequantize(packed, scales, global_scale)
    return dequant.view(indices.shape[0], indices.shape[1], NVFP4_MLA_HEAD_DIM)


def nvfp4_mla_sparse_attention(
    query: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float,
    sinks: torch.Tensor | None = None,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_lens: torch.Tensor | None = None,
) -> None:
    swa_indices = swa_indices.reshape(query.shape[0], -1)
    values = [_gather_dequant_nvfp4_rows(swa_cache, swa_indices)]
    valid = [
        (swa_indices >= 0)
        & (
            torch.arange(swa_indices.shape[1], device=query.device)[None, :]
            < swa_lens[:, None]
        )
    ]

    if extra_cache is not None:
        if extra_indices is None or extra_lens is None:
            raise ValueError("extra NVFP4 MLA cache requires indices and lengths")
        extra_indices = extra_indices.reshape(query.shape[0], -1)
        values.append(_gather_dequant_nvfp4_rows(extra_cache, extra_indices))
        valid.append(
            (extra_indices >= 0)
            & (
                torch.arange(extra_indices.shape[1], device=query.device)[None, :]
                < extra_lens[:, None]
            )
        )

    kv = torch.cat(values, dim=1).float()
    valid_mask = torch.cat(valid, dim=1)
    scores = torch.einsum("thd,tkd->thk", query.float(), kv) * sm_scale
    scores = scores.masked_fill(~valid_mask[:, None, :], float("-inf"))

    row_max = scores.amax(dim=-1, keepdim=True)
    if sinks is not None:
        sink_logits = sinks.float().view(1, -1, 1)
        row_max = torch.maximum(row_max, sink_logits)
    else:
        sink_logits = None
        row_max = torch.where(torch.isfinite(row_max), row_max, 0.0)

    weights = torch.where(valid_mask[:, None, :], torch.exp(scores - row_max), 0.0)
    denominator = weights.sum(dim=-1, keepdim=True)
    if sink_logits is not None:
        denominator = denominator + torch.exp(sink_logits - row_max)
    weights = weights / denominator.clamp_min(torch.finfo(torch.float32).tiny)
    result = torch.einsum("thk,tkd->thd", weights, kv)
    output.copy_(result.to(output.dtype))
