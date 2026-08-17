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
# Bound the PyTorch fallback's temporary [query, sparse_width, head_dim]
# tensors explicitly: a single 8K-token prefill can otherwise request tens of
# GiB on unified-memory systems before the allocator reports a useful error.
NVFP4_MLA_FALLBACK_MAX_QUERY_ROWS = 16
_NVFP4_MLA_USE_FUSED_KERNEL = True


@triton.jit
def _dequant_nvfp4_mla_rows(
    cache_ptr,
    row_indices,
    row_valid,
    dim_offsets,
    cache_block_stride,
    cache_block_size,
    DATA_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
):
    """Load a candidate tile from the block-planar vLLM NVFP4 cache."""
    safe_indices = tl.maximum(row_indices, 0)
    block_indices = safe_indices // cache_block_size
    block_offsets = safe_indices % cache_block_size

    packed_offsets = dim_offsets // 2
    packed = tl.load(
        cache_ptr
        + block_indices[:, None] * cache_block_stride
        + block_offsets[:, None] * DATA_BYTES
        + packed_offsets[None, :],
        mask=row_valid[:, None],
        other=0,
    ).to(tl.uint32)
    shift = (dim_offsets & 1) * 4
    code = (packed >> shift[None, :]) & 0xF
    magnitude = code & 0x7
    exponent = magnitude >> 1
    mantissa = magnitude & 1
    normal = (1.0 + mantissa.to(tl.float32) * 0.5) * tl.exp2(
        exponent.to(tl.float32) - 1.0
    )
    fp4 = tl.where(exponent == 0, mantissa.to(tl.float32) * 0.5, normal)
    fp4 = tl.where((code & 0x8) != 0, -fp4, fp4)

    scale_offsets = dim_offsets // 16
    scale_raw = tl.load(
        cache_ptr
        + block_indices[:, None] * cache_block_stride
        + cache_block_size * DATA_BYTES
        + block_offsets[:, None] * SCALE_BYTES
        + scale_offsets[None, :],
        mask=row_valid[:, None],
        other=0,
    ).to(tl.uint8)
    scale = scale_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    return fp4 * scale


@triton.jit
def _nvfp4_mla_sparse_attention_kernel(
    query_ptr,
    swa_cache_ptr,
    swa_indices_ptr,
    swa_lens_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_lens_ptr,
    sinks_ptr,
    output_ptr,
    query_stride_t,
    query_stride_h,
    swa_cache_block_stride,
    swa_cache_block_size,
    swa_indices_stride_t,
    extra_cache_block_stride,
    extra_cache_block_size,
    extra_indices_stride_t,
    output_stride_t,
    output_stride_h,
    num_heads: tl.constexpr,
    sm_scale: tl.constexpr,
    SWA_WIDTH: tl.constexpr,
    EXTRA_WIDTH: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, HEAD_DIM)
    head_valid = head_offsets < num_heads

    q = tl.load(
        query_ptr
        + token_idx * query_stride_t
        + head_offsets[:, None] * query_stride_h
        + dim_offsets[None, :],
        mask=head_valid[:, None],
        other=0.0,
    )
    acc = tl.zeros((BLOCK_H, HEAD_DIM), tl.float32)
    row_max = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_H,), tl.float32)
    log2_scale = sm_scale * 1.4426950408889634

    swa_len = tl.load(swa_lens_ptr + token_idx).to(tl.int32)
    for candidate_start in tl.range(0, SWA_WIDTH, BLOCK_N):
        candidate_offsets = candidate_start + tl.arange(0, BLOCK_N)
        candidate_valid = candidate_offsets < tl.minimum(swa_len, SWA_WIDTH)
        candidate_indices = tl.load(
            swa_indices_ptr + token_idx * swa_indices_stride_t + candidate_offsets,
            mask=candidate_offsets < SWA_WIDTH,
            other=-1,
        ).to(tl.int32)
        candidate_valid &= candidate_indices >= 0
        kv = _dequant_nvfp4_mla_rows(
            swa_cache_ptr,
            candidate_indices,
            candidate_valid,
            dim_offsets,
            swa_cache_block_stride,
            swa_cache_block_size,
            256,
            32,
        ).to(tl.bfloat16)
        scores = tl.dot(q, tl.trans(kv), out_dtype=tl.float32) * log2_scale
        scores = tl.where(candidate_valid[None, :], scores, -float("inf"))
        tile_max = tl.max(scores, axis=1)
        has_candidates = tl.sum(candidate_valid.to(tl.int32), axis=0) > 0
        next_max = tl.where(has_candidates, tl.maximum(row_max, tile_max), row_max)
        alpha = tl.where(has_candidates, tl.exp2(row_max - next_max), 1.0)
        probabilities = tl.exp2(scores - next_max[:, None])
        probabilities = tl.where(candidate_valid[None, :], probabilities, 0.0)
        acc = acc * alpha[:, None] + tl.dot(
            probabilities.to(tl.bfloat16), kv, out_dtype=tl.float32
        )
        row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
        row_max = next_max

    if HAS_EXTRA:
        extra_len = tl.load(extra_lens_ptr + token_idx).to(tl.int32)
        for candidate_start in tl.range(0, EXTRA_WIDTH, BLOCK_N):
            candidate_offsets = candidate_start + tl.arange(0, BLOCK_N)
            candidate_valid = candidate_offsets < tl.minimum(extra_len, EXTRA_WIDTH)
            candidate_indices = tl.load(
                extra_indices_ptr
                + token_idx * extra_indices_stride_t
                + candidate_offsets,
                mask=candidate_offsets < EXTRA_WIDTH,
                other=-1,
            ).to(tl.int32)
            candidate_valid &= candidate_indices >= 0
            kv = _dequant_nvfp4_mla_rows(
                extra_cache_ptr,
                candidate_indices,
                candidate_valid,
                dim_offsets,
                extra_cache_block_stride,
                extra_cache_block_size,
                256,
                32,
            ).to(tl.bfloat16)
            scores = tl.dot(q, tl.trans(kv), out_dtype=tl.float32) * log2_scale
            scores = tl.where(candidate_valid[None, :], scores, -float("inf"))
            tile_max = tl.max(scores, axis=1)
            has_candidates = tl.sum(candidate_valid.to(tl.int32), axis=0) > 0
            next_max = tl.where(has_candidates, tl.maximum(row_max, tile_max), row_max)
            alpha = tl.where(has_candidates, tl.exp2(row_max - next_max), 1.0)
            probabilities = tl.exp2(scores - next_max[:, None])
            probabilities = tl.where(candidate_valid[None, :], probabilities, 0.0)
            acc = acc * alpha[:, None] + tl.dot(
                probabilities.to(tl.bfloat16), kv, out_dtype=tl.float32
            )
            row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
            row_max = next_max

    if HAS_SINKS:
        sink = tl.load(sinks_ptr + head_offsets, mask=head_valid, other=-float("inf"))
        sink *= 1.4426950408889634
        sink_valid = sink != -float("inf")
        next_max = tl.where(sink_valid, tl.maximum(row_max, sink), row_max)
        alpha = tl.where(sink_valid, tl.exp2(row_max - next_max), 1.0)
        acc *= alpha[:, None]
        sink_weight = tl.where(sink_valid, tl.exp2(sink - next_max), 0.0)
        row_sum = row_sum * alpha + sink_weight

    result = tl.where(row_sum[:, None] > 0, acc / row_sum[:, None], 0.0)
    tl.store(
        output_ptr
        + token_idx * output_stride_t
        + head_offsets[:, None] * output_stride_h
        + dim_offsets[None, :],
        result,
        mask=head_valid[:, None],
    )


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
    if _NVFP4_MLA_USE_FUSED_KERNEL and query.is_cuda:
        if query.shape[0] == 0:
            return
        if query.shape[-1] != NVFP4_MLA_HEAD_DIM:
            raise ValueError(
                f"NVFP4 MLA query head dim must be {NVFP4_MLA_HEAD_DIM}, "
                f"got {query.shape[-1]}"
            )
        if query.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            raise TypeError("fused NVFP4 MLA requires bf16 query and output")
        swa_indices = swa_indices.reshape(query.shape[0], -1).contiguous()
        swa_lens = swa_lens.contiguous()
        if extra_cache is not None:
            if extra_indices is None or extra_lens is None:
                raise ValueError("extra NVFP4 MLA cache requires indices and lengths")
            extra_indices = extra_indices.reshape(query.shape[0], -1).contiguous()
            extra_cache_arg = extra_cache
            extra_indices_arg = extra_indices
            extra_lens_arg = extra_lens.contiguous()
            extra_width = extra_indices.shape[1]
        else:
            extra_cache_arg = swa_cache
            extra_indices_arg = swa_indices
            extra_lens_arg = swa_lens
            extra_width = 0
        sinks_arg = sinks if sinks is not None else swa_lens
        block_h = 16
        num_warps = 8 if query.shape[0] <= 6 else 4
        grid = (query.shape[0], triton.cdiv(query.shape[1], block_h))
        _nvfp4_mla_sparse_attention_kernel[grid](
            query,
            swa_cache,
            swa_indices,
            swa_lens,
            extra_cache_arg,
            extra_indices_arg,
            extra_lens_arg,
            sinks_arg,
            output,
            query.stride(0),
            query.stride(1),
            swa_cache.stride(0),
            swa_cache.shape[1],
            swa_indices.stride(0),
            extra_cache_arg.stride(0),
            extra_cache_arg.shape[1],
            extra_indices_arg.stride(0),
            output.stride(0),
            output.stride(1),
            num_heads=query.shape[1],
            sm_scale=sm_scale,
            SWA_WIDTH=swa_indices.shape[1],
            EXTRA_WIDTH=extra_width,
            HAS_EXTRA=extra_cache is not None,
            HAS_SINKS=sinks is not None,
            HEAD_DIM=NVFP4_MLA_HEAD_DIM,
            BLOCK_H=block_h,
            BLOCK_N=16,
            num_warps=num_warps,
            num_stages=1,
        )
        return

    if query.shape[0] > NVFP4_MLA_FALLBACK_MAX_QUERY_ROWS:
        for start in range(0, query.shape[0], NVFP4_MLA_FALLBACK_MAX_QUERY_ROWS):
            end = min(start + NVFP4_MLA_FALLBACK_MAX_QUERY_ROWS, query.shape[0])
            nvfp4_mla_sparse_attention(
                query=query[start:end],
                swa_cache=swa_cache,
                swa_indices=swa_indices[start:end],
                swa_lens=swa_lens[start:end],
                output=output[start:end],
                sm_scale=sm_scale,
                sinks=sinks,
                extra_cache=extra_cache,
                extra_indices=(
                    extra_indices[start:end] if extra_indices is not None else None
                ),
                extra_lens=extra_lens[start:end] if extra_lens is not None else None,
            )
        return

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
