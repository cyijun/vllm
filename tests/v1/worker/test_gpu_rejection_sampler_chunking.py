# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from typing import get_args

import numpy as np
import pytest
import torch

from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    _FP32_BYTES,
    MAX_CHUNK_BYTES,
    RejectionSampler,
    _iter_request_chunks,
)


def test_iter_request_chunks_preserves_request_boundaries():
    cu_num_logits = np.array([0, 3, 4, 11, 13], dtype=np.int32)

    assert list(_iter_request_chunks(cu_num_logits, max_chunk_logits=5)) == [
        (0, 2),
        (2, 3),
        (3, 4),
    ]


@pytest.mark.parametrize(
    ("logprobs_mode", "use_processed_logits"),
    [("raw_logprobs", False), ("processed_logprobs", True)],
)
def test_single_chunk_avoids_chunk_orchestration(
    monkeypatch: pytest.MonkeyPatch,
    logprobs_mode: str,
    use_processed_logits: bool,
):
    vocab_size = 17
    logits = torch.arange(102, dtype=torch.float32).view(6, vocab_size)
    processed_logits = logits + 1
    sampled = torch.arange(8, dtype=torch.int64).view(2, 4)
    num_sampled = torch.tensor([3, 4], dtype=torch.int32)
    cu_num_logits_np = np.array([0, 3, 6], dtype=np.int32)
    cu_num_logits = torch.from_numpy(cu_num_logits_np)
    input_batch = SimpleNamespace(
        input_ids=torch.arange(6),
        logits_indices=torch.arange(6),
        positions=torch.arange(6),
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
        expanded_idx_mapping=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32),
        expanded_local_pos=torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32),
        cu_num_logits_np=cu_num_logits_np,
        cu_num_logits=cu_num_logits,
        seq_lens=torch.tensor([3, 3], dtype=torch.int32),
    )
    rejection_sampler = object.__new__(RejectionSampler)
    rejection_sampler.sampler = SimpleNamespace(
        compute_nans=False,
        logprobs_mode=logprobs_mode,
        sampling_states=SimpleNamespace(max_num_logprobs=lambda _: -1),
        req_states=SimpleNamespace(
            prefill_len=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int32))
        ),
    )

    def fake_verify(
        self,
        verify_logits,
        _draft_logits,
        _draft_sampled,
        _pos,
        verify_cu_num_logits,
        *_mappings,
    ):
        assert verify_logits is logits
        assert verify_cu_num_logits is cu_num_logits
        return processed_logits, sampled, num_sampled

    def fake_get_logprobs_tensors(
        self,
        verify_sampled,
        verify_num_sampled,
        score_logits,
        score_cu_num_logits,
        score_cu_num_logits_np,
        _max_num_logprobs,
    ):
        assert verify_sampled is sampled
        assert verify_num_sampled is num_sampled
        assert score_logits is (processed_logits if use_processed_logits else logits)
        assert score_cu_num_logits is cu_num_logits
        assert score_cu_num_logits_np is cu_num_logits_np
        return None

    def fail_chunked_verify(*_args, **_kwargs):
        pytest.fail("single-chunk verification used chunk orchestration")

    rejection_sampler._verify = MethodType(fake_verify, rejection_sampler)
    rejection_sampler._get_logprobs_tensors = MethodType(
        fake_get_logprobs_tensors, rejection_sampler
    )
    rejection_sampler._verify_in_chunks = MethodType(
        fail_chunked_verify, rejection_sampler
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler.get_num_sampled_and_rejected",
        lambda values, *_args: (values, torch.zeros_like(values)),
    )

    assert logits.shape[0] <= MAX_CHUNK_BYTES // (vocab_size * _FP32_BYTES)
    output = rejection_sampler(logits, input_batch)

    assert output.sampled_token_ids is sampled
    assert output.num_sampled is num_sampled


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("logprobs_mode", get_args(LogprobsMode))
def test_chunked_scores_match_full_batch(logprobs_mode: str):
    device = torch.device("cuda")
    cu_num_logits_np = np.array([0, 3, 4, 8, 10], dtype=np.int32)
    num_logits_per_req = np.diff(cu_num_logits_np)
    idx_mapping_np = np.array([7, 2, 9, 1], dtype=np.int32)
    input_batch = SimpleNamespace(
        num_reqs=4,
        cu_num_logits_np=cu_num_logits_np,
        cu_num_logits=torch.from_numpy(cu_num_logits_np).to(device),
        idx_mapping_np=idx_mapping_np,
        idx_mapping=torch.from_numpy(idx_mapping_np).to(device),
        expanded_idx_mapping=torch.from_numpy(
            np.repeat(idx_mapping_np, num_logits_per_req)
        ).to(device),
        expanded_local_pos=torch.from_numpy(
            np.concatenate(
                [np.arange(count, dtype=np.int32) for count in num_logits_per_req]
            )
        ).to(device),
    )
    rejection_sampler = object.__new__(RejectionSampler)
    rejection_sampler.sampler = SimpleNamespace(logprobs_mode=logprobs_mode)
    rejection_sampler.num_speculative_steps = 3

    def fake_verify(
        self,
        logits,
        _draft_logits,
        _draft_sampled,
        _pos,
        cu_num_logits,
        idx_mapping,
        *_mappings,
    ):
        num_sampled = torch.diff(cu_num_logits).to(torch.int32)
        sampled = (
            idx_mapping.to(torch.int64).unsqueeze(1) + torch.arange(4, device=device)
        ) % logits.shape[1]
        return logits.float() + 1, sampled, num_sampled

    rejection_sampler._verify = MethodType(fake_verify, rejection_sampler)
    logits = torch.arange(170, dtype=torch.float32, device=device).view(10, 17)

    sampled, num_sampled, chunked_logprobs = rejection_sampler._verify_in_chunks(
        logits,
        input_batch,
        draft_logits=None,
        draft_sampled=torch.arange(10, device=device),
        pos=torch.arange(10, device=device),
        max_chunk_logits=5,
        max_num_logprobs=2,
    )
    score_logits = logits + 1 if logprobs_mode in PROCESSED_LOGPROBS_MODES else logits
    full_logprobs = rejection_sampler._get_logprobs_tensors(
        sampled,
        num_sampled,
        score_logits,
        input_batch.cu_num_logits,
        input_batch.cu_num_logits_np,
        max_num_logprobs=2,
    )

    assert sampled[:, 0].tolist() == idx_mapping_np.tolist()
    assert num_sampled.tolist() == num_logits_per_req.tolist()
    assert chunked_logprobs is not None
    assert full_logprobs is not None
    assert torch.equal(
        chunked_logprobs.logprob_token_ids,
        full_logprobs.logprob_token_ids,
    )
    assert torch.equal(chunked_logprobs.logprobs, full_logprobs.logprobs)
    assert torch.equal(
        chunked_logprobs.selected_token_ranks,
        full_logprobs.selected_token_ranks,
    )
    assert (
        chunked_logprobs.cu_num_generated_tokens
        == full_logprobs.cu_num_generated_tokens
    )
