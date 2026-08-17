# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm import b12x_block
from vllm.model_executor.kernels.linear.scaled_mm.b12x_block import (
    B12xFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

pytestmark = pytest.mark.cpu_test


def make_kernel() -> B12xFp8BlockScaledMMKernel:
    kernel = object.__new__(B12xFp8BlockScaledMMKernel)
    kernel.weight_group_shape = GroupShape(128, 128)
    kernel.config = SimpleNamespace(out_dtype=torch.bfloat16)
    return kernel


def test_m12_uses_triton_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = torch.tensor([12])
    calls = []

    def fake_triton(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(b12x_block, "_triton_block_fp8", fake_triton)
    kernel = make_kernel()
    operands = (
        torch.empty(12, 128),
        torch.empty(128, 128),
        torch.empty(12, 1),
        torch.empty(1, 1),
    )

    assert kernel.apply_block_scaled_mm(*operands) is expected
    assert len(calls) == 1


def test_other_m_uses_b12x(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = torch.tensor([6])
    calls = []

    def fake_mm_block_fp8(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(
        b12x_block,
        "get_b12x_blockscaled",
        lambda: SimpleNamespace(mm_block_fp8=fake_mm_block_fp8),
    )
    kernel = make_kernel()
    operands = (
        torch.empty(6, 128),
        torch.empty(128, 128),
        torch.empty(6, 1),
        torch.empty(1, 1),
    )

    assert kernel.apply_block_scaled_mm(*operands) is expected
    assert len(calls) == 1
