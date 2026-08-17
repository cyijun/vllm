# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.b12x import get_b12x_blockscaled

from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


def _triton_block_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    weight_group_shape: GroupShape,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.ops.vllm.w8a8_triton_block_scaled_mm_func(
        A,
        B,
        As,
        Bs,
        list(weight_group_shape),
        out_dtype,
    )


class B12xFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    """K128 block-FP8 linear through the native B12X SM12x GEMM."""

    # Capturing the complete DeepSeek V4 decode graph at M=12 can leave the
    # B12X block-FP8 op with an illegal-access error on SM121.  The same GEMM
    # shapes are stable through vLLM's Triton implementation.  Keep the B12X
    # fast path for every other M, including the common C1/C4/C6 shapes.
    _TRITON_FALLBACK_M = frozenset((12,))

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "B12X FP8 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "B12X FP8 kernels require a Blackwell 12x device"
        blockscaled = get_b12x_blockscaled()
        if blockscaled is None:
            return False, "Install the B12X backend with `pip install vllm[b12x]`"
        if not blockscaled.is_supported():
            return False, "B12X regular block-FP8 GEMM is not supported"
        return True, None

    @classmethod
    def can_implement(
        cls,
        config: FP8ScaledMMLinearLayerConfig,
    ) -> tuple[bool, str | None]:
        supported, reason = super().can_implement(config)
        if not supported:
            return supported, reason
        if config.input_dtype not in (torch.bfloat16, torch.float16):
            return False, "Supports only bf16/fp16 input dtype"
        if config.input_dtype != config.out_dtype:
            return False, "Input and output dtype must match"
        if config.activation_quant_key.scale.group_shape != GroupShape(1, 128):
            return False, "Supports only (1, 128) activation quantization"
        if config.weight_quant_key.scale.group_shape != GroupShape(128, 128):
            return False, "Supports only 128x128 block-scaled FP8 weights"
        out_features, in_features = config.weight_shape
        if in_features <= 0 or in_features % 128 != 0:
            return False, "Input features must be a positive multiple of 128"
        if out_features <= 0 or out_features % 128 != 0:
            return False, "Output features must be a positive multiple of 128"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        params = self._get_layer_params(layer)
        if params.weight_scale_inv is not None:
            weight_scale = params.weight_scale_inv
            scale_attr = params.WEIGHT_SCALE_INV
        else:
            weight_scale = params.weight_scale
            scale_attr = params.WEIGHT_SCALE
        if weight_scale is not None and weight_scale.dtype in (
            torch.float8_e8m0fnu,
            torch.uint8,
        ):
            replace_parameter(
                layer,
                scale_attr,
                _upcast_e8m0_to_fp32(weight_scale).contiguous(),
            )

        # DeepSeek V4 consumes wo_a as a grouped BMM in its fused projection.
        # DeepGEMM does not support SM121, so keep a dequantized BF16 copy for
        # the portable grouped-BMM fallback used by that projection.
        if getattr(layer, "is_bmm", False):
            params = self._get_layer_params(layer)
            weight_scale = (
                params.weight_scale_inv
                if params.weight_scale_inv is not None
                else params.weight_scale
            )
            assert weight_scale is not None
            block_n, block_k = layer.weight_block_size
            expanded_scale = weight_scale.repeat_interleave(
                block_n, dim=0
            ).repeat_interleave(block_k, dim=1)
            weight = (
                params.weight.float() * expanded_scale[: params.weight.shape[0], :]
            ).to(torch.bfloat16)
            num_groups = getattr(layer, "bmm_batch_size", 0)
            replace_parameter(
                layer,
                params.WEIGHT,
                weight.view(num_groups, weight.shape[0] // num_groups, weight.shape[1]),
            )
            layer.b12x_bf16_bmm = True

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        if A.shape[0] in self._TRITON_FALLBACK_M:
            return _triton_block_fp8(
                A,
                B,
                As,
                Bs,
                self.weight_group_shape,
                self.config.out_dtype,
            )
        blockscaled = get_b12x_blockscaled()
        assert blockscaled is not None
        return blockscaled.mm_block_fp8(
            A,
            As,
            B,
            Bs,
            out_dtype=self.config.out_dtype,
        )
