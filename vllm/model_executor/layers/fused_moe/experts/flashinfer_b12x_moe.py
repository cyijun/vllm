# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Dynamic,
    kMxfp4Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_convert_sf_to_mma_layout,
    has_flashinfer_b12x_moe,
)


def _sanitize_b12x_topk(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map vLLM's padding sentinel to a safe, zero-weight B12X route."""
    is_padding = topk_ids < 0
    safe_ids = topk_ids.masked_fill(is_padding, 0).to(torch.int32)
    safe_weights = topk_weights.masked_fill(is_padding, 0.0)
    return safe_ids, safe_weights


class FlashInferB12xExperts(mk.FusedMoEExpertsModular):
    """FlashInfer CuteDSL fused MoE expert for SM12x (SM120/SM121,
    RTX Pro 6000 / DGX Spark).

    Uses ``b12x_fused_moe`` from FlashInfer PR #3080 which fuses token
    dispatch, two GEMMs, SwiGLU activation, and topk-weight reduction into a
    single kernel call.  Input quantization (BF16→FP4) is performed inside the
    kernel so BF16 hidden states are passed directly.

    NVFP4 weight scale factors are converted to the MMA layout produced by
    ``convert_sf_to_mma_layout`` once during ``process_weights_after_loading``.
    Native MXFP4 checkpoints instead use B12X's W4A16 path so the draft model
    keeps BF16 activations and speculative acceptance is not degraded by an
    additional activation quantization step.

    NVFP4 W4A4 and native MXFP4 W4A16 quantization are supported.
    """

    _ACTIVATION_MAP: dict[MoEActivation, str] = {
        MoEActivation.SILU: "silu",
        MoEActivation.RELU2_NO_MUL: "relu2",
    }

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.quant_dtype in ("nvfp4", "mxfp4"), (
            "FlashInferB12xExperts only supports nvfp4 or mxfp4 quantization."
        )
        self.checkpoint_quant_mode = quant_config.quant_dtype
        self.quant_mode = "w4a16" if self.checkpoint_quant_mode == "mxfp4" else "nvfp4"
        self.source_format = (
            "fp4_e8m0_k32" if self.checkpoint_quant_mode == "mxfp4" else "modelopt"
        )
        self.out_dtype = moe_config.in_dtype
        self.num_local_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        # FC2 input scale tensor bound in process_weights_after_loading: the
        # calibrated (now-zeroed) a2_gscale for static-quant checkpoints, or
        # a synthesized uniform-1.0 tensor for W4A16 checkpoints that lack
        # one. Holding it on the instance keeps apply() alloc-free.
        self._fc2_input_scale: torch.Tensor | None = None
        self._w1_alpha: torch.Tensor | None = None
        self._w2_alpha: torch.Tensor | None = None

        # Shape params for B12xMoEWrapper construction.
        self.global_num_experts = moe_config.num_experts
        self.topk = moe_config.experts_per_token
        self.hidden_dim = moe_config.hidden_dim
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.max_num_tokens = moe_config.max_num_tokens
        self.local_expert_offset = self.ep_rank * self.num_local_experts
        self.swiglu_limit = moe_config.swiglu_limit

        activation = moe_config.activation
        if activation not in self._ACTIVATION_MAP:
            raise ValueError(
                f"FlashInferB12xExperts does not support "
                f"activation {activation!r}. "
                f"Supported: {list(self._ACTIVATION_MAP.keys())}"
            )
        self._activation_str = self._ACTIVATION_MAP[activation]

        self.w1_sf_mma: torch.Tensor | None = None
        self.w2_sf_mma: torch.Tensor | None = None
        self._prepared_w4a16: object | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.checkpoint_quant_mode == "nvfp4":
            # Absorb NVFP4's per-expert global scale into its block scales.
            layer.w13_weight_scale.data = (
                layer.w13_weight_scale.float() * layer.w13_weight_scale_2.view(-1, 1, 1)
            ).to(layer.w13_weight_scale.dtype)
            layer.w13_weight_scale_2.data.fill_(1.0)
            layer.w2_weight_scale.data = (
                layer.w2_weight_scale.float() * layer.w2_weight_scale_2.view(-1, 1, 1)
            ).to(layer.w2_weight_scale.dtype)
            layer.w2_weight_scale_2.data.fill_(1.0)

            # B12X dynamically quantizes the FC2 input, so calibrated static
            # activation scales must not be applied a second time.
            if self.a2_gscale is not None:
                self.a2_gscale.fill_(1.0)
                self._fc2_input_scale = self.a2_gscale
            else:
                self._fc2_input_scale = torch.ones(
                    self.num_local_experts,
                    device=layer.w13_weight.device,
                    dtype=torch.float32,
                )
            self._w1_alpha = self.g1_alphas
            self._w2_alpha = self.g2_alphas
        else:
            # OCP MXFP4 carries its complete scale in each UE8M0 block-scale
            # byte. B12X W4A16 consumes the linear K/32 scale grid directly.
            ones = torch.ones(
                self.num_local_experts,
                device=layer.w13_weight.device,
                dtype=torch.float32,
            )
            self._w1_alpha = ones
            self._w2_alpha = ones
            self._fc2_input_scale = None
            assert self.w1_scale is not None and self.w2_scale is not None
            self.w1_sf_mma = self.w1_scale
            self.w2_sf_mma = self.w2_scale
            # Native MXFP4 tensors already have exactly the storage required by
            # the W4A16 runtime layout. Repack them in place once during model
            # loading so every forward hits FlashInfer's prepared-weight cache
            # without retaining a second copy of the draft expert weights.
            from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
                _get_w4a16_packed_weights,
            )

            self._prepared_w4a16 = _get_w4a16_packed_weights(
                w1_weight=layer.w13_weight,
                w1_weight_sf=self.w1_sf_mma,
                w1_alpha=self._w1_alpha,
                w2_weight=layer.w2_weight,
                w2_weight_sf=self.w2_sf_mma,
                w2_alpha=self._w2_alpha,
                activation=self._activation_str,
                params_dtype=self.out_dtype,
                source_format=self.source_format,
                reuse_input_storage=True,
            )
            return

        # Precompute MMA-layout views of the weight scale factors once here
        # rather than recomputing on every forward pass.
        assert self.w1_scale is not None
        num_experts_w1, m1_padded, k1_sf_padded = self.w1_scale.shape
        m1 = m1_padded
        k1 = k1_sf_padded * 16
        self.w1_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w1_scale.reshape(num_experts_w1 * m1_padded, k1_sf_padded),
            m=m1,
            k=k1,
            num_groups=num_experts_w1,
            sf_vec_size=16,
        )

        assert self.w2_scale is not None
        num_experts_w2, m2_padded, k2_sf_padded = self.w2_scale.shape
        m2 = m2_padded
        k2 = k2_sf_padded * 16
        self.w2_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w2_scale.reshape(num_experts_w2 * m2_padded, k2_sf_padded),
            m=m2,
            k=k2,
            num_groups=num_experts_w2,
            sf_vec_size=16,
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(120)
            and has_flashinfer_b12x_moe()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # b12x performs in-kernel BF16->FP4 activation quant, so W4A16
        # NVFP4 checkpoints (activation_key=None, e.g. mixed-precision
        # compressed-tensors layouts) are runtime-compatible.
        return (weight_key, activation_key) in (
            (kNvfp4Static, kNvfp4Dynamic),
            (kNvfp4Static, None),
            (kMxfp4Static, kMxfp4Dynamic),
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.RELU2_NO_MUL)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # B12xMoEWrapper does not yet support expert parallelism: its local
        # expert count must equal the global expert count.
        return not moe_parallel_config.use_ep

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # b12x_fused_moe applies topk weights internally.
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # b12x_fused_moe manages its own internal workspace.
        workspace1 = (1,)
        workspace2 = (0,)
        output_shape = (M, K)
        return (workspace1, workspace2, output_shape)

    @property
    def expects_unquantized_inputs(self) -> bool:
        # B12xMoEWrapper expects BF16 hidden states and performs its own FP4
        # quantization internally.  Returning True prevents the modular kernel
        # from pre-quantizing activations.
        return True

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        assert self._w1_alpha is not None and self._w2_alpha is not None, (
            "process_weights_after_loading must initialize B12X expert alphas"
        )
        assert self.quant_mode != "nvfp4" or self._fc2_input_scale is not None, (
            "NVFP4 requires process_weights_after_loading to set FC2 input scale"
        )
        assert self.w1_sf_mma is not None and self.w2_sf_mma is not None, (
            "process_weights_after_loading must run before FlashInferB12xExperts.apply"
        )

        # vLLM uses -1 expert IDs for cudagraph/profile padding.  Most MoE
        # backends consume that sentinel directly, while B12X requires every
        # expert ID to be in range.  Route padding through expert 0 with a zero
        # scale so the output remains zero without an out-of-bounds access.
        token_selected_experts, token_final_scales = _sanitize_b12x_topk(
            topk_ids, topk_weights
        )

        # The functional API reuses FlashInfer's process-wide workspace cache.
        # vLLM warms the largest capture shape before CUDA graph capture, and
        # supplying ``output`` keeps the captured call allocation-free.  Avoid
        # a process-wide CUDA Event here: every layer executes in order on the
        # current stream, while recording the same event repeatedly inside
        # multiple captured graphs creates invalid cross-graph dependencies.
        from flashinfer.fused_moe import b12x_fused_moe

        b12x_fused_moe(
            x=hidden_states,
            output=output,
            w1_weight=w1,
            w1_weight_sf=self.w1_sf_mma,
            w1_alpha=self._w1_alpha,
            fc2_input_scale=self._fc2_input_scale,
            w2_weight=w2,
            w2_weight_sf=self.w2_sf_mma,
            w2_alpha=self._w2_alpha,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            num_experts=self.global_num_experts,
            top_k=self.topk,
            num_local_experts=self.num_local_experts,
            activation=self._activation_str,
            swiglu_limit=self.swiglu_limit,
            quant_mode=self.quant_mode,
            source_format=self.source_format,
        )
