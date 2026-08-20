# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import importlib.metadata
from dataclasses import replace
from typing import Any

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
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

logger = init_logger(__name__)


_B12X_ROUTE_PACK_WARMED: set[tuple[str, int, int, int, int]] = set()


def _apply_b12x_w4a16_ultrawide_compile_compat() -> None:
    """Keep b12x 1.2.4's valid M=1 ultra-wide tile self-consistent.

    Its first compile intentionally validates the E8M0 FC2 (K=32, N=512)
    tile by footprint because the generic selector has a K>=64 floor.  The
    registered launch compiles the same tile a second time as an explicit
    pin, where 1.2.4 accidentally sends it through that generic validator.
    Re-run only this exact pin through the original automatic selection and
    accept it only when the selected tile is byte-for-byte identical.
    """
    if importlib.metadata.version("b12x") != "1.2.4":
        return

    from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    original_attr = "_vllm_original_compile_w4a16_fused_moe"
    if hasattr(w4a16_kernel, original_attr):
        return
    original_compile = w4a16_kernel.compile_w4a16_fused_moe
    setattr(w4a16_kernel, original_attr, original_compile)

    @functools.wraps(original_compile)
    def compile_compat(*args: Any, **kwargs: Any) -> Any:
        forced = kwargs.get("force_tile_config")
        forced_tuple = None if forced is None else tuple(int(value) for value in forced)
        is_ultrawide_decode = (
            forced_tuple is not None
            and forced_tuple[2:] == (32, 512)
            and int(kwargs.get("size_m", -1)) == 1
            and bool(kwargs.get("tc_decode_fused_sum", False))
            and kwargs.get("weight_layout") == "packed"
            and kwargs.get("scale_format") == "e8m0_k32"
        )
        if is_ultrawide_decode:
            automatic_kwargs = dict(kwargs)
            automatic_kwargs["force_tile_config"] = None
            compiled = original_compile(*args, **automatic_kwargs)
            selected = (
                int(compiled.fc1_tile_k),
                int(compiled.fc1_tile_n),
                int(compiled.fc2_tile_k),
                int(compiled.fc2_tile_n),
            )
            if selected == forced_tuple:
                return compiled
        return original_compile(*args, **kwargs)

    w4a16_kernel.compile_w4a16_fused_moe = compile_compat
    logger.info("Enabled b12x 1.2.4 M=1 ultra-wide W4A16 compile compatibility")


@functools.lru_cache
def _standalone_b12x_weight_plan(
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation: str,
    params_dtype: torch.dtype,
):
    """Return the tensor-free standalone b12x MXFP4 weight plan."""
    from b12x.moe import fused_moe

    return fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="fp4_e8m0_k32",
        activation=activation,
        params_dtype=params_dtype,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        # vLLM's B12X conversion has already changed checkpoint [gate, up]
        # rows to the B12X logical [up, gate] order.
        w13_layout="w13",
    )


@functools.lru_cache
def _standalone_b12x_execution_plan(
    *,
    max_tokens: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    device: str,
    activation: str,
    params_dtype: torch.dtype,
    swiglu_limit: float | None,
):
    """Plan one token-capacity bucket shared by all equivalent MoE layers."""
    from b12x.moe import fused_moe

    weight_plan = _standalone_b12x_weight_plan(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        params_dtype=params_dtype,
    )
    return fused_moe.plan(
        fused_moe.Caps(
            max_tokens=max(int(max_tokens), 1),
            num_topk=topk,
            device=device,
            weight_plan=weight_plan,
            core_token_counts=(max(int(max_tokens), 1),),
            # Routing ids/weights arrive preselected from vLLM; disable the
            # unrelated logits-routing arena. Packed-route buffers remain in
            # the core plan, and expert counts use vLLM's fixed scratch tail.
            route_num_experts=0,
            quant_mode="w4a16",
            apply_router_weight_on_input=False,
            swiglu_limit=swiglu_limit,
            frozen=True,
        )
    )


def _standalone_b12x_scratch_nbytes(plan: Any) -> int:
    specs = plan.scratch_specs()
    if len(specs) != 1:
        raise RuntimeError(
            f"expected one standalone b12x scratch buffer, got {len(specs)}"
        )
    spec = specs[0]
    if spec.dtype != torch.uint8:
        raise TypeError(
            f"expected standalone b12x scratch dtype uint8, got {spec.dtype}"
        )
    return int(spec.shape[0])


def _workspace_as_standalone_b12x_scratch(
    workspace: torch.Tensor | None,
    plan: Any,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if workspace is None:
        raise RuntimeError("standalone b12x MXFP4 requires workspace2 scratch")
    if not workspace.is_contiguous():
        raise ValueError("standalone b12x MXFP4 workspace2 must be contiguous")
    scratch = workspace.reshape(-1).view(torch.uint8)
    required_nbytes = _standalone_b12x_scratch_nbytes(plan)
    expert_counts_nbytes = int(num_experts) * 4
    total_nbytes = required_nbytes + expert_counts_nbytes
    if scratch.numel() < total_nbytes:
        raise ValueError(
            "standalone b12x MXFP4 workspace2 is too small: "
            f"have={scratch.numel()} bytes, need={total_nbytes} bytes"
        )
    expert_counts = scratch[
        required_nbytes : required_nbytes + expert_counts_nbytes
    ].view(torch.int32)
    return scratch[:required_nbytes], expert_counts


def _run_standalone_b12x(fused_moe: Any, binding: Any) -> torch.Tensor:
    """Run a planned binding with its caller-owned route-count workspace.

    b12x 1.2.4 binds ``expert_counts`` but its public runner forwards that
    tensor only for Trellis weights.  General packed W4A16 then tries to make
    a temporary route-count tensor and fails during CUDA graph capture.  Use
    the same planned launch with the already-bound tensor until the public
    runner forwards it for every packed W4A16 layout.
    """
    if binding.implementation != "w4a16" or binding.expert_counts is None:
        return fused_moe.run(binding=binding)

    prepared = binding.experts.representation_for("w4a16")
    if getattr(prepared, "weight_layout", "") == "trellis3_t256":
        return fused_moe.run(binding=binding)

    from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe

    required_fields = (
        "output",
        "intermediate_cache13",
        "intermediate_cache2",
        "packed_route_indices",
        "block_expert_ids",
        "packed_route_count",
        "expert_offsets",
    )
    missing = [name for name in required_fields if getattr(binding, name) is None]
    if missing:
        raise RuntimeError(
            "standalone b12x W4A16 binding is missing: " + ", ".join(missing)
        )

    return run_w4a16_moe(
        binding.a,
        prepared,
        binding.topk_weights,
        binding.topk_ids,
        activation=binding.experts.activation,
        apply_router_weight_on_input=binding.apply_router_weight_on_input,
        fast_math=binding.fast_math,
        intermediate_cache13=binding.intermediate_cache13,
        intermediate_cache2=binding.intermediate_cache2,
        output=binding.output,
        fc1_c_tmp=binding.fc1_c_tmp,
        fc2_c_tmp=binding.fc2_c_tmp,
        packed_route_indices=binding.packed_route_indices,
        block_expert_ids=binding.block_expert_ids,
        packed_route_count=binding.packed_route_count,
        expert_offsets=binding.expert_offsets,
        expert_counts=binding.expert_counts,
        expert_map=binding.route_expert_map,
        output_expert_map=binding.output_expert_map,
        activation_amax=binding.activation_amax,
        layer_idx=binding.layer_idx,
        swiglu_limit=binding.swiglu_limit,
        swiglu_alpha=binding.swiglu_alpha,
        swiglu_beta=binding.swiglu_beta,
        fused_launch=binding.fused_launch,
        topk_sum_launch=binding.topk_sum_launch,
        route_block_size_m=binding.route_block_size_m,
    )


def _b12x_route_pack_token_capacities(max_tokens: int) -> tuple[int, ...]:
    """Return every power-of-two route-pack capacity through ``max_tokens``."""
    max_tokens = max(int(max_tokens), 1)
    max_capacity = 1 << (max_tokens - 1).bit_length()
    return tuple(1 << shift for shift in range(max_capacity.bit_length()))


def _prewarm_b12x_route_pack(
    *,
    device: torch.device,
    num_experts: int,
    topk: int,
    max_tokens: int,
) -> None:
    """Resolve every reachable B12X route-pack specialization."""
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError(f"B12X route-pack warmup requires CUDA, got {device}")

    num_experts = max(int(num_experts), 1)
    topk = max(int(topk), 1)
    capacities = _b12x_route_pack_token_capacities(max_tokens)

    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_host import (
        select_route_block_size_m,
    )
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_kernel import (
        pack_topk_routes_by_expert,
    )

    with torch.accelerator.device_index(device.index):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "B12X route-pack warmup must run before CUDA graph capture"
            )
        device_index = int(torch.accelerator.current_device_index())
        cache_key = (
            device.type,
            device_index,
            num_experts,
            topk,
            capacities[-1],
        )
        if cache_key in _B12X_ROUTE_PACK_WARMED:
            return

        topk_ids = torch.zeros(
            (capacities[-1], topk),
            dtype=torch.int32,
            device=device,
        )
        for token_capacity in capacities:
            block_size = select_route_block_size_m(
                token_capacity,
                topk,
                num_experts,
            )
            live_token_counts = (
                (token_capacity, token_capacity - 1)
                if token_capacity > 2
                else (token_capacity,)
            )
            for live_tokens in live_token_counts:
                pack_topk_routes_by_expert(
                    topk_ids[:live_tokens],
                    block_size,
                    num_experts,
                )
        torch.accelerator.synchronize(device)
        _B12X_ROUTE_PACK_WARMED.add(cache_key)

    logger.info(
        "Prewarmed B12X route-pack capacities %s on cuda:%d (experts=%d, topk=%d)",
        capacities,
        device_index,
        num_experts,
        topk,
    )


def _sanitize_b12x_topk(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map vLLM's padding sentinel to a safe, zero-weight B12X route."""
    is_padding = topk_ids < 0
    safe_ids = topk_ids.masked_fill(is_padding, 0).to(torch.int32)
    safe_weights = topk_weights.masked_fill(is_padding, 0.0)
    return safe_ids, safe_weights


def _prepare_b12x_topk(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    quant_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare routes without masking work already handled by W4A16.

    W4A16's direct and packed route paths both skip negative expert IDs.  Keep
    the sanitizing fallback for the NVFP4/W4A4 implementation, whose kernels
    still require every expert ID to be in range.
    """
    if quant_mode == "w4a16":
        return topk_ids.to(torch.int32), topk_weights
    return _sanitize_b12x_topk(topk_ids, topk_weights)


class FlashInferB12xExperts(mk.FusedMoEExpertsModular):
    """FlashInfer CuteDSL fused MoE expert for SM12x (SM120/SM121,
    RTX Pro 6000 / DGX Spark).

    Uses ``b12x_fused_moe`` from FlashInfer PR #3080 which fuses token
    dispatch, two GEMMs, SwiGLU activation, and topk-weight reduction into a
    single kernel call.  Input quantization (BF16→FP4) is performed inside the
    kernel so BF16 hidden states are passed directly.

    NVFP4 weight scale factors are converted to the MMA layout produced by
    ``convert_sf_to_mma_layout`` once during ``process_weights_after_loading``.
    Native MXFP4 checkpoints use B12X's W4A16 path so the draft model keeps
    BF16 activations. ModelOpt NVFP4 checkpoints can opt into the same packed
    W4A16 execution path with ``VLLM_B12X_NVFP4_W4A16=1``. The opt-in repacks
    weights in place and replaces the source block scales, so it is fixed for
    the lifetime of the loaded model.

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
        self._nvfp4_w4a16 = (
            self.checkpoint_quant_mode == "nvfp4" and envs.VLLM_B12X_NVFP4_W4A16
        )
        self._standalone_mxfp4 = (
            self.checkpoint_quant_mode == "mxfp4" and envs.VLLM_B12X_STANDALONE_MXFP4
        )
        self.quant_mode = (
            "w4a16"
            if self.checkpoint_quant_mode == "mxfp4" or self._nvfp4_w4a16
            else "nvfp4"
        )
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
        self._route_pack_max_tokens = int(moe_config.max_num_tokens)
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
        self._standalone_b12x_experts: object | None = None
        self._standalone_b12x_device: torch.device | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self._standalone_mxfp4:
            from b12x.moe import fused_moe

            _apply_b12x_w4a16_ultrawide_compile_compat()

            unit_scale = torch.ones(
                self.num_local_experts,
                device=layer.w13_weight.device,
                dtype=torch.float32,
            )
            weight_plan = _standalone_b12x_weight_plan(
                num_experts=self.global_num_experts,
                hidden_size=self.hidden_dim,
                intermediate_size=self.intermediate_size_per_partition,
                activation=self._activation_str,
                params_dtype=self.out_dtype,
            )
            experts = fused_moe.prepare_weights(
                plan=weight_plan,
                params_dtype=self.out_dtype,
                w1_fp4=layer.w13_weight,
                w2_fp4=layer.w2_weight,
                w1_global_scale=unit_scale,
                w2_global_scale=unit_scale,
                w1_blockscale=layer.w13_weight_scale,
                w2_blockscale=layer.w2_weight_scale,
                a1_gscale=unit_scale,
                a2_gscale=unit_scale,
            )
            self._standalone_b12x_experts = experts
            self._standalone_b12x_device = layer.w13_weight.device
            self._prepared_w4a16 = experts.representation_for("w4a16")
            self._w1_alpha = experts.w1_alphas
            self._w2_alpha = experts.w2_alphas
            self.w1_sf_mma = experts.w1_blockscale
            self.w2_sf_mma = experts.w2_blockscale
            self._fc2_input_scale = None
            return

        if self._nvfp4_w4a16:
            assert self.w1_scale is not None and self.w2_scale is not None
            assert self.g1_alphas is not None and self.g2_alphas is not None

            # vLLM has already converted the linear K/16 ModelOpt scales to
            # FlashInfer's swizzled source layout. Pack both projections into
            # the B12X W4A16 MMA layout, reusing the FP4 weight storage so the
            # full model never keeps a second expert-weight copy.
            from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (  # noqa: E501
                prepare_w4a16_packed_weights,
            )

            prepared = prepare_w4a16_packed_weights(
                layer.w13_weight,
                self.w1_scale,
                self.g1_alphas,
                layer.w2_weight,
                self.w2_scale,
                self.g2_alphas,
                activation=self._activation_str,
                params_dtype=self.out_dtype,
                source_format=self.source_format,
                w13_layout="w13",
                reuse_input_storage=True,
            )
            self._prepared_w4a16 = prepared

            # Rebind the existing Parameters so the source-layout scales are
            # released layer by layer instead of retaining a second full scale
            # grid across the model. The quant config references the same
            # Parameter objects, so its views stay synchronized.
            layer.w13_weight_scale.data = prepared.w13_scale
            layer.w2_weight_scale.data = prepared.w2_scale
            layer.w13_weight_scale_2.data = prepared.w13_global_scale
            layer.w2_weight_scale_2.data = prepared.w2_global_scale

            self.w1_sf_mma = layer.w13_weight_scale
            self.w2_sf_mma = layer.w2_weight_scale
            self._w1_alpha = layer.w13_weight_scale_2
            self._w2_alpha = layer.w2_weight_scale_2
            self._fc2_input_scale = None
            _prewarm_b12x_route_pack(
                device=layer.w13_weight.device,
                num_experts=self.global_num_experts,
                topk=self.topk,
                max_tokens=self._route_pack_max_tokens,
            )
            return

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
            _prewarm_b12x_route_pack(
                device=layer.w13_weight.device,
                num_experts=self.global_num_experts,
                topk=self.topk,
                max_tokens=self._route_pack_max_tokens,
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
        if self._standalone_mxfp4:
            if self._standalone_b12x_device is None:
                raise RuntimeError(
                    "process_weights_after_loading must initialize standalone b12x"
                )
            plan = _standalone_b12x_execution_plan(
                max_tokens=M,
                num_experts=self.global_num_experts,
                hidden_size=self.hidden_dim,
                intermediate_size=self.intermediate_size_per_partition,
                topk=self.topk,
                device=str(self._standalone_b12x_device),
                activation=self._activation_str,
                params_dtype=self.out_dtype,
                swiglu_limit=self.swiglu_limit,
            )
            scratch_nbytes = _standalone_b12x_scratch_nbytes(plan)
            # b12x's route workspace scales with M and can be smaller than an
            # E-wide count array at decode.  Keep expert counts in a fixed,
            # caller-owned tail so both eager warmup and CUDA graphs are
            # allocation-free without depending on b12x's private layout.
            scratch_nbytes += self.global_num_experts * 4
            element_size = torch.empty((), dtype=self.out_dtype).element_size()
            workspace2 = ((scratch_nbytes + element_size - 1) // element_size,)
            return (1,), workspace2, (M, K)

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

        # vLLM uses -1 expert IDs for cudagraph/profile padding. W4A16 skips
        # these routes in-kernel; NVFP4/W4A4 retains the safe expert-0 fallback.
        token_selected_experts, token_final_scales = _prepare_b12x_topk(
            topk_ids, topk_weights, self.quant_mode
        )

        if self._standalone_mxfp4:
            if apply_router_weight_on_input:
                raise RuntimeError(
                    "standalone b12x MXFP4 does not support applying router "
                    "weights on input"
                )
            assert self._standalone_b12x_experts is not None
            assert self._standalone_b12x_device is not None
            from b12x.moe import fused_moe

            plan = _standalone_b12x_execution_plan(
                max_tokens=hidden_states.shape[0],
                num_experts=self.global_num_experts,
                hidden_size=self.hidden_dim,
                intermediate_size=self.intermediate_size_per_partition,
                topk=self.topk,
                device=str(self._standalone_b12x_device),
                activation=self._activation_str,
                params_dtype=self.out_dtype,
                swiglu_limit=self.swiglu_limit,
            )
            scratch, expert_counts = _workspace_as_standalone_b12x_scratch(
                workspace2, plan, self.global_num_experts
            )
            binding = fused_moe.bind(
                plan,
                scratch=scratch,
                a=hidden_states,
                experts=self._standalone_b12x_experts,
                topk_weights=token_final_scales,
                topk_ids=token_selected_experts,
                output=output,
                input_scales_static=True,
                unit_scale_contract=True,
            )
            uses_packed_routes = not bool(
                getattr(binding.fused_launch, "direct_topk_routes", False)
            )
            if (
                binding.implementation == "w4a16"
                and uses_packed_routes
                and binding.expert_counts is None
            ):
                binding = replace(binding, expert_counts=expert_counts)
            _run_standalone_b12x(fused_moe, binding)
            return

        # The functional API reuses FlashInfer's process-wide workspace cache.
        # vLLM warms the largest capture shape before CUDA graph capture, and
        # supplying ``output`` keeps the captured call allocation-free.  Avoid
        # a process-wide CUDA Event here: every layer executes in order on the
        # current stream, while recording the same event repeatedly inside
        # multiple captured graphs creates invalid cross-graph dependencies.
        if self._nvfp4_w4a16:
            assert self._prepared_w4a16 is not None
            from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (  # noqa: E501
                launch_sm120_moe,
            )

            launch_sm120_moe(
                a=hidden_states,
                topk_ids=token_selected_experts,
                topk_weights=token_final_scales,
                w1_weight=w1,
                w1_weight_sf=self.w1_sf_mma,
                w1_alpha=self._w1_alpha,
                w2_weight=w2,
                w2_weight_sf=self.w2_sf_mma,
                w2_alpha=self._w2_alpha,
                num_experts=self.global_num_experts,
                top_k=self.topk,
                num_local_experts=self.num_local_experts,
                scatter_output=output,
                activation=self._activation_str,
                swiglu_limit=self.swiglu_limit,
                quant_mode="w4a16",
                source_format=self.source_format,
                _prepared_weights=self._prepared_w4a16,
            )
            return

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
