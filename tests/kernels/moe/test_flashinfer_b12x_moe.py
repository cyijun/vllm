# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(120):
    pytest.skip(
        reason="FlashInfer B12x MoE requires SM120 (RTX Pro 6000 / DGX Spark).",
        allow_module_level=True,
    )

from vllm.utils.flashinfer import has_flashinfer_b12x_moe

if not has_flashinfer_b12x_moe():
    pytest.skip(
        reason=(
            "FlashInfer B12xMoEWrapper not available in installed "
            "FlashInfer (needs PR #3080)."
        ),
        allow_module_level=True,
    )

# Import fp4_quantize after the skip guard — FlashInfer must be installed.
from flashinfer.fp4_quantization import fp4_quantize
from flashinfer.quantization import SfLayout, mxfp4_dequantize, mxfp4_quantize

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.utils import torch_moe
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    nvfp4_moe_quant_config,
    ocp_mx_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import (
    FlashInferB12xExperts,
    _b12x_route_pack_token_capacities,
    _prepare_b12x_topk,
    _sanitize_b12x_topk,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    convert_weight_to_mxfp4_moe_kernel_format,
    select_deepseek_v4_mxfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    reorder_w1w3_to_w3w1,
)
from vllm.utils.torch_utils import set_random_seed

# Dimensions chosen to satisfy FP4 alignment requirements (k multiple of 256,
# n multiple of 128) while keeping tests fast.
MNK_FACTORS = [
    (2, 128, 256),
    (2, 256, 512),
    (16, 128, 256),
    (64, 256, 512),
]


@pytest.mark.parametrize(
    ("max_tokens", "expected"),
    [
        (1, (1,)),
        (6, (1, 2, 4, 8)),
        (512, tuple(1 << shift for shift in range(10))),
    ],
)
def test_b12x_route_pack_token_capacities(max_tokens, expected):
    assert _b12x_route_pack_token_capacities(max_tokens) == expected


def _process_b12x_weights(
    experts: FlashInferB12xExperts,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_scale_2: torch.Tensor,
    w2_scale_2: torch.Tensor,
) -> None:
    layer = SimpleNamespace(
        w13_weight_scale=w1_scale,
        w13_weight_scale_2=w1_scale_2,
        w2_weight_scale=w2_scale,
        w2_weight_scale_2=w2_scale_2,
    )
    experts.process_weights_after_loading(layer)


def test_flashinfer_b12x_functional_call_receives_swiglu_limit(monkeypatch):
    monkeypatch.delenv("VLLM_B12X_NVFP4_W4A16", raising=False)
    captured = {}

    def fake_b12x_fused_moe(**kwargs):
        captured.update(kwargs)
        return kwargs["output"]

    monkeypatch.setattr("flashinfer.fused_moe.b12x_fused_moe", fake_b12x_fused_moe)
    ones = torch.ones(1, dtype=torch.float32, device="cuda")
    quant_config = nvfp4_moe_quant_config(
        g1_alphas=ones,
        g2_alphas=ones,
        a1_gscale=ones,
        a2_gscale=ones,
        w1_scale=ones,
        w2_scale=ones,
    )
    moe_config = make_dummy_moe_config(
        hidden_dim=256,
        intermediate_size=128,
        swiglu_limit=10.0,
    )
    experts = FlashInferB12xExperts(moe_config, quant_config)
    experts._fc2_input_scale = ones
    experts._w1_alpha = ones
    experts._w2_alpha = ones
    experts.w1_sf_mma = ones
    experts.w2_sf_mma = ones

    hidden_states = torch.zeros((1, 256), dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(hidden_states)
    topk_ids = torch.zeros((1, 2), dtype=torch.int64, device="cuda")
    topk_weights = torch.ones((1, 2), dtype=torch.float32, device="cuda")
    weight = torch.empty(1, dtype=torch.uint8, device="cuda")
    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=weight,
        w2=weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=moe_config.num_experts,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert captured["activation"] == "silu"
    assert captured["swiglu_limit"] == 10.0
    assert captured["output"] is output
    assert captured["quant_mode"] == "nvfp4"


@torch.inference_mode()
def test_flashinfer_b12x_nvfp4_w4a16_reuses_weight_storage(monkeypatch):
    """The opt-in packs ModelOpt NVFP4 in place and runs BF16 activations."""
    monkeypatch.setenv("VLLM_B12X_NVFP4_W4A16", "1")
    prewarm_calls = []
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe."
        "_prewarm_b12x_route_pack",
        lambda **kwargs: prewarm_calls.append(kwargs),
    )
    m, n, k, e, topk = 6, 128, 256, 8, 2
    dtype = torch.bfloat16
    set_random_seed(23)

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        hidden_states = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        # The B12X source contract is [up; gate], matching vLLM's post-load
        # [w3; w1] reorder.
        w13_bf16 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 15
        w2_bf16 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 15
        global_scale = torch.ones(1, device="cuda", dtype=torch.float32)
        w13_q_flat, w13_sf_flat = fp4_quantize(
            w13_bf16.reshape(e * 2 * n, k),
            global_scale=global_scale,
            sf_vec_size=16,
            is_sf_swizzled_layout=True,
        )
        w2_q_flat, w2_sf_flat = fp4_quantize(
            w2_bf16.reshape(e * k, n),
            global_scale=global_scale,
            sf_vec_size=16,
            is_sf_swizzled_layout=True,
        )
        w13_q = w13_q_flat.view(e, 2 * n, k // 2)
        w2_q = w2_q_flat.view(e, k, n // 2)
        w13_sf = w13_sf_flat.view(e, 2 * n, w13_sf_flat.shape[1])
        w2_sf = w2_sf_flat.view(e, k, w2_sf_flat.shape[1])
        ones_e = torch.ones(e, device="cuda", dtype=torch.float32)
        layer = SimpleNamespace(
            w13_weight=w13_q,
            w2_weight=w2_q,
            w13_weight_scale=w13_sf,
            w2_weight_scale=w2_sf,
            w13_weight_scale_2=ones_e.clone(),
            w2_weight_scale_2=ones_e.clone(),
        )
        quant_config = nvfp4_moe_quant_config(
            g1_alphas=layer.w13_weight_scale_2,
            g2_alphas=layer.w2_weight_scale_2,
            a1_gscale=ones_e,
            a2_gscale=ones_e,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )
        moe_config = make_dummy_moe_config(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            in_dtype=dtype,
            swiglu_limit=10.0,
        )
        experts = FlashInferB12xExperts(moe_config, quant_config)
        w13_ptr = w13_q.data_ptr()
        w2_ptr = w2_q.data_ptr()
        source_scale_elements = w13_sf.numel() + w2_sf.numel()
        experts.process_weights_after_loading(layer)

        assert experts.quant_mode == "w4a16"
        assert experts._prepared_w4a16 is not None
        assert experts._prepared_w4a16.w13.data_ptr() == w13_ptr
        assert experts._prepared_w4a16.w2.data_ptr() == w2_ptr
        assert layer.w13_weight_scale.data_ptr() == (
            experts._prepared_w4a16.w13_scale.data_ptr()
        )
        assert layer.w2_weight_scale.data_ptr() == (
            experts._prepared_w4a16.w2_scale.data_ptr()
        )
        assert (
            layer.w13_weight_scale.numel() + layer.w2_weight_scale.numel()
            == source_scale_elements
        )
        assert prewarm_calls == [
            {
                "device": layer.w13_weight.device,
                "num_experts": e,
                "topk": topk,
                "max_tokens": moe_config.max_num_tokens,
            }
        ]

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states, score, topk, renormalize=False
        )
        output = torch.empty_like(hidden_states)
        experts.apply(
            output=output,
            hidden_states=hidden_states,
            w1=w13_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=e,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
        assert torch.isfinite(output).all()


def test_flashinfer_b12x_sanitizes_padding_routes():
    topk_ids = torch.tensor([[3, -1, 5], [-1, -1, 2]], dtype=torch.int64, device="cuda")
    topk_weights = torch.tensor(
        [[0.5, 0.25, 0.125], [1.0, -2.0, 0.75]],
        dtype=torch.float32,
        device="cuda",
    )

    safe_ids, safe_weights = _sanitize_b12x_topk(topk_ids, topk_weights)

    torch.testing.assert_close(
        safe_ids,
        torch.tensor([[3, 0, 5], [0, 0, 2]], dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(
        safe_weights,
        torch.tensor(
            [[0.5, 0.0, 0.125], [0.0, 0.0, 0.75]],
            dtype=torch.float32,
            device="cuda",
        ),
    )
    # Sanitization must not mutate tensors potentially reused by other stages.
    assert (topk_ids == -1).sum().item() == 3
    assert topk_weights[1, 1].item() == -2.0


def test_flashinfer_b12x_w4a16_preserves_padding_routes():
    topk_ids = torch.tensor([[3, -1, 5], [-1, -1, 2]], dtype=torch.int64, device="cuda")
    topk_weights = torch.tensor(
        [[0.5, 0.25, 0.125], [1.0, -2.0, 0.75]],
        dtype=torch.float32,
        device="cuda",
    )

    prepared_ids, prepared_weights = _prepare_b12x_topk(topk_ids, topk_weights, "w4a16")

    torch.testing.assert_close(prepared_ids, topk_ids.to(torch.int32))
    assert prepared_weights is topk_weights


@pytest.mark.parametrize("requested_backend", ["auto", "flashinfer_b12x"])
def test_flashinfer_b12x_selected_for_mxfp4(requested_backend):
    moe_config = make_dummy_moe_config(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=128,
        swiglu_limit=10.0,
    )
    moe_config.moe_backend = requested_backend

    backend, experts_cls = select_deepseek_v4_mxfp4_moe_backend(moe_config)

    assert backend == Mxfp4MoeBackend.FLASHINFER_B12X
    assert experts_cls is FlashInferB12xExperts


@torch.inference_mode()
def test_flashinfer_b12x_functional_adapter_cuda_graph(workspace_init):
    """The shared functional workspace must survive capture and replay."""
    m, n, k, e, topk = 8, 128, 256, 8, 2
    dtype = torch.bfloat16
    set_random_seed(11)

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        hidden_states = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 15
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 15

        dummy_scale = torch.ones((e, 2 * n, 1), device="cuda", dtype=torch.float32)
        w1, _ = reorder_w1w3_to_w3w1(w1, dummy_scale)
        global_scale = torch.ones(1, device="cuda", dtype=torch.float32)
        w1_q_flat, w1_sf_flat = fp4_quantize(
            w1.reshape(e * 2 * n, k),
            global_scale=global_scale,
            sf_vec_size=16,
            is_sf_swizzled_layout=True,
        )
        w2_q_flat, w2_sf_flat = fp4_quantize(
            w2.reshape(e * k, n),
            global_scale=global_scale,
            sf_vec_size=16,
            is_sf_swizzled_layout=True,
        )
        w1_q = w1_q_flat.view(e, 2 * n, k // 2)
        w2_q = w2_q_flat.view(e, k, n // 2)
        w1_sf = w1_sf_flat.view(e, 2 * n, w1_sf_flat.shape[1])
        w2_sf = w2_sf_flat.view(e, k, w2_sf_flat.shape[1])
        ones_e = torch.ones(e, device="cuda", dtype=torch.float32)

        quant_config = nvfp4_moe_quant_config(
            g1_alphas=ones_e,
            g2_alphas=ones_e,
            a1_gscale=ones_e,
            a2_gscale=ones_e,
            w1_scale=w1_sf,
            w2_scale=w2_sf,
        )
        moe_config = make_dummy_moe_config(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            in_dtype=dtype,
        )
        experts = FlashInferB12xExperts(moe_config, quant_config)
        _process_b12x_weights(experts, w1_sf, w2_sf, ones_e, ones_e)

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states, score, topk, renormalize=False
        )
        output = torch.empty_like(hidden_states)

        def apply() -> None:
            experts.apply(
                output=output,
                hidden_states=hidden_states,
                w1=w1_q,
                w2=w2_q,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=MoEActivation.SILU,
                global_num_experts=e,
                expert_map=None,
                a1q_scale=None,
                a2_scale=None,
                workspace13=None,
                workspace2=None,
                expert_tokens_meta=None,
                apply_router_weight_on_input=False,
            )

        # Populate FlashInfer's process-wide weight/workspace caches before
        # capture; capture itself must not grow either cache.
        apply()
        torch.accelerator.synchronize()
        eager_output = output.clone()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            apply()
        graph.replay()
        torch.accelerator.synchronize()

        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, eager_output, atol=1e-2, rtol=1e-2)


@torch.inference_mode()
def test_flashinfer_b12x_mxfp4_moe(workspace_init):
    """Checkpoint-layout MXFP4 weights run through the B12X W4A16 path."""
    m, n, k, e, topk = 8, 128, 256, 8, 2
    dtype = torch.bfloat16
    set_random_seed(19)

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        hidden_states = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w13_bf16 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2_bf16 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

        def quantize_checkpoint_weights(
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            packed, scales, reference = [], [], []
            for expert_weight in weight:
                expert_packed, expert_scale = mxfp4_quantize(
                    expert_weight,
                    sfLayout=SfLayout.layout_linear,
                )
                packed.append(expert_packed)
                scales.append(expert_scale)
                reference.append(
                    mxfp4_dequantize(
                        expert_packed,
                        expert_scale,
                        sfLayout=SfLayout.layout_linear,
                    ).to(device=weight.device, dtype=weight.dtype)
                )
            return torch.stack(packed), torch.stack(scales), torch.stack(reference)

        w13_q, w13_scale, w13_reference = quantize_checkpoint_weights(w13_bf16)
        w2_q, w2_scale, w2_reference = quantize_checkpoint_weights(w2_bf16)
        layer = SimpleNamespace()
        w13_q, w2_q, w13_scale, w2_scale, _, _ = (
            convert_weight_to_mxfp4_moe_kernel_format(
                mxfp4_backend=Mxfp4MoeBackend.FLASHINFER_B12X,
                layer=layer,
                w13_weight=w13_q,
                w2_weight=w2_q,
                w13_weight_scale=w13_scale,
                w2_weight_scale=w2_scale,
            )
        )
        layer.w13_weight = w13_q
        layer.w2_weight = w2_q
        layer.w13_weight_scale = w13_scale
        layer.w2_weight_scale = w2_scale

        quant_config = ocp_mx_moe_quant_config(
            quant_dtype="mxfp4",
            w1_scale=w13_scale,
            w2_scale=w2_scale,
        )
        moe_config = make_dummy_moe_config(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            in_dtype=dtype,
        )
        experts = FlashInferB12xExperts(moe_config, quant_config)
        experts.process_weights_after_loading(layer)
        assert experts._prepared_w4a16 is not None
        assert experts._prepared_w4a16.w13.data_ptr() == w13_q.data_ptr()
        assert experts._prepared_w4a16.w2.data_ptr() == w2_q.data_ptr()
        assert experts._prepared_w4a16.w13_scale.data_ptr() == w13_scale.data_ptr()
        assert experts._prepared_w4a16.w2_scale.data_ptr() == w2_scale.data_ptr()
        kernel = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            experts,
        )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states, score, topk, renormalize=False
        )
        b12x_output = kernel.apply(
            hidden_states=hidden_states,
            w1=w13_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=e,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            expert_map=None,
        )
        reference = torch_moe(
            hidden_states,
            w13_reference,
            w2_reference,
            score,
            topk,
        )

        assert experts.checkpoint_quant_mode == "mxfp4"
        assert experts.quant_mode == "w4a16"
        assert experts.source_format == "fp4_e8m0_k32"
        torch.testing.assert_close(b12x_output, reference, atol=2e-1, rtol=2e-1)

        graph_output = torch.empty_like(hidden_states)

        def apply() -> None:
            experts.apply(
                output=graph_output,
                hidden_states=hidden_states,
                w1=w13_q,
                w2=w2_q,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=MoEActivation.SILU,
                global_num_experts=e,
                expert_map=None,
                a1q_scale=None,
                a2_scale=None,
                workspace13=None,
                workspace2=None,
                expert_tokens_meta=None,
                apply_router_weight_on_input=False,
            )

        apply()
        torch.accelerator.synchronize()
        eager_output = graph_output.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            apply()
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(graph_output, eager_output, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [8, 16])
@pytest.mark.parametrize("topk", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_flashinfer_b12x_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    workspace_init,
):
    """Test FlashInferB12xExperts against a BF16 torch reference.

    The SM12x kernel takes BF16 hidden states directly and fuses token
    dispatch, W1 GEMM, SwiGLU, and W2 GEMM into one call.  We verify
    correctness against ``torch_moe`` using generous tolerances to account
    for the internal FP4 quantization of activations and weights.

    Scale convention
    ----------------
    The SM12x kernel uses ``w1_alpha`` as *both* the activation-quantisation
    global scale and the weight dequantisation factor.  These two roles are
    conflated into a single parameter in ``launch_sm120_moe``, so they must
    equal the same value.  We use ``global_scale = 1.0`` for
    ``fp4_quantize`` so that ``w1_alpha = ones`` satisfies both roles
    simultaneously.  The alternative — vLLM's convention of baking a large
    ``w_gs`` into block-scale values and compensating with
    ``g1_alphas = 1/w_gs`` — is incompatible with this kernel.
    """
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        # Generate BF16 reference weights in [gate, up] order.
        # Shape: w1=(e, 2n, k), w2=(e, k, n).
        w1_bf16 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 15
        w2_bf16 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 15

        # ------------------------------------------------------------------ #
        # Quantise weights for the SM12x kernel using FlashInfer's convention:
        #   global_scale = 1.0   →   block_scale = max_abs_block / fp4_max
        #   w1_alpha = 1.0       (no extra global factor to compensate)
        #
        # The scale factors returned by fp4_quantize(..., is_sf_swizzled_layout=True)
        # are already in the swizzled 2D layout expected by convert_sf_to_mma_layout.
        # No additional swizzle_blockscale() call is needed.
        # ------------------------------------------------------------------ #
        gs = torch.ones(1, device="cuda", dtype=torch.float32)
        sf_vec_size = 16

        # W1: reorder BF16 from [gate, up] → [up, gate], then quantise.
        # Note: in reorder_w1w3_to_w3w1, "w1" refers to the gate projection
        # and "w3" refers to the up projection.
        # A dummy scale is passed and discarded; real scales come from
        # fp4_quantize after reordering.
        w1_reordered, _ = reorder_w1w3_to_w3w1(
            w1_bf16.clone(),
            torch.ones((e, 2 * n, 1), device="cuda", dtype=torch.float32),
        )
        w1_flat = w1_reordered.reshape(e * 2 * n, k)
        w1_q_flat, w1_sf_flat = fp4_quantize(
            w1_flat,
            global_scale=gs,
            sf_vec_size=sf_vec_size,
            is_sf_swizzled_layout=True,
        )
        w1_q = w1_q_flat.view(e, 2 * n, k // 2)  # uint8, packed FP4
        w1_blockscale = w1_sf_flat.view(e, 2 * n, w1_sf_flat.shape[1])  # float8

        # W2: no row reordering needed for the down-projection.
        w2_flat = w2_bf16.reshape(e * k, n)
        w2_q_flat, w2_sf_flat = fp4_quantize(
            w2_flat,
            global_scale=gs,
            sf_vec_size=sf_vec_size,
            is_sf_swizzled_layout=True,
        )
        w2_q = w2_q_flat.view(e, k, n // 2)  # uint8, packed FP4
        w2_blockscale = w2_sf_flat.view(e, k, w2_sf_flat.shape[1])  # float8

        # All per-expert alphas are 1.0 (global_scale = 1.0, no compensation).
        ones_e = torch.ones(e, device="cuda", dtype=torch.float32)

        quant_config = nvfp4_moe_quant_config(
            g1_alphas=ones_e,
            g2_alphas=ones_e,
            a1_gscale=ones_e,
            a2_gscale=ones_e,
            w1_scale=w1_blockscale,
            w2_scale=w2_blockscale,
        )

        moe_config = make_dummy_moe_config(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            in_dtype=dtype,
        )

        experts = FlashInferB12xExperts(
            moe_config=moe_config,
            quant_config=quant_config,
        )

        _process_b12x_weights(
            experts,
            w1_blockscale,
            w2_blockscale,
            ones_e,
            ones_e,
        )

        kernel = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            experts,
        )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        sm12x_output = kernel.apply(
            hidden_states=a,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=e,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            expert_map=None,
        )

        # Reference: BF16 torch MoE using original [gate, up] BF16 weights.
        # torch_moe's SiluAndMul expects [gate, up] order, matching w1_bf16.
        torch_output = torch_moe(a, w1_bf16, w2_bf16, score, topk)

        torch.testing.assert_close(sm12x_output, torch_output, atol=2e-1, rtol=2e-1)


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [8, 16])
@pytest.mark.parametrize("topk", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_flashinfer_b12x_moe_relu2(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    workspace_init,
):
    """Test FlashInferB12xExperts with ReLU2 (non-gated) activation.

    ReLU2 is used by Nemotron-H style models.  Unlike the gated SiLU
    path, w1 has shape [E, N, K] (not [E, 2N, K]) and the activation
    is relu(x)^2 without a gate/up split.
    """
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        # Non-gated: w1 shape is (e, n, k), not (e, 2n, k).
        w1_bf16 = torch.randn((e, n, k), device="cuda", dtype=dtype) / 15
        w2_bf16 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 15

        gs = torch.ones(1, device="cuda", dtype=torch.float32)
        sf_vec_size = 16

        # W1: no gate/up reordering for non-gated.
        w1_flat = w1_bf16.reshape(e * n, k)
        w1_q_flat, w1_sf_flat = fp4_quantize(
            w1_flat,
            global_scale=gs,
            sf_vec_size=sf_vec_size,
            is_sf_swizzled_layout=True,
        )
        w1_q = w1_q_flat.view(e, n, k // 2)
        w1_blockscale = w1_sf_flat.view(e, n, w1_sf_flat.shape[1])

        w2_flat = w2_bf16.reshape(e * k, n)
        w2_q_flat, w2_sf_flat = fp4_quantize(
            w2_flat,
            global_scale=gs,
            sf_vec_size=sf_vec_size,
            is_sf_swizzled_layout=True,
        )
        w2_q = w2_q_flat.view(e, k, n // 2)
        w2_blockscale = w2_sf_flat.view(e, k, w2_sf_flat.shape[1])

        ones_e = torch.ones(e, device="cuda", dtype=torch.float32)

        quant_config = nvfp4_moe_quant_config(
            g1_alphas=ones_e,
            g2_alphas=ones_e,
            a1_gscale=ones_e,
            a2_gscale=ones_e,
            w1_scale=w1_blockscale,
            w2_scale=w2_blockscale,
        )

        moe_config = make_dummy_moe_config(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            in_dtype=dtype,
            activation=MoEActivation.RELU2_NO_MUL,
        )

        experts = FlashInferB12xExperts(
            moe_config=moe_config,
            quant_config=quant_config,
        )
        _process_b12x_weights(
            experts,
            w1_blockscale,
            w2_blockscale,
            ones_e,
            ones_e,
        )

        kernel = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            experts,
        )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        b12x_output = kernel.apply(
            hidden_states=a,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=e,
            activation=MoEActivation.RELU2_NO_MUL,
            apply_router_weight_on_input=False,
            expert_map=None,
        )

        torch_output = torch_moe(
            a,
            w1_bf16,
            w2_bf16,
            score,
            topk,
            activation=MoEActivation.RELU2_NO_MUL,
        )

        torch.testing.assert_close(
            b12x_output,
            torch_output,
            atol=2e-1,
            rtol=2e-1,
        )


if __name__ == "__main__":
    test_flashinfer_b12x_moe(16, 128, 256, 8, 2, torch.bfloat16)
