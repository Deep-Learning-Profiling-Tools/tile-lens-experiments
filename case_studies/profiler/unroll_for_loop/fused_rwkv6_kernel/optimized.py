
import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def fused_recurrent_rwkv6_fwd_kernel(
    q, k, v, w, u, o, h0, ht, s_k_h, s_v_h, scale, B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr, REVERSE: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_o = o + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_w = w + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_u = u + i_h * K + tl.arange(0, BK) + i_k * BK

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    mask_kv = mask_bv[:, None] & mask_bk[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    b_u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)
    # for _ in range(0, T):
    #     b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    #     b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    #     b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
    #     b_w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
    #     b_w = tl.exp(b_w)
    #     b_kv = b_k[None, :] * b_v[:, None]
    #     b_o = (b_h + b_kv * b_u[None, :]) * b_q[None, :]
    #     b_o = tl.sum(b_o, axis=1)
    #     b_h = b_h * b_w[None, :]
    #     b_h += b_kv
    #     tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_bv)
    #     p_q += -K if REVERSE else K
    #     p_k += -K if REVERSE else K
    #     p_o += -V if REVERSE else V
    #     p_v += -V if REVERSE else V
    #     p_w += -K if REVERSE else K

    # Initialization
    b_h_0 = b_h

    # Precompute pointers needed for each iteration
    p_k_0, p_v_0, p_q_0, p_o_0, p_w_0 = p_k, p_v, p_q, p_o, p_w

    p_k_1 = p_k_0 + (-K if REVERSE else K)
    p_v_1 = p_v_0 + (-V if REVERSE else V)
    p_q_1 = p_q_0 + (-K if REVERSE else K)
    p_o_1 = p_o_0 + (-V if REVERSE else V)
    p_w_1 = p_w_0 + (-K if REVERSE else K)

    p_k_2 = p_k_1 + (-K if REVERSE else K)
    p_v_2 = p_v_1 + (-V if REVERSE else V)
    p_q_2 = p_q_1 + (-K if REVERSE else K)
    p_o_2 = p_o_1 + (-V if REVERSE else V)
    p_w_2 = p_w_1 + (-K if REVERSE else K)

    p_k_3 = p_k_2 + (-K if REVERSE else K)
    p_v_3 = p_v_2 + (-V if REVERSE else V)
    p_q_3 = p_q_2 + (-K if REVERSE else K)
    p_o_3 = p_o_2 + (-V if REVERSE else V)
    p_w_3 = p_w_2 + (-K if REVERSE else K)

    # =========================
    # Pull all tl.load as early as possible
    # =========================
    b_k_0 = tl.load(p_k_0, mask=mask_bk, other=0).to(tl.float32)
    b_v_0 = tl.load(p_v_0, mask=mask_bv, other=0).to(tl.float32)
    b_q_0 = tl.load(p_q_0, mask=mask_bk, other=0).to(tl.float32) * scale
    b_w_0 = tl.load(p_w_0, mask=mask_bk, other=0).to(tl.float32)

    b_k_1 = tl.load(p_k_1, mask=mask_bk, other=0).to(tl.float32)
    b_v_1 = tl.load(p_v_1, mask=mask_bv, other=0).to(tl.float32)
    b_q_1 = tl.load(p_q_1, mask=mask_bk, other=0).to(tl.float32) * scale
    b_w_1 = tl.load(p_w_1, mask=mask_bk, other=0).to(tl.float32)

    b_k_2 = tl.load(p_k_2, mask=mask_bk, other=0).to(tl.float32)
    b_v_2 = tl.load(p_v_2, mask=mask_bv, other=0).to(tl.float32)
    b_q_2 = tl.load(p_q_2, mask=mask_bk, other=0).to(tl.float32) * scale
    b_w_2 = tl.load(p_w_2, mask=mask_bk, other=0).to(tl.float32)

    b_k_3 = tl.load(p_k_3, mask=mask_bk, other=0).to(tl.float32)
    b_v_3 = tl.load(p_v_3, mask=mask_bv, other=0).to(tl.float32)
    b_q_3 = tl.load(p_q_3, mask=mask_bk, other=0).to(tl.float32) * scale
    b_w_3 = tl.load(p_w_3, mask=mask_bk, other=0).to(tl.float32)

    # =========================
    # Iterative computation (no more loads), arithmetic and exp/sum only
    # =========================
    # iter 0
    b_w_0 = tl.exp(b_w_0)
    b_kv_0 = b_k_0[None, :] * b_v_0[:, None]
    b_o_0  = (b_h_0 + b_kv_0 * b_u[None, :]) * b_q_0[None, :]
    b_o_0  = tl.sum(b_o_0, axis=1)
    b_h_1  = b_h_0 * b_w_0[None, :]
    b_h_1  = b_h_1 + b_kv_0

    # iter 1
    b_w_1 = tl.exp(b_w_1)
    b_kv_1 = b_k_1[None, :] * b_v_1[:, None]
    b_o_1  = (b_h_1 + b_kv_1 * b_u[None, :]) * b_q_1[None, :]
    b_o_1  = tl.sum(b_o_1, axis=1)
    b_h_2  = b_h_1 * b_w_1[None, :]
    b_h_2  = b_h_2 + b_kv_1

    # iter 2
    b_w_2 = tl.exp(b_w_2)
    b_kv_2 = b_k_2[None, :] * b_v_2[:, None]
    b_o_2  = (b_h_2 + b_kv_2 * b_u[None, :]) * b_q_2[None, :]
    b_o_2  = tl.sum(b_o_2, axis=1)
    b_h_3  = b_h_2 * b_w_2[None, :]
    b_h_3  = b_h_3 + b_kv_2

    # iter 3
    b_w_3 = tl.exp(b_w_3)
    b_kv_3 = b_k_3[None, :] * b_v_3[:, None]
    b_o_3  = (b_h_3 + b_kv_3 * b_u[None, :]) * b_q_3[None, :]
    b_o_3  = tl.sum(b_o_3, axis=1)
    b_h_4  = b_h_3 * b_w_3[None, :]
    b_h_4  = b_h_4 + b_kv_3

    # =========================
    # Defer all tl.store until the end
    # =========================
    tl.store(p_o_0, b_o_0.to(p_o.dtype.element_ty), mask=mask_bv)
    tl.store(p_o_1, b_o_1.to(p_o.dtype.element_ty), mask=mask_bv)
    tl.store(p_o_2, b_o_2.to(p_o.dtype.element_ty), mask=mask_bv)
    tl.store(p_o_3, b_o_3.to(p_o.dtype.element_ty), mask=mask_bv)

    # If later iterations need post-loop values/pointers, update base variables at the end
    b_h = b_h_4
    # p_q = p_q_3 + (-K if REVERSE else K)
    # p_k = p_k_3 + (-K if REVERSE else K)
    # p_o = p_o_3 + (-V if REVERSE else V)
    # p_v = p_v_3 + (-V if REVERSE else V)
    # p_w = p_w_3 + (-K if REVERSE else K)


    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_kv)

class FusedRecurrentRWKV6Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, r, k, v, w, u, scale=None, initial_state=None, output_final_state=False, reverse=False):
        q = r
        B, H, T, K, V = *q.shape, v.shape[-1]

        BK, BV = min(triton.next_power_of_2(K), 32), min(triton.next_power_of_2(V), 32)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 1

        final_state = q.new_empty(B, H, K, V) if output_final_state else None

        o = q.new_empty(NK, B, H, T, V, dtype=torch.float32)
        grid = (NV, NK, B * H)
        fused_recurrent_rwkv6_fwd_kernel[grid](
            q, k, v, w, u, o, initial_state, final_state,
            k.stride(1),
            v.stride(1),
            scale,
            B=B, H=H, T=T, K=K, V=V, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None,
            REVERSE=reverse,
            num_warps=num_warps,
            num_stages=num_stages
        )

        o = o.sum(0)
        ctx.save_for_backward(q, k, v, w, u, initial_state)
        ctx.scale = scale
        ctx.reverse = reverse
        return o.to(q.dtype), final_state

def fused_recurrent_rwkv6(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: float = -1,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale == -1:
        scale = r.shape[-1] ** -0.5
    o, final_state = FusedRecurrentRWKV6Function.apply(r, k, v, w, u, scale, initial_state, output_final_state)
    return o, final_state




##################################################################################################################################################


import torch

def test_fused_recurrent_rwkv6():
    # Define input dimensions
    B, H, T, K, V = 2, 3, 4, 8, 8

    # Create random input tensors
    r = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    k = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    v = torch.randn(B, H, T, V, dtype=torch.float32, device='cuda')
    w = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    u = torch.randn(H, K, dtype=torch.float32, device='cuda')

    # Prepare a dictionary to store results
    results = {}

    # Test without initial state, without final state, forward
    o, final_state = fused_recurrent_rwkv6(r, k, v, w, u, scale=0.5, initial_state=None, output_final_state=False)
    results["test_case_1"] = {"output": o.shape, "final_state": final_state}

    # Test with initial state, without final state, forward
    initial_state = torch.randn(B, H, K, V, dtype=torch.float32, device='cuda')
    o, final_state = fused_recurrent_rwkv6(r, k, v, w, u, scale=0.5, initial_state=initial_state, output_final_state=False)
    results["test_case_2"] = {"output": o.shape, "final_state": final_state}

    # Test without initial state, with final state, forward
    o, final_state = fused_recurrent_rwkv6(r, k, v, w, u, scale=0.5, initial_state=None, output_final_state=True)
    results["test_case_3"] = {"output": o.shape, "final_state": final_state.shape}

    # Test with initial state, with final state, forward
    o, final_state = fused_recurrent_rwkv6(r, k, v, w, u, scale=0.5, initial_state=initial_state, output_final_state=True)
    results["test_case_4"] = {"output": o.shape, "final_state": final_state.shape}

    return results

result_gold = test_fused_recurrent_rwkv6()
