
import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def fused_recurrent_retention_fwd_kernel(
    q, k, v, o, initial_state, final_state,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    B, H, T, scale,
    BK: tl.constexpr, BV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr
):
    # Kernel logic
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = (1 - tl.math.exp2(-5 - i_h * 1.0))

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < DK
    mask_bv = (i_v * BV + tl.arange(0, BV)) < DV
    mask_kv = mask_bk[None, :] & mask_bv[:, None]

    h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_init_s = initial_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[None, :]) * \
            DV + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)

    # Initialization
    h_0 = h

    # Precompute pointers needed for each iteration
    p_k_0, p_v_0, p_q_0, p_o_0 = p_k, p_v, p_q, p_o

    p_k_1 = p_k_0 + DK
    p_v_1 = p_v_0 + DV
    p_q_1 = p_q_0 + DK
    p_o_1 = p_o_0 + DV

    p_k_2 = p_k_1 + DK
    p_v_2 = p_v_1 + DV
    p_q_2 = p_q_1 + DK
    p_o_2 = p_o_1 + DV

    p_k_3 = p_k_2 + DK
    p_v_3 = p_v_2 + DV
    p_q_3 = p_q_2 + DK
    p_o_3 = p_o_2 + DV

    p_k_4 = p_k_3 + DK
    p_v_4 = p_v_3 + DV
    p_q_4 = p_q_3 + DK
    p_o_4 = p_o_3 + DV

    p_k_5 = p_k_4 + DK
    p_v_5 = p_v_4 + DV
    p_q_5 = p_q_4 + DK
    p_o_5 = p_o_4 + DV

    p_k_6 = p_k_5 + DK
    p_v_6 = p_v_5 + DV
    p_q_6 = p_q_5 + DK
    p_o_6 = p_o_5 + DV

    p_k_7 = p_k_6 + DK
    p_v_7 = p_v_6 + DV
    p_q_7 = p_q_6 + DK
    p_o_7 = p_o_6 + DV

    # =========================
    # Pull all tl.load as early as possible
    # =========================
    _k_0 = tl.load(p_k_0, mask=mask_bk, other=0).to(tl.float32)
    _v_0 = tl.load(p_v_0, mask=mask_bv, other=0).to(tl.float32)
    _q_0 = tl.load(p_q_0, mask=mask_bk, other=0).to(tl.float32) * scale

    _k_1 = tl.load(p_k_1, mask=mask_bk, other=0).to(tl.float32)
    _v_1 = tl.load(p_v_1, mask=mask_bv, other=0).to(tl.float32)
    _q_1 = tl.load(p_q_1, mask=mask_bk, other=0).to(tl.float32) * scale

    _k_2 = tl.load(p_k_2, mask=mask_bk, other=0).to(tl.float32)
    _v_2 = tl.load(p_v_2, mask=mask_bv, other=0).to(tl.float32)
    _q_2 = tl.load(p_q_2, mask=mask_bk, other=0).to(tl.float32) * scale

    _k_3 = tl.load(p_k_3, mask=mask_bk, other=0).to(tl.float32)
    _v_3 = tl.load(p_v_3, mask=mask_bv, other=0).to(tl.float32)
    _q_3 = tl.load(p_q_3, mask=mask_bk, other=0).to(tl.float32) * scale

    _k_4 = tl.load(p_k_4, mask=mask_bk, other=0).to(tl.float32)
    _v_4 = tl.load(p_v_4, mask=mask_bv, other=0).to(tl.float32)
    _q_4 = tl.load(p_q_4, mask=mask_bk, other=0).to(tl.float32) * scale

    _k_5 = tl.load(p_k_5, mask=mask_bk, other=0).to(tl.float32)
    _v_5 = tl.load(p_v_5, mask=mask_bv, other=0).to(tl.float32)
    _q_5 = tl.load(p_q_5, mask=mask_bk, other=0).to(tl.float32) * scale

    _k_6 = tl.load(p_k_6, mask=mask_bk, other=0).to(tl.float32)
    _v_6 = tl.load(p_v_6, mask=mask_bv, other=0).to(tl.float32)
    _q_6 = tl.load(p_q_6, mask=mask_bk, other=0).to(tl.float32) * scale

    _k_7 = tl.load(p_k_7, mask=mask_bk, other=0).to(tl.float32)
    _v_7 = tl.load(p_v_7, mask=mask_bv, other=0).to(tl.float32)
    _q_7 = tl.load(p_q_7, mask=mask_bk, other=0).to(tl.float32) * scale

    # =========================
    # Iterative computation (no more loads), arithmetic only
    # =========================
    # iter 0
    h_1 = b_b * h_0 + _k_0[None, :] * _v_0[:, None]
    _o_0 = h_1 * _q_0[None, :]
    _o_0 = tl.sum(_o_0, axis=1)

    # iter 1
    h_2 = b_b * h_1 + _k_1[None, :] * _v_1[:, None]
    _o_1 = h_2 * _q_1[None, :]
    _o_1 = tl.sum(_o_1, axis=1)

    # iter 2
    h_3 = b_b * h_2 + _k_2[None, :] * _v_2[:, None]
    _o_2 = h_3 * _q_2[None, :]
    _o_2 = tl.sum(_o_2, axis=1)

    # iter 3
    h_4 = b_b * h_3 + _k_3[None, :] * _v_3[:, None]
    _o_3 = h_4 * _q_3[None, :]
    _o_3 = tl.sum(_o_3, axis=1)

    # iter 4
    h_5 = b_b * h_4 + _k_4[None, :] * _v_4[:, None]
    _o_4 = h_5 * _q_4[None, :]
    _o_4 = tl.sum(_o_4, axis=1)

    # iter 5
    h_6 = b_b * h_5 + _k_5[None, :] * _v_5[:, None]
    _o_5 = h_6 * _q_5[None, :]
    _o_5 = tl.sum(_o_5, axis=1)

    # iter 6
    h_7 = b_b * h_6 + _k_6[None, :] * _v_6[:, None]
    _o_6 = h_7 * _q_6[None, :]
    _o_6 = tl.sum(_o_6, axis=1)

    # iter 7
    h_8 = b_b * h_7 + _k_7[None, :] * _v_7[:, None]
    _o_7 = h_8 * _q_7[None, :]
    _o_7 = tl.sum(_o_7, axis=1)

    # =========================
    # Defer all tl.store until the end
    # =========================
    tl.store(p_o_0, _o_0.to(p_o.dtype.element_ty), mask=mask_bv)
    tl.store(p_o_1, _o_1.to(p_o.dtype.element_ty), mask=mask_bv)
    tl.store(p_o_2, _o_2.to(p_o.dtype.element_ty), mask=mask_bv)
    tl.store(p_o_3, _o_3.to(p_o.dtype.element_ty), mask=mask_bv)
    tl.store(p_o_4, _o_4.to(p_o.dtype.element_ty), mask=mask_bv)
    tl.store(p_o_5, _o_5.to(p_o.dtype.element_ty), mask=mask_bv)
    tl.store(p_o_6, _o_6.to(p_o.dtype.element_ty), mask=mask_bv)
    tl.store(p_o_7, _o_7.to(p_o.dtype.element_ty), mask=mask_bv)

    # If later iterations need post-loop values/pointers, update base variables at the end
    h = h_8

    if STORE_FINAL_STATE:
        p_final_s = final_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[None, :]) * \
            DV + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_final_s, h.to(p_final_s.dtype.element_ty), mask=mask_kv)

@triton.jit
def fused_recurrent_retention_bwd_kernel(
    q, k, v, do, dq, dk, dv, initial_state,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    B, H, T, scale,
    BK: tl.constexpr, BV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr
):
    # Kernel logic
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)

    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    mask_bk = i_k * BK + tl.arange(0, BK) < DK
    mask_bv = i_v * BV + tl.arange(0, BV) < DV

    h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        mask_kv = mask_bk[:, None] & mask_bv[None, :]
        p_init_s = initial_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[:, None]) * \
            DV + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)

    # Initialization for first loop (forward pass for dq)
    h_0 = h

    # Precompute pointers needed for each iteration
    p_k_0, p_v_0, p_do_0, p_dq_0 = p_k, p_v, p_do, p_dq

    p_k_1 = p_k_0 + DK
    p_v_1 = p_v_0 + DV
    p_do_1 = p_do_0 + DV
    p_dq_1 = p_dq_0 + DK

    p_k_2 = p_k_1 + DK
    p_v_2 = p_v_1 + DV
    p_do_2 = p_do_1 + DV
    p_dq_2 = p_dq_1 + DK

    p_k_3 = p_k_2 + DK
    p_v_3 = p_v_2 + DV
    p_do_3 = p_do_2 + DV
    p_dq_3 = p_dq_2 + DK

    p_k_4 = p_k_3 + DK
    p_v_4 = p_v_3 + DV
    p_do_4 = p_do_3 + DV
    p_dq_4 = p_dq_3 + DK

    p_k_5 = p_k_4 + DK
    p_v_5 = p_v_4 + DV
    p_do_5 = p_do_4 + DV
    p_dq_5 = p_dq_4 + DK

    p_k_6 = p_k_5 + DK
    p_v_6 = p_v_5 + DV
    p_do_6 = p_do_5 + DV
    p_dq_6 = p_dq_5 + DK

    p_k_7 = p_k_6 + DK
    p_v_7 = p_v_6 + DV
    p_do_7 = p_do_6 + DV
    p_dq_7 = p_dq_6 + DK

    # =========================
    # Pull all tl.load as early as possible (first loop)
    # =========================
    _k_0 = tl.load(p_k_0, mask=mask_bk, other=0).to(tl.float32)
    _v_0 = tl.load(p_v_0, mask=mask_bv, other=0).to(tl.float32)
    _do_0 = tl.load(p_do_0, mask=mask_bv, other=0).to(tl.float32)

    _k_1 = tl.load(p_k_1, mask=mask_bk, other=0).to(tl.float32)
    _v_1 = tl.load(p_v_1, mask=mask_bv, other=0).to(tl.float32)
    _do_1 = tl.load(p_do_1, mask=mask_bv, other=0).to(tl.float32)

    _k_2 = tl.load(p_k_2, mask=mask_bk, other=0).to(tl.float32)
    _v_2 = tl.load(p_v_2, mask=mask_bv, other=0).to(tl.float32)
    _do_2 = tl.load(p_do_2, mask=mask_bv, other=0).to(tl.float32)

    _k_3 = tl.load(p_k_3, mask=mask_bk, other=0).to(tl.float32)
    _v_3 = tl.load(p_v_3, mask=mask_bv, other=0).to(tl.float32)
    _do_3 = tl.load(p_do_3, mask=mask_bv, other=0).to(tl.float32)

    _k_4 = tl.load(p_k_4, mask=mask_bk, other=0).to(tl.float32)
    _v_4 = tl.load(p_v_4, mask=mask_bv, other=0).to(tl.float32)
    _do_4 = tl.load(p_do_4, mask=mask_bv, other=0).to(tl.float32)

    _k_5 = tl.load(p_k_5, mask=mask_bk, other=0).to(tl.float32)
    _v_5 = tl.load(p_v_5, mask=mask_bv, other=0).to(tl.float32)
    _do_5 = tl.load(p_do_5, mask=mask_bv, other=0).to(tl.float32)

    _k_6 = tl.load(p_k_6, mask=mask_bk, other=0).to(tl.float32)
    _v_6 = tl.load(p_v_6, mask=mask_bv, other=0).to(tl.float32)
    _do_6 = tl.load(p_do_6, mask=mask_bv, other=0).to(tl.float32)

    _k_7 = tl.load(p_k_7, mask=mask_bk, other=0).to(tl.float32)
    _v_7 = tl.load(p_v_7, mask=mask_bv, other=0).to(tl.float32)
    _do_7 = tl.load(p_do_7, mask=mask_bv, other=0).to(tl.float32)

    # =========================
    # Iterative computation (first loop - computing dq)
    # =========================
    # iter 0
    h_1 = b_b * h_0 + _k_0[:, None] * _v_0[None, :]
    _d_q_0 = h_1 * _do_0[None, :]
    d_q_0 = tl.sum(_d_q_0, axis=1) * scale

    # iter 1
    h_2 = b_b * h_1 + _k_1[:, None] * _v_1[None, :]
    _d_q_1 = h_2 * _do_1[None, :]
    d_q_1 = tl.sum(_d_q_1, axis=1) * scale

    # iter 2
    h_3 = b_b * h_2 + _k_2[:, None] * _v_2[None, :]
    _d_q_2 = h_3 * _do_2[None, :]
    d_q_2 = tl.sum(_d_q_2, axis=1) * scale

    # iter 3
    h_4 = b_b * h_3 + _k_3[:, None] * _v_3[None, :]
    _d_q_3 = h_4 * _do_3[None, :]
    d_q_3 = tl.sum(_d_q_3, axis=1) * scale

    # iter 4
    h_5 = b_b * h_4 + _k_4[:, None] * _v_4[None, :]
    _d_q_4 = h_5 * _do_4[None, :]
    d_q_4 = tl.sum(_d_q_4, axis=1) * scale

    # iter 5
    h_6 = b_b * h_5 + _k_5[:, None] * _v_5[None, :]
    _d_q_5 = h_6 * _do_5[None, :]
    d_q_5 = tl.sum(_d_q_5, axis=1) * scale

    # iter 6
    h_7 = b_b * h_6 + _k_6[:, None] * _v_6[None, :]
    _d_q_6 = h_7 * _do_6[None, :]
    d_q_6 = tl.sum(_d_q_6, axis=1) * scale

    # iter 7
    h_8 = b_b * h_7 + _k_7[:, None] * _v_7[None, :]
    _d_q_7 = h_8 * _do_7[None, :]
    d_q_7 = tl.sum(_d_q_7, axis=1) * scale

    # =========================
    # Defer all tl.store until the end (first loop)
    # =========================
    tl.store(p_dq_0, d_q_0.to(p_dq.dtype.element_ty), mask=mask_bk)
    tl.store(p_dq_1, d_q_1.to(p_dq.dtype.element_ty), mask=mask_bk)
    tl.store(p_dq_2, d_q_2.to(p_dq.dtype.element_ty), mask=mask_bk)
    tl.store(p_dq_3, d_q_3.to(p_dq.dtype.element_ty), mask=mask_bk)
    tl.store(p_dq_4, d_q_4.to(p_dq.dtype.element_ty), mask=mask_bk)
    tl.store(p_dq_5, d_q_5.to(p_dq.dtype.element_ty), mask=mask_bk)
    tl.store(p_dq_6, d_q_6.to(p_dq.dtype.element_ty), mask=mask_bk)
    tl.store(p_dq_7, d_q_7.to(p_dq.dtype.element_ty), mask=mask_bk)

    tl.debug_barrier()

    # Second loop (backward pass for dk, dv)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * DV
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * DV
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * \
        BK + tl.arange(0, BK) + (T - 1) * DK
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * \
        BV + tl.arange(0, BV) + (T - 1) * DV
    d_h = tl.zeros([BK, BV], dtype=tl.float32)

    # Initialization for second loop
    d_h_0 = d_h

    # Precompute pointers needed for each iteration (reverse traversal)
    p_do_0, p_q_0, p_k_0, p_v_0, p_dk_0, p_dv_0 = p_do, p_q, p_k, p_v, p_dk, p_dv

    p_do_1 = p_do_0 - DV
    p_q_1 = p_q_0 - DK
    p_k_1 = p_k_0 - DK
    p_v_1 = p_v_0 - DV
    p_dk_1 = p_dk_0 - DK
    p_dv_1 = p_dv_0 - DV

    p_do_2 = p_do_1 - DV
    p_q_2 = p_q_1 - DK
    p_k_2 = p_k_1 - DK
    p_v_2 = p_v_1 - DV
    p_dk_2 = p_dk_1 - DK
    p_dv_2 = p_dv_1 - DV

    p_do_3 = p_do_2 - DV
    p_q_3 = p_q_2 - DK
    p_k_3 = p_k_2 - DK
    p_v_3 = p_v_2 - DV
    p_dk_3 = p_dk_2 - DK
    p_dv_3 = p_dv_2 - DV

    p_do_4 = p_do_3 - DV
    p_q_4 = p_q_3 - DK
    p_k_4 = p_k_3 - DK
    p_v_4 = p_v_3 - DV
    p_dk_4 = p_dk_3 - DK
    p_dv_4 = p_dv_3 - DV

    p_do_5 = p_do_4 - DV
    p_q_5 = p_q_4 - DK
    p_k_5 = p_k_4 - DK
    p_v_5 = p_v_4 - DV
    p_dk_5 = p_dk_4 - DK
    p_dv_5 = p_dv_4 - DV

    p_do_6 = p_do_5 - DV
    p_q_6 = p_q_5 - DK
    p_k_6 = p_k_5 - DK
    p_v_6 = p_v_5 - DV
    p_dk_6 = p_dk_5 - DK
    p_dv_6 = p_dv_5 - DV

    p_do_7 = p_do_6 - DV
    p_q_7 = p_q_6 - DK
    p_k_7 = p_k_6 - DK
    p_v_7 = p_v_6 - DV
    p_dk_7 = p_dk_6 - DK
    p_dv_7 = p_dv_6 - DV

    # =========================
    # Pull all tl.load as early as possible (second loop)
    # =========================
    _do_0 = tl.load(p_do_0, mask=mask_bv, other=0).to(tl.float32)
    _q_0 = tl.load(p_q_0, mask=mask_bk, other=0).to(tl.float32) * scale
    _k_0 = tl.load(p_k_0, mask=mask_bk, other=0).to(tl.float32)
    _v_0 = tl.load(p_v_0, mask=mask_bv, other=0).to(tl.float32)

    _do_1 = tl.load(p_do_1, mask=mask_bv, other=0).to(tl.float32)
    _q_1 = tl.load(p_q_1, mask=mask_bk, other=0).to(tl.float32) * scale
    _k_1 = tl.load(p_k_1, mask=mask_bk, other=0).to(tl.float32)
    _v_1 = tl.load(p_v_1, mask=mask_bv, other=0).to(tl.float32)

    _do_2 = tl.load(p_do_2, mask=mask_bv, other=0).to(tl.float32)
    _q_2 = tl.load(p_q_2, mask=mask_bk, other=0).to(tl.float32) * scale
    _k_2 = tl.load(p_k_2, mask=mask_bk, other=0).to(tl.float32)
    _v_2 = tl.load(p_v_2, mask=mask_bv, other=0).to(tl.float32)

    _do_3 = tl.load(p_do_3, mask=mask_bv, other=0).to(tl.float32)
    _q_3 = tl.load(p_q_3, mask=mask_bk, other=0).to(tl.float32) * scale
    _k_3 = tl.load(p_k_3, mask=mask_bk, other=0).to(tl.float32)
    _v_3 = tl.load(p_v_3, mask=mask_bv, other=0).to(tl.float32)

    _do_4 = tl.load(p_do_4, mask=mask_bv, other=0).to(tl.float32)
    _q_4 = tl.load(p_q_4, mask=mask_bk, other=0).to(tl.float32) * scale
    _k_4 = tl.load(p_k_4, mask=mask_bk, other=0).to(tl.float32)
    _v_4 = tl.load(p_v_4, mask=mask_bv, other=0).to(tl.float32)

    _do_5 = tl.load(p_do_5, mask=mask_bv, other=0).to(tl.float32)
    _q_5 = tl.load(p_q_5, mask=mask_bk, other=0).to(tl.float32) * scale
    _k_5 = tl.load(p_k_5, mask=mask_bk, other=0).to(tl.float32)
    _v_5 = tl.load(p_v_5, mask=mask_bv, other=0).to(tl.float32)

    _do_6 = tl.load(p_do_6, mask=mask_bv, other=0).to(tl.float32)
    _q_6 = tl.load(p_q_6, mask=mask_bk, other=0).to(tl.float32) * scale
    _k_6 = tl.load(p_k_6, mask=mask_bk, other=0).to(tl.float32)
    _v_6 = tl.load(p_v_6, mask=mask_bv, other=0).to(tl.float32)

    _do_7 = tl.load(p_do_7, mask=mask_bv, other=0).to(tl.float32)
    _q_7 = tl.load(p_q_7, mask=mask_bk, other=0).to(tl.float32) * scale
    _k_7 = tl.load(p_k_7, mask=mask_bk, other=0).to(tl.float32)
    _v_7 = tl.load(p_v_7, mask=mask_bv, other=0).to(tl.float32)

    # =========================
    # Iterative computation (second loop - computing dk, dv)
    # =========================
    # iter 0
    d_h_0 += _q_0[:, None] * _do_0[None, :]
    d_k_0 = tl.sum(d_h_0 * _v_0[None, :], axis=1)
    d_v_0 = tl.sum(d_h_0 * _k_0[:, None], axis=0)
    d_h_1 = d_h_0 * b_b

    # iter 1
    d_h_1 += _q_1[:, None] * _do_1[None, :]
    d_k_1 = tl.sum(d_h_1 * _v_1[None, :], axis=1)
    d_v_1 = tl.sum(d_h_1 * _k_1[:, None], axis=0)
    d_h_2 = d_h_1 * b_b

    # iter 2
    d_h_2 += _q_2[:, None] * _do_2[None, :]
    d_k_2 = tl.sum(d_h_2 * _v_2[None, :], axis=1)
    d_v_2 = tl.sum(d_h_2 * _k_2[:, None], axis=0)
    d_h_3 = d_h_2 * b_b

    # iter 3
    d_h_3 += _q_3[:, None] * _do_3[None, :]
    d_k_3 = tl.sum(d_h_3 * _v_3[None, :], axis=1)
    d_v_3 = tl.sum(d_h_3 * _k_3[:, None], axis=0)
    d_h_4 = d_h_3 * b_b

    # iter 4
    d_h_4 += _q_4[:, None] * _do_4[None, :]
    d_k_4 = tl.sum(d_h_4 * _v_4[None, :], axis=1)
    d_v_4 = tl.sum(d_h_4 * _k_4[:, None], axis=0)
    d_h_5 = d_h_4 * b_b

    # iter 5
    d_h_5 += _q_5[:, None] * _do_5[None, :]
    d_k_5 = tl.sum(d_h_5 * _v_5[None, :], axis=1)
    d_v_5 = tl.sum(d_h_5 * _k_5[:, None], axis=0)
    d_h_6 = d_h_5 * b_b

    # iter 6
    d_h_6 += _q_6[:, None] * _do_6[None, :]
    d_k_6 = tl.sum(d_h_6 * _v_6[None, :], axis=1)
    d_v_6 = tl.sum(d_h_6 * _k_6[:, None], axis=0)
    d_h_7 = d_h_6 * b_b

    # iter 7
    d_h_7 += _q_7[:, None] * _do_7[None, :]
    d_k_7 = tl.sum(d_h_7 * _v_7[None, :], axis=1)
    d_v_7 = tl.sum(d_h_7 * _k_7[:, None], axis=0)

    # =========================
    # Defer all tl.store until the end (second loop)
    # =========================
    tl.store(p_dk_0, d_k_0.to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv_0, d_v_0.to(p_dv.dtype.element_ty), mask=mask_bv)

    tl.store(p_dk_1, d_k_1.to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv_1, d_v_1.to(p_dv.dtype.element_ty), mask=mask_bv)

    tl.store(p_dk_2, d_k_2.to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv_2, d_v_2.to(p_dv.dtype.element_ty), mask=mask_bv)

    tl.store(p_dk_3, d_k_3.to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv_3, d_v_3.to(p_dv.dtype.element_ty), mask=mask_bv)

    tl.store(p_dk_4, d_k_4.to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv_4, d_v_4.to(p_dv.dtype.element_ty), mask=mask_bv)

    tl.store(p_dk_5, d_k_5.to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv_5, d_v_5.to(p_dv.dtype.element_ty), mask=mask_bv)

    tl.store(p_dk_6, d_k_6.to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv_6, d_v_6.to(p_dv.dtype.element_ty), mask=mask_bv)

    tl.store(p_dk_7, d_k_7.to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv_7, d_v_7.to(p_dv.dtype.element_ty), mask=mask_bv)

class FusedRecurrentRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, initial_state=None, output_final_state=False):
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]

        scale = d_head_qk ** -0.5
        BK, BV = min(d_head_qk, 32), min(d_head_v, 32)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 1

        o = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)

        if output_final_state:
            final_state = q.new_empty(batch_size, n_heads, d_head_qk, d_head_v)
        else:
            final_state = None

        grid = (NV, NK, batch_size * n_heads)
        fused_recurrent_retention_fwd_kernel[grid](
            q, k, v, o, initial_state, final_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, scale,
            DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None
        )

        o = o.sum(0)
        ctx.save_for_backward(q, k, v, initial_state)
        return o, final_state

    @staticmethod
    def backward(ctx, do, d_final_state=None):
        q, k, v, initial_state = ctx.saved_tensors
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        scale = d_head_qk ** -0.5

        BK, BV = min(d_head_qk, 32), min(d_head_v, 32)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 1

        dq = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk)
        dk = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk)
        dv = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)
        grid = (NV, NK, batch_size * n_heads)

        fused_recurrent_retention_bwd_kernel[grid](
            q, k, v, do, dq, dk, dv, initial_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, scale,
            DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages,
            USE_INITIAL_STATE=initial_state is not None
        )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        return dq, dk, dv, None, None

def fused_recurrent_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = FusedRecurrentRetentionFunction.apply(q, k, v, initial_state, output_final_state)
    return o, final_state




##################################################################################################################################################


import torch

# Extended test function with backward propagation
def test_fused_recurrent_retention_with_backward():
    test_results = {}

    # Test parameters
    batch_size = 2
    n_heads = 4
    seq_len = 8
    d_head_qk = 16
    d_head_v = 16

    # Create random input tensors
    q = torch.randn(batch_size, n_heads, seq_len, d_head_qk, dtype=torch.float32, requires_grad=True, device='cuda')
    k = torch.randn(batch_size, n_heads, seq_len, d_head_qk, dtype=torch.float32, requires_grad=True, device='cuda')
    v = torch.randn(batch_size, n_heads, seq_len, d_head_v, dtype=torch.float32, requires_grad=True, device='cuda')

    # Test 1: Without initial state and without final state
    initial_state = None
    output_final_state = False
    o, final_state = fused_recurrent_retention(q, k, v, initial_state=initial_state, output_final_state=output_final_state)
    loss = o.sum()  # Define a simple loss function
    loss.backward()  # Perform backward pass
    test_results['test_case_1'] = {
        "output_shape": o.shape,
        "final_state": final_state,
        "loss": loss.item(),
        "gradients_q": q.grad.norm().item(),
        "gradients_k": k.grad.norm().item(),
        "gradients_v": v.grad.norm().item()
    }

    # Reset gradients for the next test
    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()

    # Test 2: With initial state and without final state
    initial_state = torch.randn(batch_size, n_heads, d_head_qk, d_head_v, dtype=torch.float32, device='cuda', requires_grad=True)
    o, final_state = fused_recurrent_retention(q, k, v, initial_state=initial_state, output_final_state=False)
    loss = o.sum()
    loss.backward()
    test_results['test_case_2'] = {
        "output_shape": o.shape,
        "final_state": final_state,
        "loss": loss.item(),
        "gradients_q": q.grad.norm().item(),
        "gradients_k": k.grad.norm().item(),
        "gradients_v": v.grad.norm().item(),
    }

    # Reset gradients for the next test
    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()

    # Test 3: With initial state and with final state
    o, final_state = fused_recurrent_retention(q, k, v, initial_state=initial_state, output_final_state=True)
    loss = o.sum() + final_state.sum()
    loss.backward()
    test_results['test_case_3'] = {
        "output_shape": o.shape,
        "final_state_shape": final_state.shape,
        "loss": loss.item(),
        "gradients_q": q.grad.norm().item(),
        "gradients_k": k.grad.norm().item(),
        "gradients_v": v.grad.norm().item()
    }

    # Test 4: Without initial state and with final state
    initial_state = None
    output_final_state = True
    o, final_state = fused_recurrent_retention(q, k, v, initial_state=initial_state, output_final_state=output_final_state)
    loss = o.sum() + final_state.sum()
    loss.backward()
    test_results['test_case_4'] = {
        "output_shape": o.shape,
        "final_state_shape": final_state.shape,
        "loss": loss.item(),
        "gradients_q": q.grad.norm().item(),
        "gradients_k": k.grad.norm().item(),
        "gradients_v": v.grad.norm().item()
    }

    return test_results

# Run the test function with backward propagation
result_gold = test_fused_recurrent_retention_with_backward()
