
import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def fused_recurrent_fwd_kernel(
    q, k, v, beta, o, h0, ht, s_qk_h, s_vo_h, scale, B, H, T, K: tl.constexpr, V: tl.constexpr, 
    BK: tl.constexpr, BV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr, 
    IS_HEADWISE_BETA: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    if IS_HEADWISE_BETA:
        p_beta = beta + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    else:
        p_beta = beta + i_bh * T
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    mask_kv = mask_bk[None, :] & mask_bv[:, None]

    h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    # Initialization
    h_0 = h

    # Precompute pointers needed for each iteration
    p_k_0, p_v_0, p_q_0, p_o_0, p_beta_0 = p_k, p_v, p_q, p_o, p_beta

    p_k_1 = p_k_0 + K
    p_v_1 = p_v_0 + V
    p_q_1 = p_q_0 + K
    p_o_1 = p_o_0 + V
    p_beta_1 = p_beta_0 + (V if IS_HEADWISE_BETA else 1)

    p_k_2 = p_k_1 + K
    p_v_2 = p_v_1 + V
    p_q_2 = p_q_1 + K
    p_o_2 = p_o_1 + V
    p_beta_2 = p_beta_1 + (V if IS_HEADWISE_BETA else 1)

    p_k_3 = p_k_2 + K
    p_v_3 = p_v_2 + V
    p_q_3 = p_q_2 + K
    p_o_3 = p_o_2 + V
    p_beta_3 = p_beta_2 + (V if IS_HEADWISE_BETA else 1)

    p_k_4 = p_k_3 + K
    p_v_4 = p_v_3 + V
    p_q_4 = p_q_3 + K
    p_o_4 = p_o_3 + V
    p_beta_4 = p_beta_3 + (V if IS_HEADWISE_BETA else 1)

    p_k_5 = p_k_4 + K
    p_v_5 = p_v_4 + V
    p_q_5 = p_q_4 + K
    p_o_5 = p_o_4 + V
    p_beta_5 = p_beta_4 + (V if IS_HEADWISE_BETA else 1)

    p_k_6 = p_k_5 + K
    p_v_6 = p_v_5 + V
    p_q_6 = p_q_5 + K
    p_o_6 = p_o_5 + V
    p_beta_6 = p_beta_5 + (V if IS_HEADWISE_BETA else 1)

    p_k_7 = p_k_6 + K
    p_v_7 = p_v_6 + V
    p_q_7 = p_q_6 + K
    p_o_7 = p_o_6 + V
    p_beta_7 = p_beta_6 + (V if IS_HEADWISE_BETA else 1)

    # =========================
    # Pull all tl.load as early as possible
    # =========================
    b_k_0 = tl.load(p_k_0, mask=mask_bk, other=0).to(tl.float32)
    b_v_0 = tl.load(p_v_0, mask=mask_bv, other=0).to(tl.float32)
    b_q_0 = tl.load(p_q_0, mask=mask_bk, other=0).to(tl.float32) * scale
    if IS_HEADWISE_BETA:
        b_beta_0 = tl.load(p_beta_0, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_0 = tl.load(p_beta_0).to(tl.float32)

    b_k_1 = tl.load(p_k_1, mask=mask_bk, other=0).to(tl.float32)
    b_v_1 = tl.load(p_v_1, mask=mask_bv, other=0).to(tl.float32)
    b_q_1 = tl.load(p_q_1, mask=mask_bk, other=0).to(tl.float32) * scale
    if IS_HEADWISE_BETA:
        b_beta_1 = tl.load(p_beta_1, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_1 = tl.load(p_beta_1).to(tl.float32)

    b_k_2 = tl.load(p_k_2, mask=mask_bk, other=0).to(tl.float32)
    b_v_2 = tl.load(p_v_2, mask=mask_bv, other=0).to(tl.float32)
    b_q_2 = tl.load(p_q_2, mask=mask_bk, other=0).to(tl.float32) * scale
    if IS_HEADWISE_BETA:
        b_beta_2 = tl.load(p_beta_2, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_2 = tl.load(p_beta_2).to(tl.float32)

    b_k_3 = tl.load(p_k_3, mask=mask_bk, other=0).to(tl.float32)
    b_v_3 = tl.load(p_v_3, mask=mask_bv, other=0).to(tl.float32)
    b_q_3 = tl.load(p_q_3, mask=mask_bk, other=0).to(tl.float32) * scale
    if IS_HEADWISE_BETA:
        b_beta_3 = tl.load(p_beta_3, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_3 = tl.load(p_beta_3).to(tl.float32)

    b_k_4 = tl.load(p_k_4, mask=mask_bk, other=0).to(tl.float32)
    b_v_4 = tl.load(p_v_4, mask=mask_bv, other=0).to(tl.float32)
    b_q_4 = tl.load(p_q_4, mask=mask_bk, other=0).to(tl.float32) * scale
    if IS_HEADWISE_BETA:
        b_beta_4 = tl.load(p_beta_4, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_4 = tl.load(p_beta_4).to(tl.float32)

    b_k_5 = tl.load(p_k_5, mask=mask_bk, other=0).to(tl.float32)
    b_v_5 = tl.load(p_v_5, mask=mask_bv, other=0).to(tl.float32)
    b_q_5 = tl.load(p_q_5, mask=mask_bk, other=0).to(tl.float32) * scale
    if IS_HEADWISE_BETA:
        b_beta_5 = tl.load(p_beta_5, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_5 = tl.load(p_beta_5).to(tl.float32)

    b_k_6 = tl.load(p_k_6, mask=mask_bk, other=0).to(tl.float32)
    b_v_6 = tl.load(p_v_6, mask=mask_bv, other=0).to(tl.float32)
    b_q_6 = tl.load(p_q_6, mask=mask_bk, other=0).to(tl.float32) * scale
    if IS_HEADWISE_BETA:
        b_beta_6 = tl.load(p_beta_6, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_6 = tl.load(p_beta_6).to(tl.float32)

    b_k_7 = tl.load(p_k_7, mask=mask_bk, other=0).to(tl.float32)
    b_v_7 = tl.load(p_v_7, mask=mask_bv, other=0).to(tl.float32)
    b_q_7 = tl.load(p_q_7, mask=mask_bk, other=0).to(tl.float32) * scale
    if IS_HEADWISE_BETA:
        b_beta_7 = tl.load(p_beta_7, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_7 = tl.load(p_beta_7).to(tl.float32)

    # =========================
    # Iterative computation (no more loads), arithmetic only
    # =========================
    # iter 0
    _v_minus_0 = tl.sum(h_0 * b_k_0[None, :], axis=1)
    b_v_0 -= _v_minus_0
    b_v_0_store = b_v_0  # For writing back to memory
    b_v_0 *= b_beta_0
    h_1 = h_0 + b_k_0[None, :] * b_v_0[:, None]
    _o_0 = h_1 * b_q_0[None, :]
    _o_0 = tl.sum(_o_0, axis=1)

    # iter 1
    _v_minus_1 = tl.sum(h_1 * b_k_1[None, :], axis=1)
    b_v_1 -= _v_minus_1
    b_v_1_store = b_v_1
    b_v_1 *= b_beta_1
    h_2 = h_1 + b_k_1[None, :] * b_v_1[:, None]
    _o_1 = h_2 * b_q_1[None, :]
    _o_1 = tl.sum(_o_1, axis=1)

    # iter 2
    _v_minus_2 = tl.sum(h_2 * b_k_2[None, :], axis=1)
    b_v_2 -= _v_minus_2
    b_v_2_store = b_v_2
    b_v_2 *= b_beta_2
    h_3 = h_2 + b_k_2[None, :] * b_v_2[:, None]
    _o_2 = h_3 * b_q_2[None, :]
    _o_2 = tl.sum(_o_2, axis=1)

    # iter 3
    _v_minus_3 = tl.sum(h_3 * b_k_3[None, :], axis=1)
    b_v_3 -= _v_minus_3
    b_v_3_store = b_v_3
    b_v_3 *= b_beta_3
    h_4 = h_3 + b_k_3[None, :] * b_v_3[:, None]
    _o_3 = h_4 * b_q_3[None, :]
    _o_3 = tl.sum(_o_3, axis=1)

    # iter 4
    _v_minus_4 = tl.sum(h_4 * b_k_4[None, :], axis=1)
    b_v_4 -= _v_minus_4
    b_v_4_store = b_v_4
    b_v_4 *= b_beta_4
    h_5 = h_4 + b_k_4[None, :] * b_v_4[:, None]
    _o_4 = h_5 * b_q_4[None, :]
    _o_4 = tl.sum(_o_4, axis=1)

    # iter 5
    _v_minus_5 = tl.sum(h_5 * b_k_5[None, :], axis=1)
    b_v_5 -= _v_minus_5
    b_v_5_store = b_v_5
    b_v_5 *= b_beta_5
    h_6 = h_5 + b_k_5[None, :] * b_v_5[:, None]
    _o_5 = h_6 * b_q_5[None, :]
    _o_5 = tl.sum(_o_5, axis=1)

    # iter 6
    _v_minus_6 = tl.sum(h_6 * b_k_6[None, :], axis=1)
    b_v_6 -= _v_minus_6
    b_v_6_store = b_v_6
    b_v_6 *= b_beta_6
    h_7 = h_6 + b_k_6[None, :] * b_v_6[:, None]
    _o_6 = h_7 * b_q_6[None, :]
    _o_6 = tl.sum(_o_6, axis=1)

    # iter 7
    _v_minus_7 = tl.sum(h_7 * b_k_7[None, :], axis=1)
    b_v_7 -= _v_minus_7
    b_v_7_store = b_v_7
    b_v_7 *= b_beta_7
    h_8 = h_7 + b_k_7[None, :] * b_v_7[:, None]
    _o_7 = h_8 * b_q_7[None, :]
    _o_7 = tl.sum(_o_7, axis=1)

    # =========================
    # Defer all tl.store until the end
    # =========================
    tl.store(p_v_0, b_v_0_store.to(p_v.dtype.element_ty), mask=mask_bv)
    tl.store(p_o_0, _o_0.to(p_o.dtype.element_ty), mask=mask_bv)

    tl.store(p_v_1, b_v_1_store.to(p_v.dtype.element_ty), mask=mask_bv)
    tl.store(p_o_1, _o_1.to(p_o.dtype.element_ty), mask=mask_bv)

    tl.store(p_v_2, b_v_2_store.to(p_v.dtype.element_ty), mask=mask_bv)
    tl.store(p_o_2, _o_2.to(p_o.dtype.element_ty), mask=mask_bv)

    tl.store(p_v_3, b_v_3_store.to(p_v.dtype.element_ty), mask=mask_bv)
    tl.store(p_o_3, _o_3.to(p_o.dtype.element_ty), mask=mask_bv)

    tl.store(p_v_4, b_v_4_store.to(p_v.dtype.element_ty), mask=mask_bv)
    tl.store(p_o_4, _o_4.to(p_o.dtype.element_ty), mask=mask_bv)

    tl.store(p_v_5, b_v_5_store.to(p_v.dtype.element_ty), mask=mask_bv)
    tl.store(p_o_5, _o_5.to(p_o.dtype.element_ty), mask=mask_bv)

    tl.store(p_v_6, b_v_6_store.to(p_v.dtype.element_ty), mask=mask_bv)
    tl.store(p_o_6, _o_6.to(p_o.dtype.element_ty), mask=mask_bv)

    tl.store(p_v_7, b_v_7_store.to(p_v.dtype.element_ty), mask=mask_bv)
    tl.store(p_o_7, _o_7.to(p_o.dtype.element_ty), mask=mask_bv)

    # If later iterations need post-loop values/pointers, update base variables at the end
    h = h_8

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, h.to(p_ht.dtype.element_ty), mask=mask_kv)

@triton.jit
def fused_recurrent_bwd_kernel(
    q, k, v, beta, dht, dh0, do, dq, dk, dv, dbeta, h0, s_qk_h, s_vo_h, NK, scale, B, H, T, 
    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, 
    IS_HEADWISE_BETA: tl.constexpr, USE_DH0: tl.constexpr, USE_DHT: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    if IS_HEADWISE_BETA:
        p_beta = beta + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    else:
        p_beta = beta + i_bh * T + T - 1

    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    if IS_HEADWISE_BETA:
        p_dbeta = dbeta + (i_bh + i_k * B * H + i_v * B * H * NK) * s_vo_h + tl.arange(0, BV) + (T - 1) * V
    else:
        p_dbeta = dbeta + (i_bh + i_v * B * H) * T + T - 1
    d_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_DHT:
        p_ht = dht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        d_h += tl.load(p_ht, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32)

    # Initialization
    d_h_0 = d_h

    # Precompute pointers needed for each iteration (reverse traversal)
    p_q_0, p_k_0, p_v_0, p_do_0, p_dk_0, p_dv_0, p_dbeta_0, p_beta_0 = p_q, p_k, p_v, p_do, p_dk, p_dv, p_dbeta, p_beta

    p_q_1 = p_q_0 - K
    p_k_1 = p_k_0 - K
    p_v_1 = p_v_0 - V
    p_do_1 = p_do_0 - V
    p_dk_1 = p_dk_0 - K
    p_dv_1 = p_dv_0 - V
    p_dbeta_1 = p_dbeta_0 - (V if IS_HEADWISE_BETA else 1)
    p_beta_1 = p_beta_0 - (V if IS_HEADWISE_BETA else 1)

    p_q_2 = p_q_1 - K
    p_k_2 = p_k_1 - K
    p_v_2 = p_v_1 - V
    p_do_2 = p_do_1 - V
    p_dk_2 = p_dk_1 - K
    p_dv_2 = p_dv_1 - V
    p_dbeta_2 = p_dbeta_1 - (V if IS_HEADWISE_BETA else 1)
    p_beta_2 = p_beta_1 - (V if IS_HEADWISE_BETA else 1)

    p_q_3 = p_q_2 - K
    p_k_3 = p_k_2 - K
    p_v_3 = p_v_2 - V
    p_do_3 = p_do_2 - V
    p_dk_3 = p_dk_2 - K
    p_dv_3 = p_dv_2 - V
    p_dbeta_3 = p_dbeta_2 - (V if IS_HEADWISE_BETA else 1)
    p_beta_3 = p_beta_2 - (V if IS_HEADWISE_BETA else 1)

    p_q_4 = p_q_3 - K
    p_k_4 = p_k_3 - K
    p_v_4 = p_v_3 - V
    p_do_4 = p_do_3 - V
    p_dk_4 = p_dk_3 - K
    p_dv_4 = p_dv_3 - V
    p_dbeta_4 = p_dbeta_3 - (V if IS_HEADWISE_BETA else 1)
    p_beta_4 = p_beta_3 - (V if IS_HEADWISE_BETA else 1)

    p_q_5 = p_q_4 - K
    p_k_5 = p_k_4 - K
    p_v_5 = p_v_4 - V
    p_do_5 = p_do_4 - V
    p_dk_5 = p_dk_4 - K
    p_dv_5 = p_dv_4 - V
    p_dbeta_5 = p_dbeta_4 - (V if IS_HEADWISE_BETA else 1)
    p_beta_5 = p_beta_4 - (V if IS_HEADWISE_BETA else 1)

    p_q_6 = p_q_5 - K
    p_k_6 = p_k_5 - K
    p_v_6 = p_v_5 - V
    p_do_6 = p_do_5 - V
    p_dk_6 = p_dk_5 - K
    p_dv_6 = p_dv_5 - V
    p_dbeta_6 = p_dbeta_5 - (V if IS_HEADWISE_BETA else 1)
    p_beta_6 = p_beta_5 - (V if IS_HEADWISE_BETA else 1)

    p_q_7 = p_q_6 - K
    p_k_7 = p_k_6 - K
    p_v_7 = p_v_6 - V
    p_do_7 = p_do_6 - V
    p_dk_7 = p_dk_6 - K
    p_dv_7 = p_dv_6 - V
    p_dbeta_7 = p_dbeta_6 - (V if IS_HEADWISE_BETA else 1)
    p_beta_7 = p_beta_6 - (V if IS_HEADWISE_BETA else 1)

    # =========================
    # Pull all tl.load as early as possible
    # =========================
    b_q_0 = tl.load(p_q_0, mask=mask_bk, other=0).to(tl.float32) * scale
    b_k_0 = tl.load(p_k_0, mask=mask_bk, other=0).to(tl.float32)
    b_v_0 = tl.load(p_v_0, mask=mask_bv, other=0).to(tl.float32)
    b_do_0 = tl.load(p_do_0, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA:
        b_beta_0 = tl.load(p_beta_0, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_0 = tl.load(p_beta_0).to(tl.float32)

    b_q_1 = tl.load(p_q_1, mask=mask_bk, other=0).to(tl.float32) * scale
    b_k_1 = tl.load(p_k_1, mask=mask_bk, other=0).to(tl.float32)
    b_v_1 = tl.load(p_v_1, mask=mask_bv, other=0).to(tl.float32)
    b_do_1 = tl.load(p_do_1, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA:
        b_beta_1 = tl.load(p_beta_1, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_1 = tl.load(p_beta_1).to(tl.float32)

    b_q_2 = tl.load(p_q_2, mask=mask_bk, other=0).to(tl.float32) * scale
    b_k_2 = tl.load(p_k_2, mask=mask_bk, other=0).to(tl.float32)
    b_v_2 = tl.load(p_v_2, mask=mask_bv, other=0).to(tl.float32)
    b_do_2 = tl.load(p_do_2, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA:
        b_beta_2 = tl.load(p_beta_2, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_2 = tl.load(p_beta_2).to(tl.float32)

    b_q_3 = tl.load(p_q_3, mask=mask_bk, other=0).to(tl.float32) * scale
    b_k_3 = tl.load(p_k_3, mask=mask_bk, other=0).to(tl.float32)
    b_v_3 = tl.load(p_v_3, mask=mask_bv, other=0).to(tl.float32)
    b_do_3 = tl.load(p_do_3, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA:
        b_beta_3 = tl.load(p_beta_3, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_3 = tl.load(p_beta_3).to(tl.float32)

    b_q_4 = tl.load(p_q_4, mask=mask_bk, other=0).to(tl.float32) * scale
    b_k_4 = tl.load(p_k_4, mask=mask_bk, other=0).to(tl.float32)
    b_v_4 = tl.load(p_v_4, mask=mask_bv, other=0).to(tl.float32)
    b_do_4 = tl.load(p_do_4, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA:
        b_beta_4 = tl.load(p_beta_4, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_4 = tl.load(p_beta_4).to(tl.float32)

    b_q_5 = tl.load(p_q_5, mask=mask_bk, other=0).to(tl.float32) * scale
    b_k_5 = tl.load(p_k_5, mask=mask_bk, other=0).to(tl.float32)
    b_v_5 = tl.load(p_v_5, mask=mask_bv, other=0).to(tl.float32)
    b_do_5 = tl.load(p_do_5, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA:
        b_beta_5 = tl.load(p_beta_5, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_5 = tl.load(p_beta_5).to(tl.float32)

    b_q_6 = tl.load(p_q_6, mask=mask_bk, other=0).to(tl.float32) * scale
    b_k_6 = tl.load(p_k_6, mask=mask_bk, other=0).to(tl.float32)
    b_v_6 = tl.load(p_v_6, mask=mask_bv, other=0).to(tl.float32)
    b_do_6 = tl.load(p_do_6, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA:
        b_beta_6 = tl.load(p_beta_6, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_6 = tl.load(p_beta_6).to(tl.float32)

    b_q_7 = tl.load(p_q_7, mask=mask_bk, other=0).to(tl.float32) * scale
    b_k_7 = tl.load(p_k_7, mask=mask_bk, other=0).to(tl.float32)
    b_v_7 = tl.load(p_v_7, mask=mask_bv, other=0).to(tl.float32)
    b_do_7 = tl.load(p_do_7, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA:
        b_beta_7 = tl.load(p_beta_7, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_7 = tl.load(p_beta_7).to(tl.float32)

    # =========================
    # Iterative computation (no more loads), arithmetic only
    # =========================
    # iter 0
    d_h_0 += b_q_0[:, None] * b_do_0[None, :]
    d_k_0 = tl.sum(d_h_0 * (b_v_0 * b_beta_0)[None, :], axis=1)
    d_v_0 = tl.sum(d_h_0 * b_k_0[:, None], axis=0)
    d_beta_0 = d_v_0 * b_v_0 if IS_HEADWISE_BETA else tl.sum(d_v_0 * b_v_0)
    d_v_0_store = d_v_0 * b_beta_0
    d_h_1 = d_h_0 - b_k_0[:, None] * d_v_0_store[None, :]

    # iter 1
    d_h_1 += b_q_1[:, None] * b_do_1[None, :]
    d_k_1 = tl.sum(d_h_1 * (b_v_1 * b_beta_1)[None, :], axis=1)
    d_v_1 = tl.sum(d_h_1 * b_k_1[:, None], axis=0)
    d_beta_1 = d_v_1 * b_v_1 if IS_HEADWISE_BETA else tl.sum(d_v_1 * b_v_1)
    d_v_1_store = d_v_1 * b_beta_1
    d_h_2 = d_h_1 - b_k_1[:, None] * d_v_1_store[None, :]

    # iter 2
    d_h_2 += b_q_2[:, None] * b_do_2[None, :]
    d_k_2 = tl.sum(d_h_2 * (b_v_2 * b_beta_2)[None, :], axis=1)
    d_v_2 = tl.sum(d_h_2 * b_k_2[:, None], axis=0)
    d_beta_2 = d_v_2 * b_v_2 if IS_HEADWISE_BETA else tl.sum(d_v_2 * b_v_2)
    d_v_2_store = d_v_2 * b_beta_2
    d_h_3 = d_h_2 - b_k_2[:, None] * d_v_2_store[None, :]

    # iter 3
    d_h_3 += b_q_3[:, None] * b_do_3[None, :]
    d_k_3 = tl.sum(d_h_3 * (b_v_3 * b_beta_3)[None, :], axis=1)
    d_v_3 = tl.sum(d_h_3 * b_k_3[:, None], axis=0)
    d_beta_3 = d_v_3 * b_v_3 if IS_HEADWISE_BETA else tl.sum(d_v_3 * b_v_3)
    d_v_3_store = d_v_3 * b_beta_3
    d_h_4 = d_h_3 - b_k_3[:, None] * d_v_3_store[None, :]

    # iter 4
    d_h_4 += b_q_4[:, None] * b_do_4[None, :]
    d_k_4 = tl.sum(d_h_4 * (b_v_4 * b_beta_4)[None, :], axis=1)
    d_v_4 = tl.sum(d_h_4 * b_k_4[:, None], axis=0)
    d_beta_4 = d_v_4 * b_v_4 if IS_HEADWISE_BETA else tl.sum(d_v_4 * b_v_4)
    d_v_4_store = d_v_4 * b_beta_4
    d_h_5 = d_h_4 - b_k_4[:, None] * d_v_4_store[None, :]

    # iter 5
    d_h_5 += b_q_5[:, None] * b_do_5[None, :]
    d_k_5 = tl.sum(d_h_5 * (b_v_5 * b_beta_5)[None, :], axis=1)
    d_v_5 = tl.sum(d_h_5 * b_k_5[:, None], axis=0)
    d_beta_5 = d_v_5 * b_v_5 if IS_HEADWISE_BETA else tl.sum(d_v_5 * b_v_5)
    d_v_5_store = d_v_5 * b_beta_5
    d_h_6 = d_h_5 - b_k_5[:, None] * d_v_5_store[None, :]

    # iter 6
    d_h_6 += b_q_6[:, None] * b_do_6[None, :]
    d_k_6 = tl.sum(d_h_6 * (b_v_6 * b_beta_6)[None, :], axis=1)
    d_v_6 = tl.sum(d_h_6 * b_k_6[:, None], axis=0)
    d_beta_6 = d_v_6 * b_v_6 if IS_HEADWISE_BETA else tl.sum(d_v_6 * b_v_6)
    d_v_6_store = d_v_6 * b_beta_6
    d_h_7 = d_h_6 - b_k_6[:, None] * d_v_6_store[None, :]

    # iter 7
    d_h_7 += b_q_7[:, None] * b_do_7[None, :]
    d_k_7 = tl.sum(d_h_7 * (b_v_7 * b_beta_7)[None, :], axis=1)
    d_v_7 = tl.sum(d_h_7 * b_k_7[:, None], axis=0)
    d_beta_7 = d_v_7 * b_v_7 if IS_HEADWISE_BETA else tl.sum(d_v_7 * b_v_7)
    d_v_7_store = d_v_7 * b_beta_7
    d_h_8 = d_h_7 - b_k_7[:, None] * d_v_7_store[None, :]

    # =========================
    # Defer all tl.store until the end
    # =========================
    tl.store(p_dk_0, d_k_0.to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv_0, d_v_0_store.to(p_dv.dtype.element_ty), mask=mask_bv)
    if IS_HEADWISE_BETA:
        tl.store(p_dbeta_0, d_beta_0.to(p_dbeta.dtype.element_ty), mask=mask_bv)
    else:
        tl.store(p_dbeta_0, d_beta_0.to(p_dbeta.dtype.element_ty))

    tl.store(p_dk_1, d_k_1.to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv_1, d_v_1_store.to(p_dv.dtype.element_ty), mask=mask_bv)
    if IS_HEADWISE_BETA:
        tl.store(p_dbeta_1, d_beta_1.to(p_dbeta.dtype.element_ty), mask=mask_bv)
    else:
        tl.store(p_dbeta_1, d_beta_1.to(p_dbeta.dtype.element_ty))

    tl.store(p_dk_2, d_k_2.to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv_2, d_v_2_store.to(p_dv.dtype.element_ty), mask=mask_bv)
    if IS_HEADWISE_BETA:
        tl.store(p_dbeta_2, d_beta_2.to(p_dbeta.dtype.element_ty), mask=mask_bv)
    else:
        tl.store(p_dbeta_2, d_beta_2.to(p_dbeta.dtype.element_ty))

    tl.store(p_dk_3, d_k_3.to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv_3, d_v_3_store.to(p_dv.dtype.element_ty), mask=mask_bv)
    if IS_HEADWISE_BETA:
        tl.store(p_dbeta_3, d_beta_3.to(p_dbeta.dtype.element_ty), mask=mask_bv)
    else:
        tl.store(p_dbeta_3, d_beta_3.to(p_dbeta.dtype.element_ty))

    tl.store(p_dk_4, d_k_4.to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv_4, d_v_4_store.to(p_dv.dtype.element_ty), mask=mask_bv)
    if IS_HEADWISE_BETA:
        tl.store(p_dbeta_4, d_beta_4.to(p_dbeta.dtype.element_ty), mask=mask_bv)
    else:
        tl.store(p_dbeta_4, d_beta_4.to(p_dbeta.dtype.element_ty))

    tl.store(p_dk_5, d_k_5.to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv_5, d_v_5_store.to(p_dv.dtype.element_ty), mask=mask_bv)
    if IS_HEADWISE_BETA:
        tl.store(p_dbeta_5, d_beta_5.to(p_dbeta.dtype.element_ty), mask=mask_bv)
    else:
        tl.store(p_dbeta_5, d_beta_5.to(p_dbeta.dtype.element_ty))

    tl.store(p_dk_6, d_k_6.to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv_6, d_v_6_store.to(p_dv.dtype.element_ty), mask=mask_bv)
    if IS_HEADWISE_BETA:
        tl.store(p_dbeta_6, d_beta_6.to(p_dbeta.dtype.element_ty), mask=mask_bv)
    else:
        tl.store(p_dbeta_6, d_beta_6.to(p_dbeta.dtype.element_ty))

    tl.store(p_dk_7, d_k_7.to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv_7, d_v_7_store.to(p_dv.dtype.element_ty), mask=mask_bv)
    if IS_HEADWISE_BETA:
        tl.store(p_dbeta_7, d_beta_7.to(p_dbeta.dtype.element_ty), mask=mask_bv)
    else:
        tl.store(p_dbeta_7, d_beta_7.to(p_dbeta.dtype.element_ty))

    # If later iterations need post-loop values/pointers, update base variables at the end
    d_h = d_h_8

    if USE_DH0:
        p_dh0 = dh0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, d_h.to(p_dh0.dtype.element_ty), mask=mask_bk[:, None] & mask_bv[None, :])

    tl.debug_barrier()

    h = tl.zeros([BK, BV], dtype=tl.float32)

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    if IS_HEADWISE_BETA:
        p_beta = beta + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    else:
        p_beta = beta + i_bh * T
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)

    if USE_INITIAL_STATE:
        mask_kv = mask_bk[:, None] & mask_bv[None, :]
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    # Initialization
    h_0 = h

    # Precompute pointers needed for each iteration
    p_k_0, p_v_0, p_do_0, p_dk_0, p_dv_0, p_dq_0, p_beta_0 = p_k, p_v, p_do, p_dk, p_dv, p_dq, p_beta

    p_k_1 = p_k_0 + K
    p_v_1 = p_v_0 + V
    p_do_1 = p_do_0 + V
    p_dk_1 = p_dk_0 + K
    p_dv_1 = p_dv_0 + V
    p_dq_1 = p_dq_0 + K
    p_beta_1 = p_beta_0 + (V if IS_HEADWISE_BETA else 1)

    p_k_2 = p_k_1 + K
    p_v_2 = p_v_1 + V
    p_do_2 = p_do_1 + V
    p_dk_2 = p_dk_1 + K
    p_dv_2 = p_dv_1 + V
    p_dq_2 = p_dq_1 + K
    p_beta_2 = p_beta_1 + (V if IS_HEADWISE_BETA else 1)

    p_k_3 = p_k_2 + K
    p_v_3 = p_v_2 + V
    p_do_3 = p_do_2 + V
    p_dk_3 = p_dk_2 + K
    p_dv_3 = p_dv_2 + V
    p_dq_3 = p_dq_2 + K
    p_beta_3 = p_beta_2 + (V if IS_HEADWISE_BETA else 1)

    p_k_4 = p_k_3 + K
    p_v_4 = p_v_3 + V
    p_do_4 = p_do_3 + V
    p_dk_4 = p_dk_3 + K
    p_dv_4 = p_dv_3 + V
    p_dq_4 = p_dq_3 + K
    p_beta_4 = p_beta_3 + (V if IS_HEADWISE_BETA else 1)

    p_k_5 = p_k_4 + K
    p_v_5 = p_v_4 + V
    p_do_5 = p_do_4 + V
    p_dk_5 = p_dk_4 + K
    p_dv_5 = p_dv_4 + V
    p_dq_5 = p_dq_4 + K
    p_beta_5 = p_beta_4 + (V if IS_HEADWISE_BETA else 1)

    p_k_6 = p_k_5 + K
    p_v_6 = p_v_5 + V
    p_do_6 = p_do_5 + V
    p_dk_6 = p_dk_5 + K
    p_dv_6 = p_dv_5 + V
    p_dq_6 = p_dq_5 + K
    p_beta_6 = p_beta_5 + (V if IS_HEADWISE_BETA else 1)

    p_k_7 = p_k_6 + K
    p_v_7 = p_v_6 + V
    p_do_7 = p_do_6 + V
    p_dk_7 = p_dk_6 + K
    p_dv_7 = p_dv_6 + V
    p_dq_7 = p_dq_6 + K
    p_beta_7 = p_beta_6 + (V if IS_HEADWISE_BETA else 1)

    # =========================
    # Pull all tl.load as early as possible
    # =========================
    d_k_0 = tl.load(p_dk_0, mask=mask_bk, other=0).to(tl.float32)
    d_v_0 = tl.load(p_dv_0, mask=mask_bv, other=0).to(tl.float32)
    b_k_0 = tl.load(p_k_0, mask=mask_bk, other=0).to(tl.float32)
    b_v_0 = tl.load(p_v_0, mask=mask_bv, other=0).to(tl.float32)
    b_do_0 = tl.load(p_do_0, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA:
        b_beta_0 = tl.load(p_beta_0, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_0 = tl.load(p_beta_0).to(tl.float32)

    d_k_1 = tl.load(p_dk_1, mask=mask_bk, other=0).to(tl.float32)
    d_v_1 = tl.load(p_dv_1, mask=mask_bv, other=0).to(tl.float32)
    b_k_1 = tl.load(p_k_1, mask=mask_bk, other=0).to(tl.float32)
    b_v_1 = tl.load(p_v_1, mask=mask_bv, other=0).to(tl.float32)
    b_do_1 = tl.load(p_do_1, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA:
        b_beta_1 = tl.load(p_beta_1, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_1 = tl.load(p_beta_1).to(tl.float32)

    d_k_2 = tl.load(p_dk_2, mask=mask_bk, other=0).to(tl.float32)
    d_v_2 = tl.load(p_dv_2, mask=mask_bv, other=0).to(tl.float32)
    b_k_2 = tl.load(p_k_2, mask=mask_bk, other=0).to(tl.float32)
    b_v_2 = tl.load(p_v_2, mask=mask_bv, other=0).to(tl.float32)
    b_do_2 = tl.load(p_do_2, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA:
        b_beta_2 = tl.load(p_beta_2, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_2 = tl.load(p_beta_2).to(tl.float32)

    d_k_3 = tl.load(p_dk_3, mask=mask_bk, other=0).to(tl.float32)
    d_v_3 = tl.load(p_dv_3, mask=mask_bv, other=0).to(tl.float32)
    b_k_3 = tl.load(p_k_3, mask=mask_bk, other=0).to(tl.float32)
    b_v_3 = tl.load(p_v_3, mask=mask_bv, other=0).to(tl.float32)
    b_do_3 = tl.load(p_do_3, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA:
        b_beta_3 = tl.load(p_beta_3, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_3 = tl.load(p_beta_3).to(tl.float32)

    d_k_4 = tl.load(p_dk_4, mask=mask_bk, other=0).to(tl.float32)
    d_v_4 = tl.load(p_dv_4, mask=mask_bv, other=0).to(tl.float32)
    b_k_4 = tl.load(p_k_4, mask=mask_bk, other=0).to(tl.float32)
    b_v_4 = tl.load(p_v_4, mask=mask_bv, other=0).to(tl.float32)
    b_do_4 = tl.load(p_do_4, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA:
        b_beta_4 = tl.load(p_beta_4, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_4 = tl.load(p_beta_4).to(tl.float32)

    d_k_5 = tl.load(p_dk_5, mask=mask_bk, other=0).to(tl.float32)
    d_v_5 = tl.load(p_dv_5, mask=mask_bv, other=0).to(tl.float32)
    b_k_5 = tl.load(p_k_5, mask=mask_bk, other=0).to(tl.float32)
    b_v_5 = tl.load(p_v_5, mask=mask_bv, other=0).to(tl.float32)
    b_do_5 = tl.load(p_do_5, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA:
        b_beta_5 = tl.load(p_beta_5, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_5 = tl.load(p_beta_5).to(tl.float32)

    d_k_6 = tl.load(p_dk_6, mask=mask_bk, other=0).to(tl.float32)
    d_v_6 = tl.load(p_dv_6, mask=mask_bv, other=0).to(tl.float32)
    b_k_6 = tl.load(p_k_6, mask=mask_bk, other=0).to(tl.float32)
    b_v_6 = tl.load(p_v_6, mask=mask_bv, other=0).to(tl.float32)
    b_do_6 = tl.load(p_do_6, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA:
        b_beta_6 = tl.load(p_beta_6, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_6 = tl.load(p_beta_6).to(tl.float32)

    d_k_7 = tl.load(p_dk_7, mask=mask_bk, other=0).to(tl.float32)
    d_v_7 = tl.load(p_dv_7, mask=mask_bv, other=0).to(tl.float32)
    b_k_7 = tl.load(p_k_7, mask=mask_bk, other=0).to(tl.float32)
    b_v_7 = tl.load(p_v_7, mask=mask_bv, other=0).to(tl.float32)
    b_do_7 = tl.load(p_do_7, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA:
        b_beta_7 = tl.load(p_beta_7, mask=mask_bv, other=0).to(tl.float32)
    else:
        b_beta_7 = tl.load(p_beta_7).to(tl.float32)

    # =========================
    # Iterative computation (no more loads), arithmetic only
    # =========================
    # iter 0
    d_k_0 -= tl.sum(d_v_0[None, :] * h_0, axis=1)
    d_k_0_store = d_k_0
    b_v_0 *= b_beta_0
    h_1 = h_0 + b_k_0[:, None] * b_v_0[None, :]
    _d_q_0 = h_1 * b_do_0[None, :]
    d_q_0 = tl.sum(_d_q_0, axis=1) * scale

    # iter 1
    d_k_1 -= tl.sum(d_v_1[None, :] * h_1, axis=1)
    d_k_1_store = d_k_1
    b_v_1 *= b_beta_1
    h_2 = h_1 + b_k_1[:, None] * b_v_1[None, :]
    _d_q_1 = h_2 * b_do_1[None, :]
    d_q_1 = tl.sum(_d_q_1, axis=1) * scale

    # iter 2
    d_k_2 -= tl.sum(d_v_2[None, :] * h_2, axis=1)
    d_k_2_store = d_k_2
    b_v_2 *= b_beta_2
    h_3 = h_2 + b_k_2[:, None] * b_v_2[None, :]
    _d_q_2 = h_3 * b_do_2[None, :]
    d_q_2 = tl.sum(_d_q_2, axis=1) * scale

    # iter 3
    d_k_3 -= tl.sum(d_v_3[None, :] * h_3, axis=1)
    d_k_3_store = d_k_3
    b_v_3 *= b_beta_3
    h_4 = h_3 + b_k_3[:, None] * b_v_3[None, :]
    _d_q_3 = h_4 * b_do_3[None, :]
    d_q_3 = tl.sum(_d_q_3, axis=1) * scale

    # iter 4
    d_k_4 -= tl.sum(d_v_4[None, :] * h_4, axis=1)
    d_k_4_store = d_k_4
    b_v_4 *= b_beta_4
    h_5 = h_4 + b_k_4[:, None] * b_v_4[None, :]
    _d_q_4 = h_5 * b_do_4[None, :]
    d_q_4 = tl.sum(_d_q_4, axis=1) * scale

    # iter 5
    d_k_5 -= tl.sum(d_v_5[None, :] * h_5, axis=1)
    d_k_5_store = d_k_5
    b_v_5 *= b_beta_5
    h_6 = h_5 + b_k_5[:, None] * b_v_5[None, :]
    _d_q_5 = h_6 * b_do_5[None, :]
    d_q_5 = tl.sum(_d_q_5, axis=1) * scale

    # iter 6
    d_k_6 -= tl.sum(d_v_6[None, :] * h_6, axis=1)
    d_k_6_store = d_k_6
    b_v_6 *= b_beta_6
    h_7 = h_6 + b_k_6[:, None] * b_v_6[None, :]
    _d_q_6 = h_7 * b_do_6[None, :]
    d_q_6 = tl.sum(_d_q_6, axis=1) * scale

    # iter 7
    d_k_7 -= tl.sum(d_v_7[None, :] * h_7, axis=1)
    d_k_7_store = d_k_7
    b_v_7 *= b_beta_7
    h_8 = h_7 + b_k_7[:, None] * b_v_7[None, :]
    _d_q_7 = h_8 * b_do_7[None, :]
    d_q_7 = tl.sum(_d_q_7, axis=1) * scale

    # =========================
    # Defer all tl.store until the end
    # =========================
    tl.store(p_dk_0, d_k_0_store.to(p_dk_0.dtype.element_ty), mask=mask_bk)
    tl.store(p_dq_0, d_q_0.to(p_dq_0.dtype.element_ty), mask=mask_bk)

    tl.store(p_dk_1, d_k_1_store.to(p_dk_1.dtype.element_ty), mask=mask_bk)
    tl.store(p_dq_1, d_q_1.to(p_dq_1.dtype.element_ty), mask=mask_bk)

    tl.store(p_dk_2, d_k_2_store.to(p_dk_2.dtype.element_ty), mask=mask_bk)
    tl.store(p_dq_2, d_q_2.to(p_dq_2.dtype.element_ty), mask=mask_bk)

    tl.store(p_dk_3, d_k_3_store.to(p_dk_3.dtype.element_ty), mask=mask_bk)
    tl.store(p_dq_3, d_q_3.to(p_dq_3.dtype.element_ty), mask=mask_bk)

    tl.store(p_dk_4, d_k_4_store.to(p_dk_4.dtype.element_ty), mask=mask_bk)
    tl.store(p_dq_4, d_q_4.to(p_dq_4.dtype.element_ty), mask=mask_bk)

    tl.store(p_dk_5, d_k_5_store.to(p_dk_5.dtype.element_ty), mask=mask_bk)
    tl.store(p_dq_5, d_q_5.to(p_dq_5.dtype.element_ty), mask=mask_bk)

    tl.store(p_dk_6, d_k_6_store.to(p_dk_6.dtype.element_ty), mask=mask_bk)
    tl.store(p_dq_6, d_q_6.to(p_dq_6.dtype.element_ty), mask=mask_bk)

    tl.store(p_dk_7, d_k_7_store.to(p_dk_7.dtype.element_ty), mask=mask_bk)
    tl.store(p_dq_7, d_q_7.to(p_dq_7.dtype.element_ty), mask=mask_bk)

    # If later iterations need post-loop values/pointers, update base variables at the end
    h = h_8

class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, beta, scale=None, initial_state=None, output_final_state=False):
        B, H, T, K, V = *q.shape, v.shape[-1]

        BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 1
        assert NK == 1, "NK > 1 is not supported yet"
        o = q.new_empty(NK, B, H, T, V)

        if output_final_state:
            final_state = q.new_empty(B, H, K, V, dtype=torch.float32)
        else:
            final_state = None

        grid = (NV, NK, B * H)
        fused_recurrent_fwd_kernel[grid](
            q, k, v, beta, o, initial_state, final_state,
            q.stride(1),
            v.stride(1),
            scale,
            B=B, H=H, T=T, K=K, V=V,
            BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None,
            IS_HEADWISE_BETA=beta.ndim == v.ndim,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        o = o.squeeze(0)
        ctx.save_for_backward(q, k, v, beta, initial_state)
        ctx.scale = scale
        return o, final_state

    @staticmethod
    def backward(ctx, do, dht):
        q, k, v, beta, initial_state = ctx.saved_tensors
        B, H, T, K, V = *q.shape, v.shape[-1]
        scale = ctx.scale
        BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        assert NK == 1, "NK > 1 is not supported yet"
        num_stages = 1
        num_warps = 2

        beta_vector = beta.ndim == v.ndim

        dq = q.new_empty(NV, B, H, T, K)
        dk = q.new_empty(NV, B, H, T, K)
        dv = q.new_empty(NK, B, H, T, V)
        if beta_vector:
            dbeta = q.new_empty(NV, NK, B, H, T, V)
        else:
            dbeta = q.new_empty(NV, B, H, T)
        grid = (NV, NK, B * H)

        if initial_state is not None and initial_state.requires_grad:
            dh0 = torch.empty_like(initial_state, dtype=torch.float32)
        else:
            dh0 = None

        fused_recurrent_bwd_kernel[grid](
            q, k, v, beta, dht, dh0, do, dq, dk, dv, dbeta, initial_state,
            q.stride(1),
            v.stride(1),
            NK, scale,
            B=B, H=H, T=T, K=K, V=V,
            BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            USE_DH0=dh0 is not None,
            USE_DHT=dht is not None,
            IS_HEADWISE_BETA=beta_vector,
            num_warps=num_warps,
            num_stages=num_stages
        )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        dbeta = dbeta.sum((0, 1)) if beta_vector else dbeta.sum(0)
        return dq.to(q), dk.to(k), dv.to(v), dbeta.to(beta), None, dh0, None

def fused_recurrent_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = FusedRecurrentFunction.apply(q, k, v, beta, scale, initial_state, output_final_state)
    return o, final_state




##################################################################################################################################################


import torch

def test_fused_recurrent_delta_rule_with_backward():
    # Define dimensions
    B, H, T, K, V = 2, 4, 8, 16, 32

    # Ensure inputs are leaf tensors with requires_grad=True
    q = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda', requires_grad=True)
    k = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda', requires_grad=True)
    v = torch.randn(B, H, T, V, dtype=torch.float32, device='cuda', requires_grad=True)
    beta_headwise = torch.randn(B, H, T, V, dtype=torch.float32, device='cuda', requires_grad=True)
    beta_non_headwise = torch.randn(B, H, T, dtype=torch.float32, device='cuda', requires_grad=True)
    initial_state = torch.randn(B, H, K, V, dtype=torch.float32, device='cuda', requires_grad=True)

    # Test 1: Headwise beta, with initial_state and final_state
    o, final_state = fused_recurrent_delta_rule(q, k, v, beta=beta_headwise, scale=0.1, initial_state=initial_state, output_final_state=True)

    loss = o.sum() + final_state.sum()
    loss.backward()

    result_1 = {
        "grad_q": q.grad.norm().item(),
        "grad_k": k.grad.norm().item(),
        "grad_v": v.grad.norm().item(),
        "grad_beta_headwise": beta_headwise.grad.norm().item(),
        "grad_initial_state": initial_state.grad.norm().item()
    }

    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()
    beta_headwise.grad.zero_()
    initial_state.grad.zero_()

    # Test 2: Non-headwise beta, with initial_state and final_state
    o, final_state = fused_recurrent_delta_rule(q, k, v, beta=beta_non_headwise, scale=0.1, initial_state=initial_state, output_final_state=True)

    loss = o.sum() + final_state.sum()
    loss.backward()

    result_2 = {
        "grad_q": q.grad.norm().item(),
        "grad_k": k.grad.norm().item(),
        "grad_v": v.grad.norm().item(),
        "grad_beta_non_headwise": beta_non_headwise.grad.norm().item(),
        "grad_initial_state": initial_state.grad.norm().item()
    }

    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()
    beta_non_headwise.grad.zero_()
    initial_state.grad.zero_()

    # Test 3: No initial state, with final state
    o, final_state = fused_recurrent_delta_rule(q, k, v, beta=beta_headwise, scale=0.1, initial_state=None, output_final_state=True)

    loss = o.sum() + final_state.sum()
    loss.backward()

    result_3 = {
        "grad_q": q.grad.norm().item(),
        "grad_k": k.grad.norm().item(),
        "grad_v": v.grad.norm().item(),
        "grad_beta_headwise": beta_headwise.grad.norm().item()
    }

    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()
    beta_headwise.grad.zero_()

    # Test 4: With initial state, no final state output
    o, _ = fused_recurrent_delta_rule(q, k, v, beta=beta_headwise, scale=0.1, initial_state=initial_state, output_final_state=False)

    loss = o.sum()
    loss.backward()

    result_4 = {
        "grad_q": q.grad.norm().item(),
        "grad_k": k.grad.norm().item(),
        "grad_v": v.grad.norm().item(),
        "grad_beta_headwise": beta_headwise.grad.norm().item(),
        "grad_initial_state": initial_state.grad.norm().item()
    }

    return {
        "test_case_1": result_1,
        "test_case_2": result_2,
        "test_case_3": result_3,
        "test_case_4": result_4
    }

result_gold = test_fused_recurrent_delta_rule_with_backward()
