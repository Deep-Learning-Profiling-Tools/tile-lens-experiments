import argparse
import os
import sys
import importlib.util
import torch
from typing import Dict, List, Callable

ROOT = os.path.dirname(__file__)

STUDY_CASES: Dict[str, List[str]] = {
    "unroll_for_loop": [
        "diag_ssm_triton",
        "fused_recurrent_retention",
        "fused_recurrent_delta",
        "fast_rope_embedding",
        "flash_decode2_llama",
        "iv_dependent_matmul",
        "rmsnorm_fused",
        "rmsnorm_fused_llama",
        "rmsnorm_implementation",
        "layernorm_fwd_triton",
        "var_len_copy",
        "matmul_leakyrelu",
        "flash_decode2_phi",
        "kldiv_ops",
        "mean_reduction",
        "softmax_optimize",
        "triton_conv2d_fwd",
        "triton_matmul",
        "matmul_triton1",
        "lora_expand_gemv",
    ],
    "mask_percentage": [
        "quantize_kv_transform",
        "context_attn_llama",
        "context_attn_fwd",
        "bgmv_shrink_kernel",
        "sin_kernel",
        "add_value",
        "rmsnorm_fused_llama",
        "relu_triton_kernel",
        "lora_expand_gemv",
    ],
}

STUDY_NAMES = list(STUDY_CASES.keys())


def _load_module(name: str, rel_path: str):
    path = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================================
# unroll_for_loop modules
# ============================================================================
diag_baseline = _load_module("diag_ssm_baseline", "unroll_for_loop/diag_ssm_triton/baseline.py")
diag_optimized = _load_module("diag_ssm_optimized", "unroll_for_loop/diag_ssm_triton/optimized.py")

fr_baseline = _load_module("frr_baseline", "unroll_for_loop/fused_recurrent_retention/baseline.py")
fr_optimized = _load_module("frr_optimized", "unroll_for_loop/fused_recurrent_retention/optimized.py")

frd_baseline = _load_module("frd_baseline", "unroll_for_loop/fused_recurrent_delta/baseline.py")
frd_optimized = _load_module("frd_optimized", "unroll_for_loop/fused_recurrent_delta/optimized.py")

fre_baseline = _load_module("fre_baseline", "unroll_for_loop/fast_rope_embedding/baseline.py")
fre_optimized = _load_module("fre_optimized", "unroll_for_loop/fast_rope_embedding/optimized.py")

fdll_baseline = _load_module("fdll_baseline", "unroll_for_loop/flash_decode2_llama/baseline.py")
fdll_optimized = _load_module("fdll_optimized", "unroll_for_loop/flash_decode2_llama/optimized.py")

ivdm_baseline = _load_module("ivdm_baseline", "unroll_for_loop/iv_dependent_matmul/baseline.py")
ivdm_optimized = _load_module("ivdm_optimized", "unroll_for_loop/iv_dependent_matmul/optimized.py")

rmsn_baseline = _load_module("rmsn_baseline", "unroll_for_loop/rmsnorm_fused/baseline.py")
rmsn_optimized = _load_module("rmsn_optimized", "unroll_for_loop/rmsnorm_fused/optimized.py")

rmsnll_baseline = _load_module("rmsnll_baseline", "unroll_for_loop/rmsnorm_fused_llama/baseline.py")
rmsnll_optimized = _load_module("rmsnll_optimized", "unroll_for_loop/rmsnorm_fused_llama/optimized.py")

rmsni_baseline = _load_module("rmsni_baseline", "unroll_for_loop/rmsnorm_implementation/baseline.py")
rmsni_optimized = _load_module("rmsni_optimized", "unroll_for_loop/rmsnorm_implementation/optimized.py")

lnfwd_baseline = _load_module("lnfwd_baseline", "unroll_for_loop/layernorm_fwd_triton/baseline.py")
lnfwd_optimized = _load_module("lnfwd_optimized", "unroll_for_loop/layernorm_fwd_triton/optimized.py")

vlc_baseline = _load_module("vlc_baseline", "unroll_for_loop/var_len_copy/baseline.py")
vlc_optimized = _load_module("vlc_optimized", "unroll_for_loop/var_len_copy/optimized.py")

mmlr_baseline = _load_module("mmlr_baseline", "unroll_for_loop/matmul_leakyrelu/baseline.py")
mmlr_optimized = _load_module("mmlr_optimized", "unroll_for_loop/matmul_leakyrelu/optimized.py")

fdphi_baseline = _load_module("fdphi_baseline", "unroll_for_loop/flash_decode2_phi/baseline.py")
fdphi_optimized = _load_module("fdphi_optimized", "unroll_for_loop/flash_decode2_phi/optimized.py")

kldiv_baseline = _load_module("kldiv_baseline", "unroll_for_loop/kldiv_ops/baseline.py")
kldiv_optimized = _load_module("kldiv_optimized", "unroll_for_loop/kldiv_ops/optimized.py")

meanred_baseline = _load_module("meanred_baseline", "unroll_for_loop/mean_reduction/baseline.py")
meanred_optimized = _load_module("meanred_optimized", "unroll_for_loop/mean_reduction/optimized.py")

smopt_baseline = _load_module("smopt_baseline", "unroll_for_loop/softmax_optimize/baseline.py")
smopt_optimized = _load_module("smopt_optimized", "unroll_for_loop/softmax_optimize/optimized.py")

conv2d_baseline = _load_module("conv2d_baseline", "unroll_for_loop/triton_conv2d_fwd/baseline.py")
conv2d_optimized = _load_module("conv2d_optimized", "unroll_for_loop/triton_conv2d_fwd/optimized.py")

trmm_baseline = _load_module("trmm_baseline", "unroll_for_loop/triton_matmul/baseline.py")
trmm_optimized = _load_module("trmm_optimized", "unroll_for_loop/triton_matmul/optimized.py")

mm1_baseline = _load_module("mm1_baseline", "unroll_for_loop/matmul_triton1/baseline.py")
mm1_optimized = _load_module("mm1_optimized", "unroll_for_loop/matmul_triton1/optimized.py")

lora_baseline = _load_module("lora_baseline", "unroll_for_loop/lora_expand_gemv/baseline.py")
lora_optimized = _load_module("lora_optimized", "unroll_for_loop/lora_expand_gemv/optimized.py")

# ============================================================================
# mask_percentage modules
# ============================================================================
qkv_baseline = _load_module("qkv_baseline", "mask_percentage/quantize_kv_transform/baseline.py")
qkv_optimized = _load_module("qkv_optimized", "mask_percentage/quantize_kv_transform/optimized.py")

ctx_attn_baseline = _load_module("ctx_attn_baseline", "mask_percentage/context_attn_llama/baseline.py")
ctx_attn_optimized = _load_module("ctx_attn_optimized", "mask_percentage/context_attn_llama/optimized.py")

ctx_fwd_baseline = _load_module("ctx_fwd_baseline", "mask_percentage/context_attn_fwd/baseline.py")
ctx_fwd_optimized = _load_module("ctx_fwd_optimized", "mask_percentage/context_attn_fwd/optimized.py")

bgmv_shrink_baseline = _load_module("bgmv_shrink_baseline", "mask_percentage/bgmv_shrink_kernel/baseline.py")
bgmv_shrink_optimized = _load_module("bgmv_shrink_optimized", "mask_percentage/bgmv_shrink_kernel/optimized.py")

sin_kernel_baseline = _load_module("sin_kernel_baseline", "mask_percentage/sin_kernel/baseline.py")
sin_kernel_optimized = _load_module("sin_kernel_optimized", "mask_percentage/sin_kernel/optimized.py")

add_value_baseline = _load_module("add_value_baseline", "mask_percentage/add_value/baseline.py")
add_value_optimized = _load_module("add_value_optimized", "mask_percentage/add_value/optimized.py")

mp_rmsnorm_baseline = _load_module("mp_rmsnorm_baseline", "mask_percentage/rmsnorm_fused_llama/baseline.py")
mp_rmsnorm_optimized = _load_module("mp_rmsnorm_optimized", "mask_percentage/rmsnorm_fused_llama/optimized.py")

relu_kernel_baseline = _load_module("relu_kernel_baseline", "mask_percentage/relu_triton_kernel/baseline.py")
relu_kernel_optimized = _load_module("relu_kernel_optimized", "mask_percentage/relu_triton_kernel/optimized.py")
mp_lora_expand_gemv_baseline = _load_module("mp_lora_expand_gemv_baseline", "mask_percentage/lora_expand_gemv/baseline.py")
mp_lora_expand_gemv_optimized = _load_module("mp_lora_expand_gemv_optimized", "mask_percentage/lora_expand_gemv/optimized.py")


def _report(title: str, ok: bool):
    mark = "✓" if ok else "✗"
    print(f"{mark} {title}: {'PASSED' if ok else 'FAILED'}")


# ============================================================================
# unroll_for_loop tests
# ============================================================================

def test_diag_ssm():
    print("\n" + "=" * 80)
    print("Testing Diagonal SSM Triton (baseline vs optimized)")
    print("=" * 80)

    batch_size, dim, length = 2, 3, 5  # length must stay 5 for unrolled kernel
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # -------- Real tensors --------
    s_real = torch.randn((batch_size, dim), dtype=torch.float32, device="cuda", requires_grad=True)
    x_real = torch.randn((length, batch_size, dim), dtype=torch.float32, device="cuda", requires_grad=True)
    lam_real = torch.rand((dim,), dtype=torch.float32, device="cuda", requires_grad=True)

    s_real_opt = s_real.detach().clone().requires_grad_(True)
    x_real_opt = x_real.detach().clone().requires_grad_(True)
    lam_real_opt = lam_real.detach().clone().requires_grad_(True)

    y_base_real = diag_baseline.diag_ssm_forward_triton(s_real, x_real, lam_real)
    y_opt_real = diag_optimized.diag_ssm_forward_triton(s_real_opt, x_real_opt, lam_real_opt)

    fwd_real_ok = torch.allclose(y_base_real, y_opt_real, rtol=1e-5, atol=1e-6)
    if not fwd_real_ok:
        diff = torch.max(torch.abs(y_base_real - y_opt_real)).item()
        print(f"Real forward max diff: {diff:.2e}")

    y_base_real.backward(torch.ones_like(y_base_real))
    y_opt_real.backward(torch.ones_like(y_opt_real))

    grad_s_ok = torch.allclose(s_real.grad, s_real_opt.grad, rtol=1e-5, atol=1e-6)
    grad_x_ok = torch.allclose(x_real.grad, x_real_opt.grad, rtol=1e-5, atol=1e-6)
    grad_lam_ok = torch.allclose(lam_real.grad, lam_real_opt.grad, rtol=1e-5, atol=1e-6)
    if not grad_s_ok:
        print("Real grad_s max diff:", torch.max(torch.abs(s_real.grad - s_real_opt.grad)).item())
    if not grad_x_ok:
        print("Real grad_x max diff:", torch.max(torch.abs(x_real.grad - x_real_opt.grad)).item())
    if not grad_lam_ok:
        print("Real grad_lambda max diff:", torch.max(torch.abs(lam_real.grad - lam_real_opt.grad)).item())

    real_ok = fwd_real_ok and grad_s_ok and grad_x_ok and grad_lam_ok
    _report("Diag SSM real", real_ok)

    # -------- Complex tensors --------
    s_c = torch.randn((batch_size, dim), dtype=torch.complex64, device="cuda", requires_grad=True)
    x_c = torch.randn((length, batch_size, dim), dtype=torch.complex64, device="cuda", requires_grad=True)
    lam_c = torch.rand((dim,), dtype=torch.complex64, device="cuda", requires_grad=True)

    s_c_opt = s_c.detach().clone().requires_grad_(True)
    x_c_opt = x_c.detach().clone().requires_grad_(True)
    lam_c_opt = lam_c.detach().clone().requires_grad_(True)

    y_base_c = diag_baseline.diag_ssm_forward_triton(s_c, x_c, lam_c)
    y_opt_c = diag_optimized.diag_ssm_forward_triton(s_c_opt, x_c_opt, lam_c_opt)

    fwd_c_ok = torch.allclose(y_base_c, y_opt_c, rtol=1e-5, atol=1e-6)
    if not fwd_c_ok:
        diff = torch.max(torch.abs(y_base_c - y_opt_c)).item()
        print(f"Complex forward max diff: {diff:.2e}")

    y_base_c.backward(torch.ones_like(y_base_c))
    y_opt_c.backward(torch.ones_like(y_opt_c))

    grad_s_c_ok = torch.allclose(s_c.grad, s_c_opt.grad, rtol=1e-5, atol=1e-6)
    grad_x_c_ok = torch.allclose(x_c.grad, x_c_opt.grad, rtol=1e-5, atol=1e-6)
    grad_lam_c_ok = torch.allclose(lam_c.grad, lam_c_opt.grad, rtol=1e-5, atol=1e-6)
    if not grad_s_c_ok:
        print("Complex grad_s max diff:", torch.max(torch.abs(s_c.grad - s_c_opt.grad)).item())
    if not grad_x_c_ok:
        print("Complex grad_x max diff:", torch.max(torch.abs(x_c.grad - x_c_opt.grad)).item())
    if not grad_lam_c_ok:
        print("Complex grad_lambda max diff:", torch.max(torch.abs(lam_c.grad - lam_c_opt.grad)).item())

    complex_ok = fwd_c_ok and grad_s_c_ok and grad_x_c_ok and grad_lam_c_ok
    _report("Diag SSM complex", complex_ok)

    return real_ok and complex_ok


def test_fused_recurrent_retention():
    print("\n" + "=" * 80)
    print("Testing Fused Recurrent Retention (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    batch_size, n_heads, seq_len, d_head_qk, d_head_v = 2, 4, 8, 16, 16

    q = torch.randn(batch_size, n_heads, seq_len, d_head_qk, dtype=torch.float32, requires_grad=True, device="cuda")
    k = torch.randn(batch_size, n_heads, seq_len, d_head_qk, dtype=torch.float32, requires_grad=True, device="cuda")
    v = torch.randn(batch_size, n_heads, seq_len, d_head_v, dtype=torch.float32, requires_grad=True, device="cuda")

    q_opt = q.detach().clone().requires_grad_(True)
    k_opt = k.detach().clone().requires_grad_(True)
    v_opt = v.detach().clone().requires_grad_(True)

    o_base, fs_base = fr_baseline.fused_recurrent_retention(q, k, v, output_final_state=True)
    o_opt, fs_opt = fr_optimized.fused_recurrent_retention(q_opt, k_opt, v_opt, output_final_state=True)

    fwd_o_ok = torch.allclose(o_base, o_opt, rtol=1e-4, atol=1e-5)
    fwd_fs_ok = torch.allclose(fs_base, fs_opt, rtol=1e-4, atol=1e-5)
    if not fwd_o_ok:
        print("Retention forward o max diff:", torch.max(torch.abs(o_base - o_opt)).item())
    if not fwd_fs_ok:
        print("Retention forward final_state max diff:", torch.max(torch.abs(fs_base - fs_opt)).item())

    loss_base = o_base.sum() + fs_base.sum()
    loss_opt = o_opt.sum() + fs_opt.sum()
    loss_base.backward()
    loss_opt.backward()

    grad_q_ok = torch.allclose(q.grad, q_opt.grad, rtol=1e-4, atol=1e-5)
    grad_k_ok = torch.allclose(k.grad, k_opt.grad, rtol=1e-4, atol=1e-5)
    grad_v_ok = torch.allclose(v.grad, v_opt.grad, rtol=1e-4, atol=1e-5)
    if not grad_q_ok:
        print("Retention grad_q max diff:", torch.max(torch.abs(q.grad - q_opt.grad)).item())
    if not grad_k_ok:
        print("Retention grad_k max diff:", torch.max(torch.abs(k.grad - k_opt.grad)).item())
    if not grad_v_ok:
        print("Retention grad_v max diff:", torch.max(torch.abs(v.grad - v_opt.grad)).item())

    retention_ok = fwd_o_ok and fwd_fs_ok and grad_q_ok and grad_k_ok and grad_v_ok
    _report("Fused recurrent retention", retention_ok)

    return retention_ok


def test_fused_recurrent_delta():
    print("\n" + "=" * 80)
    print("Testing Fused Recurrent Delta (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    B, H, T, K, V = 2, 4, 8, 16, 32
    rtol, atol = 1e-4, 1e-5

    # ---------- Headwise beta with initial state ----------
    q = torch.randn(B, H, T, K, dtype=torch.float32, device="cuda", requires_grad=True)
    k = torch.randn(B, H, T, K, dtype=torch.float32, device="cuda", requires_grad=True)
    v = torch.randn(B, H, T, V, dtype=torch.float32, device="cuda", requires_grad=True)
    beta = torch.randn(B, H, T, V, dtype=torch.float32, device="cuda", requires_grad=True)
    init = torch.randn(B, H, K, V, dtype=torch.float32, device="cuda", requires_grad=True)

    q_opt = q.detach().clone().requires_grad_(True)
    k_opt = k.detach().clone().requires_grad_(True)
    v_opt = v.detach().clone().requires_grad_(True)
    beta_opt = beta.detach().clone().requires_grad_(True)
    init_opt = init.detach().clone().requires_grad_(True)

    o_base, fs_base = frd_baseline.fused_recurrent_delta_rule(
        q, k, v, beta=beta, initial_state=init, output_final_state=True
    )
    o_opt, fs_opt = frd_optimized.fused_recurrent_delta_rule(
        q_opt, k_opt, v_opt, beta=beta_opt, initial_state=init_opt, output_final_state=True
    )

    head_fwd_o_ok = torch.allclose(o_base, o_opt, rtol=rtol, atol=atol)
    head_fwd_fs_ok = torch.allclose(fs_base, fs_opt, rtol=rtol, atol=atol)

    loss_base = o_base.sum() + fs_base.sum()
    loss_opt = o_opt.sum() + fs_opt.sum()
    loss_base.backward()
    loss_opt.backward()

    head_grad_q_ok = torch.allclose(q.grad, q_opt.grad, rtol=rtol, atol=atol)
    head_grad_k_ok = torch.allclose(k.grad, k_opt.grad, rtol=rtol, atol=atol)
    head_grad_v_ok = torch.allclose(v.grad, v_opt.grad, rtol=rtol, atol=atol)
    head_grad_beta_ok = torch.allclose(beta.grad, beta_opt.grad, rtol=rtol, atol=atol)
    head_grad_init_ok = torch.allclose(init.grad, init_opt.grad, rtol=rtol, atol=atol)

    headwise_ok = head_fwd_o_ok and head_fwd_fs_ok and head_grad_q_ok and head_grad_k_ok and head_grad_v_ok and head_grad_beta_ok and head_grad_init_ok
    if not headwise_ok:
        if not head_fwd_o_ok:
            print("Delta headwise forward o max diff:", torch.max(torch.abs(o_base - o_opt)).item())
        if not head_fwd_fs_ok:
            print("Delta headwise forward fs max diff:", torch.max(torch.abs(fs_base - fs_opt)).item())
        if not head_grad_q_ok:
            print("Delta headwise grad_q max diff:", torch.max(torch.abs(q.grad - q_opt.grad)).item())
        if not head_grad_k_ok:
            print("Delta headwise grad_k max diff:", torch.max(torch.abs(k.grad - k_opt.grad)).item())
        if not head_grad_v_ok:
            print("Delta headwise grad_v max diff:", torch.max(torch.abs(v.grad - v_opt.grad)).item())
        if not head_grad_beta_ok:
            print("Delta headwise grad_beta max diff:", torch.max(torch.abs(beta.grad - beta_opt.grad)).item())
        if not head_grad_init_ok:
            print("Delta headwise grad_init max diff:", torch.max(torch.abs(init.grad - init_opt.grad)).item())

    # ---------- Non-headwise beta without initial state ----------
    q2 = torch.randn(B, H, T, K, dtype=torch.float32, device="cuda", requires_grad=True)
    k2 = torch.randn(B, H, T, K, dtype=torch.float32, device="cuda", requires_grad=True)
    v2 = torch.randn(B, H, T, V, dtype=torch.float32, device="cuda", requires_grad=True)
    beta2 = torch.randn(B, H, T, dtype=torch.float32, device="cuda", requires_grad=True)

    q2_opt = q2.detach().clone().requires_grad_(True)
    k2_opt = k2.detach().clone().requires_grad_(True)
    v2_opt = v2.detach().clone().requires_grad_(True)
    beta2_opt = beta2.detach().clone().requires_grad_(True)

    o2_base, fs2_base = frd_baseline.fused_recurrent_delta_rule(
        q2, k2, v2, beta=beta2, initial_state=None, output_final_state=True
    )
    o2_opt, fs2_opt = frd_optimized.fused_recurrent_delta_rule(
        q2_opt, k2_opt, v2_opt, beta=beta2_opt, initial_state=None, output_final_state=True
    )

    non_fwd_o_ok = torch.allclose(o2_base, o2_opt, rtol=rtol, atol=atol)
    non_fwd_fs_ok = torch.allclose(fs2_base, fs2_opt, rtol=rtol, atol=atol)

    loss2_base = o2_base.sum() + fs2_base.sum()
    loss2_opt = o2_opt.sum() + fs2_opt.sum()
    loss2_base.backward()
    loss2_opt.backward()

    grad_q2_ok = torch.allclose(q2.grad, q2_opt.grad, rtol=rtol, atol=atol)
    grad_k2_ok = torch.allclose(k2.grad, k2_opt.grad, rtol=rtol, atol=atol)
    grad_v2_ok = torch.allclose(v2.grad, v2_opt.grad, rtol=rtol, atol=atol)
    grad_beta2_ok = torch.allclose(beta2.grad, beta2_opt.grad, rtol=rtol, atol=atol)

    non_headwise_ok = non_fwd_o_ok and non_fwd_fs_ok and grad_q2_ok and grad_k2_ok and grad_v2_ok and grad_beta2_ok
    if not non_headwise_ok:
        if not non_fwd_o_ok:
            print("Delta non-head forward o max diff:", torch.max(torch.abs(o2_base - o2_opt)).item())
        if not non_fwd_fs_ok:
            print("Delta non-head forward fs max diff:", torch.max(torch.abs(fs2_base - fs2_opt)).item())
        if not grad_q2_ok:
            print("Delta non-head grad_q max diff:", torch.max(torch.abs(q2.grad - q2_opt.grad)).item())
        if not grad_k2_ok:
            print("Delta non-head grad_k max diff:", torch.max(torch.abs(k2.grad - k2_opt.grad)).item())
        if not grad_v2_ok:
            print("Delta non-head grad_v max diff:", torch.max(torch.abs(v2.grad - v2_opt.grad)).item())
        if not grad_beta2_ok:
            print("Delta non-head grad_beta max diff:", torch.max(torch.abs(beta2.grad - beta2_opt.grad)).item())

    delta_ok = headwise_ok and non_headwise_ok
    _report("Fused recurrent delta (headwise + non-headwise)", delta_ok)
    return delta_ok


def test_fast_rope_embedding():
    print("\n" + "=" * 80)
    print("Testing Fast RoPE Embedding (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    batch_size, seq_len, n_heads, head_dim = 2, 4, 8, 16
    rtol, atol = 1e-5, 1e-6

    q = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=torch.float32, device="cuda", requires_grad=True)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=torch.float32, device="cuda", requires_grad=True)
    q_opt = q.detach().clone().requires_grad_(True)
    k_opt = k.detach().clone().requires_grad_(True)

    cos = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device="cuda")

    q_out_base, k_out_base = fre_baseline.fast_rope_embedding(q, k, cos, sin)
    q_out_opt, k_out_opt = fre_optimized.fast_rope_embedding(q_opt, k_opt, cos, sin)

    q_fwd_ok = torch.allclose(q_out_base, q_out_opt, rtol=rtol, atol=atol)
    k_fwd_ok = torch.allclose(k_out_base, k_out_opt, rtol=rtol, atol=atol)
    if not q_fwd_ok:
        print("RoPE Q forward max diff:", torch.max(torch.abs(q_out_base - q_out_opt)).item())
    if not k_fwd_ok:
        print("RoPE K forward max diff:", torch.max(torch.abs(k_out_base - k_out_opt)).item())

    (q_out_base.mean() + k_out_base.mean()).backward()
    (q_out_opt.mean() + k_out_opt.mean()).backward()

    q_grad_ok = torch.allclose(q.grad, q_opt.grad, rtol=rtol, atol=atol)
    k_grad_ok = torch.allclose(k.grad, k_opt.grad, rtol=rtol, atol=atol)
    if not q_grad_ok:
        print("RoPE Q grad max diff:", torch.max(torch.abs(q.grad - q_opt.grad)).item())
    if not k_grad_ok:
        print("RoPE K grad max diff:", torch.max(torch.abs(k.grad - k_opt.grad)).item())

    rope_ok = q_fwd_ok and k_fwd_ok and q_grad_ok and k_grad_ok
    _report("Fast RoPE embedding", rope_ok)
    return rope_ok


def test_flash_decode2_llama():
    print("\n" + "=" * 80)
    print("Testing Flash Decode2 LLaMA Stage2 (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    batch_size, head_num = 2, 4
    seq_block_num = 8  # support up to block_n_size == 8
    head_dim = 32
    block_seq = 8
    rtol, atol = 1e-5, 1e-6

    cases = [
        ("block_n_size_3", torch.tensor([24, 24], dtype=torch.int32, device="cuda")),
        ("block_n_size_4", torch.tensor([32, 32], dtype=torch.int32, device="cuda")),
        ("block_n_size_8", torch.tensor([64, 64], dtype=torch.int32, device="cuda")),
    ]

    all_ok = True
    for name, seqlen in cases:
        mid = torch.randn(batch_size, head_num, seq_block_num, head_dim, dtype=torch.float32, device="cuda")
        mid_log = torch.randn(batch_size, head_num, seq_block_num, dtype=torch.float32, device="cuda")

        mid_opt = mid.clone()
        mid_log_opt = mid_log.clone()

        o_base = torch.empty(batch_size, head_num, head_dim, dtype=torch.float32, device="cuda")
        o_opt = torch.empty_like(o_base)

        fdll_baseline.flash_decode_stage2(mid, mid_log, seqlen, o_base, block_seq)
        fdll_optimized.flash_decode_stage2(mid_opt, mid_log_opt, seqlen, o_opt, block_seq)

        ok = torch.allclose(o_base, o_opt, rtol=rtol, atol=atol)
        if not ok:
            diff = torch.max(torch.abs(o_base - o_opt)).item()
            print(f"{name} max diff: {diff:.2e}")
        _report(f"Flash Decode2 LLaMA {name}", ok)
        all_ok = all_ok and ok

    return all_ok


def test_iv_dependent_matmul():
    print("\n" + "=" * 80)
    print("Testing IV Dependent MatMul (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    M, K, N = 256, 256, 256
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 32, 32
    rtol, atol = 1e-3, 1e-3  # fp16 output needs looser tolerance

    # Generate input matrices
    a = torch.rand((M, K), device="cuda")
    b = torch.rand((K, N), device="cuda")

    # Baseline output
    out_base = torch.empty((M, N), device="cuda")
    grid = ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,)
    ivdm_baseline.iv_dependent_matmul_kernel[grid](
        a, b, out_base, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out_base.stride(0), out_base.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        type="pre_load",
        num_stages=3
    )

    # Optimized output
    out_opt = torch.empty((M, N), device="cuda")
    ivdm_optimized.iv_dependent_matmul_kernel[grid](
        a, b, out_opt, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out_opt.stride(0), out_opt.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        type="pre_load",
        num_stages=3
    )

    fwd_ok = torch.allclose(out_base, out_opt, rtol=rtol, atol=atol)
    if not fwd_ok:
        diff = torch.max(torch.abs(out_base.float() - out_opt.float())).item()
        print(f"Max diff: {diff:.2e}")

    _report("IV Dependent MatMul", fwd_ok)
    return fwd_ok


def test_rmsnorm_fused():
    print("\n" + "=" * 80)
    print("Testing RMSNorm Fused (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-5, 1e-6
    all_ok = True

    test_cases = [
        ("small_input", (2, 16)),
        ("medium_input", (4, 256)),
    ]

    for name, shape in test_cases:
        x = torch.randn(*shape, dtype=torch.float32, device="cuda")
        weight = torch.ones(shape[-1], dtype=torch.float32, device="cuda")

        norm_base = rmsn_baseline.TritonLlamaRMSNorm(weight)
        norm_opt = rmsn_optimized.TritonLlamaRMSNorm(weight)

        y_base = norm_base(x)
        y_opt = norm_opt(x)

        ok = torch.allclose(y_base, y_opt, rtol=rtol, atol=atol)
        if not ok:
            diff = torch.max(torch.abs(y_base - y_opt)).item()
            print(f"{name} max diff: {diff:.2e}")
        _report(f"RMSNorm Fused {name}", ok)
        all_ok = all_ok and ok

    return all_ok


def test_rmsnorm_fused_llama():
    print("\n" + "=" * 80)
    print("Testing RMSNorm Fused LLaMA (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-3, 1e-3  # fp16 output needs looser tolerance
    all_ok = True
    eps = 1e-5

    test_cases = [
        ("small_input", (2, 64)),
        ("medium_input", (4, 128)),
    ]

    for name, shape in test_cases:
        x = torch.randn(*shape, dtype=torch.float16, device="cuda")
        weight = torch.randn(shape[-1], dtype=torch.float16, device="cuda")

        y_base = rmsnll_baseline.rmsnorm_forward(x, weight, eps)
        y_opt = rmsnll_optimized.rmsnorm_forward(x, weight, eps)

        ok = torch.allclose(y_base, y_opt, rtol=rtol, atol=atol)
        if not ok:
            diff = torch.max(torch.abs(y_base.float() - y_opt.float())).item()
            print(f"{name} max diff: {diff:.2e}")
        _report(f"RMSNorm Fused LLaMA {name}", ok)
        all_ok = all_ok and ok

    return all_ok


def test_rmsnorm_implementation():
    print("\n" + "=" * 80)
    print("Testing RMSNorm Implementation (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-3, 1e-3  # fp16 output needs looser tolerance
    all_ok = True
    eps = 1e-6

    # Test with K=4096 (optimized version has special handling)
    batch, M, K = 2, 3, 4096
    x = torch.randn((batch, M, K), dtype=torch.float16, device="cuda")
    rms_weights = torch.randn((K,), dtype=torch.float16, device="cuda")

    y_base = rmsni_baseline.rmsnorm_wrapper(x, rms_weights, eps)
    y_opt = rmsni_optimized.rmsnorm_wrapper(x, rms_weights, eps)

    ok = torch.allclose(y_base, y_opt, rtol=rtol, atol=atol)
    if not ok:
        diff = torch.max(torch.abs(y_base.float() - y_opt.float())).item()
        print(f"K=4096 max diff: {diff:.2e}")
    _report("RMSNorm Implementation K=4096", ok)
    all_ok = all_ok and ok

    # Test with K=8192 (optimized version has special handling)
    K = 8192
    x = torch.randn((batch, M, K), dtype=torch.float16, device="cuda")
    rms_weights = torch.randn((K,), dtype=torch.float16, device="cuda")

    y_base = rmsni_baseline.rmsnorm_wrapper(x, rms_weights, eps)
    y_opt = rmsni_optimized.rmsnorm_wrapper(x, rms_weights, eps)

    ok = torch.allclose(y_base, y_opt, rtol=rtol, atol=atol)
    if not ok:
        diff = torch.max(torch.abs(y_base.float() - y_opt.float())).item()
        print(f"K=8192 max diff: {diff:.2e}")
    _report("RMSNorm Implementation K=8192", ok)
    all_ok = all_ok and ok

    return all_ok


def test_layernorm_fwd_triton():
    print("\n" + "=" * 80)
    print("Testing LayerNorm Forward Triton (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-5, 1e-6
    all_ok = True
    eps = 1e-5

    test_cases = [
        ("N=128", (2, 3, 128)),
        ("N=256", (2, 3, 256)),
    ]

    for name, shape in test_cases:
        X = torch.randn(*shape, dtype=torch.float32, device="cuda")
        W = torch.randn(shape[1], shape[2], dtype=torch.float32, device="cuda")

        y_base = lnfwd_baseline.layernorm_forward(X, W, eps)
        y_opt = lnfwd_optimized.layernorm_forward(X, W, eps)

        ok = torch.allclose(y_base, y_opt, rtol=rtol, atol=atol)
        if not ok:
            diff = torch.max(torch.abs(y_base - y_opt)).item()
            print(f"{name} max diff: {diff:.2e}")
        _report(f"LayerNorm Forward {name}", ok)
        all_ok = all_ok and ok

    return all_ok


def test_var_len_copy():
    print("\n" + "=" * 80)
    print("Testing Var Len Copy (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Test with lengths that fit in one BLOCK_SIZE (256)
    old_a_start = torch.tensor([0, 100, 300], dtype=torch.int32, device="cuda")
    old_a_len = torch.tensor([50, 100, 200], dtype=torch.int32, device="cuda")
    old_a_location = torch.arange(500, dtype=torch.float32, device="cuda")
    new_a_start = torch.tensor([0, 60, 260], dtype=torch.int32, device="cuda")

    new_a_location_base = torch.zeros(500, dtype=torch.float32, device="cuda")
    new_a_location_opt = torch.zeros(500, dtype=torch.float32, device="cuda")

    vlc_baseline.launch_var_len_copy_triton(
        old_a_start, old_a_len, old_a_location, new_a_start, new_a_location_base
    )
    vlc_optimized.launch_var_len_copy_triton(
        old_a_start, old_a_len, old_a_location, new_a_start, new_a_location_opt
    )

    ok = torch.allclose(new_a_location_base, new_a_location_opt)
    if not ok:
        diff = torch.max(torch.abs(new_a_location_base - new_a_location_opt)).item()
        print(f"Max diff: {diff:.2e}")

    _report("Var Len Copy", ok)
    return ok


def test_matmul_leakyrelu():
    print("\n" + "=" * 80)
    print("Testing MatMul LeakyReLU (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-3, 1e-3  # fp16 output
    all_ok = True

    M, K, N = 64, 128, 64
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    # Test with leaky_relu activation
    c_base_lr = mmlr_baseline.matmul(a, b, activation="leaky_relu")
    c_opt_lr = mmlr_optimized.matmul(a, b, activation="leaky_relu")

    ok_lr = torch.allclose(c_base_lr, c_opt_lr, rtol=rtol, atol=atol)
    if not ok_lr:
        diff = torch.max(torch.abs(c_base_lr.float() - c_opt_lr.float())).item()
        print(f"leaky_relu max diff: {diff:.2e}")
    _report("MatMul LeakyReLU (leaky_relu)", ok_lr)
    all_ok = all_ok and ok_lr

    # Test without activation
    c_base_no = mmlr_baseline.matmul(a, b, activation="")
    c_opt_no = mmlr_optimized.matmul(a, b, activation="")

    ok_no = torch.allclose(c_base_no, c_opt_no, rtol=rtol, atol=atol)
    if not ok_no:
        diff = torch.max(torch.abs(c_base_no.float() - c_opt_no.float())).item()
        print(f"no activation max diff: {diff:.2e}")
    _report("MatMul LeakyReLU (no activation)", ok_no)
    all_ok = all_ok and ok_no

    return all_ok


def test_flash_decode2_phi():
    print("\n" + "=" * 80)
    print("Testing Flash Decode2 Phi Stage2 (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    batch_size, head_num = 2, 4
    seq_block_num = 1  # optimized version only handles 1 iteration
    head_dim = 64
    block_seq = 16
    rtol, atol = 1e-5, 1e-6

    # Use seqlen that results in block_n_size = 1
    seqlen = torch.tensor([block_seq, block_seq], dtype=torch.int32, device="cuda")

    mid_out = torch.randn(batch_size, head_num, seq_block_num, head_dim, dtype=torch.float32, device="cuda")
    mid_out_logexpsum = torch.randn(batch_size, head_num, seq_block_num, dtype=torch.float32, device="cuda")

    out_base = torch.zeros(batch_size, head_num, head_dim, dtype=torch.float32, device="cuda")
    out_opt = torch.zeros(batch_size, head_num, head_dim, dtype=torch.float32, device="cuda")

    fdphi_baseline.flash_decode_stage2(mid_out, mid_out_logexpsum, seqlen, out_base, block_seq)
    fdphi_optimized.flash_decode_stage2(mid_out, mid_out_logexpsum, seqlen, out_opt, block_seq)

    ok = torch.allclose(out_base, out_opt, rtol=rtol, atol=atol)
    if not ok:
        diff = torch.max(torch.abs(out_base - out_opt)).item()
        print(f"Max diff: {diff:.2e}")

    _report("Flash Decode2 Phi", ok)
    return ok


def test_kldiv_ops():
    print("\n" + "=" * 80)
    print("Testing KLDiv Ops (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-5, 1e-6
    all_ok = True
    eps = 1e-6

    # Small input that fits in one BLOCK_SIZE iteration
    y_pred = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]], device="cuda", dtype=torch.float32).log()
    y_true = torch.tensor([[0.1, 0.4, 0.5], [0.2, 0.5, 0.3]], device="cuda", dtype=torch.float32)

    # Test forward with batchmean reduction
    out_base = kldiv_baseline.kldiv_forward_triton(y_pred, y_true, log_target=False, reduction="batchmean", eps=eps)
    out_opt = kldiv_optimized.kldiv_forward_triton(y_pred, y_true, log_target=False, reduction="batchmean", eps=eps)

    ok = torch.allclose(out_base, out_opt, rtol=rtol, atol=atol)
    if not ok:
        diff = torch.abs(out_base - out_opt).item()
        print(f"forward batchmean diff: {diff:.2e}")
    _report("KLDiv forward (batchmean)", ok)
    all_ok = all_ok and ok

    # Test forward with none reduction
    out_base_none = kldiv_baseline.kldiv_forward_triton(y_pred, y_true, log_target=False, reduction="none", eps=eps)
    out_opt_none = kldiv_optimized.kldiv_forward_triton(y_pred, y_true, log_target=False, reduction="none", eps=eps)

    ok_none = torch.allclose(out_base_none, out_opt_none, rtol=rtol, atol=atol)
    if not ok_none:
        diff = torch.max(torch.abs(out_base_none - out_opt_none)).item()
        print(f"forward none max diff: {diff:.2e}")
    _report("KLDiv forward (none)", ok_none)
    all_ok = all_ok and ok_none

    return all_ok


def test_mean_reduction():
    print("\n" + "=" * 80)
    print("Testing Mean Reduction (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-5, 1e-6
    all_ok = True

    # Test with dims [1, 2] on shape (2, 3, 4, 5) -> N = 3*4 = 12
    # This matches the optimized version's special handling
    x = torch.randn(2, 3, 4, 5, dtype=torch.float32, device="cuda")

    y_base = meanred_baseline.mean_dim(x, [1, 2])
    y_opt = meanred_optimized.mean_dim(x, [1, 2])

    ok = torch.allclose(y_base, y_opt, rtol=rtol, atol=atol)
    if not ok:
        diff = torch.max(torch.abs(y_base - y_opt)).item()
        print(f"dims=[1,2] max diff: {diff:.2e}")
    _report("Mean Reduction dims=[1,2]", ok)
    all_ok = all_ok and ok

    # Test with keepdim=True
    y_base_kd = meanred_baseline.mean_dim(x, [1, 2], keepdim=True)
    y_opt_kd = meanred_optimized.mean_dim(x, [1, 2], keepdim=True)

    ok_kd = torch.allclose(y_base_kd, y_opt_kd, rtol=rtol, atol=atol)
    if not ok_kd:
        diff = torch.max(torch.abs(y_base_kd - y_opt_kd)).item()
        print(f"keepdim=True max diff: {diff:.2e}")
    _report("Mean Reduction keepdim=True", ok_kd)
    all_ok = all_ok and ok_kd

    return all_ok


def test_softmax_optimize():
    print("\n" + "=" * 80)
    print("Testing Softmax Optimize (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-5, 1e-6
    all_ok = True

    # Test with small N that fits in single TILE_N iteration
    test_cases = [
        ("M=128, N=128", (128, 128)),
        ("M=64, N=256", (64, 256)),
    ]

    for name, shape in test_cases:
        x = torch.randn(*shape, dtype=torch.float32, device="cuda")

        y_base = smopt_baseline.softmax(x)
        y_opt = smopt_optimized.softmax(x)

        ok = torch.allclose(y_base, y_opt, rtol=rtol, atol=atol)
        if not ok:
            diff = torch.max(torch.abs(y_base - y_opt)).item()
            print(f"{name} max diff: {diff:.2e}")
        _report(f"Softmax Optimize {name}", ok)
        all_ok = all_ok and ok

    return all_ok


def test_triton_conv2d_fwd():
    print("\n" + "=" * 80)
    print("Testing Triton Conv2D Forward (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-4, 1e-4
    all_ok = True

    # Test case 1: Basic 3x3 conv
    input_t = torch.randn(1, 3, 32, 32, device="cuda", dtype=torch.float32)
    weight_t = torch.randn(16, 3, 3, 3, device="cuda", dtype=torch.float32)

    y_base = conv2d_baseline.conv2d_forward(input_t, weight_t, 3, 3, 1, 1, 0, 0, 1)
    y_opt = conv2d_optimized.conv2d_forward(input_t, weight_t, 3, 3, 1, 1, 0, 0, 1)

    ok = torch.allclose(y_base, y_opt, rtol=rtol, atol=atol)
    if not ok:
        diff = torch.max(torch.abs(y_base - y_opt)).item()
        print(f"3x3 conv max diff: {diff:.2e}")
    _report("Conv2D 3x3 basic", ok)
    all_ok = all_ok and ok

    # Test case 2: With padding and stride
    y_base2 = conv2d_baseline.conv2d_forward(input_t, weight_t, 3, 3, 2, 2, 1, 1, 1)
    y_opt2 = conv2d_optimized.conv2d_forward(input_t, weight_t, 3, 3, 2, 2, 1, 1, 1)

    ok2 = torch.allclose(y_base2, y_opt2, rtol=rtol, atol=atol)
    if not ok2:
        diff = torch.max(torch.abs(y_base2 - y_opt2)).item()
        print(f"3x3 with stride/pad max diff: {diff:.2e}")
    _report("Conv2D 3x3 stride/pad", ok2)
    all_ok = all_ok and ok2

    return all_ok


def test_triton_matmul():
    print("\n" + "=" * 80)
    print("Testing Triton MatMul (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-3, 1e-3  # fp16 output
    all_ok = True

    # M=256, K=128, N=256 with BLOCK_SIZE_K=64 -> 2 iterations (matches unrolled version)
    M, K, N = 256, 128, 256
    a = torch.randn((M, K), dtype=torch.float16, device="cuda")
    b = torch.randn((K, N), dtype=torch.float16, device="cuda")

    c_base = trmm_baseline.matmul(a, b)
    c_opt = trmm_optimized.matmul(a, b)

    ok = torch.allclose(c_base, c_opt, rtol=rtol, atol=atol)
    if not ok:
        diff = torch.max(torch.abs(c_base.float() - c_opt.float())).item()
        print(f"Max diff: {diff:.2e}")
    _report("Triton MatMul fp16", ok)
    all_ok = all_ok and ok

    return all_ok


def test_matmul_triton1():
    print("\n" + "=" * 80)
    print("Testing MatMul Triton1 (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-5, 1e-6
    all_ok = True

    # 16x16 matmul where k_size = k_block_size (single iteration)
    x = torch.randn(16, 16, device="cuda", dtype=torch.float32)
    y = torch.randn(16, 16, device="cuda", dtype=torch.float32)

    z_base = mm1_baseline.matmul(x, y)
    z_opt = mm1_optimized.matmul(x, y)

    ok = torch.allclose(z_base, z_opt, rtol=rtol, atol=atol)
    if not ok:
        diff = torch.max(torch.abs(z_base - z_opt)).item()
        print(f"Max diff: {diff:.2e}")
    _report("MatMul Triton1 16x16", ok)
    all_ok = all_ok and ok

    return all_ok


def test_lora_expand_gemv():
    print("\n" + "=" * 80)
    print("Testing LoRA Expand GEMV (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-3, 1e-3  # fp16 output
    all_ok = True

    batch_size = 4
    hidden_size = 128
    rank = 64
    lora_num = 3

    inputs = torch.randn(batch_size, hidden_size, dtype=torch.float16, device="cuda")
    lora_b_weights = torch.randn(lora_num, rank, hidden_size, dtype=torch.float16, device="cuda")
    lora_indices_tensor = torch.tensor([0, 1, -1, 2], dtype=torch.int32, device="cuda")

    # Test with add_inputs=False
    output_base = torch.zeros(batch_size, rank, dtype=torch.float16, device="cuda")
    output_opt = torch.zeros(batch_size, rank, dtype=torch.float16, device="cuda")

    lora_baseline._bgmv_expand(inputs, lora_b_weights, output_base, lora_indices_tensor, add_inputs=False)
    lora_optimized._bgmv_expand(inputs, lora_b_weights, output_opt, lora_indices_tensor, add_inputs=False)

    ok = torch.allclose(output_base, output_opt, rtol=rtol, atol=atol)
    if not ok:
        diff = torch.max(torch.abs(output_base.float() - output_opt.float())).item()
        print(f"add_inputs=False max diff: {diff:.2e}")
    _report("LoRA Expand GEMV (add_inputs=False)", ok)
    all_ok = all_ok and ok

    return all_ok


# ============================================================================
# mask_percentage tests
# ============================================================================

def test_quantize_kv_transform():
    print("\n" + "=" * 80)
    print("Testing Quantize KV Transform (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-5, 1e-6
    all_ok = True

    test_cases = [
        ("B=32, N_CTX=1024, H=12, D=96", 32, 1024, 12, 96),
        ("B=16, N_CTX=512, H=8, D=64", 16, 512, 8, 64),
        ("B=1, N_CTX=1, H=1, D=1", 1, 1, 1, 1),
    ]

    for name, B, N_CTX, H, D in test_cases:
        src = torch.randn((B * N_CTX, H, D), dtype=torch.float16, device="cuda")
        dest_loc = torch.arange(0, B * N_CTX, dtype=torch.int32, device="cuda")

        # Baseline outputs
        value_dest_base = torch.zeros((B * N_CTX, H, D), dtype=torch.int8, device="cuda")
        scale_dest_base = torch.zeros((B * N_CTX, H, 1), dtype=torch.float16, device="cuda")

        # Optimized outputs
        value_dest_opt = torch.zeros((B * N_CTX, H, D), dtype=torch.int8, device="cuda")
        scale_dest_opt = torch.zeros((B * N_CTX, H, 1), dtype=torch.float16, device="cuda")

        qkv_baseline.destindex_copy_quantize_kv(src, dest_loc, value_dest_base, scale_dest_base)
        qkv_optimized.destindex_copy_quantize_kv(src, dest_loc, value_dest_opt, scale_dest_opt)

        value_ok = torch.equal(value_dest_base, value_dest_opt)
        scale_ok = torch.allclose(scale_dest_base, scale_dest_opt, rtol=rtol, atol=atol)

        ok = value_ok and scale_ok
        if not ok:
            if not value_ok:
                diff_count = (value_dest_base != value_dest_opt).sum().item()
                print(f"{name} value mismatch count: {diff_count}")
            if not scale_ok:
                diff = torch.max(torch.abs(scale_dest_base.float() - scale_dest_opt.float())).item()
                print(f"{name} scale max diff: {diff:.2e}")
        _report(f"Quantize KV Transform {name}", ok)
        all_ok = all_ok and ok

    return all_ok


def test_context_attn_llama():
    print("\n" + "=" * 80)
    print("Testing Context Attention LLaMA (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-2, 1e-2  # fp16 attention needs looser tolerance
    all_ok = True

    # Use small parameters for testing
    Z, H, N_CTX, D_HEAD = 2, 4, 128, 64
    dtype = torch.float16

    # Total KV cache size
    total_kv_len = Z * N_CTX

    q = torch.randn((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda")
    k = torch.randn((total_kv_len, H, D_HEAD), dtype=dtype, device="cuda")
    v = torch.randn((total_kv_len, H, D_HEAD), dtype=dtype, device="cuda")

    # req_to_token_indexs maps (request_idx, position) -> kv_cache_index
    # Each request i has tokens at positions [i*N_CTX, (i+1)*N_CTX)
    req_to_token_indexs = torch.zeros((Z, N_CTX + 100), dtype=torch.int32, device="cuda")
    for i in range(Z):
        req_to_token_indexs[i, :N_CTX] = torch.arange(i * N_CTX, (i + 1) * N_CTX, dtype=torch.int32)

    max_input_len = N_CTX
    # b_start_loc: start position in q for each batch
    b_start_loc = torch.arange(0, Z * N_CTX, N_CTX, dtype=torch.int32, device="cuda")
    b_seq_len = torch.full((Z,), N_CTX, dtype=torch.int32, device="cuda")
    b_req_idx = torch.arange(Z, dtype=torch.int32, device="cuda")
    b_prompt_cache_len = torch.zeros(Z, dtype=torch.int32, device="cuda")

    o_base = torch.zeros((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda")
    o_opt = torch.zeros((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda")

    ctx_attn_baseline.context_attention_fwd(
        q, k, v, o_base, b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, max_input_len, req_to_token_indexs
    )
    ctx_attn_optimized.context_attention_fwd(
        q, k, v, o_opt, b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, max_input_len, req_to_token_indexs
    )

    ok = torch.allclose(o_base, o_opt, rtol=rtol, atol=atol)
    if not ok:
        diff = torch.max(torch.abs(o_base.float() - o_opt.float())).item()
        print(f"Max diff: {diff:.2e}")
    _report("Context Attention LLaMA", ok)
    all_ok = all_ok and ok

    return all_ok


def test_context_attn_fwd():
    print("\n" + "=" * 80)
    print("Testing Context Attention Forward (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-2, 1e-2  # fp16 attention needs looser tolerance
    all_ok = True

    # Use small parameters for testing
    Z, H, N_CTX, D_HEAD = 2, 4, 128, 64
    dtype = torch.float16

    q = torch.randn((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda")
    # K and V have shape (batch, heads, seq_len, head_dim)
    k = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
    v = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")

    max_input_len = N_CTX
    b_start_loc = torch.arange(0, Z * N_CTX, N_CTX, dtype=torch.int32, device="cuda")
    b_seq_len = torch.full((Z,), N_CTX, dtype=torch.int32, device="cuda")
    b_prompt_cache_len = torch.zeros(Z, dtype=torch.int32, device="cuda")

    o_base = torch.zeros((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda")
    o_opt = torch.zeros((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda")

    ctx_fwd_baseline.context_attention_fwd_ppl_int8kv(
        q, k, v, o_base, b_start_loc, b_seq_len, max_input_len, b_prompt_cache_len
    )
    ctx_fwd_optimized.context_attention_fwd_ppl_int8kv(
        q, k, v, o_opt, b_start_loc, b_seq_len, max_input_len, b_prompt_cache_len
    )

    ok = torch.allclose(o_base, o_opt, rtol=rtol, atol=atol)
    if not ok:
        diff = torch.max(torch.abs(o_base.float() - o_opt.float())).item()
        print(f"Max diff: {diff:.2e}")
    _report("Context Attention Forward", ok)
    all_ok = all_ok and ok

    return all_ok


def test_bgmv_shrink_kernel():
    print("\n" + "=" * 80)
    print("Testing BGMV Shrink Kernel (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-2, 1e-2  # fp16 output
    all_ok = True

    batch_size = 4
    N = 16
    K = 64
    lora_num = 3
    scaling = 1.0

    inputs = torch.randn((batch_size, K), dtype=torch.float16, device="cuda")
    lora_a_weights = torch.randn((lora_num, N, K), dtype=torch.float16, device="cuda")
    lora_indices_tensor = torch.tensor([0, 1, -1, 2], dtype=torch.int32, device="cuda")

    output_base = torch.zeros((batch_size, N), dtype=torch.float16, device="cuda")
    output_opt = torch.zeros((batch_size, N), dtype=torch.float16, device="cuda")

    bgmv_shrink_baseline._bgmv_shrink(inputs, lora_a_weights, output_base, lora_indices_tensor, scaling)
    bgmv_shrink_optimized._bgmv_shrink(inputs, lora_a_weights, output_opt, lora_indices_tensor, scaling)

    ok = torch.allclose(output_base, output_opt, rtol=rtol, atol=atol)
    if not ok:
        diff = torch.max(torch.abs(output_base.float() - output_opt.float())).item()
        print(f"Max diff: {diff:.2e}")
    _report("BGMV Shrink Kernel", ok)
    all_ok = all_ok and ok

    return all_ok


def test_sin_kernel():
    print("\n" + "=" * 80)
    print("Testing Sin Kernel (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-5, 1e-6
    all_ok = True

    test_cases = [
        ("small", torch.randn(16, device="cuda")),
        ("medium", torch.randn(1024, device="cuda")),
        ("large", torch.randn(4096, device="cuda")),
    ]

    for name, x in test_cases:
        y_base = sin_kernel_baseline.call_kernel(x)
        y_opt = sin_kernel_optimized.call_kernel(x)

        ok = torch.allclose(y_base, y_opt, rtol=rtol, atol=atol)
        if not ok:
            diff = torch.max(torch.abs(y_base - y_opt)).item()
            print(f"{name} max diff: {diff:.2e}")
        _report(f"Sin Kernel {name}", ok)
        all_ok = all_ok and ok

    return all_ok


def test_add_value():
    print("\n" + "=" * 80)
    print("Testing Add Value (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-5, 1e-6
    all_ok = True

    test_cases = [
        ("small", torch.randn(16, device="cuda")),
        ("medium", torch.randn(1024, device="cuda")),
        ("large", torch.randn(4096, device="cuda")),
    ]

    for name, x in test_cases:
        y_base = add_value_baseline.puzzle1(x)
        y_opt = add_value_optimized.puzzle1(x)

        ok = torch.allclose(y_base, y_opt, rtol=rtol, atol=atol)
        if not ok:
            diff = torch.max(torch.abs(y_base - y_opt)).item()
            print(f"{name} max diff: {diff:.2e}")
        _report(f"Add Value {name}", ok)
        all_ok = all_ok and ok

    return all_ok


def test_mp_rmsnorm_fused_llama():
    print("\n" + "=" * 80)
    print("Testing RMSNorm Fused LLaMA [mask_percentage] (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-2, 1e-2  # fp16 output
    all_ok = True
    eps = 1e-5

    test_cases = [
        ("small", (2, 64)),
        ("medium", (4, 128)),
    ]

    for name, shape in test_cases:
        x = torch.randn(*shape, dtype=torch.float16, device="cuda")
        weight = torch.randn(shape[-1], dtype=torch.float16, device="cuda")

        y_base = mp_rmsnorm_baseline.rmsnorm_forward(x, weight, eps)
        y_opt = mp_rmsnorm_optimized.rmsnorm_forward(x, weight, eps)

        ok = torch.allclose(y_base, y_opt, rtol=rtol, atol=atol)
        if not ok:
            diff = torch.max(torch.abs(y_base.float() - y_opt.float())).item()
            print(f"{name} max diff: {diff:.2e}")
        _report(f"RMSNorm Fused LLaMA [mp] {name}", ok)
        all_ok = all_ok and ok

    return all_ok


def test_relu_triton_kernel():
    print("\n" + "=" * 80)
    print("Testing ReLU Triton Kernel (baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-5, 1e-6
    all_ok = True

    test_cases = [
        ("small", torch.randn(16, device="cuda")),
        ("medium", torch.randn(1024, device="cuda")),
        ("mixed", torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0], device="cuda")),
    ]

    for name, x in test_cases:
        y_base = relu_kernel_baseline.relu(x)
        y_opt = relu_kernel_optimized.relu(x)

        ok = torch.allclose(y_base, y_opt, rtol=rtol, atol=atol)
        if not ok:
            diff = torch.max(torch.abs(y_base - y_opt)).item()
            print(f"{name} max diff: {diff:.2e}")
        _report(f"ReLU Kernel {name}", ok)
        all_ok = all_ok and ok

    return all_ok


def test_mp_lora_expand_gemv():
    print("\n" + "=" * 80)
    print("Testing LoRA Expand GEMV (mask_percentage: baseline vs optimized)")
    print("=" * 80)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtol, atol = 1e-2, 1e-3
    all_ok = True

    test_cases = [
        ("batch4_hidden128_rank64", 4, 128, 64, 3),
        ("batch8_hidden256_rank32", 8, 256, 32, 4),
    ]

    for name, batch_size, hidden_size, rank, lora_num in test_cases:
        inputs = torch.randn(batch_size, hidden_size, dtype=torch.float16, device="cuda")
        lora_b_weights = torch.randn(lora_num, rank, hidden_size, dtype=torch.float16, device="cuda")
        lora_indices = torch.randint(0, lora_num, (batch_size,), dtype=torch.int32, device="cuda")

        out_base = torch.zeros(batch_size, rank, dtype=torch.float16, device="cuda")
        out_opt = torch.zeros(batch_size, rank, dtype=torch.float16, device="cuda")

        mp_lora_expand_gemv_baseline._bgmv_expand(inputs, lora_b_weights, out_base, lora_indices, add_inputs=False)
        mp_lora_expand_gemv_optimized._bgmv_expand(inputs, lora_b_weights, out_opt, lora_indices, add_inputs=False)

        ok = torch.allclose(out_base, out_opt, rtol=rtol, atol=atol)
        if not ok:
            diff = torch.max(torch.abs(out_base - out_opt)).item()
            print(f"{name} max diff: {diff:.2e}")
        _report(f"LoRA Expand GEMV {name}", ok)
        all_ok = all_ok and ok

    return all_ok


# ============================================================================
# Test registry organized by study
# ============================================================================

STUDY_TEST_FUNCS: Dict[str, Dict[str, Callable[[], bool]]] = {
    "unroll_for_loop": {
        "diag_ssm_triton": test_diag_ssm,
        "fused_recurrent_retention": test_fused_recurrent_retention,
        "fused_recurrent_delta": test_fused_recurrent_delta,
        "fast_rope_embedding": test_fast_rope_embedding,
        "flash_decode2_llama": test_flash_decode2_llama,
        "iv_dependent_matmul": test_iv_dependent_matmul,
        "rmsnorm_fused": test_rmsnorm_fused,
        "rmsnorm_fused_llama": test_rmsnorm_fused_llama,
        "rmsnorm_implementation": test_rmsnorm_implementation,
        "layernorm_fwd_triton": test_layernorm_fwd_triton,
        "var_len_copy": test_var_len_copy,
        "matmul_leakyrelu": test_matmul_leakyrelu,
        "flash_decode2_phi": test_flash_decode2_phi,
        "kldiv_ops": test_kldiv_ops,
        "mean_reduction": test_mean_reduction,
        "softmax_optimize": test_softmax_optimize,
        "triton_conv2d_fwd": test_triton_conv2d_fwd,
        "triton_matmul": test_triton_matmul,
        "matmul_triton1": test_matmul_triton1,
        "lora_expand_gemv": test_lora_expand_gemv,
    },
    "mask_percentage": {
        "quantize_kv_transform": test_quantize_kv_transform,
        "context_attn_llama": test_context_attn_llama,
        "context_attn_fwd": test_context_attn_fwd,
        "bgmv_shrink_kernel": test_bgmv_shrink_kernel,
        "sin_kernel": test_sin_kernel,
        "add_value": test_add_value,
        "rmsnorm_fused_llama": test_mp_rmsnorm_fused_llama,
        "relu_triton_kernel": test_relu_triton_kernel,
        "lora_expand_gemv": test_mp_lora_expand_gemv,
    },
}


def get_all_cases() -> List[str]:
    """Return all case names across all studies."""
    all_cases = []
    for cases in STUDY_CASES.values():
        all_cases.extend(cases)
    return all_cases


def find_studies_for_case(case_name: str) -> List[str]:
    """Return list of studies that contain the given case."""
    return [study for study, cases in STUDY_CASES.items() if case_name in cases]


def main():
    parser = argparse.ArgumentParser(description="Test correctness of optimized Triton kernels against baseline")
    parser.add_argument(
        "-s", "--study",
        type=str,
        choices=STUDY_NAMES + ["all"],
        default="all",
        help="Study to run (default: all)",
    )
    parser.add_argument(
        "-c", "--case",
        type=str,
        choices=get_all_cases(),
        help="Run only the specified case (default: run all cases in selected study)",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all available studies and cases, then exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available studies and cases:")
        for study_name, cases in STUDY_CASES.items():
            print(f"\n{study_name}:")
            for case in cases:
                print(f"  {case}")
        return

    # Determine which studies to run
    if args.study == "all":
        studies_to_run = STUDY_NAMES
    else:
        studies_to_run = [args.study]

    # If a specific case is given, find which study it belongs to
    if args.case:
        matching_studies = find_studies_for_case(args.case)
        if args.study != "all":
            # User specified both --study and --case, use the specified study
            if args.study not in matching_studies:
                print(f"Error: case '{args.case}' not found in study '{args.study}'", file=sys.stderr)
                sys.exit(1)
            studies_to_run = [args.study]
        elif len(matching_studies) > 1:
            # Case exists in multiple studies, require explicit --study
            print(f"Error: case '{args.case}' exists in multiple studies: {', '.join(matching_studies)}", file=sys.stderr)
            print(f"Please specify --study to disambiguate, e.g.: -s {matching_studies[0]} -c {args.case}", file=sys.stderr)
            sys.exit(1)
        else:
            studies_to_run = matching_studies

    all_results = {}

    for study_name in studies_to_run:
        print(f"\n{'=' * 80}")
        print(f"Study: {study_name}")
        print('=' * 80)

        test_funcs = STUDY_TEST_FUNCS[study_name]
        cases_to_run = [args.case] if args.case and args.case in STUDY_CASES[study_name] else STUDY_CASES[study_name]

        for case_name in cases_to_run:
            if case_name in test_funcs:
                all_results[f"{study_name}/{case_name}"] = test_funcs[case_name]()

    all_ok = all(all_results.values())
    print("\n" + "=" * 80)
    print("Overall result:", "PASSED" if all_ok else "FAILED")
    print("=" * 80)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
