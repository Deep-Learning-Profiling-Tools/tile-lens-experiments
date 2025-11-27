import argparse
import os
import sys
import importlib.util
import torch

ROOT = os.path.dirname(__file__)

CASE_NAMES = [
    "diag_ssm_triton",
    "fused_recurrent_retention",
    "fused_recurrent_delta",
    "fast_rope_embedding",
    "flash_decode2_llama",
    "iv_dependent_matmul",
    "rmsnorm_fused",
    "rmsnorm_fused_llama",
    "rmsnorm_implementation",
]


def _load_module(name: str, rel_path: str):
    path = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# diag_ssm_triton modules
diag_baseline = _load_module("diag_ssm_baseline", "diag_ssm_triton/baseline.py")
diag_optimized = _load_module("diag_ssm_optimized", "diag_ssm_triton/optimized.py")

# fused_recurrent_retention modules
fr_baseline = _load_module("frr_baseline", "fused_recurrent_retention/baseline.py")
fr_optimized = _load_module("frr_optimized", "fused_recurrent_retention/optimized.py")

# fused_recurrent_delta modules
frd_baseline = _load_module("frd_baseline", "fused_recurrent_delta/baseline.py")
frd_optimized = _load_module("frd_optimized", "fused_recurrent_delta/optimized.py")

# fast_rope_embedding modules
fre_baseline = _load_module("fre_baseline", "fast_rope_embedding/baseline.py")
fre_optimized = _load_module("fre_optimized", "fast_rope_embedding/optimized.py")

# flash_decode2_llama modules
fdll_baseline = _load_module("fdll_baseline", "flash_decode2_llama/baseline.py")
fdll_optimized = _load_module("fdll_optimized", "flash_decode2_llama/optimized.py")

# iv_dependent_matmul modules
ivdm_baseline = _load_module("ivdm_baseline", "iv_dependent_matmul/baseline.py")
ivdm_optimized = _load_module("ivdm_optimized", "iv_dependent_matmul/optimized.py")

# rmsnorm_fused modules
rmsn_baseline = _load_module("rmsn_baseline", "rmsnorm_fused/baseline.py")
rmsn_optimized = _load_module("rmsn_optimized", "rmsnorm_fused/optimized.py")

# rmsnorm_fused_llama modules
rmsnll_baseline = _load_module("rmsnll_baseline", "rmsnorm_fused_llama/baseline.py")
rmsnll_optimized = _load_module("rmsnll_optimized", "rmsnorm_fused_llama/optimized.py")

# rmsnorm_implementation modules
rmsni_baseline = _load_module("rmsni_baseline", "rmsnorm_implementation/baseline.py")
rmsni_optimized = _load_module("rmsni_optimized", "rmsnorm_implementation/optimized.py")


def _report(title: str, ok: bool):
    mark = "✓" if ok else "✗"
    print(f"{mark} {title}: {'PASSED' if ok else 'FAILED'}")


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


TEST_FUNCS = {
    "diag_ssm_triton": test_diag_ssm,
    "fused_recurrent_retention": test_fused_recurrent_retention,
    "fused_recurrent_delta": test_fused_recurrent_delta,
    "fast_rope_embedding": test_fast_rope_embedding,
    "flash_decode2_llama": test_flash_decode2_llama,
    "iv_dependent_matmul": test_iv_dependent_matmul,
    "rmsnorm_fused": test_rmsnorm_fused,
    "rmsnorm_fused_llama": test_rmsnorm_fused_llama,
    "rmsnorm_implementation": test_rmsnorm_implementation,
}


def main():
    parser = argparse.ArgumentParser(description="Test correctness of optimized Triton kernels against baseline")
    parser.add_argument(
        "-c", "--case",
        type=str,
        choices=CASE_NAMES,
        help="Run only the specified case (default: run all cases)",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all available cases and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available cases:")
        for name in CASE_NAMES:
            print(f"  {name}")
        return

    cases_to_run = [args.case] if args.case else CASE_NAMES
    results = {}

    for case_name in cases_to_run:
        test_func = TEST_FUNCS[case_name]
        results[case_name] = test_func()

    all_ok = all(results.values())
    print("\n" + "=" * 80)
    print("Overall result:", "PASSED" if all_ok else "FAILED")
    print("=" * 80)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
