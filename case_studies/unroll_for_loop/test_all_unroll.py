import os
import sys
import importlib.util
import torch

ROOT = os.path.dirname(__file__)


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


def main():
    diag_ok = test_diag_ssm()
    retention_ok = test_fused_recurrent_retention()

    all_ok = diag_ok and retention_ok
    print("\n" + "=" * 80)
    print("Overall result:", "PASSED" if all_ok else "FAILED")
    print("=" * 80)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
