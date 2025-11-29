import torch
import triton
import triton.language as tl


@triton.jit
def iv_dependent_matmul_kernel(a_ptr, b_ptr, c_ptr,  #
                                M, N, K,  #
                                stride_am, stride_ak,  #
                                stride_bk, stride_bn,  #
                                stride_cm, stride_cn,  #
                                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
                                type: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptr = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptr = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Precompute pointers for all 8 iterations
    a_ptrs_0 = a_ptr + 0 * BLOCK_SIZE_K * stride_ak
    b_ptrs_0 = b_ptr + 0 * BLOCK_SIZE_K * stride_bk
    a_ptrs_1 = a_ptr + 1 * BLOCK_SIZE_K * stride_ak
    b_ptrs_1 = b_ptr + 1 * BLOCK_SIZE_K * stride_bk
    a_ptrs_2 = a_ptr + 2 * BLOCK_SIZE_K * stride_ak
    b_ptrs_2 = b_ptr + 2 * BLOCK_SIZE_K * stride_bk
    a_ptrs_3 = a_ptr + 3 * BLOCK_SIZE_K * stride_ak
    b_ptrs_3 = b_ptr + 3 * BLOCK_SIZE_K * stride_bk
    a_ptrs_4 = a_ptr + 4 * BLOCK_SIZE_K * stride_ak
    b_ptrs_4 = b_ptr + 4 * BLOCK_SIZE_K * stride_bk
    a_ptrs_5 = a_ptr + 5 * BLOCK_SIZE_K * stride_ak
    b_ptrs_5 = b_ptr + 5 * BLOCK_SIZE_K * stride_bk
    a_ptrs_6 = a_ptr + 6 * BLOCK_SIZE_K * stride_ak
    b_ptrs_6 = b_ptr + 6 * BLOCK_SIZE_K * stride_bk
    a_ptrs_7 = a_ptr + 7 * BLOCK_SIZE_K * stride_ak
    b_ptrs_7 = b_ptr + 7 * BLOCK_SIZE_K * stride_bk

    # Precompute all masks
    mask_a_0 = offs_k[None, :] < K - 0 * BLOCK_SIZE_K
    mask_b_0 = offs_k[:, None] < K - 0 * BLOCK_SIZE_K
    mask_a_1 = offs_k[None, :] < K - 1 * BLOCK_SIZE_K
    mask_b_1 = offs_k[:, None] < K - 1 * BLOCK_SIZE_K
    mask_a_2 = offs_k[None, :] < K - 2 * BLOCK_SIZE_K
    mask_b_2 = offs_k[:, None] < K - 2 * BLOCK_SIZE_K
    mask_a_3 = offs_k[None, :] < K - 3 * BLOCK_SIZE_K
    mask_b_3 = offs_k[:, None] < K - 3 * BLOCK_SIZE_K
    mask_a_4 = offs_k[None, :] < K - 4 * BLOCK_SIZE_K
    mask_b_4 = offs_k[:, None] < K - 4 * BLOCK_SIZE_K
    mask_a_5 = offs_k[None, :] < K - 5 * BLOCK_SIZE_K
    mask_b_5 = offs_k[:, None] < K - 5 * BLOCK_SIZE_K
    mask_a_6 = offs_k[None, :] < K - 6 * BLOCK_SIZE_K
    mask_b_6 = offs_k[:, None] < K - 6 * BLOCK_SIZE_K
    mask_a_7 = offs_k[None, :] < K - 7 * BLOCK_SIZE_K
    mask_b_7 = offs_k[:, None] < K - 7 * BLOCK_SIZE_K

    # Load all data
    a_0 = tl.load(a_ptrs_0, mask=mask_a_0, other=0.0)
    b_0 = tl.load(b_ptrs_0, mask=mask_b_0, other=0.0)
    a_1 = tl.load(a_ptrs_1, mask=mask_a_1, other=0.0)
    b_1 = tl.load(b_ptrs_1, mask=mask_b_1, other=0.0)
    a_2 = tl.load(a_ptrs_2, mask=mask_a_2, other=0.0)
    b_2 = tl.load(b_ptrs_2, mask=mask_b_2, other=0.0)
    a_3 = tl.load(a_ptrs_3, mask=mask_a_3, other=0.0)
    b_3 = tl.load(b_ptrs_3, mask=mask_b_3, other=0.0)
    a_4 = tl.load(a_ptrs_4, mask=mask_a_4, other=0.0)
    b_4 = tl.load(b_ptrs_4, mask=mask_b_4, other=0.0)
    a_5 = tl.load(a_ptrs_5, mask=mask_a_5, other=0.0)
    b_5 = tl.load(b_ptrs_5, mask=mask_b_5, other=0.0)
    a_6 = tl.load(a_ptrs_6, mask=mask_a_6, other=0.0)
    b_6 = tl.load(b_ptrs_6, mask=mask_b_6, other=0.0)
    a_7 = tl.load(a_ptrs_7, mask=mask_a_7, other=0.0)
    b_7 = tl.load(b_ptrs_7, mask=mask_b_7, other=0.0)

    # Compute all dots
    accumulator += tl.dot(a_0, b_0)
    accumulator += tl.dot(a_1, b_1)
    accumulator += tl.dot(a_2, b_2)
    accumulator += tl.dot(a_3, b_3)
    accumulator += tl.dot(a_4, b_4)
    accumulator += tl.dot(a_5, b_5)
    accumulator += tl.dot(a_6, b_6)
    accumulator += tl.dot(a_7, b_7)

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def iv_dependent_matmul_wrapper(
    M: int, 
    K: int, 
    N: int, 
    BLOCK_SIZE_M: int, 
    BLOCK_SIZE_N: int, 
    BLOCK_SIZE_K: int, 
    type: str = "pre_load",  # Kernel type for scheduling ("pre_load", "post_load", etc.)
    device: torch.device = "cuda"  # Device to run the test (defaults to "cuda")
):
    # Ensure the device is correct
    device = torch.device(device)

    # Generate random input matrices a and b on the specified device
    a = torch.rand((M, K), device=device)
    b = torch.rand((K, N), device=device)

    # Create an empty tensor to store the Triton result
    triton_output = torch.empty((M, N), device=device)

    # Define Triton grid configuration
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    # Set the number of stages based on the kernel type
    num_stages = 4 if type == "post_load_three_iters" else 3

    # Run the Triton kernel
    iv_dependent_matmul_kernel[grid](
        a, b, triton_output, M, N, K,  #
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),  #
        triton_output.stride(0), triton_output.stride(1),  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, type=type,  #
        num_stages=num_stages
    )

    # Optionally print the result for inspection
    # print(triton_output)

    return triton_output




##################################################################################################################################################


import torch

# 封装 IV Dependent MatMul 测试的函数
def test_iv_dependent_matmul_kernel():
    # 定义矩阵维度和块大小
    M = 256
    K = 256
    N = 256
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    # 创建 CUDA 设备
    device = torch.device('cuda')

    # 定义所有类型的内核配置
    types = [
        "pre_load",
        "post_load",
        "post_pre_mixed",
        "post_load_two_iters",
        "post_load_three_iters"
    ]

    # 字典用于存储每个测试用例的结果
    results = {}

    # 遍历每种内核类型，进行测试
    for i, type in enumerate(types):
        # 调用封装函数运行 Triton 核心
        triton_output = iv_dependent_matmul_wrapper(
            M=M,
            K=K,
            N=N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            type=type,
            device=device
        )

        # 确保输出的大小正确
        assert triton_output.shape == (M, N), f"Expected output shape {(M, N)} but got {triton_output.shape} for type {type}"

        # 保存结果到字典
        results[f"test_case_{i+1}"] = triton_output

    return results

# 执行测试函数进行所有类型的验证
result_gold = test_iv_dependent_matmul_kernel()
