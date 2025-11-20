import torch
import triton
import triton.language as tl

@triton.jit
def diag_ssm_forward_kernel(s_ptr, x_ptr, lambda_ptr, y_ptr, length,
                            batch_size, dim, BLOCK_SIZE: tl.constexpr):
    """
    前向传播核函数（实数版本）- 循环展开5次

    参数:
        s_ptr: [batch_size, dim]
        x_ptr: [length, batch_size, dim]
        lambda_ptr: [dim]
        y_ptr: [length, batch_size, dim]
    """
    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim
    s = tl.load(s_ptr + col_offsets, mask=mask, other=0)
    Lambda = tl.load(lambda_ptr + col_offsets % dim, mask=mask, other=0)

    # Initialization
    s_0 = s

    # Precompute offsets for each iteration
    offsets_0 = 0 * batch_size * dim + col_offsets
    offsets_1 = 1 * batch_size * dim + col_offsets
    offsets_2 = 2 * batch_size * dim + col_offsets
    offsets_3 = 3 * batch_size * dim + col_offsets
    offsets_4 = 4 * batch_size * dim + col_offsets

    # =========================
    # Pull all tl.load as early as possible
    # =========================
    x_0 = tl.load(x_ptr + offsets_0, mask=mask, other=0)
    x_1 = tl.load(x_ptr + offsets_1, mask=mask, other=0)
    x_2 = tl.load(x_ptr + offsets_2, mask=mask, other=0)
    x_3 = tl.load(x_ptr + offsets_3, mask=mask, other=0)
    x_4 = tl.load(x_ptr + offsets_4, mask=mask, other=0)

    # =========================
    # Iterative computation
    # =========================
    # iter 0
    s_1 = s_0 * Lambda + x_0

    # iter 1
    s_2 = s_1 * Lambda + x_1

    # iter 2
    s_3 = s_2 * Lambda + x_2

    # iter 3
    s_4 = s_3 * Lambda + x_3

    # iter 4
    s_5 = s_4 * Lambda + x_4

    # =========================
    # Defer all tl.store until the end
    # =========================
    tl.store(y_ptr + offsets_0, s_1, mask=mask)
    tl.store(y_ptr + offsets_1, s_2, mask=mask)
    tl.store(y_ptr + offsets_2, s_3, mask=mask)
    tl.store(y_ptr + offsets_3, s_4, mask=mask)
    tl.store(y_ptr + offsets_4, s_5, mask=mask)

@triton.jit
def diag_ssm_backward_kernel(
        s_ptr, lambda_ptr, y_ptr, grad_s_ptr, grad_x_ptr, grad_lambda_ptr,
        grad_y_ptr, length, batch_size, dim, BLOCK_SIZE: tl.constexpr):
    """
    反向传播核函数（实数版本）- 循环展开5次

    参数:
        s_ptr: [batch_size, dim]
        lambda_ptr: [dim]
        y_ptr: [length, batch_size, dim]
        grad_s_ptr: [batch_size, dim]
        grad_x_ptr: [length, batch_size, dim]
        grad_lambda_ptr: [batch_size, dim]
        grad_y_ptr: [length, batch_size, dim]
    """

    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim

    Lambda = tl.load(lambda_ptr + col_offsets % dim, mask=mask, other=0)

    # 初始化梯度为零
    grad_s_0 = tl.zeros_like(Lambda)
    grad_Lambda_0 = tl.zeros_like(Lambda)

    # Precompute t values (reverse traversal)
    t_0 = length - 1 - 0  # = length - 1
    t_1 = length - 1 - 1  # = length - 2
    t_2 = length - 1 - 2  # = length - 3
    t_3 = length - 1 - 3  # = length - 4
    t_4 = length - 1 - 4  # = length - 5

    # Precompute offsets
    offsets_0 = t_0 * batch_size * dim + col_offsets
    offsets_1 = t_1 * batch_size * dim + col_offsets
    offsets_2 = t_2 * batch_size * dim + col_offsets
    offsets_3 = t_3 * batch_size * dim + col_offsets
    offsets_4 = t_4 * batch_size * dim + col_offsets

    # =========================
    # Pull all tl.load as early as possible
    # =========================
    grad_y_0 = tl.load(grad_y_ptr + offsets_0, mask=mask, other=0)
    grad_y_1 = tl.load(grad_y_ptr + offsets_1, mask=mask, other=0)
    grad_y_2 = tl.load(grad_y_ptr + offsets_2, mask=mask, other=0)
    grad_y_3 = tl.load(grad_y_ptr + offsets_3, mask=mask, other=0)
    grad_y_4 = tl.load(grad_y_ptr + offsets_4, mask=mask, other=0)

    # Load s values with conditional logic
    # iter 0: t_0 = length - 1 (always > 0 for length >= 5)
    s_0 = tl.load(y_ptr + offsets_0 - batch_size * dim, mask=mask, other=0)

    # iter 1: t_1 = length - 2 (always > 0 for length >= 5)
    s_1 = tl.load(y_ptr + offsets_1 - batch_size * dim, mask=mask, other=0)

    # iter 2: t_2 = length - 3 (always > 0 for length >= 5)
    s_2 = tl.load(y_ptr + offsets_2 - batch_size * dim, mask=mask, other=0)

    # iter 3: t_3 = length - 4 (always > 0 for length >= 5)
    s_3 = tl.load(y_ptr + offsets_3 - batch_size * dim, mask=mask, other=0)

    # iter 4: t_4 = length - 5 (could be 0 if length == 5)
    # Need conditional: if t_4 > 0, load from y_ptr; else load from s_ptr
    s_4 = tl.where(t_4 > 0,
                   tl.load(y_ptr + offsets_4 - batch_size * dim, mask=mask, other=0),
                   tl.load(s_ptr + col_offsets, mask=mask, other=0))

    # =========================
    # Iterative computation
    # =========================
    # iter 0
    grad_s_0 = grad_y_0 + grad_s_0
    grad_x_0 = grad_s_0
    grad_Lambda_1 = grad_Lambda_0 + grad_s_0 * s_0
    grad_s_1 = grad_s_0 * Lambda

    # iter 1
    grad_s_1 = grad_y_1 + grad_s_1
    grad_x_1 = grad_s_1
    grad_Lambda_2 = grad_Lambda_1 + grad_s_1 * s_1
    grad_s_2 = grad_s_1 * Lambda

    # iter 2
    grad_s_2 = grad_y_2 + grad_s_2
    grad_x_2 = grad_s_2
    grad_Lambda_3 = grad_Lambda_2 + grad_s_2 * s_2
    grad_s_3 = grad_s_2 * Lambda

    # iter 3
    grad_s_3 = grad_y_3 + grad_s_3
    grad_x_3 = grad_s_3
    grad_Lambda_4 = grad_Lambda_3 + grad_s_3 * s_3
    grad_s_4 = grad_s_3 * Lambda

    # iter 4
    grad_s_4 = grad_y_4 + grad_s_4
    grad_x_4 = grad_s_4
    grad_Lambda_5 = grad_Lambda_4 + grad_s_4 * s_4
    grad_s_5 = grad_s_4 * Lambda

    # =========================
    # Defer all tl.store until the end
    # =========================
    tl.store(grad_x_ptr + offsets_0, grad_x_0, mask=mask)
    tl.store(grad_x_ptr + offsets_1, grad_x_1, mask=mask)
    tl.store(grad_x_ptr + offsets_2, grad_x_2, mask=mask)
    tl.store(grad_x_ptr + offsets_3, grad_x_3, mask=mask)
    tl.store(grad_x_ptr + offsets_4, grad_x_4, mask=mask)

    tl.store(grad_s_ptr + col_offsets, grad_s_5, mask=mask)
    tl.store(grad_lambda_ptr + col_offsets, grad_Lambda_5, mask=mask)

@triton.jit
def diag_ssm_forward_kernel_complex(s_ptr, x_ptr, y_ptr, lambda_ptr,
                                    length, batch_size, dim,
                                    BLOCK_SIZE: tl.constexpr):
    """
    前向传播核函数（复数版本）- 循环展开5次

    参数:
        s_ptr: [batch_size, dim, 2]
        x_ptr: [length, batch_size, dim, 2]
        lambda_ptr: [dim, 2]
        y_ptr: [length, batch_size, dim, 2]
    """
    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim

    # 加载's'和'Lambda'的实部和虚部
    s_real_0 = tl.load(s_ptr + col_offsets * 2, mask=mask, other=0)
    s_imag_0 = tl.load(s_ptr + col_offsets * 2 + 1, mask=mask, other=0)
    lambda_real = tl.load(
        lambda_ptr + (col_offsets % dim) * 2, mask=mask, other=0)
    lambda_imag = tl.load(
        lambda_ptr + (col_offsets % dim) * 2 + 1, mask=mask, other=0)

    # Precompute offsets
    offsets_0 = (0 * batch_size * dim + col_offsets) * 2
    offsets_1 = (1 * batch_size * dim + col_offsets) * 2
    offsets_2 = (2 * batch_size * dim + col_offsets) * 2
    offsets_3 = (3 * batch_size * dim + col_offsets) * 2
    offsets_4 = (4 * batch_size * dim + col_offsets) * 2

    # =========================
    # Pull all tl.load as early as possible
    # =========================
    x_real_0 = tl.load(x_ptr + offsets_0, mask=mask, other=0)
    x_imag_0 = tl.load(x_ptr + offsets_0 + 1, mask=mask, other=0)

    x_real_1 = tl.load(x_ptr + offsets_1, mask=mask, other=0)
    x_imag_1 = tl.load(x_ptr + offsets_1 + 1, mask=mask, other=0)

    x_real_2 = tl.load(x_ptr + offsets_2, mask=mask, other=0)
    x_imag_2 = tl.load(x_ptr + offsets_2 + 1, mask=mask, other=0)

    x_real_3 = tl.load(x_ptr + offsets_3, mask=mask, other=0)
    x_imag_3 = tl.load(x_ptr + offsets_3 + 1, mask=mask, other=0)

    x_real_4 = tl.load(x_ptr + offsets_4, mask=mask, other=0)
    x_imag_4 = tl.load(x_ptr + offsets_4 + 1, mask=mask, other=0)

    # =========================
    # Iterative computation (complex multiplication and addition)
    # =========================
    # iter 0
    new_s_real_1 = s_real_0 * lambda_real - s_imag_0 * lambda_imag + x_real_0
    new_s_imag_1 = s_real_0 * lambda_imag + s_imag_0 * lambda_real + x_imag_0

    # iter 1
    new_s_real_2 = new_s_real_1 * lambda_real - new_s_imag_1 * lambda_imag + x_real_1
    new_s_imag_2 = new_s_real_1 * lambda_imag + new_s_imag_1 * lambda_real + x_imag_1

    # iter 2
    new_s_real_3 = new_s_real_2 * lambda_real - new_s_imag_2 * lambda_imag + x_real_2
    new_s_imag_3 = new_s_real_2 * lambda_imag + new_s_imag_2 * lambda_real + x_imag_2

    # iter 3
    new_s_real_4 = new_s_real_3 * lambda_real - new_s_imag_3 * lambda_imag + x_real_3
    new_s_imag_4 = new_s_real_3 * lambda_imag + new_s_imag_3 * lambda_real + x_imag_3

    # iter 4
    new_s_real_5 = new_s_real_4 * lambda_real - new_s_imag_4 * lambda_imag + x_real_4
    new_s_imag_5 = new_s_real_4 * lambda_imag + new_s_imag_4 * lambda_real + x_imag_4

    # =========================
    # Defer all tl.store until the end
    # =========================
    tl.store(y_ptr + offsets_0, new_s_real_1, mask=mask)
    tl.store(y_ptr + offsets_0 + 1, new_s_imag_1, mask=mask)

    tl.store(y_ptr + offsets_1, new_s_real_2, mask=mask)
    tl.store(y_ptr + offsets_1 + 1, new_s_imag_2, mask=mask)

    tl.store(y_ptr + offsets_2, new_s_real_3, mask=mask)
    tl.store(y_ptr + offsets_2 + 1, new_s_imag_3, mask=mask)

    tl.store(y_ptr + offsets_3, new_s_real_4, mask=mask)
    tl.store(y_ptr + offsets_3 + 1, new_s_imag_4, mask=mask)

    tl.store(y_ptr + offsets_4, new_s_real_5, mask=mask)
    tl.store(y_ptr + offsets_4 + 1, new_s_imag_5, mask=mask)

@triton.jit
def diag_ssm_backward_kernel_complex(
        s_ptr, lambda_ptr, y_ptr, grad_s_ptr, grad_x_ptr, grad_lambda_ptr,
        grad_y_ptr, length, batch_size, dim, BLOCK_SIZE: tl.constexpr):
    """
    反向传播核函数（复数版本）- 循环展开5次

    参数:
        s_ptr: [batch_size, dim, 2]
        lambda_ptr: [dim, 2]
        y_ptr: [length, batch_size, dim, 2]
        grad_s_ptr: [batch_size, dim, 2]
        grad_x_ptr: [length, batch_size, dim, 2]
        grad_lambda_ptr: [batch_size, dim, 2]
        grad_y_ptr: [length, batch_size, dim, 2]
    """

    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim

    # 加载'Lambda'的实部和虚部
    lambda_real = tl.load(
        lambda_ptr + (col_offsets % dim) * 2, mask=mask, other=0)
    lambda_imag = tl.load(
        lambda_ptr + (col_offsets % dim) * 2 + 1, mask=mask, other=0)

    # 初始化梯度为零
    grad_s_real_0 = tl.zeros_like(lambda_real)
    grad_s_imag_0 = tl.zeros_like(lambda_imag)
    grad_lambda_real_0 = tl.zeros_like(lambda_real)
    grad_lambda_imag_0 = tl.zeros_like(lambda_imag)

    # Precompute t values (reverse traversal)
    t_0 = length - 1 - 0
    t_1 = length - 1 - 1
    t_2 = length - 1 - 2
    t_3 = length - 1 - 3
    t_4 = length - 1 - 4

    # Precompute offsets
    offsets_0 = (t_0 * batch_size * dim + col_offsets) * 2
    offsets_1 = (t_1 * batch_size * dim + col_offsets) * 2
    offsets_2 = (t_2 * batch_size * dim + col_offsets) * 2
    offsets_3 = (t_3 * batch_size * dim + col_offsets) * 2
    offsets_4 = (t_4 * batch_size * dim + col_offsets) * 2

    # =========================
    # Pull all tl.load as early as possible
    # =========================
    grad_y_real_0 = tl.load(grad_y_ptr + offsets_0, mask=mask, other=0)
    grad_y_imag_0 = -tl.load(grad_y_ptr + offsets_0 + 1, mask=mask, other=0)

    grad_y_real_1 = tl.load(grad_y_ptr + offsets_1, mask=mask, other=0)
    grad_y_imag_1 = -tl.load(grad_y_ptr + offsets_1 + 1, mask=mask, other=0)

    grad_y_real_2 = tl.load(grad_y_ptr + offsets_2, mask=mask, other=0)
    grad_y_imag_2 = -tl.load(grad_y_ptr + offsets_2 + 1, mask=mask, other=0)

    grad_y_real_3 = tl.load(grad_y_ptr + offsets_3, mask=mask, other=0)
    grad_y_imag_3 = -tl.load(grad_y_ptr + offsets_3 + 1, mask=mask, other=0)

    grad_y_real_4 = tl.load(grad_y_ptr + offsets_4, mask=mask, other=0)
    grad_y_imag_4 = -tl.load(grad_y_ptr + offsets_4 + 1, mask=mask, other=0)

    # Load s values with conditional logic
    # iter 0: t_0 = length - 1 (always > 0 for length >= 5)
    s_real_0 = tl.load(y_ptr + offsets_0 - 2 * batch_size * dim, mask=mask, other=0)
    s_imag_0 = tl.load(y_ptr + offsets_0 - 2 * batch_size * dim + 1, mask=mask, other=0)

    # iter 1: t_1 = length - 2 (always > 0 for length >= 5)
    s_real_1 = tl.load(y_ptr + offsets_1 - 2 * batch_size * dim, mask=mask, other=0)
    s_imag_1 = tl.load(y_ptr + offsets_1 - 2 * batch_size * dim + 1, mask=mask, other=0)

    # iter 2: t_2 = length - 3 (always > 0 for length >= 5)
    s_real_2 = tl.load(y_ptr + offsets_2 - 2 * batch_size * dim, mask=mask, other=0)
    s_imag_2 = tl.load(y_ptr + offsets_2 - 2 * batch_size * dim + 1, mask=mask, other=0)

    # iter 3: t_3 = length - 4 (always > 0 for length >= 5)
    s_real_3 = tl.load(y_ptr + offsets_3 - 2 * batch_size * dim, mask=mask, other=0)
    s_imag_3 = tl.load(y_ptr + offsets_3 - 2 * batch_size * dim + 1, mask=mask, other=0)

    # iter 4: t_4 = length - 5 (could be 0 if length == 5)
    s_real_4 = tl.where(t_4 > 0,
                        tl.load(y_ptr + offsets_4 - 2 * batch_size * dim, mask=mask, other=0),
                        tl.load(s_ptr + 2 * col_offsets, mask=mask, other=0))
    s_imag_4 = tl.where(t_4 > 0,
                        tl.load(y_ptr + offsets_4 - 2 * batch_size * dim + 1, mask=mask, other=0),
                        tl.load(s_ptr + 2 * col_offsets + 1, mask=mask, other=0))

    # =========================
    # Iterative computation
    # =========================
    # iter 0
    grad_s_real_0 = grad_y_real_0 + grad_s_real_0
    grad_s_imag_0 = grad_y_imag_0 + grad_s_imag_0
    grad_x_real_0 = grad_s_real_0
    grad_x_imag_0 = grad_s_imag_0
    grad_lambda_real_1 = grad_lambda_real_0 + (grad_s_real_0 * s_real_0 - grad_s_imag_0 * s_imag_0)
    grad_lambda_imag_1 = grad_lambda_imag_0 + (grad_s_real_0 * s_imag_0 + grad_s_imag_0 * s_real_0)
    grad_s_real_1 = grad_x_real_0 * lambda_real - grad_x_imag_0 * lambda_imag
    grad_s_imag_1 = grad_x_real_0 * lambda_imag + grad_x_imag_0 * lambda_real

    # iter 1
    grad_s_real_1 = grad_y_real_1 + grad_s_real_1
    grad_s_imag_1 = grad_y_imag_1 + grad_s_imag_1
    grad_x_real_1 = grad_s_real_1
    grad_x_imag_1 = grad_s_imag_1
    grad_lambda_real_2 = grad_lambda_real_1 + (grad_s_real_1 * s_real_1 - grad_s_imag_1 * s_imag_1)
    grad_lambda_imag_2 = grad_lambda_imag_1 + (grad_s_real_1 * s_imag_1 + grad_s_imag_1 * s_real_1)
    grad_s_real_2 = grad_x_real_1 * lambda_real - grad_x_imag_1 * lambda_imag
    grad_s_imag_2 = grad_x_real_1 * lambda_imag + grad_x_imag_1 * lambda_real

    # iter 2
    grad_s_real_2 = grad_y_real_2 + grad_s_real_2
    grad_s_imag_2 = grad_y_imag_2 + grad_s_imag_2
    grad_x_real_2 = grad_s_real_2
    grad_x_imag_2 = grad_s_imag_2
    grad_lambda_real_3 = grad_lambda_real_2 + (grad_s_real_2 * s_real_2 - grad_s_imag_2 * s_imag_2)
    grad_lambda_imag_3 = grad_lambda_imag_2 + (grad_s_real_2 * s_imag_2 + grad_s_imag_2 * s_real_2)
    grad_s_real_3 = grad_x_real_2 * lambda_real - grad_x_imag_2 * lambda_imag
    grad_s_imag_3 = grad_x_real_2 * lambda_imag + grad_x_imag_2 * lambda_real

    # iter 3
    grad_s_real_3 = grad_y_real_3 + grad_s_real_3
    grad_s_imag_3 = grad_y_imag_3 + grad_s_imag_3
    grad_x_real_3 = grad_s_real_3
    grad_x_imag_3 = grad_s_imag_3
    grad_lambda_real_4 = grad_lambda_real_3 + (grad_s_real_3 * s_real_3 - grad_s_imag_3 * s_imag_3)
    grad_lambda_imag_4 = grad_lambda_imag_3 + (grad_s_real_3 * s_imag_3 + grad_s_imag_3 * s_real_3)
    grad_s_real_4 = grad_x_real_3 * lambda_real - grad_x_imag_3 * lambda_imag
    grad_s_imag_4 = grad_x_real_3 * lambda_imag + grad_x_imag_3 * lambda_real

    # iter 4
    grad_s_real_4 = grad_y_real_4 + grad_s_real_4
    grad_s_imag_4 = grad_y_imag_4 + grad_s_imag_4
    grad_x_real_4 = grad_s_real_4
    grad_x_imag_4 = grad_s_imag_4
    grad_lambda_real_5 = grad_lambda_real_4 + (grad_s_real_4 * s_real_4 - grad_s_imag_4 * s_imag_4)
    grad_lambda_imag_5 = grad_lambda_imag_4 + (grad_s_real_4 * s_imag_4 + grad_s_imag_4 * s_real_4)
    grad_s_real_5 = grad_x_real_4 * lambda_real - grad_x_imag_4 * lambda_imag
    grad_s_imag_5 = grad_x_real_4 * lambda_imag + grad_x_imag_4 * lambda_real

    # =========================
    # Defer all tl.store until the end
    # =========================
    tl.store(grad_x_ptr + offsets_0, grad_x_real_0, mask=mask)
    tl.store(grad_x_ptr + offsets_0 + 1, -grad_x_imag_0, mask=mask)

    tl.store(grad_x_ptr + offsets_1, grad_x_real_1, mask=mask)
    tl.store(grad_x_ptr + offsets_1 + 1, -grad_x_imag_1, mask=mask)

    tl.store(grad_x_ptr + offsets_2, grad_x_real_2, mask=mask)
    tl.store(grad_x_ptr + offsets_2 + 1, -grad_x_imag_2, mask=mask)

    tl.store(grad_x_ptr + offsets_3, grad_x_real_3, mask=mask)
    tl.store(grad_x_ptr + offsets_3 + 1, -grad_x_imag_3, mask=mask)

    tl.store(grad_x_ptr + offsets_4, grad_x_real_4, mask=mask)
    tl.store(grad_x_ptr + offsets_4 + 1, -grad_x_imag_4, mask=mask)

    # 存储最终的梯度
    tl.store(grad_s_ptr + col_offsets * 2, grad_s_real_5, mask=mask)
    tl.store(grad_s_ptr + col_offsets * 2 + 1, -grad_s_imag_5, mask=mask)
    tl.store(
        grad_lambda_ptr + col_offsets * 2, grad_lambda_real_5, mask=mask)
    tl.store(
        grad_lambda_ptr + col_offsets * 2 + 1,
        -grad_lambda_imag_5,
        mask=mask)

class _ssm_forward(torch.autograd.Function):
    # TODO 使用 @triton.autotune 选择最佳的 BLOCK_SIZE
    # 对于3090，BLOCK_SIZE = 128似乎效果良好
    BLOCK_SIZE = 128

    @staticmethod
    def forward(ctx, s, x, Lambda):
        assert s.is_contiguous() and x.is_contiguous() and Lambda.is_contiguous()
        length, batch_size, dim = x.shape
        n = batch_size * dim
        y = torch.zeros_like(x)
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )

        if Lambda.dtype == torch.complex64:
            # 确保s和x是复数张量
            if not torch.is_complex(s):
                raise ValueError("当Lambda为复数时，s必须是复数张量")
            if not torch.is_complex(x):
                raise ValueError("当Lambda为复数时，x必须是复数张量")
            diag_ssm_forward_kernel_complex[grid](
                torch.view_as_real(s), torch.view_as_real(x),
                torch.view_as_real(y), torch.view_as_real(Lambda), length,
                batch_size, dim, _ssm_forward.BLOCK_SIZE)
        elif Lambda.dtype.is_floating_point:
            diag_ssm_forward_kernel[grid](s, x, Lambda, y, length,
                                          batch_size, dim,
                                          _ssm_forward.BLOCK_SIZE)
        else:
            raise ValueError("不支持的 dtype: %s" % Lambda.dtype)
        ctx.save_for_backward(s, y, Lambda)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        s, y, Lambda = ctx.saved_tensors
        length, batch_size, dim = y.shape
        grad_y = grad_y.contiguous()
        n = batch_size * dim
        grad_s = torch.empty_like(s)
        grad_x = torch.empty_like(grad_y)
        # grad_lambda 存储每个批次中 Lambda 的梯度
        # 我们将在内核完成后进行求和
        grad_lambda = torch.empty_like(s)
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )
        if Lambda.dtype == torch.complex64:
            diag_ssm_backward_kernel_complex[grid](
                torch.view_as_real(s), torch.view_as_real(Lambda),
                torch.view_as_real(y), torch.view_as_real(grad_s),
                torch.view_as_real(grad_x),
                torch.view_as_real(grad_lambda),
                torch.view_as_real(grad_y), length, batch_size, dim,
                _ssm_forward.BLOCK_SIZE)
        else:
            diag_ssm_backward_kernel[grid](
                s, Lambda, y, grad_s, grad_x, grad_lambda, grad_y, length,
                batch_size, dim, _ssm_forward.BLOCK_SIZE)
        return grad_s, grad_x, grad_lambda.sum(dim=0)

diag_ssm_forward_triton = _ssm_forward.apply

##################################################################################################################################################

def test_diag_ssm_triton():
    # 测试参数
    batch_size, dim, length = 2, 3, 5  # 定义测试张量的维度
    BLOCK_SIZE = 128  # Triton核的块大小

    # 初始化输入张量，确保 requires_grad=True
    # 实数张量
    s_real = torch.randn((batch_size, dim), dtype=torch.float32, device="cuda", requires_grad=True)
    x_real = torch.randn((length, batch_size, dim), dtype=torch.float32, device="cuda", requires_grad=True)
    Lambda_real = torch.rand((dim,), dtype=torch.float32, device="cuda", requires_grad=True)

    # 复数张量
    s_complex = torch.randn((batch_size, dim), dtype=torch.complex64, device="cuda", requires_grad=True)
    x_complex = torch.randn((length, batch_size, dim), dtype=torch.complex64, device="cuda", requires_grad=True)
    Lambda_complex = torch.rand((dim,), dtype=torch.complex64, device="cuda", requires_grad=True)

    # Triton前向传播，对于实数Lambda
    y_triton_real = diag_ssm_forward_triton(s_real, x_real, Lambda_real)
    # Triton前向传播，对于复数Lambda
    y_triton_complex = diag_ssm_forward_triton(s_complex, x_complex, Lambda_complex)

    # Triton反向传播，对于实数Lambda
    grad_output_real = torch.ones_like(y_triton_real, device="cuda")
    y_triton_real.backward(grad_output_real)
    # Triton反向传播，对于复数Lambda
    grad_output_complex = torch.ones_like(y_triton_complex, device="cuda")
    y_triton_complex.backward(grad_output_complex)

    results = {
        "test_case_1": {
            "y_triton_real": y_triton_real,
            "grad_s_real": s_real.grad.clone(),
            "grad_x_real": x_real.grad.clone(),
            "grad_Lambda_real": Lambda_real.grad.clone(),
        },
        "test_case_2": {
            "y_triton_complex": y_triton_complex,
            "grad_s_complex": s_complex.grad.clone(),
            "grad_x_complex": x_complex.grad.clone(),
            "grad_Lambda_complex": Lambda_complex.grad.clone(),
        }
    }

    return results

if __name__ == "__main__":
    result_gold = test_diag_ssm_triton()
    # 输出结果
    for test_case, outputs in result_gold.items():
        print(f"{test_case}:")
        for name, tensor in outputs.items():
            print(f"  {name}: {tensor}")
