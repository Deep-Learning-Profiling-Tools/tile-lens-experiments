import triton
import triton.language as tl
import torch

ROPE_GROUP_SIZE = 4
MAX_FUSED_SIZE : int = 65536

def calculate_settings(n : int) -> (int, int,):
    BLOCK_SIZE : int = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps : int = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps

@triton.jit
def _rope_embedding(
    Q,     Q_row_stride,
    cos, cos_row_stride,
    sin, sin_row_stride,
    seqlen,
    head_dim      : tl.constexpr,
    n_heads       : tl.constexpr,
    BACKWARD_PASS : tl.constexpr,
    BLOCK_SIZE    : tl.constexpr,
):
    """
        Calculates the RoPE Embedding quickly
        RoPE is Q * cos + rotate_half(Q) * sin
        See our blog post for more info
    """
    ROPE_GROUP_SIZE = 4
    row_position  = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets  = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    sin1 = tl.load(sin + (row_position % seqlen)*sin_row_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)
    cos1 = tl.load(cos + (row_position % seqlen)*cos_row_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)

    if BACKWARD_PASS:
        # See our blog post for more info.
        sin1 = -sin1

    # [TODO] Autotune ROPE_GROUP_SIZE to be 1, 2, 4, 8
    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)

    # Loop unrolled - 8 iterations for n_heads=8, ROPE_GROUP_SIZE=4
    # This assumes n_heads=8 from the test case
    # For general case, we check if k < head_end to handle edge cases

    # =========================
    # Precompute all offsets
    # =========================
    # Group 0 (head_start=0, head_end=4)
    k_0 = head_start + 0
    k_1 = head_start + 1
    k_2 = head_start + 2
    k_3 = head_start + 3

    # Since ROPE_GROUP_SIZE=4, the second group starts at head_start+4 only if group_head_position=1
    # But we'll handle both cases using conditional logic

    # Calculate offsets for all 8 iterations
    # First group (k=0,1,2,3)
    offs_q1_0 = row_position * Q_row_stride + k_0 * head_dim + col_offsets
    offs_q2_0 = row_position * Q_row_stride + k_0 * head_dim + col_offsets + half_head_dim

    offs_q1_1 = row_position * Q_row_stride + k_1 * head_dim + col_offsets
    offs_q2_1 = row_position * Q_row_stride + k_1 * head_dim + col_offsets + half_head_dim

    offs_q1_2 = row_position * Q_row_stride + k_2 * head_dim + col_offsets
    offs_q2_2 = row_position * Q_row_stride + k_2 * head_dim + col_offsets + half_head_dim

    offs_q1_3 = row_position * Q_row_stride + k_3 * head_dim + col_offsets
    offs_q2_3 = row_position * Q_row_stride + k_3 * head_dim + col_offsets + half_head_dim

    # =========================
    # Unrolled loop iterations
    # Each iteration is self-contained with its loads, computations, and stores
    # =========================

    # Iteration 0 (k=0 or k=4 depending on group)
    if k_0 < head_end:
        Q1_0 = tl.load(Q + offs_q1_0, mask = mask, other = 0).to(sin1.dtype)
        Q2_0 = tl.load(Q + offs_q2_0, mask = mask, other = 0).to(sin1.dtype)
        rotated_Q1_0 = Q1_0*cos1 - Q2_0*sin1
        rotated_Q2_0 = Q2_0*cos1 + Q1_0*sin1
        tl.store(Q + offs_q1_0, rotated_Q1_0, mask = mask)
        tl.store(Q + offs_q2_0, rotated_Q2_0, mask = mask)

    # Iteration 1 (k=1 or k=5)
    if k_1 < head_end:
        Q1_1 = tl.load(Q + offs_q1_1, mask = mask, other = 0).to(sin1.dtype)
        Q2_1 = tl.load(Q + offs_q2_1, mask = mask, other = 0).to(sin1.dtype)
        rotated_Q1_1 = Q1_1*cos1 - Q2_1*sin1
        rotated_Q2_1 = Q2_1*cos1 + Q1_1*sin1
        tl.store(Q + offs_q1_1, rotated_Q1_1, mask = mask)
        tl.store(Q + offs_q2_1, rotated_Q2_1, mask = mask)

    # Iteration 2 (k=2 or k=6)
    if k_2 < head_end:
        Q1_2 = tl.load(Q + offs_q1_2, mask = mask, other = 0).to(sin1.dtype)
        Q2_2 = tl.load(Q + offs_q2_2, mask = mask, other = 0).to(sin1.dtype)
        rotated_Q1_2 = Q1_2*cos1 - Q2_2*sin1
        rotated_Q2_2 = Q2_2*cos1 + Q1_2*sin1
        tl.store(Q + offs_q1_2, rotated_Q1_2, mask = mask)
        tl.store(Q + offs_q2_2, rotated_Q2_2, mask = mask)

    # Iteration 3 (k=3 or k=7)
    if k_3 < head_end:
        Q1_3 = tl.load(Q + offs_q1_3, mask = mask, other = 0).to(sin1.dtype)
        Q2_3 = tl.load(Q + offs_q2_3, mask = mask, other = 0).to(sin1.dtype)
        rotated_Q1_3 = Q1_3*cos1 - Q2_3*sin1
        rotated_Q2_3 = Q2_3*cos1 + Q1_3*sin1
        tl.store(Q + offs_q1_3, rotated_Q1_3, mask = mask)
        tl.store(Q + offs_q2_3, rotated_Q2_3, mask = mask)


class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        cos, sin = cos.squeeze(), sin.squeeze()
        batch, seq_len, n_heads, head_dim = Q.shape
        Q = Q.reshape(batch*seq_len, n_heads*head_dim)
        n_rows, n_cols = Q.shape
        assert(seq_len <= cos.shape[0])

        # [TODO] Changing blocksize to head_dim//2 seems to have
        # some concurrency / un-deterministic issues.
        BLOCK_SIZE, num_warps = calculate_settings(head_dim//2) # (head_dim//2)

        # group_size = 4 # 4 or 8, too large group_size can hurt performance.
        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        n_groups = div + (mod != 0)

        _rope_embedding[(n_rows, n_groups, )](
              Q,   Q.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len,
            head_dim, n_heads,
            BACKWARD_PASS = False,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.n_groups = n_groups
        ctx.cos = cos
        ctx.sin = sin
        return Q.view(batch, seq_len, n_heads, head_dim)

    @staticmethod
    def backward(ctx, dY):
        batch, seq_len, n_heads, head_dim = dY.shape
        dY = dY.reshape(batch*seq_len, n_heads*head_dim)
        # Must be reshape not view
        n_rows, n_cols = dY.shape

        cos = ctx.cos
        sin = ctx.sin

        _rope_embedding[(n_rows, ctx.n_groups, )](
            dY,  dY .stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len, head_dim, n_heads,
            BACKWARD_PASS = True,
            BLOCK_SIZE = ctx.BLOCK_SIZE,
            num_warps  = ctx.num_warps,
        )
        dY = dY.view(batch, seq_len, n_heads, head_dim)
        return dY, None, None,


def fast_rope_embedding(Q, K, cos, sin):
    Q = Fast_RoPE_Embedding.apply(Q.transpose(1, 2), cos, sin).transpose(1, 2)
    K = Fast_RoPE_Embedding.apply(K.transpose(1, 2), cos, sin).transpose(1, 2)
    return Q, K




##################################################################################################################################################


import torch

def test_fast_rope_embedding_with_backward():
    # Define the test parameters
    batch_size = 2
    seq_len = 4
    n_heads = 8
    head_dim = 16

    # Create random input tensors with requires_grad=True for gradient computation
    Q = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=torch.float32, device='cuda', requires_grad=True)
    K = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=torch.float32, device='cuda', requires_grad=True)

    # Create cos and sin tensors
    cos = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device='cuda')
    sin = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device='cuda')

    # Forward pass using fast_rope_embedding
    Q_out, K_out = fast_rope_embedding(Q, K, cos, sin)

    # Compute a dummy loss function (mean of the outputs)
    loss = Q_out.mean() + K_out.mean()

    # Perform backward propagation
    loss.backward()

    # Collect gradients
    result = {
        "Q_grad": Q.grad,
        "K_grad": K.grad
    }

    return result


# Run the backward test
result_gold = test_fast_rope_embedding_with_backward()
