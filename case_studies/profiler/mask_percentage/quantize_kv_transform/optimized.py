import torch

import triton
import triton.language as tl

# Add profiling
import triton_viz
from triton_viz.clients.profiler.profiler import Profiler

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K,
    Dest_loc,
    Out,
    Out_scale,
    stride_k_bs,
    stride_k_h,
    stride_k_d,
    stride_o_bs,
    stride_o_h,
    stride_o_d,
    stride_os_bs,
    stride_os_h,
    stride_os_d,
    head_num,
    head_dim,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    N_D_TILES: tl.constexpr,
):
    cur_index = tl.program_id(0)
    pid_hg = tl.program_id(1)
    h_start = pid_hg * BLOCK_HEAD
    offs_h = h_start + tl.arange(0, BLOCK_HEAD)
    mask_h = offs_h < head_num

    dest_index = tl.load(Dest_loc + cur_index)

    running_max = tl.full((BLOCK_HEAD,), 0.0, tl.float32)
    for t in tl.static_range(N_D_TILES):
        d_start = t * BLOCK_DMODEL
        offs_d = d_start + tl.arange(0, BLOCK_DMODEL)
        mask_d = offs_d < head_dim
        k_ptrs = K + cur_index * stride_k_bs + offs_h[:, None] * stride_k_h + stride_k_d * offs_d[None, :]
        vals = tl.load(k_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0)
        tile_max = tl.max(tl.abs(vals).to(tl.float32), axis=1)
        running_max = tl.maximum(running_max, tile_max)

    data_scale = (running_max / 127.0).to(Out_scale.dtype.element_ty)[:, None]

    for t in tl.static_range(N_D_TILES):
        d_start = t * BLOCK_DMODEL
        offs_d = d_start + tl.arange(0, BLOCK_DMODEL)
        mask_d = offs_d < head_dim
        k_ptrs = K + cur_index * stride_k_bs + offs_h[:, None] * stride_k_h + stride_k_d * offs_d[None, :]
        src_data = tl.load(k_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0)
        q_src_data = (src_data / data_scale).to(tl.int8)
        o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]
        tl.store(o_ptrs, q_src_data, mask=mask_h[:, None] & mask_d[None, :])

    os_ptrs = Out_scale + dest_index * stride_os_bs + stride_os_h * offs_h[:, None]
    tl.store(os_ptrs, data_scale, mask=mask_h[:, None])


@torch.no_grad()
def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale):
    seq_len = DestLoc.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    assert K.shape[1] == Out.shape[1] and K.shape[2] == Out.shape[2]

    BLOCK_HEAD = 1 << (head_num.bit_length() - 1) if head_num > 0 else 1
    tmp = 1 << (head_dim.bit_length() - 1) if head_dim > 0 else 1
    while head_dim % tmp != 0 and tmp > 1:
        tmp >>= 1
    BLOCK_DMODEL = tmp
    N_D_TILES = (head_dim + BLOCK_DMODEL - 1) // BLOCK_DMODEL

    grid = (seq_len, (head_num + BLOCK_HEAD - 1) // BLOCK_HEAD)
    num_warps = 1

    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K,
        DestLoc,
        Out,
        Out_scale,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        Out_scale.stride(0),
        Out_scale.stride(1),
        Out_scale.stride(2),
        head_num,
        head_dim,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_HEAD=BLOCK_HEAD,
        N_D_TILES=N_D_TILES,
        num_warps=num_warps,
        num_stages=1,
    )
    return




##################################################################################################################################################


def test_destindex_copy_quantize_kv():
    B, N_CTX, H, D = 32, 1024, 12, 96
    src = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda()
    dest_loc = torch.arange(0, B * N_CTX, dtype=torch.int32).cuda()
    value_dest = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda().to(torch.int8)
    scale_dest = torch.randn((B * N_CTX, H, 1), dtype=torch.float16).cuda()

    # Test case 1
    destindex_copy_quantize_kv(src, dest_loc, value_dest, scale_dest)
    result_1 = {
        "value_dest": value_dest.clone(),
        "scale_dest": scale_dest.clone()
    }

    # Test case 2: Different dimensions
    B, N_CTX, H, D = 16, 512, 8, 64
    src = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda()
    dest_loc = torch.arange(0, B * N_CTX, dtype=torch.int32).cuda()
    value_dest = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda().to(torch.int8)
    scale_dest = torch.randn((B * N_CTX, H, 1), dtype=torch.float16).cuda()
    destindex_copy_quantize_kv(src, dest_loc, value_dest, scale_dest)
    result_2 = {
        "value_dest": value_dest.clone(),
        "scale_dest": scale_dest.clone()
    }

    # Test case 3: Different data types
    src = torch.randn((B * N_CTX, H, D), dtype=torch.float32).cuda()
    destindex_copy_quantize_kv(src, dest_loc, value_dest, scale_dest)
    result_3 = {
        "value_dest": value_dest.clone(),
        "scale_dest": scale_dest.clone()
    }

    # Test case 4: Edge case with minimal dimensions
    B, N_CTX, H, D = 1, 1, 1, 1
    src = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda()
    dest_loc = torch.arange(0, B * N_CTX, dtype=torch.int32).cuda()
    value_dest = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda().to(torch.int8)
    scale_dest = torch.randn((B * N_CTX, H, 1), dtype=torch.float16).cuda()
    destindex_copy_quantize_kv(src, dest_loc, value_dest, scale_dest)
    result_4 = {
        "value_dest": value_dest.clone(),
        "scale_dest": scale_dest.clone()
    }

    return {
        "test_case_1": result_1,
        "test_case_2": result_2,
        "test_case_3": result_3,
        "test_case_4": result_4
    }

result_gold = test_destindex_copy_quantize_kv()
