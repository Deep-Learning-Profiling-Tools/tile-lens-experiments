# Triton Kernel Profiler Case Studies

This directory contains case studies for profiling and testing optimized Triton kernels against their baseline implementations.

## Studies

- **unroll_for_loop**: Loop unrolling optimizations
- **mask_percentage**: Mask percentage optimizations

## Usage

### Test Correctness
Verify that optimized kernels produce the same results as baseline:
```bash
# Run all studies and cases
python test_correctness.py

# Run a specific study
python test_correctness.py -s unroll_for_loop
python test_correctness.py -s mask_percentage

# Run a specific case (auto-detects study if unique)
python test_correctness.py -c iv_dependent_matmul
python test_correctness.py -c quantize_kv_transform

# If a case exists in multiple studies, specify both -s and -c
python test_correctness.py -s unroll_for_loop -c some_shared_case

# List available studies and cases
python test_correctness.py -l
```

### Profile Performance
Measure execution times of baseline vs optimized kernels:
```bash
# Run all studies and cases
python collect_proton_times.py

# Run a specific study
python collect_proton_times.py -s unroll_for_loop
python collect_proton_times.py -s mask_percentage

# Run a specific case (auto-detects study if unique)
python collect_proton_times.py -c iv_dependent_matmul
python collect_proton_times.py -c quantize_kv_transform

# If a case exists in multiple studies, specify both -s and -c
python collect_proton_times.py -s unroll_for_loop -c some_shared_case

# List available studies and cases
python collect_proton_times.py -l
```

This generates `proton_times.csv` in each study directory with timing comparisons.

## Directory Structure
```
profiler/
├── test_correctness.py      # Correctness testing script
├── collect_proton_times.py  # Performance profiling script
├── README.md
├── unroll_for_loop/         # Loop unrolling study
│   ├── diag_ssm_triton/
│   │   ├── baseline.py
│   │   └── optimized.py
│   ├── ...
│   └── proton_times.csv     # Generated timing results
└── mask_percentage/         # Mask percentage study
    ├── quantize_kv_transform/
    │   ├── baseline.py
    │   └── optimized.py
    └── proton_times.csv     # Generated timing results
```

Each case directory should contain:
- `baseline.py` - Original Triton kernel implementation
- `optimized.py` - Optimized version

## Supported Cases

### unroll_for_loop
- `diag_ssm_triton`
- `fused_recurrent_retention`
- `fused_recurrent_delta`
- `fast_rope_embedding`
- `flash_decode2_llama`
- `fused_rwkv6_kernel`
- `iv_dependent_matmul`
- `rmsnorm_fused`
- `rmsnorm_fused_llama`
- `rmsnorm_implementation`
- `layernorm_fwd_triton`
- `var_len_copy`
- `matmul_leakyrelu`
- `flash_decode2_phi`
- `kldiv_ops`
- `mean_reduction`
- `softmax_optimize`
- `triton_conv2d_fwd`
- `triton_matmul`
- `matmul_triton1`
- `lora_expand_gemv`

### mask_percentage
- `quantize_kv_transform`
