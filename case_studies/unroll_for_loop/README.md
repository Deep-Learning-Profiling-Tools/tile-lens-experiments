# Loop Unroll Case Studies

## Usage

### Test Correctness
Verify that optimized kernels produce the same results as baseline:
```bash
# Run all cases
python test_correctness.py

# Run a specific case
python test_correctness.py -c iv_dependent_matmul

# List available cases
python test_correctness.py -l
```

### Profile Performance
Measure execution times of baseline vs optimized kernels:
```bash
# Run all cases
python collect_proton_times.py

# Run a specific case
python collect_proton_times.py -c iv_dependent_matmul

# List available cases
python collect_proton_times.py -l
```

This generates `proton_times.csv` with timing comparisons.

## Directory Structure
Each case study directory should contain:
- `baseline.py` - Original Triton kernel implementation
- `optimized.py` - Loop-unrolled version

## Supported Cases
- `diag_ssm_triton`
- `fused_recurrent_retention`
- `fused_recurrent_delta`
- `fast_rope_embedding`
- `flash_decode2_llama`
- `fused_rwkv6_kernel`
- `iv_dependent_matmul`