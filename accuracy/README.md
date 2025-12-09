# Accuracy Experiments

This directory contains scripts for evaluating the accuracy of block sampling in the Triton profiler. It compares profiler results between `PROFILER_ENABLE_BLOCK_SAMPLING=0` (ground truth) and `PROFILER_ENABLE_BLOCK_SAMPLING=1` (test).

## Directory Structure

```
accuracy/
├── runner.py      # Run experiments and collect logs
├── analyzer.py    # Analyze results and generate confusion matrix
├── results/       # Output directory for logs and CSV
│   ├── ground_truth/    # Logs with PROFILER_ENABLE_BLOCK_SAMPLING=0
│   └── block_sampling/  # Logs with PROFILER_ENABLE_BLOCK_SAMPLING=1
└── README.md
```

## Usage

### Step 1: Run Experiments

```bash
# Run all tests (both ground_truth and block_sampling configurations)
python runner.py

# Run specific test case
python runner.py --case matmul_triton1

# Run specific repository
python runner.py --repo tritonbench
python runner.py --repo liger_kernel

# Run only one configuration
python runner.py --configs ground_truth
python runner.py --configs block_sampling

# Specify custom output directory
python runner.py --output-dir ./my_results
```

### Step 2: Analyze Results

```bash
# Analyze with default settings (mask_ratio, top-15)
python analyzer.py

# Analyze mask_ratio metric with custom top-k
python analyzer.py --metric mask_ratio --top-k 15

# Analyze unroll_for_loop metric
python analyzer.py --metric unroll_for_loop

# Specify custom results directory
python analyzer.py --results-dir ./my_results
```

## Supported Metrics

### 1. `mask_ratio` (Ranking Comparison)

Compares the mask percentage for load/store operations. Uses top-k ranking to determine selection.

| Column | Description |
|--------|-------------|
| `ground_truth` | `True` if test is in top-k by mask ratio value |
| `sampled_results` | `True` if test is in top-k by mask ratio value |

**Confusion Matrix Interpretation:**
- **TP**: Ground truth in top-k AND block sampling in top-k
- **FN**: Ground truth in top-k AND block sampling NOT in top-k
- **FP**: Ground truth NOT in top-k AND block sampling in top-k
- **TN**: Ground truth NOT in top-k AND block sampling NOT in top-k

### 2. `unroll_for_loop` (Exact Match Comparison)

Compares for-loop unrolling statistics between ground truth and block sampling. Checks if the detection results are identical.

| Column | Description |
|--------|-------------|
| `ground_truth` | `True` if loops detected (total_loops > 0) |
| `sampled_results` | `True` if block sampling result matches ground truth exactly |

**Matching Criteria:**
- Same number of statistics blocks
- Same number of loops in each block
- Same line number and total steps for each loop

**Confusion Matrix Interpretation:**
- **TP**: Has loops AND sampling correctly detected them
- **FN**: Has loops AND sampling got it wrong
- **TN**: No loops AND sampling correctly found none
- **FP**: No loops AND sampling incorrectly detected some

## Output Files

### CSV Output

A single CSV file is generated: `comparison_{metric}.csv`

```csv
test_name,ground_truth,sampled_results
matmul_triton1,True,True
add_value,False,True
...
```

### Console Output

The analyzer prints:
1. Confusion matrix with TP/FP/TN/FN counts
2. Metrics: Accuracy, Precision, Recall, F1 Score
3. Detailed comparison table

## Examples

### Example 1: Full Pipeline

```bash
# Run experiments
python runner.py --repo tritonbench

# Analyze mask_ratio
python analyzer.py --metric mask_ratio --top-k 15

# Analyze unroll_for_loop
python analyzer.py --metric unroll_for_loop
```

### Example 2: Single Test Case

```bash
# Run single case
python runner.py --case matmul_triton1

# Analyze results
python analyzer.py
```

### Example 3: Custom Configuration

```bash
# Run only ground truth first
python runner.py --configs ground_truth

# Then run block sampling
python runner.py --configs block_sampling

# Analyze with custom top-k
python analyzer.py --metric mask_ratio --top-k 20
```

## Environment Variables

The runner sets the following environment variables:

| Variable | Ground Truth | Block Sampling |
|----------|--------------|----------------|
| `TRITON_INTERPRET` | 1 | 1 |
| `ENABLE_TIMING` | 1 | 1 |
| `PROFILER_ENABLE_BLOCK_SAMPLING` | 0 | 1 |
| `PROFILER_DISABLE_BUFFER_LOAD_CHECK` | 1 | 1 |
| `SANITIZER_ENABLE_FAKE_TENSOR` | 1 | 1 |
