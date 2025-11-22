#!/bin/bash
# Bash wrapper for profiler ablation study
# Run profiler ablation with different configurations

echo "========================================"
echo "Running Profiler Ablation Study"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed or not in PATH"
    exit 1
fi

# Check if triton-profiler is available
if ! command -v triton-profiler &> /dev/null; then
    echo "Error: triton-profiler not found in PATH"
    echo "Make sure triton-viz is installed with profiler support"
    exit 1
fi

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Timestamp: ${TIMESTAMP}"
echo ""

# Base output directory
OUTPUT_BASE="profiler_ablation_${TIMESTAMP}"

# Check for whitelist
WHITELIST_ARG=""
if [ -f "whitelist.txt" ]; then
    echo "✓ Using whitelist.txt"
    WHITELIST_ARG="--whitelist whitelist.txt"
    echo "  Files in whitelist:"
    grep -v "^#" whitelist.txt | grep -v "^$" | sed 's/^/    - /'
else
    echo "✗ No whitelist.txt found, will run all files"
    echo "  Warning: This may take a long time!"
    echo "  Create whitelist.txt to run specific files only"
fi
echo ""

# Configuration explanation
echo "Profiler Configurations (all with TRITON_INTERPRET=1 and ENABLE_TIMING=1):"
echo "  1. both_enabled:             Load/store skipping ON, Block sampling ON"
echo "  2. only_load_store_skipping: Load/store skipping ON, Block sampling OFF"
echo "  3. only_block_sampling:      Load/store skipping OFF, Block sampling ON"
echo "  4. both_disabled:            Load/store skipping OFF, Block sampling OFF (baseline)"
echo ""

# Parse command line arguments
CONFIGS="all"
if [ $# -gt 0 ]; then
    CONFIGS="$@"
    echo "Running specific configurations: $CONFIGS"
else
    echo "Running all configurations"
fi
echo ""

echo "========================================"
echo "Starting Tests"
echo "========================================"
echo ""

# Run the Python script
python3 ablation_runner.py \
    ${WHITELIST_ARG} \
    --output-dir "${OUTPUT_BASE}" \
    --configs ${CONFIGS}

TEST_EXIT_CODE=$?

echo ""
echo "========================================"
echo "Test Runs Complete"
echo "========================================"
echo ""

if [ ${TEST_EXIT_CODE} -eq 0 ]; then
    echo "✓ All tests completed successfully"
else
    echo "✗ Some tests failed or encountered errors"
fi

echo ""
echo "Results saved in: ${OUTPUT_BASE}/"
echo ""

# Run analysis script and generate CSV
echo "========================================"
echo "Analyzing Timing Results and Generating CSV"
echo "========================================"
echo ""

python3 analyze_profiler_timing.py "${OUTPUT_BASE}" --csv "profiler_timing_results.csv"

ANALYSIS_EXIT_CODE=$?

if [ ${ANALYSIS_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Analysis Complete!"
    echo "========================================"
    echo ""
    echo "Results:"
    echo "  CSV File:  ${OUTPUT_BASE}/profiler_timing_results.csv"
    echo "  Log files: ${OUTPUT_BASE}/*/"
    echo "  Summary:   ${OUTPUT_BASE}/summary.json"
    echo ""
    echo "To view CSV:"
    echo "  cat ${OUTPUT_BASE}/profiler_timing_results.csv"
    echo "  or"
    echo "  column -t -s, ${OUTPUT_BASE}/profiler_timing_results.csv | less -S"
    echo ""
    echo "To view summary:"
    echo "  cat ${OUTPUT_BASE}/summary.json | python3 -m json.tool"
else
    echo ""
    echo "⚠ Analysis script failed, but test logs are still available in ${OUTPUT_BASE}/"
fi

echo ""
echo "========================================"

# Exit with test status (analysis failure is secondary)
exit ${TEST_EXIT_CODE}