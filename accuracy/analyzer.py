#!/usr/bin/env python3
"""
Accuracy experiment analyzer.
Analyzes profiler results and generates confusion matrix comparing
ground truth (PROFILER_ENABLE_BLOCK_SAMPLING=0) vs test (PROFILER_ENABLE_BLOCK_SAMPLING=1).

Supported metrics:
  - mask_ratio: Compares mask percentage for load/store operations
  - unroll_for_loop: (Future) Compares for-loop unrolling statistics

Usage:
    python analyzer.py                           # Analyze results with default settings
    python analyzer.py --top-k 15                # Use top 15 as positive class
    python analyzer.py --metric mask_ratio       # Specify metric
    python analyzer.py --results-dir ./results   # Specify results directory
"""

import os
import re
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Script directory
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_TOP_K = 15


@dataclass
class MetricResult:
    """Result of parsing a metric from a log file."""
    test_name: str
    value: float  # Primary value used for ranking
    details: Dict[str, Any]  # Additional metric-specific details


class MetricParser(ABC):
    """Abstract base class for metric parsers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name."""
        pass

    @abstractmethod
    def parse(self, log_content: str, test_name: str) -> Optional[MetricResult]:
        """Parse metric from log content."""
        pass

    @abstractmethod
    def get_csv_headers(self) -> List[str]:
        """Get CSV column headers for this metric."""
        pass

    @abstractmethod
    def get_csv_row(self, result: MetricResult) -> List[str]:
        """Get CSV row values for a result."""
        pass


class MaskRatioParser(MetricParser):
    """Parser for mask ratio statistics."""

    @property
    def name(self) -> str:
        return "mask_ratio"

    def parse(self, log_content: str, test_name: str) -> Optional[MetricResult]:
        """Parse mask ratio from log content."""
        # Find Load mask percentage
        load_match = re.search(
            r'Overall Load Operations:.*?Masked percentage:\s+([0-9.]+)%',
            log_content, re.DOTALL
        )
        load_pct = float(load_match.group(1)) if load_match else 0.0

        # Find Store mask percentage
        store_match = re.search(
            r'Overall Store Operations:.*?Masked percentage:\s+([0-9.]+)%',
            log_content, re.DOTALL
        )
        store_pct = float(store_match.group(1)) if store_match else 0.0

        # Use max of load and store as primary value
        max_pct = max(load_pct, store_pct)

        return MetricResult(
            test_name=test_name,
            value=max_pct,
            details={
                "load_pct": load_pct,
                "store_pct": store_pct,
                "max_pct": max_pct
            }
        )

    def get_csv_headers(self) -> List[str]:
        return ["load_pct", "store_pct", "max_pct"]

    def get_csv_row(self, result: MetricResult) -> List[str]:
        d = result.details
        return [
            f"{d['load_pct']:.2f}",
            f"{d['store_pct']:.2f}",
            f"{d['max_pct']:.2f}"
        ]


class UnrollForLoopParser(MetricParser):
    """Parser for for-loop unrolling statistics.

    This metric compares exact match between ground truth and block sampling,
    not top-k ranking like mask_ratio.
    """

    @property
    def name(self) -> str:
        return "unroll_for_loop"

    @property
    def is_matching_metric(self) -> bool:
        """This metric compares exact match, not ranking."""
        return True

    def parse(self, log_content: str, test_name: str) -> Optional[MetricResult]:
        """Parse all for-loop unrolling statistics blocks from log content.

        A log may contain multiple "For-Loop Unrolling Statistics" blocks.
        Each block contains:
        - Total for-loops detected
        - For each loop: line number, range type, total steps
        """
        all_blocks = []

        # Split by "For-Loop Unrolling Statistics" sections
        # Pattern to find each statistics block
        block_pattern = re.compile(
            r'For-Loop Unrolling Statistics.*?Total for-loops detected:\s+(\d+)(.*?)(?=============)',
            re.DOTALL
        )

        for block_match in block_pattern.finditer(log_content):
            total_loops = int(block_match.group(1))
            block_content = block_match.group(2)

            # Parse individual loops in this block
            loops = []
            loop_pattern = re.compile(
                r'Loop #(\d+):\s*\n\s*Line number:\s+(\d+)\s*\n\s*Range type:\s+(\w+)\s*\n\s*Total steps:\s+(\d+)',
                re.MULTILINE
            )

            for loop_match in loop_pattern.finditer(block_content):
                loops.append({
                    "loop_num": int(loop_match.group(1)),
                    "line_num": int(loop_match.group(2)),
                    "range_type": loop_match.group(3),
                    "total_steps": int(loop_match.group(4))
                })

            all_blocks.append({
                "total_loops": total_loops,
                "loops": loops
            })

        # If no blocks found, try simpler pattern for "No for-loops detected"
        if not all_blocks:
            no_loop_pattern = re.compile(r'No for-loops detected')
            if no_loop_pattern.search(log_content):
                all_blocks.append({
                    "total_loops": 0,
                    "loops": []
                })

        # Calculate total loops across all blocks for sorting/display
        total_all_loops = sum(b["total_loops"] for b in all_blocks)
        total_all_steps = sum(
            loop["total_steps"]
            for b in all_blocks
            for loop in b["loops"]
        )

        return MetricResult(
            test_name=test_name,
            value=float(total_all_loops),  # Used for display, not ranking
            details={
                "blocks": all_blocks,
                "total_blocks": len(all_blocks),
                "total_loops": total_all_loops,
                "total_steps": total_all_steps
            }
        )

    def get_csv_headers(self) -> List[str]:
        return ["total_blocks", "total_loops", "total_steps"]

    def get_csv_row(self, result: MetricResult) -> List[str]:
        d = result.details
        return [
            str(d["total_blocks"]),
            str(d["total_loops"]),
            str(d["total_steps"])
        ]

    @staticmethod
    def results_match(gt_result: MetricResult, test_result: MetricResult) -> bool:
        """Check if two results have the same loop statistics."""
        gt_blocks = gt_result.details["blocks"]
        test_blocks = test_result.details["blocks"]

        # Must have same number of blocks
        if len(gt_blocks) != len(test_blocks):
            return False

        # Compare each block
        for gt_block, test_block in zip(gt_blocks, test_blocks):
            # Must have same number of loops
            if gt_block["total_loops"] != test_block["total_loops"]:
                return False

            # Must have same loop details
            if len(gt_block["loops"]) != len(test_block["loops"]):
                return False

            for gt_loop, test_loop in zip(gt_block["loops"], test_block["loops"]):
                # Compare line_num and total_steps (range_type might vary)
                if gt_loop["line_num"] != test_loop["line_num"]:
                    return False
                if gt_loop["total_steps"] != test_loop["total_steps"]:
                    return False

        return True


# Registry of available metric parsers
METRIC_PARSERS: Dict[str, MetricParser] = {
    "mask_ratio": MaskRatioParser(),
    "unroll_for_loop": UnrollForLoopParser(),
}


def load_results(results_dir: Path, config_name: str) -> Dict[str, str]:
    """
    Load log files from a configuration directory.

    Returns:
        Dict mapping test_name to log content
    """
    config_dir = results_dir / config_name
    if not config_dir.exists():
        print(f"Warning: Config directory not found: {config_dir}")
        return {}

    results = {}
    for log_file in sorted(config_dir.glob("*.log")):
        # Extract test name from filename (format: ID_testname.log)
        # Remove ID prefix and .log suffix
        filename = log_file.stem
        parts = filename.split("_", 1)
        if len(parts) == 2:
            test_name = parts[1]
        else:
            test_name = filename

        with open(log_file, 'r') as f:
            results[test_name] = f.read()

    return results


def analyze_results(
    results_dir: Path,
    metric: str,
    top_k: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Analyze results for both configurations.

    For ranking metrics (mask_ratio): ranks by value and selects top-k.
    For matching metrics (unroll_for_loop): compares exact match between GT and test.

    Returns:
        Tuple of (ground_truth_results, test_results)
    """
    parser = METRIC_PARSERS.get(metric)
    if not parser:
        raise ValueError(f"Unknown metric: {metric}. Available: {list(METRIC_PARSERS.keys())}")

    # Load results
    gt_logs = load_results(results_dir, "ground_truth")
    test_logs = load_results(results_dir, "block_sampling")

    if not gt_logs:
        raise ValueError("No ground truth results found")
    if not test_logs:
        raise ValueError("No test (block_sampling) results found")

    # Parse metrics
    gt_results = []
    for test_name, content in gt_logs.items():
        result = parser.parse(content, test_name)
        if result:
            gt_results.append({
                "test_name": test_name,
                "metric_result": result
            })

    test_results = []
    for test_name, content in test_logs.items():
        result = parser.parse(content, test_name)
        if result:
            test_results.append({
                "test_name": test_name,
                "metric_result": result
            })

    # Check if this is a matching metric (like unroll_for_loop)
    is_matching = getattr(parser, 'is_matching_metric', False)

    if is_matching:
        # For matching metrics: compare exact match between GT and test
        test_by_name = {r["test_name"]: r for r in test_results}

        for r in gt_results:
            test_name = r["test_name"]
            gt_result = r["metric_result"]
            test_r = test_by_name.get(test_name)

            # ground_truth "selected" = has loops (value > 0)
            r["selected"] = gt_result.value > 0
            r["rank"] = 0  # No ranking for matching metrics

            # Check if test result matches
            if test_r:
                test_result = test_r["metric_result"]
                matches = parser.results_match(gt_result, test_result)
                test_r["selected"] = matches
                test_r["rank"] = 0
                # Store match info in gt_result for later use
                r["matches_test"] = matches
            else:
                r["matches_test"] = False

        # Sort by test_name for consistent ordering
        gt_results.sort(key=lambda x: x["test_name"])
        test_results.sort(key=lambda x: x["test_name"])
    else:
        # For ranking metrics: sort by value and select top-k
        gt_results.sort(key=lambda x: x["metric_result"].value, reverse=True)
        test_results.sort(key=lambda x: x["metric_result"].value, reverse=True)

        # Assign ranks and selected status
        for i, r in enumerate(gt_results, 1):
            r["rank"] = i
            r["selected"] = i <= top_k

        for i, r in enumerate(test_results, 1):
            r["rank"] = i
            r["selected"] = i <= top_k

    return gt_results, test_results


def generate_csv(
    gt_results: List[Dict],
    test_results: List[Dict],
    metric: str,
    output_dir: Path
) -> Path:
    """Generate a single CSV file comparing ground truth and sampled results.

    For ranking metrics (mask_ratio):
      - ground_truth: True if in top-k
      - sampled_results: True if in top-k

    For matching metrics (unroll_for_loop):
      - ground_truth: True if has loops detected
      - sampled_results: True if sampling result matches ground truth
    """
    parser = METRIC_PARSERS.get(metric)
    is_matching = getattr(parser, 'is_matching_metric', False)

    # Build lookup for test results
    test_by_name = {r["test_name"]: r for r in test_results}

    # Output CSV with two boolean columns
    csv_path = output_dir / f"comparison_{metric}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["test_name", "ground_truth", "sampled_results"])
        for r in gt_results:
            test_name = r["test_name"]
            test_r = test_by_name.get(test_name)

            if is_matching:
                # For matching metrics:
                # ground_truth = has loops (value > 0)
                # sampled_results = matches ground truth
                gt_selected = r["selected"]  # has loops
                test_selected = r.get("matches_test", False)  # matches GT
            else:
                # For ranking metrics:
                # ground_truth = in top-k
                # sampled_results = in top-k
                gt_selected = r["selected"]
                test_selected = test_r["selected"] if test_r else False

            writer.writerow([test_name, gt_selected, test_selected])

    return csv_path


def compute_confusion_matrix(
    gt_results: List[Dict],
    test_results: List[Dict],
    metric: str
) -> Dict[str, int]:
    """
    Compute confusion matrix comparing ground truth selection vs test selection.

    For ranking metrics (mask_ratio):
      - TP: GT in top-k AND test in top-k
      - FN: GT in top-k AND test not in top-k
      - FP: GT not in top-k AND test in top-k
      - TN: GT not in top-k AND test not in top-k

    For matching metrics (unroll_for_loop):
      - TP: GT has loops AND sampling matches
      - FN: GT has loops AND sampling doesn't match
      - FP: GT has no loops AND sampling doesn't match
      - TN: GT has no loops AND sampling matches

    Returns:
        Dict with TP, FP, TN, FN counts
    """
    parser = METRIC_PARSERS.get(metric)
    is_matching = getattr(parser, 'is_matching_metric', False)

    tp = fp = tn = fn = 0

    if is_matching:
        # For matching metrics
        for r in gt_results:
            has_loops = r["selected"]  # GT has loops
            matches = r.get("matches_test", False)  # Sampling matches GT

            if has_loops and matches:
                tp += 1  # Has loops and correctly detected
            elif has_loops and not matches:
                fn += 1  # Has loops but sampling got it wrong
            elif not has_loops and matches:
                tn += 1  # No loops and sampling correctly found none
            else:
                fp += 1  # No loops but sampling got it wrong
    else:
        # For ranking metrics
        test_selected = {r["test_name"]: r["selected"] for r in test_results}

        for r in gt_results:
            test_name = r["test_name"]
            gt_sel = r["selected"]
            test_sel = test_selected.get(test_name, False)

            if gt_sel and test_sel:
                tp += 1  # True Positive: both selected
            elif gt_sel and not test_sel:
                fn += 1  # False Negative: GT selected, test not
            elif not gt_sel and test_sel:
                fp += 1  # False Positive: GT not selected, test selected
            else:
                tn += 1  # True Negative: both not selected

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def print_confusion_matrix(cm: Dict[str, int], metric: str, top_k: int):
    """Print confusion matrix in a formatted way."""
    parser = METRIC_PARSERS.get(metric)
    is_matching = getattr(parser, 'is_matching_metric', False)

    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    total = tp + fp + tn + fn

    print(f"\n{'=' * 60}")
    if is_matching:
        print(f"Confusion Matrix (Exact Match Comparison)")
        print(f"{'=' * 60}")
        print(f"  Ground Truth: 'has_loops' = detected loops (total > 0)")
        print(f"  Test:         'matches' = sampling result matches GT")
    else:
        print(f"Confusion Matrix (Top-{top_k} Selection)")
        print(f"{'=' * 60}")
        print(f"  Ground Truth: 'selected' = in top {top_k} by metric value")
        print(f"  Test:         'selected' = in top {top_k} by metric value")
    print(f"{'=' * 60}\n")

    # Matrix visualization
    if is_matching:
        print("                     Sampling Result")
        print("                     Matches     Not Match")
        print(f"                   +-----------+-----------+")
        print(f"  GT       HasLoop |   TP={tp:3d}  |   FN={fn:3d}  |")
        print(f"           -------+-----------+-----------+")
        print(f"           NoLoop |   FP={fp:3d}  |   TN={tn:3d}  |")
        print(f"                   +-----------+-----------+")
    else:
        print("                     Predicted (Block Sampling)")
        print("                     Selected    Not Selected")
        print(f"                   +-----------+-----------+")
        print(f"  Actual   Selected |   TP={tp:3d}  |   FN={fn:3d}  |")
        print(f"  (Ground  --------+-----------+-----------+")
        print(f"   Truth)  Not Sel |   FP={fp:3d}  |   TN={tn:3d}  |")
        print(f"                   +-----------+-----------+")

    # Metrics
    print(f"\n{'=' * 60}")
    print("Metrics:")
    print(f"{'=' * 60}")

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  Accuracy:  {accuracy:.4f} ({tp + tn}/{total})")
    print(f"  Precision: {precision:.4f} ({tp}/{tp + fp})")
    print(f"  Recall:    {recall:.4f} ({tp}/{tp + fn})")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"{'=' * 60}")


def print_comparison_table(
    gt_results: List[Dict],
    test_results: List[Dict],
    metric: str,
    top_k: int
):
    """Print side-by-side comparison."""
    parser = METRIC_PARSERS.get(metric)
    is_matching = getattr(parser, 'is_matching_metric', False)

    # Build lookup
    test_by_name = {r["test_name"]: r for r in test_results}

    print(f"\n{'=' * 90}")

    if is_matching:
        # For matching metrics: show all results with match status
        print(f"Loop Detection Comparison (Matching)")
        print(f"{'=' * 90}")
        print(f"{'Test Name':<40} {'GT Loops':>10} {'GT Steps':>10} {'Test Loops':>12} {'Test Steps':>12} {'Match':>6}")
        print(f"{'-' * 90}")

        for r in gt_results:
            test_name = r["test_name"]
            gt_result = r["metric_result"]
            gt_loops = gt_result.details["total_loops"]
            gt_steps = gt_result.details["total_steps"]

            test_r = test_by_name.get(test_name)
            if test_r:
                test_result = test_r["metric_result"]
                test_loops = test_result.details["total_loops"]
                test_steps = test_result.details["total_steps"]
            else:
                test_loops = "N/A"
                test_steps = "N/A"

            matches = r.get("matches_test", False)
            match_str = "Yes" if matches else "No"

            print(f"{test_name:<40} {gt_loops:>10} {gt_steps:>10} {test_loops:>12} {test_steps:>12} {match_str:>6}")

        # Summary
        total_match = sum(1 for r in gt_results if r.get("matches_test", False))
        total = len(gt_results)
        print(f"{'-' * 90}")
        print(f"Total: {total_match}/{total} matched ({100*total_match/total:.1f}%)")
    else:
        # For ranking metrics: show top-k
        print(f"Ranking Comparison (Top-{top_k})")
        print(f"{'=' * 90}")
        print(f"{'Rank':<6} {'Test Name':<35} {'GT Value':>10} {'Test Value':>10} {'Match':>6}")
        print(f"{'-' * 90}")

        for r in gt_results[:top_k]:
            test_name = r["test_name"]
            gt_value = r["metric_result"].value
            test_r = test_by_name.get(test_name)
            test_value = test_r["metric_result"].value if test_r else 0
            match = "Yes" if test_r and test_r["selected"] else "No"

            print(f"{r['rank']:<6} {test_name:<35} {gt_value:>9.2f}% {test_value:>9.2f}% {match:>6}")

    print(f"{'=' * 90}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze accuracy experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyzer.py                           # Analyze with defaults
    python analyzer.py --top-k 15                # Use top 15 as positive class
    python analyzer.py --metric mask_ratio       # Specify metric
    python analyzer.py --results-dir ./results   # Specify results directory
        """
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Results directory (default: {DEFAULT_RESULTS_DIR})"
    )
    parser.add_argument(
        "--metric", "-m",
        choices=list(METRIC_PARSERS.keys()),
        default="mask_ratio",
        help="Metric to analyze (default: mask_ratio)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Top K items to consider as positive class (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory for CSV files (default: same as results-dir)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    print(f"\n{'=' * 60}")
    print("Accuracy Experiment Analyzer")
    print(f"{'=' * 60}")
    print(f"Results directory: {results_dir}")
    print(f"Metric: {args.metric}")
    print(f"Top-K: {args.top_k}")
    print(f"{'=' * 60}")

    try:
        # Analyze results
        gt_results, test_results = analyze_results(
            results_dir, args.metric, args.top_k
        )

        print(f"\nLoaded {len(gt_results)} ground truth results")
        print(f"Loaded {len(test_results)} test results")

        # Generate CSV
        csv_path = generate_csv(
            gt_results, test_results, args.metric, output_dir
        )
        print(f"\nCSV file generated: {csv_path}")

        # Compute and print confusion matrix
        cm = compute_confusion_matrix(gt_results, test_results, args.metric)
        print_confusion_matrix(cm, args.metric, args.top_k)

        # Print comparison table
        print_comparison_table(gt_results, test_results, args.metric, args.top_k)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
