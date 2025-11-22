#!/usr/bin/env python3
"""
Analyze profiler ablation study timing results.
Parses output files and extracts Triton-Viz execution times for 4 configurations:
1. both_enabled: Load/store skipping ON, Block sampling ON
2. only_load_store_skipping: Load/store skipping ON, Block sampling OFF
3. only_block_sampling: Load/store skipping OFF, Block sampling ON
4. both_disabled: Load/store skipping OFF, Block sampling OFF (baseline)

All configurations use triton-profiler with ENABLE_TIMING=1.
"""

import os
import re
import sys
import csv
from pathlib import Path
import argparse
from collections import defaultdict


def parse_triton_profiler_log(log_file):
    """
    Parse triton-profiler log file for execution time.

    Example line:
    Triton-Viz: execution time for _kernel_name: 3.326 ms

    Returns:
        dict: {'total_ms': float, 'count': int, 'kernels': [{'name': str, 'time_ms': float}, ...]}
    """
    pattern = r'Triton-Viz:\s+execution time for\s+(\S+):\s+([\d.]+)\s+ms'

    kernels = []
    total_ms = 0.0

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    kernel_name = match.group(1)
                    exec_time = float(match.group(2))
                    kernels.append({
                        'name': kernel_name,
                        'time_ms': exec_time
                    })
                    total_ms += exec_time
    except FileNotFoundError:
        print(f"Warning: File not found: {log_file}")
        return {'total_ms': 0.0, 'count': 0, 'kernels': []}
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        return {'total_ms': 0.0, 'count': 0, 'kernels': []}

    return {
        'total_ms': total_ms,
        'count': len(kernels),
        'kernels': kernels
    }


def find_log_files(output_dir, config_name):
    """
    Find all log files for a given configuration.

    Args:
        output_dir: base output directory
        config_name: configuration name (both_enabled, only_load_store_skipping, etc.)

    Returns:
        list of Path objects sorted by file number
    """
    config_path = Path(output_dir) / config_name

    if not config_path.exists():
        return []

    # Look for numbered log files
    log_files = sorted(config_path.glob("*.log"))

    # Sort by file number (extract number from filename like "0001_add_value.log")
    def extract_number(path):
        match = re.search(r'^(\d+)_', path.name)
        if match:
            return int(match.group(1))
        return float('inf')  # Put files without numbers at the end

    log_files.sort(key=extract_number)

    return log_files


def analyze_configuration(output_dir, config_name):
    """
    Analyze all log files for a configuration.

    Args:
        output_dir: base output directory
        config_name: configuration name

    Returns:
        dict: {file_number: {'file_name': str, 'total_ms': float, 'count': int}}
    """
    log_files = find_log_files(output_dir, config_name)

    if not log_files:
        print(f"  No log files found for {config_name}")
        return None

    print(f"  Found {len(log_files)} log file(s)")

    # Parse each log file and extract timing data
    file_totals = {}

    for log_file in log_files:
        print(f"    Parsing: {log_file.name}")

        # Extract file number and name from filename
        match = re.search(r'^(\d+)_(.+)\.log$', log_file.name)
        if not match:
            continue

        file_number = int(match.group(1))
        base_name = match.group(2)

        # Parse timing data from this file
        results = parse_triton_profiler_log(log_file)

        file_totals[file_number] = {
            'file_name': base_name,
            'total_ms': results['total_ms'],
            'count': results['count']
        }

    return file_totals


def print_results(config_name, file_totals):
    """
    Print analysis results for a configuration.

    Args:
        config_name: configuration name
        file_totals: dict of {file_number: {'file_name': str, 'total_ms': float, 'count': int}}
    """
    if not file_totals:
        print(f"  No results to display")
        return

    print(f"\n  Results by file:")
    print(f"  {'─' * 80}")

    # Sort by file number
    sorted_files = sorted(file_totals.items())

    grand_total = 0
    total_measurements = 0

    for file_number, data in sorted_files:
        total_ms = data['total_ms']
        count = data['count']
        file_name = data['file_name']
        grand_total += total_ms
        total_measurements += count

        # Format display name
        display_name = f"{file_number:04d}_{file_name}"
        if len(display_name) > 60:
            display_name = display_name[:57] + "..."

        print(f"    {display_name}")
        print(f"      Total: {total_ms:>10.3f} ms  (from {count} kernel calls)")

    print(f"  {'─' * 80}")
    print(f"  Grand Total: {grand_total:>10.3f} ms")
    print(f"  Total Files: {len(file_totals)}")
    print(f"  Total Kernel Calls: {total_measurements}")
    if file_totals:
        print(f"  Average per File: {grand_total / len(file_totals):.3f} ms")


def export_to_csv(output_dir, config_results, csv_filename):
    """
    Export results to CSV file with format:
    Test_Name, baseline, + skipping, + sampling, + skipping, sampling

    Each unique file gets exactly one row combining all configurations.
    Each cell contains the sum of all kernel execution times for that configuration.

    Args:
        output_dir: base output directory
        config_results: dict mapping config names to their file_totals
        csv_filename: output CSV filename
    """
    # Configuration mapping (internal name -> CSV column name)
    config_mapping = [
        ('both_disabled', 'baseline'),
        ('only_load_store_skipping', '+ skipping'),
        ('only_block_sampling', '+ sampling'),
        ('both_enabled', '+ skipping, sampling')
    ]

    # Collect all unique file names from all configurations
    all_file_names = set()
    for config_name, _ in config_mapping:
        file_totals = config_results.get(config_name)
        if file_totals:
            for file_data in file_totals.values():
                all_file_names.add(file_data['file_name'])

    if not all_file_names:
        print("No test results to export")
        return None

    # Sort file names alphabetically
    sorted_file_names = sorted(all_file_names)

    # Prepare CSV data - one row per unique file
    csv_data = []
    for file_name in sorted_file_names:
        row = {'Test_Name': file_name}

        # Add timing data for each configuration
        for internal_name, csv_column in config_mapping:
            file_totals = config_results.get(internal_name, {})

            # Find the timing for this file name in this configuration
            time_ms = 0.0
            for file_data in file_totals.values():
                if file_data['file_name'] == file_name:
                    time_ms = file_data['total_ms']
                    break

            row[csv_column] = f"{time_ms:.3f}"

        csv_data.append(row)

    # Write CSV file
    csv_path = Path(output_dir) / csv_filename
    try:
        fieldnames = ['Test_Name'] + [csv_col for _, csv_col in config_mapping]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

        print(f"\n✓ CSV exported to: {csv_path}")
        print(f"  Total unique files: {len(csv_data)}")

        return csv_path
    except Exception as e:
        print(f"\n✗ Error exporting CSV: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze profiler ablation study timing results"
    )
    parser.add_argument(
        "output_dir",
        help="Output directory containing test results (e.g., profiler_ablation_20231113_120000)"
    )
    parser.add_argument(
        "--csv",
        default="profiler_timing_results.csv",
        help="Export results to CSV file (default: profiler_timing_results.csv)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)

    print("=" * 80)
    print("Profiler Ablation Study Timing Analysis")
    print("=" * 80)
    print(f"\nAnalyzing: {output_dir}")
    print()

    # Configuration names and descriptions
    configs = [
        ('both_enabled', 'Both Enabled (Load/Store Skip ON, Block Sampling ON)'),
        ('only_load_store_skipping', 'Only Load/Store Skipping (Load/Store Skip ON, Block Sampling OFF)'),
        ('only_block_sampling', 'Only Block Sampling (Load/Store Skip OFF, Block Sampling ON)'),
        ('both_disabled', 'Both Disabled (Load/Store Skip OFF, Block Sampling OFF)'),
    ]

    # Analyze each configuration
    config_results = {}

    for config_name, config_desc in configs:
        print("━" * 80)
        print(f"{config_desc.upper()}")
        print("━" * 80)
        totals = analyze_configuration(output_dir, config_name)
        config_results[config_name] = totals

        if totals:
            print_results(config_name, totals)
        else:
            print(f"  No {config_name} results found")

        print()

    print("=" * 80)

    # Summary comparison
    print("\nSUMMARY")
    print("=" * 80)

    summary_data = []
    for config_name, config_desc in configs:
        totals = config_results.get(config_name)
        if totals:
            total_ms = sum(t['total_ms'] for t in totals.values())
            summary_data.append((config_desc, total_ms))
            print(f"  {config_desc:60s} {total_ms:>12.3f} ms")

    # Calculate speedups relative to both_disabled (baseline)
    if summary_data and summary_data[-1][0] == 'Both Disabled (Load/Store Skip OFF, Block Sampling OFF)':
        baseline = summary_data[-1][1]
        if baseline > 0:
            print("\n  Speedups relative to baseline (both_disabled):")
            for desc, time_ms in summary_data[:-1]:
                speedup = baseline / time_ms if time_ms > 0 else float('inf')
                reduction = ((baseline - time_ms) / baseline) * 100 if baseline > 0 else 0
                print(f"    {desc:55s} {speedup:>6.2f}x ({reduction:>5.1f}% reduction)")

    print("=" * 80)

    # Export to CSV if requested
    if args.csv:
        csv_path = export_to_csv(output_dir, config_results, args.csv)
        if csv_path:
            print(f"\nTo view the CSV:")
            print(f"  cat {csv_path}")
            print(f"  or")
            print(f"  column -t -s, {csv_path} | less -S")

    return 0


if __name__ == "__main__":
    sys.exit(main())