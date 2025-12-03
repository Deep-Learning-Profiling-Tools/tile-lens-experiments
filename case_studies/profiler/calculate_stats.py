#!/usr/bin/env python3
"""Calculate mean and std dev from proton_times.csv for error bar plotting."""

import argparse
import csv
import statistics
from pathlib import Path

NUM_RUNS = 5


def calculate_stats(input_csv: Path, output_csv: Path) -> None:
    rows = []
    with input_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_name = row["Case Name"]

            # Extract baseline and optimized runs
            baseline_runs = [float(row[f"baseline_run{i+1}"]) for i in range(NUM_RUNS) if row[f"baseline_run{i+1}"]]
            optimized_runs = [float(row[f"optimized_run{i+1}"]) for i in range(NUM_RUNS) if row[f"optimized_run{i+1}"]]

            # Calculate statistics
            if baseline_runs:
                baseline_mean = statistics.mean(baseline_runs)
                baseline_std = statistics.stdev(baseline_runs) if len(baseline_runs) > 1 else 0.0
            else:
                baseline_mean = baseline_std = None

            if optimized_runs:
                optimized_mean = statistics.mean(optimized_runs)
                optimized_std = statistics.stdev(optimized_runs) if len(optimized_runs) > 1 else 0.0
            else:
                optimized_mean = optimized_std = None

            # Calculate speedup and its error (using error propagation)
            if baseline_mean and optimized_mean and optimized_mean > 0:
                speedup = baseline_mean / optimized_mean
                # Error propagation: σ(a/b) = |a/b| * sqrt((σa/a)² + (σb/b)²)
                if baseline_mean > 0 and optimized_mean > 0:
                    rel_err_baseline = baseline_std / baseline_mean
                    rel_err_optimized = optimized_std / optimized_mean
                    speedup_std = speedup * ((rel_err_baseline ** 2 + rel_err_optimized ** 2) ** 0.5)
                else:
                    speedup_std = 0.0
            else:
                speedup = speedup_std = None

            rows.append({
                "case_name": case_name,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "optimized_mean": optimized_mean,
                "optimized_std": optimized_std,
                "speedup": speedup,
                "speedup_std": speedup_std,
            })

    # Write output CSV
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Case Name", "baseline_mean", "baseline_std", "optimized_mean", "optimized_std", "speedup", "speedup_std"])
        for row in rows:
            writer.writerow([
                row["case_name"],
                f"{row['baseline_mean']:.3f}" if row['baseline_mean'] is not None else "",
                f"{row['baseline_std']:.3f}" if row['baseline_std'] is not None else "",
                f"{row['optimized_mean']:.3f}" if row['optimized_mean'] is not None else "",
                f"{row['optimized_std']:.3f}" if row['optimized_std'] is not None else "",
                f"{row['speedup']:.3f}" if row['speedup'] is not None else "",
                f"{row['speedup_std']:.3f}" if row['speedup_std'] is not None else "",
            ])

    print(f"Wrote {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate mean and std dev from proton_times.csv")
    parser.add_argument(
        "-i", "--input",
        type=Path,
        help="Input CSV file (default: auto-detect proton_times.csv in study directories)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output CSV file (default: proton_stats.csv in same directory as input)",
    )
    parser.add_argument(
        "-s", "--study",
        type=str,
        choices=["unroll_for_loop", "mask_percentage", "all"],
        default="all",
        help="Study to process (default: all)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    if args.input:
        # Process single file
        input_csv = args.input
        output_csv = args.output or input_csv.parent / "proton_stats.csv"
        calculate_stats(input_csv, output_csv)
    else:
        # Process study files
        study_files = {
            "unroll_for_loop": ("unroll_times.csv", "unroll_stats.csv"),
            "mask_percentage": ("mask_times.csv", "mask_stats.csv"),
        }
        studies = ["unroll_for_loop", "mask_percentage"] if args.study == "all" else [args.study]
        for study in studies:
            input_name, output_name = study_files[study]
            input_csv = root / input_name
            if input_csv.exists():
                output_csv = root / output_name
                print(f"Processing {study}...")
                calculate_stats(input_csv, output_csv)
            else:
                print(f"Skipping {study}: {input_csv} not found")


if __name__ == "__main__":
    main()
