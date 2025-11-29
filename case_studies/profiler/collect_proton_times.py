#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

STUDY_CASES: Dict[str, List[str]] = {
    "unroll_for_loop": [
        "fused_recurrent_delta",
        "fused_rwkv6_kernel",
        "fused_recurrent_retention",
        "diag_ssm_triton",
        "fast_rope_embedding",
        "flash_decode2_llama",
        "iv_dependent_matmul",
        "rmsnorm_fused",
        "rmsnorm_fused_llama",
        "rmsnorm_implementation",
        "layernorm_fwd_triton",
        "var_len_copy",
        "matmul_leakyrelu",
        "flash_decode2_phi",
        "kldiv_ops",
        "mean_reduction",
        "softmax_optimize",
        "triton_conv2d_fwd",
        "triton_matmul",
        "matmul_triton1",
        "lora_expand_gemv",
    ],
    "mask_percentage": [
        "quantize_kv_transform",
        "context_attn_llama",
        "context_attn_fwd",
        "bgmv_shrink_kernel",
    ],
}

STUDY_NAMES = list(STUDY_CASES.keys())


def find_triton_kernels(py_file: Path) -> List[str]:
    """
    Return the function names decorated with @triton.jit in the given file.
    A simple state machine is enough because @triton.jit is always followed
    by the corresponding def line.
    """
    kernels: List[str] = []
    pending_decorator = False
    for line in py_file.read_text().splitlines():
        if "@triton.jit" in line:
            pending_decorator = True
            continue
        if pending_decorator:
            match = re.match(r"\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
            if match:
                kernels.append(match.group(1))
                pending_decorator = False
    return kernels


def run_proton(py_file: Path) -> Path:
    workdir = py_file.parent
    start = time.time()
    result = subprocess.run(
        ["proton", py_file.name],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"proton failed for {py_file}: {result.stderr or result.stdout}")

    hatchet_tmp = workdir / "proton.hatchet"
    if not hatchet_tmp.exists():
        raise FileNotFoundError(f"Expected {hatchet_tmp} after running proton on {py_file}")
    if hatchet_tmp.stat().st_mtime < start:
        raise RuntimeError(f"{hatchet_tmp} looks stale after profiling {py_file}")

    destination = workdir / f"{py_file.stem}.hatchet"
    destination.unlink(missing_ok=True)
    hatchet_tmp.rename(destination)
    return destination


def parse_kernel_times(hatchet_file: Path, kernels: Iterable[str]) -> Tuple[float, List[str]]:
    result = subprocess.run(
        ["proton-viewer", "-m", "time/us", str(hatchet_file)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"proton-viewer failed for {hatchet_file}: {result.stderr or result.stdout}")

    totals = {name: 0.0 for name in kernels}
    missing: List[str] = []
    for line in result.stdout.splitlines():
        for name in totals:
            if re.search(rf"\b{name}\b", line):
                time_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
                if time_match:
                    totals[name] += float(time_match.group(1))
                break

    for name, value in totals.items():
        if value == 0.0:
            missing.append(name)

    return sum(totals.values()), missing


def get_all_cases() -> List[str]:
    """Return all case names across all studies."""
    all_cases = []
    for cases in STUDY_CASES.values():
        all_cases.extend(cases)
    return all_cases


def find_studies_for_case(case_name: str) -> List[str]:
    """Return list of studies that contain the given case."""
    return [study for study, cases in STUDY_CASES.items() if case_name in cases]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Proton profiling times for Triton kernels")
    parser.add_argument(
        "-s", "--study",
        type=str,
        choices=STUDY_NAMES + ["all"],
        default="all",
        help="Study to run (default: all)",
    )
    parser.add_argument(
        "-c", "--case",
        type=str,
        choices=get_all_cases(),
        help="Run only the specified case (default: run all cases in selected study)",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all available studies and cases, then exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available studies and cases:")
        for study_name, cases in STUDY_CASES.items():
            print(f"\n{study_name}:")
            for case in cases:
                print(f"  {case}")
        return

    root = Path(__file__).resolve().parent

    # Determine which studies to run
    if args.study == "all":
        studies_to_run = STUDY_NAMES
    else:
        studies_to_run = [args.study]

    # If a specific case is given, find which study it belongs to
    if args.case:
        matching_studies = find_studies_for_case(args.case)
        if args.study != "all":
            # User specified both --study and --case, use the specified study
            if args.study not in matching_studies:
                print(f"Error: case '{args.case}' not found in study '{args.study}'", file=sys.stderr)
                sys.exit(1)
            studies_to_run = [args.study]
        elif len(matching_studies) > 1:
            # Case exists in multiple studies, require explicit --study
            print(f"Error: case '{args.case}' exists in multiple studies: {', '.join(matching_studies)}", file=sys.stderr)
            print(f"Please specify --study to disambiguate, e.g.: -s {matching_studies[0]} -c {args.case}", file=sys.stderr)
            sys.exit(1)
        else:
            studies_to_run = matching_studies

    for study_name in studies_to_run:
        print(f"\n{'=' * 60}")
        print(f"Study: {study_name}")
        print('=' * 60)

        study_root = root / study_name
        case_names = [args.case] if args.case and args.case in STUDY_CASES[study_name] else STUDY_CASES[study_name]
        expected_dirs = [study_root / name for name in case_names]
        subdirs = [path for path in expected_dirs if path.is_dir()]
        missing_dirs = [path.name for path in expected_dirs if not path.is_dir()]
        for name in missing_dirs:
            print(f"Skipping missing case directory: {name}", file=sys.stderr)
        rows = []

        for folder in subdirs:
            per_file_times = {}
            py_files = [folder / "baseline.py", folder / "optimized.py"]
            py_files = [f for f in py_files if f.exists()]
            for py_file in py_files:
                kernels = find_triton_kernels(py_file)
                if not kernels:
                    print(f"Skipping {py_file}: no @triton.jit kernels found", file=sys.stderr)
                    continue

                print(f"Profiling {py_file} with kernels {', '.join(kernels)}")
                hatchet_file = run_proton(py_file)
                total_time, missing = parse_kernel_times(hatchet_file, kernels)
                per_file_times[py_file.stem] = total_time

                missing_note = f" (missing timings for: {', '.join(missing)})" if missing else ""
                print(f"  -> {total_time:.3f} us{missing_note}")

            rows.append(
                {
                    "case_name": folder.name,
                    "baseline": per_file_times.get("baseline", ""),
                    "optimized": per_file_times.get("optimized", ""),
                }
            )

        csv_path = study_root / "proton_times.csv"
        with csv_path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Case Name", "baseline_time_us", "optimized_time_us"])
            for row in rows:
                writer.writerow(
                    [
                        row["case_name"],
                        "" if row["baseline"] == "" else f"{row['baseline']:.3f}",
                        "" if row["optimized"] == "" else f"{row['optimized']:.3f}",
                    ]
                )

        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
