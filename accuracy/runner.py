#!/usr/bin/env python3
"""
Accuracy experiment runner.
Compares profiler results between PROFILER_ENABLE_BLOCK_SAMPLING=0 (ground truth)
and PROFILER_ENABLE_BLOCK_SAMPLING=1 (test).

Usage:
    python runner.py                          # Run all tests
    python runner.py --case matmul_triton1    # Run single case
    python runner.py --repo tritonbench       # Run specific repo
"""

import os
import subprocess
import time
import re
from pathlib import Path
from datetime import datetime
import sys
import argparse
from typing import List, Tuple, Dict, Any

# Add utils to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from test_registry import (
    load_registry, discover_tests, REPO_CONFIGS,
    TRITONBENCH_DIR, DEFAULT_REGISTRY
)

# Script directory
SCRIPT_DIR = Path(__file__).parent.resolve()

# Sampling configurations
SAMPLING_CONFIGS = {
    "ground_truth": {
        "name": "ground_truth",
        "description": "PROFILER_ENABLE_BLOCK_SAMPLING=0",
        "env": {"PROFILER_ENABLE_BLOCK_SAMPLING": "0"}
    },
    "block_sampling": {
        "name": "block_sampling",
        "description": "PROFILER_ENABLE_BLOCK_SAMPLING=1",
        "env": {"PROFILER_ENABLE_BLOCK_SAMPLING": "1"}
    }
}


def run_profiler(
    test: Dict[str, Any],
    config_name: str,
    output_base_dir: Path,
    global_id: int,
    total_registry: int,
    current: int,
    total_current: int
) -> Tuple[bool, str]:
    """
    Run triton-profiler with a specific configuration.

    Returns:
        Tuple of (success, output_file_path)
    """
    config = SAMPLING_CONFIGS[config_name]
    file_path = test["file_path"]
    test_name = test["name"]
    is_pytest = test["is_pytest"]
    test_function = test.get("test_function")

    safe_name = test_name.replace("::", "__")
    id_str = str(global_id).zfill(len(str(total_registry)))
    output_filename = f"{id_str}_{safe_name}.log"

    # Create output directory: results/{config_name}/
    output_dir = output_base_dir / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename

    # Environment variables for profiler
    env = os.environ.copy()
    env.update({
        "TRITON_INTERPRET": "1",
        "ENABLE_TIMING": "1",
        "PROFILER_DISABLE_BUFFER_LOAD_CHECK": "1",
        "SANITIZER_ENABLE_FAKE_TENSOR": "1"
    })
    # Apply config-specific env vars
    env.update(config["env"])

    if is_pytest:
        test_spec = f"{file_path.name}::{test_function}" if test_function else file_path.name
        cmd = ["triton-profiler", "pytest", "-s", "--assert=plain", test_spec]
        cwd = file_path.parent
    else:
        cmd = ["triton-profiler", str(file_path)]
        cwd = TRITONBENCH_DIR

    print(f"  [ID:{id_str}] ({current}/{total_current}) Running {test_name} [{config_name}]...")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=300
        )

        elapsed_time = time.time() - start_time
        output = result.stdout + "\n" + result.stderr

        # Write log file
        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Config: {config_name} ({config['description']})\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start time: {datetime.now().isoformat()}\n")
            f.write(f"Elapsed time: {elapsed_time:.3f} s\n")
            f.write("=" * 80 + "\n\n")
            f.write(output)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit code: {result.returncode}\n")

        if result.returncode == 0:
            print(f"    [OK] {elapsed_time:.2f}s")
        else:
            print(f"    [FAIL] Exit code {result.returncode}")

        return result.returncode == 0, str(output_file)

    except subprocess.TimeoutExpired:
        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Config: {config_name}\n")
            f.write("=" * 80 + "\n")
            f.write("TIMEOUT: Test exceeded 300 seconds\n")
        print(f"    [TIMEOUT]")
        return False, str(output_file)

    except Exception as e:
        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Config: {config_name}\n")
            f.write("=" * 80 + "\n")
            f.write(f"ERROR: {str(e)}\n")
        print(f"    [ERROR] {str(e)}")
        return False, str(output_file)


def run_config(
    config_name: str,
    tests: List[Dict[str, Any]],
    output_base_dir: Path,
    total_registry_tests: int
) -> List[Dict[str, Any]]:
    """Run tests for a specific configuration."""
    config = SAMPLING_CONFIGS[config_name]
    print(f"\n{'=' * 60}")
    print(f"Running: {config['description']}")
    print(f"Output directory: {output_base_dir / config_name}")
    print(f"{'=' * 60}\n")

    results = []
    num_tests = len(tests)

    for i, test in enumerate(tests, 1):
        global_id = test["global_id"]
        success, output_file = run_profiler(
            test, config_name, output_base_dir, global_id,
            total_registry_tests, i, num_tests
        )
        results.append({
            "name": test["name"],
            "global_id": global_id,
            "config": config_name,
            "success": success,
            "output_file": output_file
        })

    # Print summary
    successful = sum(1 for r in results if r["success"])
    print(f"\n  {config_name} summary: {successful}/{num_tests} tests passed")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Accuracy experiment runner - compares block sampling on/off",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py                          # Run all tests
  python runner.py --case matmul_triton1    # Run single case
  python runner.py --repo tritonbench       # Run specific repo
  python runner.py --configs ground_truth   # Run only ground truth
        """
    )
    parser.add_argument(
        "--repo",
        choices=["tritonbench", "liger_kernel"],
        default="tritonbench",
        help="Repository to run tests from (default: tritonbench)"
    )
    parser.add_argument(
        "--registry", "-w",
        type=str,
        help=f"Path to test registry file (default: {DEFAULT_REGISTRY})"
    )
    parser.add_argument(
        "--case", "-c",
        type=str,
        help="Run a single test case (e.g., matmul_triton1)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Base output directory for results (default: results/)"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(SAMPLING_CONFIGS.keys()),
        default=list(SAMPLING_CONFIGS.keys()),
        help="Configurations to run (default: all)"
    )
    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_base_dir = Path(args.output_dir)
    else:
        output_base_dir = SCRIPT_DIR / "results"

    # Load test registry
    registry_file = Path(args.registry) if args.registry else DEFAULT_REGISTRY
    if not registry_file.exists():
        print(f"Error: Test registry file not found: {registry_file}")
        return 1

    registry = load_registry(registry_file)
    if not registry:
        print(f"Error: No tests found in registry: {registry_file}")
        return 1

    print(f"Loaded test registry from {registry_file}: {len(registry)} total entries")

    # Discover tests for the specified repo
    tests = discover_tests(args.repo, registry, case=args.case)
    if not tests:
        print("No tests to process")
        return 1

    total_registry_tests = len(registry)

    # Print summary
    id_range = f"{tests[0]['global_id']}-{tests[-1]['global_id']}" if tests else "N/A"
    print(f"\n{'=' * 60}")
    print(f"Accuracy Experiment Runner")
    print(f"{'=' * 60}")
    print(f"Repository: {args.repo}")
    print(f"Tests to run: {len(tests)} (ID range: {id_range})")
    print(f"Configurations: {args.configs}")
    print(f"Output directory: {output_base_dir}")
    print(f"{'=' * 60}")

    # Print environment configuration
    print(f"\nEnvironment Configuration:")
    print(f"{'-' * 60}")
    for config_name in args.configs:
        config = SAMPLING_CONFIGS[config_name]
        print(f"\n  {config_name}: {config['description']}")
        print(f"    TRITON_INTERPRET=1")
        print(f"    ENABLE_TIMING=1")
        for key, val in config["env"].items():
            print(f"    {key}={val}")

    total_experiments = len(tests) * len(args.configs)
    print(f"\nTotal experiments: {total_experiments}")
    print(f"  ({len(tests)} tests x {len(args.configs)} configs)")
    print(f"\n{'=' * 60}")

    # Run tests for each configuration
    all_results = {}
    for config_name in args.configs:
        results = run_config(config_name, tests, output_base_dir, total_registry_tests)
        all_results[config_name] = results

    # Print overall summary
    print(f"\n{'=' * 60}")
    print("Overall Summary")
    print(f"{'=' * 60}")

    for config_name in args.configs:
        results = all_results[config_name]
        successful = sum(1 for r in results if r["success"])
        total = len(results)
        print(f"  {config_name}: {successful}/{total} passed ({100*successful/total:.1f}%)")

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_base_dir}")
    print(f"Run analyzer.py to generate comparison report")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
