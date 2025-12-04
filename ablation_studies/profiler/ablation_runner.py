#!/usr/bin/env python3
"""
Profiler ablation runner for TritonBench files.
Runs triton-profiler on selected files with configurable environment variables.
"""

import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import sys
import argparse
from collections import OrderedDict

# TritonBench directory
TRITONBENCH_DIR = Path("/home/hwu27/workspace/tile-lens-experiments/submodules/TritonBench/data/TritonBench_G_v1")

# Profiler configurations with different environment variables
PROFILER_CONFIGS = OrderedDict([
    ("both_enabled", {
        "name": "both_enabled",
        "description": "Both load/store skipping and block sampling enabled",
        "env": {
            "TRITON_INTERPRET": "1",
            "ENABLE_TIMING": "1",
            "PROFILER_ENABLE_LOAD_STORE_SKIPPING": "1",
            "PROFILER_ENABLE_BLOCK_SAMPLING": "1",
            "PROFILER_DISABLE_BUFFER_LOAD_CHECK": "1"
        }
    }),
    ("only_load_store_skipping", {
        "name": "only_load_store_skipping",
        "description": "Only load/store skipping enabled, block sampling disabled",
        "env": {
            "TRITON_INTERPRET": "1",
            "ENABLE_TIMING": "1",
            "PROFILER_ENABLE_LOAD_STORE_SKIPPING": "1",
            "PROFILER_ENABLE_BLOCK_SAMPLING": "0",
            "PROFILER_DISABLE_BUFFER_LOAD_CHECK": "1"
        }
    }),
    ("only_block_sampling", {
        "name": "only_block_sampling",
        "description": "Only block sampling enabled, load/store skipping disabled",
        "env": {
            "TRITON_INTERPRET": "1",
            "ENABLE_TIMING": "1",
            "PROFILER_ENABLE_LOAD_STORE_SKIPPING": "0",
            "PROFILER_ENABLE_BLOCK_SAMPLING": "1",
            "PROFILER_DISABLE_BUFFER_LOAD_CHECK": "1"
        }
    }),
    ("both_disabled", {
        "name": "both_disabled",
        "description": "Both features disabled (baseline)",
        "env": {
            "TRITON_INTERPRET": "1",
            "ENABLE_TIMING": "1",
            "PROFILER_ENABLE_LOAD_STORE_SKIPPING": "0",
            "PROFILER_ENABLE_BLOCK_SAMPLING": "0",
            "PROFILER_DISABLE_BUFFER_LOAD_CHECK": "1"
        }
    })
])


def load_whitelist(whitelist_file):
    """Load whitelist from file."""
    whitelist = []
    if whitelist_file.exists():
        with open(whitelist_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    whitelist.append(line)
    return whitelist


def discover_files(whitelist=None, case=None):
    """Discover Python files to profile."""
    files = []

    if not TRITONBENCH_DIR.exists():
        print(f"Error: TritonBench directory not found: {TRITONBENCH_DIR}")
        return files

    # Get all Python files
    all_files = sorted([f for f in TRITONBENCH_DIR.glob("*.py") if not f.name.startswith("__")])

    if case:
        # Run a single case
        case_name = case if case.endswith('.py') else f"{case}.py"
        files = [f for f in all_files if f.name == case_name]
        if files:
            print(f"Running single case: {case_name}")
        else:
            print(f"Error: Case not found: {case_name}")
            print(f"Available cases: {', '.join(f.stem for f in all_files[:10])}...")
    elif whitelist:
        # Filter by whitelist
        files = [f for f in all_files if f.name in whitelist]
        print(f"Using whitelist: {len(files)} out of {len(all_files)} files selected")
    else:
        files = all_files
        print(f"No whitelist provided, using all {len(files)} files")

    return files


def run_profiler(file_path, config, output_dir, test_number, total_tests):
    """Run triton-profiler on a single file with given configuration."""

    # Create output directory for this configuration
    config_output_dir = output_dir / config["name"]
    config_output_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename
    test_num_str = str(test_number).zfill(len(str(total_tests)))
    output_filename = f"{test_num_str}_{file_path.stem}.log"
    output_file = config_output_dir / output_filename

    # Setup environment
    env = os.environ.copy()
    env.update(config["env"])

    # Construct command
    cmd = ["triton-profiler", str(file_path)]

    # Print status
    print(f"  [{test_number}/{total_tests}] Running {file_path.name} with {config['name']}...")

    # Run the command
    start_time = time.time()

    with open(output_file, 'w') as f:
        # Write header information
        f.write(f"Test: {file_path.name}\n")
        f.write(f"Configuration: {config['name']}\n")
        f.write(f"Environment: {json.dumps(config['env'], indent=2)}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Start time: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        f.flush()

        try:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=TRITONBENCH_DIR,
                timeout=300  # 5 minute timeout
            )

            elapsed_time = time.time() - start_time

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")

            if result.returncode == 0:
                print(f"    ✓ Completed in {elapsed_time:.2f}s")
            else:
                print(f"    ✗ Failed with exit code {result.returncode}")

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"TIMEOUT: Test exceeded 300 seconds\n")
            f.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")
            print(f"    ✗ Timeout after {elapsed_time:.2f}s")
            return False

        except Exception as e:
            elapsed_time = time.time() - start_time
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"ERROR: {str(e)}\n")
            f.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")
            print(f"    ✗ Error: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Run profiler ablation study on TritonBench files")
    parser.add_argument("--whitelist", type=str, help="Path to whitelist file")
    parser.add_argument("-c", "--case", type=str, help="Run a single test case (e.g., bgmv_expand_slice)")
    parser.add_argument("--output-dir", type=str, help="Output directory for logs")
    parser.add_argument("--configs", nargs="+", choices=list(PROFILER_CONFIGS.keys()) + ["all"],
                       default=["all"], help="Configurations to run")
    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"profiler_ablation_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load whitelist if provided
    whitelist = None
    if args.whitelist:
        whitelist_file = Path(args.whitelist)
        if whitelist_file.exists():
            whitelist = load_whitelist(whitelist_file)
            print(f"Loaded whitelist from {whitelist_file}: {len(whitelist)} files")
        else:
            print(f"Warning: Whitelist file not found: {whitelist_file}")

    # Discover files
    files = discover_files(whitelist, case=args.case)
    if not files:
        print("No files to process")
        return 1

    # Determine configurations to run
    if "all" in args.configs:
        configs_to_run = list(PROFILER_CONFIGS.keys())
    else:
        configs_to_run = args.configs

    # Calculate total number of tests
    total_tests = len(files) * len(configs_to_run)

    print(f"\n{'=' * 60}")
    print(f"Profiler Ablation Study")
    print(f"{'=' * 60}")
    print(f"Files to profile: {len(files)}")
    print(f"Configurations: {', '.join(configs_to_run)}")
    print(f"Total tests: {total_tests}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}\n")

    # Run tests
    test_counter = 0
    results = {}

    for config_name in configs_to_run:
        config = PROFILER_CONFIGS[config_name]
        print(f"\n--- Configuration: {config['name']} ---")
        print(f"    {config['description']}")
        print()

        config_results = []

        for file_path in files:
            test_counter += 1
            success = run_profiler(file_path, config, output_dir, test_counter, total_tests)
            config_results.append({
                "file": file_path.name,
                "success": success
            })

        results[config_name] = config_results

        # Print summary for this configuration
        successful = sum(1 for r in config_results if r["success"])
        print(f"\n  Configuration summary: {successful}/{len(files)} tests passed\n")

    # Print overall summary
    print(f"\n{'=' * 60}")
    print("Overall Summary")
    print(f"{'=' * 60}")

    for config_name, config_results in results.items():
        successful = sum(1 for r in config_results if r["success"])
        total = len(config_results)
        print(f"{config_name:25} {successful:3}/{total:3} passed ({successful*100/total:.1f}%)")

    print(f"{'=' * 60}")
    print(f"Results saved in: {output_dir}/")
    print()

    # Write summary JSON
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "configurations": configs_to_run,
            "files": [f.name for f in files],
            "results": results
        }, f, indent=2)

    print(f"Summary written to: {summary_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())