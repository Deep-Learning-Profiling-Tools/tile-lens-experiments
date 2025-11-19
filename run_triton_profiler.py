#!/usr/bin/env python3
"""
Script to run all Python files in submodules/TritonBench/data/TritonBench_G_v1
Features:
- Resumable execution (checkpoint/resume)
- Real-time progress display
- Error handling (continues on errors)
- Log file recording

Usage:
    python run_triton_profiler.py              # Run profiler
    python run_triton_profiler.py reset        # Reset progress and logs
"""

import os
import subprocess
import json
import sys
import shutil
from pathlib import Path
from datetime import datetime

class TritonProfilerRunner:
    def __init__(self, benchmark_dir="submodules/TritonBench/data/TritonBench_G_v1",
                 work_dir="triton_profiler_results"):
        self.benchmark_dir = Path(benchmark_dir)
        self.work_dir = Path(work_dir)
        self.state_file = self.work_dir / "triton_runner_state.json"
        self.log_dir = self.work_dir / "logs"
        self.log_file = self.log_dir / f"profiler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Create subdirectories
        self.error_dir = self.log_dir / "error"
        self.success_dir = self.log_dir / "success"

        # Create all directories
        self.work_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.error_dir.mkdir(exist_ok=True)
        self.success_dir.mkdir(exist_ok=True)

        # Load checkpoint state
        self.state = self._load_state()

        # Get all Python files
        self.all_files = sorted([f for f in self.benchmark_dir.glob("*.py")])
        self.total_files = len(self.all_files)

    def _load_state(self):
        """Load checkpoint state for resumable execution"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Failed to load state file: {e}")
                return {"completed": [], "failed": []}
        return {"completed": [], "failed": []}

    def _save_state(self):
        """Save current state for resumable execution"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to save state file: {e}")

    def _log(self, message, level="INFO"):
        """Log message to both file and stdout"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        print(log_message)
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_message + "\n")
        except Exception as e:
            print(f"[ERROR] Failed to write to log file: {e}")

    def _save_error_details(self, file_name, returncode, stderr):
        """Save complete error details to error directory"""
        error_file = self.error_dir / f"{file_name}.log"
        try:
            with open(error_file, 'w') as f:
                f.write(f"File: {file_name}\n")
                f.write(f"Return code: {returncode}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n")
                f.write("Complete Error Output:\n")
                f.write("=" * 80 + "\n")
                f.write(stderr)
        except Exception as e:
            print(f"[ERROR] Failed to save error details: {e}")

    def _save_success_details(self, file_name, stdout):
        """Save complete success output to success directory"""
        success_file = self.success_dir / f"{file_name}.log"
        try:
            with open(success_file, 'w') as f:
                f.write(f"File: {file_name}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n")
                f.write("Complete Output:\n")
                f.write("=" * 80 + "\n")
                f.write(stdout)
        except Exception as e:
            print(f"[ERROR] Failed to save success details: {e}")

    def _run_profiler(self, script_path):
        """Run triton-profiler on a single Python file"""
        cmd = f"TRITON_INTERPRET=1 triton-profiler {script_path}"
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Timeout: execution exceeded 300 seconds"
        except Exception as e:
            return -1, "", str(e)

    def run(self):
        """Execute profiler on all Python files"""
        self._log("=" * 80)
        self._log("TritonBench Profiler Runner Started")
        self._log(f"Directory: {self.benchmark_dir}")
        self._log(f"Total files: {self.total_files}")
        self._log(f"Already completed: {len(self.state['completed'])}")
        self._log(f"Already failed: {len(self.state['failed'])}")
        self._log("=" * 80)

        completed_count = len(self.state['completed'])
        failed_count = len(self.state['failed'])

        for idx, file_path in enumerate(self.all_files, 1):
            file_name = file_path.name

            # Check if already completed
            if file_name in self.state['completed']:
                progress = f"[{idx}/{self.total_files}] [{(idx*100)//self.total_files}%]"
                self._log(f"{progress} Skip (already completed): {file_name}", "SKIP")
                continue

            # Check if previously failed
            if file_name in self.state['failed']:
                progress = f"[{idx}/{self.total_files}] [{(idx*100)//self.total_files}%]"
                self._log(f"{progress} Retry (previously failed): {file_name}", "RETRY")
            else:
                progress = f"[{idx}/{self.total_files}] [{(idx*100)//self.total_files}%]"
                self._log(f"{progress} Processing: {file_name}", "INFO")

            # Run profiler
            returncode, stdout, stderr = self._run_profiler(str(file_path))

            if returncode == 0:
                self._log(f"  ✓ Success: {file_name}", "SUCCESS")
                # Save success output
                if stdout:
                    self._save_success_details(file_name, stdout)
                    self._log(f"  Output saved to: success/{file_name}.log", "SUCCESS")
                if file_name in self.state['failed']:
                    self.state['failed'].remove(file_name)
                if file_name not in self.state['completed']:
                    self.state['completed'].append(file_name)
                completed_count += 1
            else:
                self._log(f"  ✗ Failed: {file_name} (return code: {returncode})", "ERROR")
                # Save complete error details to separate file
                if stderr:
                    self._save_error_details(file_name, returncode, stderr)
                    self._log(f"  Error details saved to: error/{file_name}.log", "ERROR")
                if file_name not in self.state['failed']:
                    self.state['failed'].append(file_name)
                failed_count += 1

            # Save state after each file
            self._save_state()

            # Display progress statistics
            success_rate = (completed_count * 100) // (idx) if idx > 0 else 0
            remaining = self.total_files - idx
            self._log(f"  Progress: Success={completed_count}, Failed={failed_count}, Remaining={remaining}, Success_Rate={success_rate}%", "STAT")

        # Final summary
        self._log("=" * 80)
        self._log("Execution Completed!")
        self._log(f"Total: {self.total_files}")
        self._log(f"Success: {completed_count}")
        self._log(f"Failed: {failed_count}")
        self._log(f"Success rate: {(completed_count*100)//self.total_files if self.total_files > 0 else 0}%")
        self._log(f"Log file: {self.log_file}")
        self._log("=" * 80)

def reset_progress(work_dir="triton_profiler_results"):
    """Reset all progress and logs"""
    work_dir_path = Path(work_dir)

    if not work_dir_path.exists():
        print(f"[INFO] No progress to reset. Directory '{work_dir}' does not exist.")
        return

    print(f"[WARNING] This will delete all progress and logs in '{work_dir}'")
    confirmation = input("Are you sure? (yes/no): ").strip().lower()

    if confirmation == "yes":
        try:
            shutil.rmtree(work_dir_path)
            print(f"[SUCCESS] Successfully removed '{work_dir}' directory")
            print("[INFO] All progress and logs have been reset")
        except Exception as e:
            print(f"[ERROR] Failed to remove directory: {e}")
    else:
        print("[INFO] Reset cancelled")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        reset_progress()
    else:
        runner = TritonProfilerRunner()
        runner.run()
