#!/usr/bin/env python3
"""
Run all example scripts and generate a summary report.

This script executes all example scripts in sequence and reports results.
"""

import sys
import os
import subprocess
import time

# The parent directory is added to the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_example(script_name, description):
    """Run an example script and return success status."""
    print(f"\n{'=' * 70}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print('=' * 70)
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.dirname(script_path)),
            env={**os.environ, 'PYTHONPATH': '.'},
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"Success ({elapsed:.2f}s)")
            if result.stdout:
                # The last few lines of output are printed.
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    if line.strip():
                        print(f"  {line}")
            return True, elapsed, None
        else:
            print(f"Failed ({elapsed:.2f}s)")
            if result.stderr:
                print("Error output:")
                for line in result.stderr.strip().split('\n')[-10:]:
                    print(f"  {line}")
            return False, elapsed, result.stderr
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"Timeout (>{elapsed:.2f}s)")
        return False, elapsed, "Timeout after 5 minutes"
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Error ({elapsed:.2f}s)")
        print(f"  {str(e)}")
        return False, elapsed, str(e)

def main():
    """Run all examples and generate summary."""
    print("=" * 70)
    print("PyCIB Example Scripts - Full Execution")
    print("=" * 70)
    
    examples = [
        ("example_1_basic_cib.py", "Example 1: Basic Deterministic CIB"),
        ("example_cycle_detection.py", "Example: Cycle Detection in CIB Succession"),
        ("example_transformation_matrix.py", "Example: Transformation Matrix Analysis"),
        ("run_notebook.py", "Notebook: Dynamic CIB (from dynamic_cib.ipynb)"),
        ("example_dynamic_cib_c10.py", "Example: Dynamic CIB on DATASET_C10 (workshop scale)"),
        ("example_enumeration_c10.py", "Example: Full enumeration on DATASET_C10"),
        ("example_attractor_basin_validation_c10.py", "Example: Attractor basin validation on DATASET_C10"),
        ("example_state_binning.py", "Example: State Binning (model reduction)"),
        ("example_solver_modes.py", "Example: Scaling solver modes (exact and Monte Carlo)"),
        ("example_solver_modes_c10.py", "Example: Scaling solver modes on DATASET_C10 (workshop scale)"),
        ("example_probabilistic_cia_static.py", "Example: Joint-Distribution Probabilistic CIA (static)"),
        ("example_probabilistic_cia_dynamic_refit.py", "Example: Joint-Distribution Probabilistic CIA (dynamic refit)"),
        ("example_probabilistic_cia_strict_vs_repair.py", "Example: Joint-Distribution Probabilistic CIA (strict versus repair)"),
        ("example_probabilistic_cia_dynamic_predict_update.py", "Example: Joint-Distribution Probabilistic CIA (dynamic predict–update)"),
        ("example_probabilistic_cia_sparse_kl.py", "Example: Joint-Distribution Probabilistic CIA (sparse constraints + KL + bounds)"),
        ("example_probabilistic_cia_scaling_iterative.py", "Example: Joint-Distribution Probabilistic CIA (scaling, iterative approximate)"),
    ]
    
    results = []
    total_start = time.time()
    
    for script, description in examples:
        success, elapsed, error = run_example(script, description)
        results.append((script, description, success, elapsed, error))
    
    total_elapsed = time.time() - total_start
    
    # A summary is printed.
    print("\n" + "=" * 70)
    print("Execution Summary")
    print("=" * 70)
    
    passed = sum(1 for _, _, success, _, _ in results if success)
    total = len(results)
    
    for script, description, success, elapsed, error in results:
        status = "Pass" if success else "Fail"
        print(f"{status:8} {description:45} ({elapsed:6.2f}s)")
        if not success and error:
            print(f"         Error: {error[:100]}...")
    
    print("-" * 70)
    print(f"Total: {passed}/{total} examples passed ({total_elapsed:.2f}s)")
    print("=" * 70)
    
    if passed == total:
        print("\nAll examples completed successfully.")
        return 0
    else:
        print(f"\n{total - passed} example(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
