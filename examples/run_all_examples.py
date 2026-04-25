#!/usr/bin/env python3
"""
Run all example scripts and generate a summary report.

This script executes all example scripts in sequence and reports results.
"""

import sys
import os
import subprocess
import time
import json
import uuid

# The parent directory is added to the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEBUG_LOG_PATH = ""

def _debug_log(run_id, hypothesis_id, location, message, data):
    # #region agent log
    payload = {
        "sessionId": "16c90d",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
        "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
    }
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, separators=(",", ":")) + "\n")
    # #endregion

def run_example(script_name, description):
    """Run an example script and return success status."""
    print(f"\n{'=' * 70}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print('=' * 70)
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    start_time = time.time()
    
    timeout_by_script = {
        "example_dynamic_cib_c15_rare_events.py": 600,
        "example_probabilistic_cia_static.py": 600,
        "example_probabilistic_cia_dynamic_refit.py": 600,
        "example_probabilistic_cia_sparse_kl.py": 600,
    }
    timeout_seconds = timeout_by_script.get(script_name, 300)
    run_id = f"run_example:{script_name}"
    # #region agent log
    _debug_log(
        run_id,
        "H2",
        "examples/run_all_examples.py:run_example:start",
        "Starting example subprocess",
        {"script": script_name, "timeoutSeconds": timeout_seconds},
    )
    # #endregion

    try:
        mpl_config = os.path.join(os.path.dirname(os.path.dirname(script_path)), ".mplconfig")
        os.makedirs(mpl_config, exist_ok=True)
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.dirname(script_path)),
            env={
                **os.environ,
                "PYTHONPATH": ".",
                "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": mpl_config,
                # Keep heavy probabilistic examples CI/smoke friendly.
                "PYCIB_EXAMPLE_QUICK": "1",
            },
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        elapsed = time.time() - start_time
        # #region agent log
        _debug_log(
            run_id,
            "H3",
            "examples/run_all_examples.py:run_example:result",
            "Example subprocess finished",
            {
                "script": script_name,
                "returnCode": result.returncode,
                "elapsedSeconds": round(elapsed, 3),
                "stdoutLength": len(result.stdout or ""),
                "stderrLength": len(result.stderr or ""),
            },
        )
        # #endregion
        
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
        # #region agent log
        _debug_log(
            run_id,
            "H1",
            "examples/run_all_examples.py:run_example:timeout",
            "Example subprocess timeout",
            {"script": script_name, "elapsedSeconds": round(elapsed, 3), "timeoutSeconds": timeout_seconds},
        )
        # #endregion
        print(f"Timeout (>{elapsed:.2f}s)")
        return False, elapsed, "Timeout after 5 minutes"
    except Exception as e:
        elapsed = time.time() - start_time
        # #region agent log
        _debug_log(
            run_id,
            "H4",
            "examples/run_all_examples.py:run_example:exception",
            "Runner raised exception",
            {"script": script_name, "error": str(e), "elapsedSeconds": round(elapsed, 3)},
        )
        # #endregion
        print(f"Error ({elapsed:.2f}s)")
        print(f"  {str(e)}")
        return False, elapsed, str(e)

def main():
    """Run all examples and generate summary."""
    # #region agent log
    _debug_log(
        "run_all_examples:main",
        "H5",
        "examples/run_all_examples.py:main:start",
        "Starting full examples runner",
        {"pythonExecutable": sys.executable},
    )
    # #endregion
    print("=" * 70)
    print("PyCIB Example Scripts - Full Execution")
    print("=" * 70)
    
    examples = [
        ("example_1_basic_cib.py", "Example 1: Basic Deterministic CIB"),
        ("example_cycle_detection.py", "Example: Cycle Detection in CIB Succession"),
        ("example_transformation_matrix.py", "Example: Transformation Matrix Analysis"),
        ("run_notebook.py", "Notebook: Dynamic CIB (from dynamic_cib.ipynb)"),
        ("example_dynamic_cib_c10.py", "Example: Dynamic CIB on DATASET_C10 (workshop scale)"),
        ("example_shock_robustness_completeness.py", "Example: Shock robustness completeness workflow (calibration + scaling + robustness metrics)"),
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
        ("example_dynamic_cib_c15_rare_events.py", "Example: Dynamic CIB on DATASET_C15 (rare events)"),
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
