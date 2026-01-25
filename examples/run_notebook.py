#!/usr/bin/env python3
"""
Execute the dynamic_cib.ipynb notebook as a Python script.

This script extracts code cells from the Jupyter notebook and executes them,
making the notebook runnable from the command line while preserving the
original notebook structure.
"""

import sys
import os
import json

# The parent directory is added to the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def extract_code_from_notebook(notebook_path):
    """
    Extract code cells from a Jupyter notebook.

    Args:
        notebook_path: Path to the .ipynb file.

    Returns:
        List of code cell source strings.

    Raises:
        FileNotFoundError: If notebook file does not exist.
        ValueError: If notebook format is invalid.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    code_cells = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = source
            if code.strip():
                code_cells.append(code)

    return code_cells

def main():
    """Execute the notebook code cells."""
    notebook_path = os.path.join(os.path.dirname(__file__), 'dynamic_cib.ipynb')

    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return 1

    print("=" * 70)
    print("Executing dynamic_cib.ipynb")
    print("=" * 70)

    try:
        code_cells = extract_code_from_notebook(notebook_path)
        print(f"\nFound {len(code_cells)} code cells to execute\n")

        # A non-interactive matplotlib backend is used.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # The results directory is determined (package root / results).
        try:
            import cib.example_data as mod
            package_dir = os.path.dirname(os.path.dirname(mod.__file__))
            results_dir = os.path.join(package_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)
        except Exception:
            # Fallback: the current directory is used.
            results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
            os.makedirs(results_dir, exist_ok=True)

        # plt.show() is overridden to save plots instead.
        plot_counter = [0]  # A list is used to allow modification in nested function.
        
        def save_plot_instead_of_show():
            """Save the current plot instead of showing it."""
            plot_counter[0] += 1
            plot_path = os.path.join(results_dir, f"example_dynamic_cib_notebook_plot_{plot_counter[0]}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  Saved plot to results/example_dynamic_cib_notebook_plot_{plot_counter[0]}.png")
            plt.close()

        # matplotlib.pyplot.show is patched at module level so it works after imports.
        plt.show = save_plot_instead_of_show
        matplotlib.pyplot.show = save_plot_instead_of_show

        # Each code cell is executed.
        namespace = {}
        
        for i, code in enumerate(code_cells, 1):
            print(f"Executing cell {i}/{len(code_cells)}...")
            try:
                exec(code, namespace)
            except Exception as e:
                print(f"Error in cell {i}: {e}")
                return 1

        print(f"\nNotebook execution completed successfully.")
        if plot_counter[0] > 0:
            print(f"Plots saved to: {results_dir}")
        return 0

    except Exception as e:
        print(f"Error executing notebook: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
