"""
populate_full_workflow_outputs.py
Executes every cell of full_workflow.ipynb and populates standard outputs.
"""

import os
import io
import json
import base64
import cv2
import pymupdf
import numpy as np
import pandas as pd

def run_and_populate():
    with open("full_workflow.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Setup execution namespace
    ns = {}
    code_cell_counter = 0

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue

        code_cell_counter += 1
        cell["execution_count"] = code_cell_counter
        code_src = "".join(cell["source"])

        # Capture stdout
        old_stdout = io.StringIO()
        import sys
        sys_stdout_orig = sys.stdout
        sys.stdout = old_stdout

        displayed_items = []
        def custom_display(*args, **kwargs):
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    displayed_items.append({
                        "output_type": "display_data",
                        "data": {
                            "text/html": [arg.to_html(classes="dataframe", border=1)],
                            "text/plain": [arg.to_string()]
                        },
                        "metadata": {}
                    })
                elif hasattr(arg, "filename"):
                    # IPython Image
                    with open(arg.filename, "rb") as img_f:
                        img_b64 = base64.b64encode(img_f.read()).decode("utf-8")
                    displayed_items.append({
                        "output_type": "display_data",
                        "data": {
                            "image/png": img_b64,
                            "text/plain": [f"<IPython.core.display.Image object>"]
                        },
                        "metadata": {}
                    })
                else:
                    displayed_items.append({
                        "output_type": "display_data",
                        "data": {"text/plain": [str(arg)]},
                        "metadata": {}
                    })

        ns["display"] = custom_display

        try:
            exec(code_src, ns)
        except Exception as e:
            sys.stdout = sys_stdout_orig
            print(f"Error in cell {code_cell_counter}: {e}")
            raise e
        finally:
            sys.stdout = sys_stdout_orig

        stdout_val = old_stdout.getvalue()
        outputs = []
        if stdout_val:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": [stdout_val]
            })
        outputs.extend(displayed_items)
        cell["outputs"] = outputs
        print(f"Cell {code_cell_counter} executed successfully! Outputs captured: {len(outputs)}")

    with open("full_workflow.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print("\n[SUCCESS] full_workflow.ipynb populated with all outputs!")

if __name__ == "__main__":
    run_and_populate()

