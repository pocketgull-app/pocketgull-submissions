import json

notebook_path = "contests/casmi_2026/kernel_casmi_inference/casmi-2026-molecule-id-inference.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "WORKBENCH" in src or "Interactive Specimen Workbench" in src or "casmi_siggraph_specimen_workbench" in src:
        print(f"Cell {idx} ({cell.get('cell_type')}): {src[:120]}...\n")
