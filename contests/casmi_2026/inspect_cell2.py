import json

notebook_path = "contests/casmi_2026/kernel_casmi_inference/casmi-2026-molecule-id-inference.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell2 = nb["cells"][2]
print("Cell 2 lines count:", len(cell2.get("source", [])))
print("First 20 lines of Cell 2:")
for line in cell2.get("source", [])[:20]:
    print(repr(line))
