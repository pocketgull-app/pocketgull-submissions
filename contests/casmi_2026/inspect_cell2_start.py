import json

notebook_path = "contests/casmi_2026/kernel_casmi_inference/casmi-2026-molecule-id-inference.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell2 = nb["cells"][2]
print("Cell 2 total lines:", len(cell2.get("source", [])))
for i, line in enumerate(cell2.get("source", [])[:10]):
    print(f"{i}: {line[:80]}")
