import json

with open("casmi_siggraph_specimen_workbench.html", "r", encoding="utf-8") as f:
    html_content = f.read()

notebook_path = "contests/casmi_2026/kernel_casmi_inference/casmi-2026-molecule-id-inference.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find cell defining WORKBENCH_HTML
cell2_idx = None
cell25_idx = None

for idx, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "WORKBENCH_HTML = r'''" in src or 'WORKBENCH_HTML = """' in src:
        cell2_idx = idx
    if "casmi_siggraph_specimen_workbench.html" in src and "with open" in src:
        cell25_idx = idx

print(f"Found Cell with WORKBENCH_HTML at: {cell2_idx}")
print(f"Found Cell with file write at: {cell25_idx}")

# Update Cell 2
if cell2_idx is not None:
    cell2_src = [
        "import IPython.display as ipd\n",
        "# -----------------------------------------------------------------------------\n",
        "# CASMI 2026 Interactive Specimen Workbench (SIGGRAPH / DSRP Systems Grammar)\n",
        "# -----------------------------------------------------------------------------\n",
        f"WORKBENCH_HTML = r'''{html_content}'''\n",
        "\n",
        "ipd.display(ipd.HTML(WORKBENCH_HTML))\n"
    ]
    nb["cells"][cell2_idx]["source"] = cell2_src
    print("Updated WORKBENCH_HTML cell.")

# Update Cell 25 if it exists and embeds the html directly
if cell25_idx is not None and cell25_idx != cell2_idx:
    cell25_src = [
        "# Export interactive SIGGRAPH specimen workbench artifact\n",
        "with open('casmi_siggraph_specimen_workbench.html', 'w', encoding='utf-8') as f:\n",
        "    f.write(WORKBENCH_HTML)\n",
        "print('✓ Exported casmi_siggraph_specimen_workbench.html')\n"
    ]
    nb["cells"][cell25_idx]["source"] = cell25_src
    print("Updated export cell.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("Successfully synced notebook with DSRP Systems Thinking Specimen Workbench!")
