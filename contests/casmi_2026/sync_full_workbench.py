import json

with open("casmi_siggraph_specimen_workbench.html", "r", encoding="utf-8") as f:
    html_content = f.read()

notebook_path = "contests/casmi_2026/kernel_casmi_inference/casmi-2026-molecule-id-inference.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 2 update:
cell2_src = [
    "# =========================================================================\n",
    "# 2. INTERACTIVE SIGGRAPH SPECIMEN WORKBENCH & DSRP SYSTEMS GRAMMAR\n",
    "# Renders immediately at the top of the notebook (RSNA-standard neuro-ergonomic UI)\n",
    "# 100% self-contained offline HTML5/Canvas (zero external JS/CDN dependencies)\n",
    "# =========================================================================\n",
    "from IPython.display import HTML, display\n",
    "\n",
    f"WORKBENCH_HTML = {repr(html_content)}\n",
    "\n",
    "display(HTML(WORKBENCH_HTML))\n"
]
nb["cells"][2]["source"] = cell2_src
print("Updated Cell 2 with full repr(html_content) containing DSRP Systems Thinking!")

# Cell 25 update:
cell25_src = [
    "# Export interactive SIGGRAPH specimen workbench artifact\n",
    "with open('casmi_siggraph_specimen_workbench.html', 'w', encoding='utf-8') as f:\n",
    "    f.write(WORKBENCH_HTML)\n",
    "print('✓ Exported casmi_siggraph_specimen_workbench.html')\n"
]
nb["cells"][25]["source"] = cell25_src
print("Updated Cell 25.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("Successfully written notebook!")
