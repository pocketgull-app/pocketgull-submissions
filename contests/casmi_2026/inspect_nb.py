import json
from pathlib import Path

def inspect_notebook(path_str):
    p = Path(path_str)
    print(f"=== Inspecting {p.name} ===")
    with open(p, encoding='utf-8') as f:
        nb = json.load(f)
    print(f"Total cells: {len(nb['cells'])}")
    for i, c in enumerate(nb['cells']):
        c_type = c.get('cell_type', 'unknown')
        source = c.get('source', [])
        first_line = source[0].strip() if source else 'EMPTY'
        print(f"Cell {i:02d} [{c_type}]: {first_line[:90]}")

if __name__ == '__main__':
    inspect_notebook('research_0399/casmi26-v4n-engine-fusion-union-lb-0-399.ipynb')
    print()
    inspect_notebook('research_0350/0-350-casmi-2026-v32-ensemble.ipynb')
