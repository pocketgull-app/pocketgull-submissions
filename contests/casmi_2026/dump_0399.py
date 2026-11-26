import json

with open('research_0399/casmi26-v4n-engine-fusion-union-lb-0-399.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

with open('research_0399_dump.txt', 'w', encoding='utf-8') as out:
    for i, c in enumerate(nb['cells']):
        out.write(f"\n{'='*30} CELL {i} [{c.get('cell_type')}] {'='*30}\n")
        out.write(''.join(c.get('source', [])))
        out.write('\n')

print("Dumped to research_0399_dump.txt successfully")
