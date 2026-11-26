import json

path = r'c:\Users\philg\Pocketgull\pocketgull\contests\waiting_game\3d-knee-anatomy-simulator-and-waiting-game.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 0 markdown link
src0 = nb['cells'][0]['source']
is_list0 = isinstance(src0, list)
src0_str = ''.join(src0) if is_list0 else src0

old_target0 = '<span><a href="https://font.pocketgull.app" target="_blank" style="color: #06b6d4; text-decoration: none;">PocketGull Clinical Superfamily</a></span>'
new_target0 = '<span><a href="https://www.kaggle.com/datasets/philgear/pocketgull-font" target="_blank" style="color: #06b6d4; text-decoration: none; font-weight: 600;" title="Explore the 1000 UPM PocketGull Clinical Typography Superfamily on Kaggle">🔤 PocketGull Typography Superfamily (1000 UPM)</a></span>'

if old_target0 in src0_str:
    src0_str = src0_str.replace(old_target0, new_target0)
    nb['cells'][0]['source'] = src0_str.splitlines(keepends=True) if is_list0 else src0_str
    print('Cell 0 updated successfully!')
else:
    print('Warning: old_target0 not found in cell 0')

# 2. Update Cell 3 3D HUD Header
src3 = nb['cells'][3]['source']
is_list3 = isinstance(src3, list)
src3_str = ''.join(src3) if is_list3 else src3

old_target3 = '<span id="header-mode-badge" class="font-brand-pocketgull text-xs font-bold tracking-wider text-teal-300 uppercase">POCKETGULL 3D MRI SCANNER v12</span>'
new_target3 = '<a href="https://www.kaggle.com/datasets/philgear/pocketgull-font" target="_blank" class="no-underline hover:opacity-80 transition flex items-center gap-1.5" title="Built with the PocketGull 1000 UPM Clinical Typography Superfamily (Click to view dataset on Kaggle)"><span id="header-mode-badge" class="font-brand-pocketgull text-xs font-bold tracking-wider text-teal-300 uppercase">POCKETGULL 3D MRI SCANNER v12</span><span class="text-[9px] bg-teal-900/90 text-teal-200 px-1.5 py-0.5 rounded border border-teal-600/70 font-mono">🔤 1000 UPM</span></a>'

if old_target3 in src3_str:
    src3_str = src3_str.replace(old_target3, new_target3)
    nb['cells'][3]['source'] = src3_str.splitlines(keepends=True) if is_list3 else src3_str
    print('Cell 3 updated successfully!')
else:
    print('Warning: old_target3 not found in cell 3')

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Saved notebook successfully!')
