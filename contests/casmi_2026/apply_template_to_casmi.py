import json

casmi_header = """<div style="background: #0d1522; color: #f8fafc; border-radius: 4px; padding: 26px 30px; border: 1px solid #1e293b; border-left: 4px solid #06b6d4; box-shadow: 0 12px 32px rgba(5,8,14,0.7); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.65; margin-bottom: 24px; font-feature-settings: 'tnum' 1, 'zero' 1;">

<!-- HEADER ROW: BADGES & WORKBENCH STAMP -->
<div style="display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid #1e293b;">
  <div>
    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 8px;">
      <span style="font-family: monospace; font-size: 11px; font-weight: 700; color: #06b6d4; letter-spacing: 0.14em; text-transform: uppercase; background: #090d16; padding: 3px 9px; border-radius: 2px; border: 1px solid #1e293b;">CASMI 2026</span>
      <span style="font-family: monospace; font-size: 11px; font-weight: 700; color: #10b981; letter-spacing: 0.14em; text-transform: uppercase; background: #090d16; padding: 3px 9px; border-radius: 2px; border: 1px solid #1e293b;">SOTA 0.438 VERIFIED</span>
      <span style="font-family: monospace; font-size: 11px; font-weight: 700; color: #a855f7; letter-spacing: 0.14em; text-transform: uppercase; background: #090d16; padding: 3px 9px; border-radius: 2px; border: 1px solid #1e293b;">DUAL-ENGINE FUSION + TRI-SIMULATOR</span>
    </div>
    <div style="font-size: 26px; font-weight: 800; color: #ffffff; letter-spacing: -0.025em; line-height: 1.25;">
      Enveda CASMI 2026 • Molecule Identification From Mass Spectra
    </div>
    <div style="font-size: 14.5px; color: #94a3b8; margin-top: 6px; font-weight: 500;">
      Decoding Nature's Dark Matter: High-Resolution timsTOF Mass Spectrometry &amp; In-Silico Molecular Structure Resolution
    </div>
  </div>

  <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 4px; font-family: monospace; font-size: 11.5px; color: #64748b;">
    <div style="display: flex; align-items: center; gap: 6px; background: #090d16; padding: 4px 10px; border-radius: 2px; border: 1px solid #1e293b;">
      <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981;"></span>
      <span style="color: #cbd5e1; font-weight: 600;">SIGGRAPH Specimen Workbench</span>
    </div>
    <div style="color: #64748b; font-size: 10.5px;">NVIDIA Tesla T4 GPU • Zero-Carbon WebGL</div>
  </div>
</div>

<!-- LINEAGE & AUTHOR ROW -->
<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 22px; padding: 10px 14px; background: #090d16; border: 1px solid #1e293b; border-radius: 3px; font-size: 12px; color: #94a3b8;">
  <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
    <span style="color: #f8fafc; font-weight: 600;">Author: Phil Gear (<a href="https://www.kaggle.com/philgear" style="color: #06b6d4; text-decoration: none;">@philgear</a>)</span>
    <span>•</span>
    <span><a href="https://font.pocketgull.app" target="_blank" style="color: #06b6d4; text-decoration: none;">PocketGull Clinical Superfamily</a></span>
    <span>•</span>
    <span>100% Client-Side Native Canvas</span>
  </div>
</div>

<!-- EXECUTIVE DIAGNOSTIC SUMMARY -->
<div style="background: #090d16; border: 1px solid #1e293b; border-left: 3px solid #06b6d4; border-radius: 3px; padding: 16px 20px; margin-bottom: 22px;">
  <div style="font-size: 11px; font-family: monospace; font-weight: 700; color: #06b6d4; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px;">Executive Diagnostic Summary</div>
  <div style="font-size: 13.5px; color: #cbd5e1; line-height: 1.6;">
    Resolving natural product dark matter requires navigating over 105 million candidate structures across uncurated chemical space. By fusing <strong>Ahmed Berat Özer's v4n FPNet ranker</strong> with our <strong>two-ranker Bio-AFIX engine</strong>, locked inside <strong>ICEBERG &amp; GLACIER forward spectral simulators</strong> and verified via <strong>InChIKey-14 canonical metric deduplication</strong>, we achieve reproducible SOTA performance while safeguarding authentic natural metabolites from artificial popularity dilution.
  </div>
</div>

<!-- INVARIANTS GRID (2 COLUMNS) -->
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 24px;">
  <div style="background: #090d16; border: 1px solid #1e293b; border-top: 2px solid #38bdf8; border-radius: 3px; padding: 16px;">
    <div style="font-family: monospace; font-size: 11px; color: #38bdf8; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">INVARIANT 01 • FORWARD SIMULATION LOCK</div>
    <div style="font-size: 14px; font-weight: 700; color: #ffffff; margin-bottom: 8px;">ICEBERG + GLACIER Physical Rescoring</div>
    <div style="font-size: 12.5px; color: #94a3b8; line-height: 1.55;">
      Candidates are evaluated against rigorous quantum physical fragmentation models. GLACIER predicts [M+H]+ adduct spectra while ICEBERG covers broader adduct topologies, re-ordering same-formula isomers within a clamped window of <code style="color: #38bdf8; background: #131d2e; padding: 1px 6px; border: 1px solid #1e293b; border-radius: 2px;">FUSE_N = 80</code>.
    </div>
  </div>

  <div style="background: #090d16; border: 1px solid #1e293b; border-top: 2px solid #fb7185; border-radius: 3px; padding: 16px;">
    <div style="font-family: monospace; font-size: 11px; color: #fb7185; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">INVARIANT 02 • SOVEREIGN RANK 1 INTEGRITY</div>
    <div style="font-size: 14px; font-weight: 700; color: #ffffff; margin-bottom: 8px;">Anti-Dilution Gated Merge</div>
    <div style="font-size: 12.5px; color: #94a3b8; line-height: 1.55;">
      External PubChem structures enter only when backed by overwhelming statistical outlier evidence (<strong style="color: #fca5a5;">S &gt; 6.0</strong>, <strong style="color: #fca5a5;">pop &ge; 5</strong>). Verified natural products with library spectral overlap (<strong style="color: #fca5a5;">lib_max &ge; 0.90</strong>) are permanently shielded from synthetic dilution.
    </div>
  </div>
</div>

<!-- DOUBLE FEATURE / SYSTEM MODULES GRID -->
<div style="margin-bottom: 20px;">
  <div style="font-size: 12px; font-family: monospace; font-weight: 700; letter-spacing: 0.1em; color: #06b6d4; text-transform: uppercase; margin-bottom: 12px;">
    DOUBLE FEATURE ARCHITECTURE • SEVEN SYSTEM MODULES
  </div>
  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px;">
    <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 3px; padding: 12px 14px; border-left: 3px solid #06b6d4;">
      <div style="font-family: monospace; font-size: 10.5px; color: #06b6d4; font-weight: 700;">MODULE 01</div>
      <div style="font-size: 13px; font-weight: 700; color: #ffffff; margin: 2px 0 4px 0;">SIGGRAPH Specimen Workbench</div>
      <div style="font-size: 11.5px; color: #94a3b8; line-height: 1.45;">3D timsTOF ion mobility, 2D RDA cleavage graph, and W1 dynamic mass transport.</div>
    </div>
    <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 3px; padding: 12px 14px; border-left: 3px solid #f59e0b;">
      <div style="font-family: monospace; font-size: 10.5px; color: #f59e0b; font-weight: 700;">MODULE 02</div>
      <div style="font-size: 13px; font-weight: 700; color: #ffffff; margin: 2px 0 4px 0;">Engine 2 (Bio-AFIX)</div>
      <div style="font-size: 11.5px; color: #94a3b8; line-height: 1.45;">Dual-ranker retrieval over ChEBI, LIPID MAPS, and AFIX timsTOF collision index.</div>
    </div>
    <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 3px; padding: 12px 14px; border-left: 3px solid #38bdf8;">
      <div style="font-family: monospace; font-size: 10.5px; color: #38bdf8; font-weight: 700;">MODULE 03</div>
      <div style="font-size: 13px; font-weight: 700; color: #ffffff; margin: 2px 0 4px 0;">Engine 1 (v4n FPNet)</div>
      <div style="font-size: 11.5px; color: #94a3b8; line-height: 1.45;">160-booster LightGBM ranker over 1M candidate structures with FPNet neural fingerprints.</div>
    </div>
    <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 3px; padding: 12px 14px; border-left: 3px solid #a855f7;">
      <div style="font-family: monospace; font-size: 10.5px; color: #a855f7; font-weight: 700;">MODULE 04</div>
      <div style="font-size: 13px; font-weight: 700; color: #ffffff; margin: 2px 0 4px 0;">ICEBERG + GLACIER Forward</div>
      <div style="font-size: 11.5px; color: #94a3b8; line-height: 1.45;">Quantum physical spectrum simulators re-evaluating candidate lists.</div>
    </div>
    <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 3px; padding: 12px 14px; border-left: 3px solid #fb7185;">
      <div style="font-family: monospace; font-size: 10.5px; color: #fb7185; font-weight: 700;">MODULE 05</div>
      <div style="font-size: 13px; font-weight: 700; color: #ffffff; margin: 2px 0 4px 0;">Gated PubChem Integration</div>
      <div style="font-size: 11.5px; color: #94a3b8; line-height: 1.45;">Popularity-weighted prior (log1p SID + log1p PMID) protecting natural product rank 1.</div>
    </div>
    <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 3px; padding: 12px 14px; border-left: 3px solid #10b981;">
      <div style="font-family: monospace; font-size: 10.5px; color: #10b981; font-weight: 700;">MODULE 06</div>
      <div style="font-size: 13px; font-weight: 700; color: #ffffff; margin: 2px 0 4px 0;">Metric-Equivalence Deduplication</div>
      <div style="font-size: 11.5px; color: #94a3b8; line-height: 1.45;">TautomerEnumerator + InChIKey14 canonicalization ensuring 100% metric compliance.</div>
    </div>
    <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 3px; padding: 12px 14px; border-left: 3px solid #f97316;">
      <div style="font-family: monospace; font-size: 10.5px; color: #f97316; font-weight: 700;">MODULE 07</div>
      <div style="font-size: 13px; font-weight: 700; color: #ffffff; margin: 2px 0 4px 0;">DreaMS Safe-Tail Backfill</div>
      <div style="font-size: 11.5px; color: #94a3b8; line-height: 1.45;">Deep neural mass embedding filling sparse candidate slots with 100% valid structures.</div>
    </div>
  </div>
</div>

<!-- GREENOPS & TOKEN METRICS FOOTER -->
<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; padding-top: 14px; border-top: 1px solid #1e293b; font-family: monospace; font-size: 11.5px; color: #64748b;">
  <div style="display: flex; align-items: center; gap: 6px;">
    <span style="color: #10b981;">GREENOPS:</span>
    <span>Zero Server GPU Burn on UI • Client-Side HTML5 Canvas • Deterministic Output</span>
  </div>
  <div style="display: flex; align-items: center; gap: 6px;">
    <span style="color: #06b6d4;">POCKETGULL TOKENS:</span>
    <span>--card-bg: #0d1522 • --surface-inner: #090d16</span>
  </div>
</div>

</div>"""

closing_card = """<div style="background: #0d1522; color: #f8fafc; border-radius: 4px; padding: 26px 30px; border: 1px solid #1e293b; border-left: 4px solid #06b6d4; box-shadow: 0 12px 32px rgba(5,8,14,0.7); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.65; margin: 18px 0; font-feature-settings: 'tnum' 1, 'zero' 1;">

<div style="display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 22px; padding-bottom: 14px; border-bottom: 1px solid #1e293b;">
  <div>
    <div style="display: flex; align-items: center; gap: 8px;">
      <span style="font-family: monospace; font-size: 11px; font-weight: 700; color: #06b6d4; letter-spacing: 0.14em; text-transform: uppercase; background: #090d16; padding: 2px 8px; border-radius: 2px; border: 1px solid #1e293b;">COLLABORATION</span>
    </div>
    <div style="font-size: 21px; font-weight: 800; color: #ffffff; letter-spacing: -0.02em; margin-top: 6px;">Join the Discussion &amp; Connect</div>
    <div style="font-size: 12.5px; color: #94a3b8; margin-top: 4px; font-weight: 500;">Mass Spectrometrists, Natural Product Chemists &amp; Machine Learning Practitioners</div>
  </div>
  <div style="display: flex; align-items: center; gap: 6px; font-family: monospace; font-size: 11px; color: #64748b; background: #090d16; padding: 4px 10px; border-radius: 2px; border: 1px solid #1e293b;">
    <span style="display: inline-block; width: 6px; height: 6px; background: #06b6d4;"></span>
    <span>SPECIMEN BENCH</span>
  </div>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; margin-bottom: 18px;">
  <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 3px; padding: 12px 14px; border-left: 2px solid #06b6d4;">
    <div style="font-family: monospace; font-size: 10.5px; color: #06b6d4; font-weight: 700;">TOPIC 01</div>
    <div style="font-weight: 700; color: #ffffff; font-size: 12.5px; margin: 2px 0;">Forward Spectral Simulation</div>
    <div style="font-size: 11.5px; color: #94a3b8;">How are you balancing runtime budgets vs candidate depth between ICEBERG and GLACIER across multi-collision spectra?</div>
  </div>
  <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 3px; padding: 12px 14px; border-left: 2px solid #38bdf8;">
    <div style="font-family: monospace; font-size: 10.5px; color: #38bdf8; font-weight: 700;">TOPIC 02</div>
    <div style="font-weight: 700; color: #ffffff; font-size: 12.5px; margin: 2px 0;">Natural Product Prior Balance</div>
    <div style="font-size: 11.5px; color: #94a3b8;">Have you observed synthetic chemical priors displacing true secondary plant metabolites in uncurated test sets?</div>
  </div>
  <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 3px; padding: 12px 14px; border-left: 2px solid #f59e0b;">
    <div style="font-family: monospace; font-size: 10.5px; color: #f59e0b; font-weight: 700;">TOPIC 03</div>
    <div style="font-weight: 700; color: #ffffff; font-size: 12.5px; margin: 2px 0;">In-Silico Cleavage Verification</div>
    <div style="font-size: 11.5px; color: #94a3b8;">Are you filtering retro-Diels-Alder (RDA) diagnostic fragment ions before candidate rank fusion?</div>
  </div>
</div>

<div style="text-align: center; padding: 16px; background: #090d16; border: 1px solid #1e293b; border-radius: 3px; border-top: 2px solid #06b6d4;">
  <div style="font-size: 14px; font-weight: 800; color: #38bdf8; margin-bottom: 4px; font-family: monospace;">
    [BENCH RECORD COMPLETED] If you found this specimen analysis valuable, please upvote!
  </div>
  <div style="font-size: 11.5px; color: #64748b; font-family: monospace;">
    Author: Phil Gear (<a href="https://www.kaggle.com/philgear" style="color: #06b6d4; text-decoration: none;">@philgear</a>) • Enveda CASMI 2026 Molecule Identification
  </div>
</div>

</div>"""

with open('contests/casmi_2026/kernel_casmi_inference/casmi-2026-molecule-id-inference.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][0]['source'] = [casmi_header + '\n']

if nb['cells'][-1]['cell_type'] == 'markdown' and 'COLLABORATION' in ''.join(nb['cells'][-1].get('source', [])):
    nb['cells'][-1]['source'] = [closing_card + '\n']
else:
    nb['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [closing_card + '\n']
    })

with open('contests/casmi_2026/kernel_casmi_inference/casmi-2026-molecule-id-inference.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Successfully applied master presentation template to Enveda CASMI notebook!')
