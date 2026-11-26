import json
import shutil
from pathlib import Path

# 1. Copy the notebook
src_nb = Path("research_0399/casmi26-v4n-engine-fusion-union-lb-0-399.ipynb")
dst_nb = Path("kernel_casmi_inference/casmi-2026-molecule-id-inference.ipynb")
shutil.copyfile(src_nb, dst_nb)
print(f"Copied {src_nb} to {dst_nb}")

# 2. Update kernel-metadata.json
metadata = {
  "id": "philgear/casmi-2026-molecule-id-inference",
  "title": "casmi-2026-molecule-id-inference",
  "code_file": "casmi-2026-molecule-id-inference.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": True,
  "enable_tpu": False,
  "enable_internet": False,
  "dataset_sources": [
    "prvsiyan/casmi26-fp-models-v2",
    "prvsiyan/casmi26-ranker-features",
    "megayak/casmi26-simulated-ranker-rows",
    "ahmedberatozer/casmi26-fpnet-full1",
    "ahmedberatozer/casmi26-glacier",
    "ahmedberatozer/casmi26-iceberg",
    "ahmedberatozer/casmi26-pubchem-tier",
    "ahmedberatozer/casmi26-v2-pool",
    "ahmedberatozer/casmi26-v3-models",
    "ahmedberatozer/casmi26-v4b-models",
    "prvsiyan/chebi-lipidmaps-casmi26",
    "prvsiyan/coconut-casmi26-candidates",
    "metric/rdkit-2026-3-3-wheel"
  ],
  "competition_sources": [
    "enveda-CASMI26-molecule-id-mass-spectra"
  ],
  "kernel_sources": [],
  "model_sources": []
}

meta_path = Path("kernel_casmi_inference/kernel-metadata.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"Updated {meta_path} with 13 foundation datasets and GPU enabled.")
