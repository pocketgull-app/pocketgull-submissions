# PocketGull Submissions Higher-Order Standards

Inherits global higher-order skills from `~/.gemini/config/skills/shared-constitution.json` and global `PreToolUse` safety gates.

## Repository Invariants
- **Epistemic Rigor & Leak-Free Splits:** Enforce GroupKFold/patient-level partitioning, calibrated uncertainty bounds, and explicit seed/provenance tracking across PhysioNet/Kaggle notebooks and Python pipelines.
- **Air-Gapped Reproducibility:** Test competition inference pipelines cleanly with deterministic outputs and zero external runtime network assumptions.
- **Privacy & PHI Screening:** Screen all synthetic EDF/CSV/table generators against HIPAA §164.514 Safe Harbor before staging uploads to Kaggle or external benchmarks.
- **Exit Gate:** Run test verification (`pytest` / model tests) and `node C:/Users/philg/Pocketgull/pocketgull/scripts/universal_observatory_csprng_audit.mjs --root .` before submission.
