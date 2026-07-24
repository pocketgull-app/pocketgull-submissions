# PhysioNet Challenge 2026 Official Submission Package

This package contains the official entry code, Docker container spec, and evaluation harness for the **George B. Moody PhysioNet Challenge 2026**.

## Entry Architecture
- **`team_code.py`**: Model training, loading, and inference entrypoints (`train_model`, `load_model`, `run_model`).
- **`pocketgull_features.py`**: Feature extraction engine (Wavelet denoising, SQI filter, CAISR sleep architecture, Vagal tone HRV, Hemodynamics).
- **`evaluate_model.py`**: Official challenge evaluation script computing Age-Conditioned AUROC ($s_C$) and Prevalence Reward ($r_C$).
- **`Dockerfile`**: Self-contained Docker environment for evaluation servers.

## Container Test Commands
```bash
# 1. Build evaluation container
docker build -t physionet2026-entry .

# 2. Run model inference test
python run_model.py data output
```
