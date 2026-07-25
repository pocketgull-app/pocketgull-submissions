#!/usr/bin/env python3
"""
Executes train_model and run_model on real human PSG recordings downloaded from PhysioNet.
"""

import os
import sys
import numpy as np
import pandas as pd

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from team_code import train_model, run_model, load_model, HEADERS

print("=" * 64)
print("  Real PhysioNet Clinical Dataset Execution & Scoring Suite")
print("=" * 64)

real_dataset_dir = os.path.join(PYTHON_DIR, "real_physionet_dataset")
demo_csv = os.path.join(real_dataset_dir, "demographics.csv")
real_model_dir = os.path.join(PYTHON_DIR, "real_model_output")
os.makedirs(real_model_dir, exist_ok=True)

if not os.path.exists(demo_csv):
    print("  [ERROR] demographics.csv not found!")
    sys.exit(1)

df_demo = pd.read_csv(demo_csv)
print(f"Loaded real dataset metadata: {len(df_demo)} patients across {df_demo['SiteID'].nunique()} sites.")

print("\n--- 1. Training Model on Real Human PSG Waveforms ---")
train_model(real_dataset_dir, real_model_dir, verbose=True)

print("\n--- 2. Loading Trained Real-Data Model ---")
model = load_model(real_model_dir, verbose=True)

print("\n--- 3. Running Inference & Scoring on Real Human Patients ---")
predictions = []
probabilities = []

for idx, row in df_demo.iterrows():
    record = {
        HEADERS['bids_folder']: row['BidsFolder'],
        HEADERS['site_id']: row['SiteID'],
        HEADERS['session_id']: row['SessionID'],
    }
    b_out, p_out = run_model(model, record, real_dataset_dir, verbose=False)
    predictions.append(b_out)
    probabilities.append(p_out)
    print(f"  Patient {row['BidsFolder']} ({row['SiteID']}, Age {row['Age']}): Binary={b_out}, Prob={p_out:.4f}")

df_demo['Pred_Binary'] = predictions
df_demo['Pred_Prob'] = probabilities

print("\n" + "=" * 64)
print("  Real PhysioNet Execution Summary")
print("=" * 64)
print(f"Total Patients Scored: {len(df_demo)}")
print(f"Probabilities Min={min(probabilities):.4f}, Max={max(probabilities):.4f}, Mean={np.mean(probabilities):.4f}")
print(f"Binary Predictions: {sum(predictions)} Positive (1), {len(predictions)-sum(predictions)} Negative (0)")
print("[OK] Real PhysioNet Execution Complete!")
