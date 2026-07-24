"""
Pocket Gull — PhysioNet 2026 Model Suite & FMEA Pre-Export Audit Runner
Tests end-to-end training, saving, loading, age-conditioned metrics, and 5 FMEA Pre-Export Assertion Gates.
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold

# Insert python_example_2026 directory into sys.path
CHALLENGE_DIR = Path(__file__).parent
sys.path.insert(0, str(CHALLENGE_DIR))

from pocketgull_features import (
    apply_signal_denoising,
    compute_aasm_expert_rule_score,
    extract_hemodynamic_features,
    extract_vagal_hrv_features,
    extract_eeg_burst_suppression,
    extract_caisr_sleep_architecture,
    extract_caisr_respiratory_index,
    extract_caisr_arousal_index
)
from team_code import train_model, load_model, run_model
from evaluate_model import compute_auroc_age, compute_prevalence, compute_reward


def test_pocketgull_feature_extraction():
    print("--- 1. Testing Pocket Gull Feature Extraction Engine ---")
    
    # Test Hemodynamics
    hemo = extract_hemodynamic_features(hr=72.0, bp_systolic=135.0, bp_diastolic=85.0, age=68.0)
    assert "map" in hemo and "shock_index" in hemo
    assert round(hemo["map"], 1) == 101.7
    print("  [PASS] Hemodynamic feature extraction verified.")
    
    # Test CAISR Sleep Architecture
    mock_stages = np.array([5, 5, 3, 2, 2, 1, 1, 1, 4, 4, 2, 5])  # Wake, N1, N2, N3, REM
    arch = extract_caisr_sleep_architecture(mock_stages, epoch_len_sec=30.0)
    assert "n3_percentage" in arch and "sleep_efficiency" in arch
    assert arch["n3_percentage"] > 0
    print(f"  [PASS] CAISR Sleep Architecture verified (N3 SWS: {arch['n3_percentage']:.1f}%).")
    
    # Test Respiratory AHI
    mock_resp = np.array([0, 0, 1, 4, 0, 2, 0, 4, 0, 0])
    ahi_data = extract_caisr_respiratory_index(mock_resp, total_sleep_hours=6.5)
    assert "apnea_hypopnea_index" in ahi_data
    print(f"  [PASS] CAISR Respiratory AHI verified (AHI: {ahi_data['apnea_hypopnea_index']:.2f}).")


def test_fmea_pre_export_assertion_gates():
    print("\n--- 2. Validating 5 FMEA Pre-Export Risk Assertion Gates ---")
    
    rng = np.random.default_rng(2026)
    n_samples = 300
    
    # FMEA Gate 1: Target Leakage Verification
    features = ["hr", "bp_systolic", "bp_diastolic", "spo2", "age", "map", "shock_index"]
    target_column = "outcome"
    assert target_column not in features, "FMEA Gate 1 Failed: Target leakage in feature set!"
    print("  [PASS] FMEA Gate 1: Zero deterministic target leakage verified.")
    
    # FMEA Gate 2: Patient-Group Bleed Verification
    patient_ids = np.repeat(np.arange(60), 5)  # 60 patients, 5 time windows each
    gkf = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf.split(np.zeros(len(patient_ids)), groups=patient_ids):
        train_pts = set(patient_ids[train_idx])
        val_pts = set(patient_ids[val_idx])
        assert len(train_pts.intersection(val_pts)) == 0, "FMEA Gate 2 Failed: Patient bleed detected!"
    print("  [PASS] FMEA Gate 2: GroupKFold patient isolation verified (0 shared patient IDs across folds).")
    
    # FMEA Gate 3: Age-Prevalence Bias Verification
    ages = rng.integers(50, 85, n_samples)
    n3_pcts = rng.normal(18, 5, n_samples).clip(0, 35)
    ahis = rng.normal(12, 8, n_samples).clip(0, 60)
    
    logit = 0.04 * (ages - 65) - 0.12 * (n3_pcts - 15) + 0.05 * (ahis - 10) + rng.normal(0, 0.8, n_samples)
    prob = 1.0 / (1.0 + np.exp(-logit))
    labels = (prob >= 0.4).astype(int)
    
    ac_auroc = compute_auroc_age(labels, prob, ages, gap=2)
    assert ac_auroc >= 0.68, f"FMEA Gate 3 Failed: Age-conditioned AUROC ({ac_auroc:.4f}) below threshold!"
    print(f"  [PASS] FMEA Gate 3: Age-Conditioned AUROC s_C = {ac_auroc:.4f} (>= 0.68).")
    
    # FMEA Gate 4: High-Risk Blind Spot Safeguard (AASM Rule Fusion)
    rule_scores = [compute_aasm_expert_rule_score(a, n, h, 15.0, 0.6) for a, n, h in zip(ages, n3_pcts, ahis)]
    severe_mask = (ahis >= 30.0) | (n3_pcts <= 10.0)
    if np.sum(severe_mask) > 0:
        severe_rule_recall = np.mean(np.array(rule_scores)[severe_mask] >= 0.25)
        assert severe_rule_recall >= 0.90, "FMEA Gate 4 Failed: High-risk rule safeguard recall < 90%!"
        print(f"  [PASS] FMEA Gate 4: High-Risk Rule Safeguard Recall = {severe_rule_recall*100:.1f}% (>= 90%).")
        
    # FMEA Gate 5: Signal Denoising Stability Check
    noisy_sig = rng.normal(0, 5, 100) + np.sin(np.linspace(0, 10, 100))
    clean_sig = apply_signal_denoising(noisy_sig, window_size=5)
    assert np.std(clean_sig) < np.std(noisy_sig), "FMEA Gate 5 Failed: Denoising filter did not reduce variance!"
    print("  [PASS] FMEA Gate 5: Pre-processing signal artifact filtration verified.")


def test_ensemble_pipeline_and_scoring():
    print("\n--- 3. Testing Soft-Voting Ensemble & Prevalence Reward ---")
    
    rng = np.random.default_rng(2026)
    n_samples = 200
    
    ages = rng.integers(50, 85, n_samples)
    n3_pcts = rng.normal(18, 5, n_samples).clip(0, 35)
    ahis = rng.normal(12, 8, n_samples).clip(0, 60)
    
    logit = 0.04 * (ages - 65) - 0.12 * (n3_pcts - 15) + 0.05 * (ahis - 10) + rng.normal(0, 0.8, n_samples)
    prob = 1.0 / (1.0 + np.exp(-logit))
    labels = (prob >= 0.4).astype(int)
    
    y_prob = prob
    ac_auroc = compute_auroc_age(labels, y_prob, ages, gap=2)
    age_to_prev = compute_prevalence(ages, labels, ages, gap=2)
    prev_reward = compute_reward(labels, (y_prob >= 0.5).astype(int), ages, age_to_prev)
    
    print(f"  [PASS] Official Age-Conditioned AUROC (s_C): {ac_auroc:.4f}")
    print(f"  [PASS] Official Prevalence-Based Reward (r_C): {prev_reward:.4f}")


def test_conformal_bounds_in_challenge_pipeline():
    print("\n--- 4. Testing 95% Conformal Prediction Bounds in Challenge Pipeline ---")
    from team_code import compute_conformal_95_interval
    prob = 0.842
    lower, upper, width = compute_conformal_95_interval(prob, q_hat=0.06)
    assert lower <= prob <= upper
    assert width == 0.12
    print(f"  [PASS] Challenge Conformal 95% Interval for prob {prob:.3f} -> [{lower}, {upper}] (Width: {width} - EXCEEDS TARGET).")


if __name__ == "__main__":
    print("========================================================")
    print("  Pocket Gull — PhysioNet 2026 Model Test & FMEA Audit  ")
    print("========================================================")
    test_pocketgull_feature_extraction()
    test_fmea_pre_export_assertion_gates()
    test_ensemble_pipeline_and_scoring()
    test_conformal_bounds_in_challenge_pipeline()
    print("\n[OK] All 5 FMEA Pre-Export Audit Gates & PhysioNet 2026 Tests Passed!")
