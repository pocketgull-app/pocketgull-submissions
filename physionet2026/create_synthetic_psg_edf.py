"""
Pocket Gull — Synthetic Polysomnography EDF Waveform Generator
Generates realistic multi-channel European Data Format (.edf) files using edfio for testing PhysioNet 2026 pipelines.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import edfio
from edfio import Edf, EdfSignal

SCRIPT_DIR = Path(__file__).parent


def generate_synthetic_psg_signals(duration_seconds: int = 300) -> list[EdfSignal]:
    """
    Generates realistic 10-channel PSG signals for the specified duration.
    Channels include: C3-M2, C4-M1, EOG-L, EOG-R, EMG, ECG, SpO2, Airflow, Chest, Abdomen.
    """
    rng = np.random.default_rng(2026)
    t_100 = np.linspace(0, duration_seconds, duration_seconds * 100, endpoint=False)
    t_10 = np.linspace(0, duration_seconds, duration_seconds * 10, endpoint=False)
    t_1 = np.linspace(0, duration_seconds, duration_seconds * 1, endpoint=False)

    # 1. EEG C3-M2 (100 Hz): 10Hz Alpha + 1.5Hz Delta + Gaussian Noise
    eeg_c3 = (
        15.0 * np.sin(2 * np.pi * 10.0 * t_100) +
        35.0 * np.sin(2 * np.pi * 1.5 * t_100) +
        rng.normal(0, 5.0, len(t_100))
    )

    # 2. EEG C4-M1 (100 Hz): Similar Central EEG
    eeg_c4 = (
        12.0 * np.sin(2 * np.pi * 9.5 * t_100) +
        40.0 * np.sin(2 * np.pi * 1.2 * t_100) +
        rng.normal(0, 5.0, len(t_100))
    )

    # 3. EOG L-M2 (100 Hz): Eye Movement
    eog_l = 25.0 * np.sin(2 * np.pi * 0.3 * t_100) + rng.normal(0, 8.0, len(t_100))

    # 4. EMG Chin (100 Hz): Muscle Tone
    emg_chin = rng.normal(0, 12.0, len(t_100))

    # 5. ECG (100 Hz): QRS Synthetic Spike Stream
    ecg = 10.0 * np.sin(2 * np.pi * 1.2 * t_100)  # ~72 bpm
    # Add synthetic QRS sharp spikes
    beat_indices = np.where(np.diff((t_100 * 1.2) % 1.0) < 0)[0]
    for idx in beat_indices:
        if idx + 2 < len(ecg):
            ecg[idx:idx+3] += np.array([20.0, 120.0, -30.0])

    # 6. Airflow (10 Hz): Respiratory Wave (0.25 Hz -> 15 breaths/min)
    airflow = 50.0 * np.sin(2 * np.pi * 0.25 * t_10) + rng.normal(0, 2.0, len(t_10))

    # 7. Chest Effort (10 Hz)
    chest = 45.0 * np.sin(2 * np.pi * 0.25 * t_10 + 0.1) + rng.normal(0, 1.5, len(t_10))

    # 8. SpO2 (1 Hz): Continuous Oximetry
    spo2 = 97.0 - 1.5 * np.sin(2 * np.pi * 0.01 * t_1) + rng.normal(0, 0.2, len(t_1))
    spo2 = np.clip(spo2, 70.0, 100.0)

    # Build EdfSignal objects with standard units and sampling rates
    signals = [
        EdfSignal(eeg_c3, sampling_frequency=100, label="EEG C3-M2", physical_dimension="uV"),
        EdfSignal(eeg_c4, sampling_frequency=100, label="EEG C4-M1", physical_dimension="uV"),
        EdfSignal(eog_l, sampling_frequency=100, label="EOG L-M2", physical_dimension="uV"),
        EdfSignal(emg_chin, sampling_frequency=100, label="EMG Chin", physical_dimension="uV"),
        EdfSignal(ecg, sampling_frequency=100, label="ECG", physical_dimension="uV"),
        EdfSignal(airflow, sampling_frequency=10, label="Airflow", physical_dimension="uV"),
        EdfSignal(chest, sampling_frequency=10, label="Chest", physical_dimension="uV"),
        EdfSignal(spo2, sampling_frequency=1, label="SpO2", physical_dimension="%"),
    ]
    return signals


def create_synthetic_psg_dataset(
    output_dir: Path | str = SCRIPT_DIR / "synthetic_dataset",
    n_patients: int = 150
) -> Path:
    """
    Creates a full synthetic dataset directory structure matching PhysioNet 2026 specification:
      - demographics.csv
      - physiological_data/{site_id}/{patient_id}_ses-01.edf
      - algorithmic_annotations/{site_id}/{patient_id}_ses-01_caisr_annotations.edf
    Generates n_patients across 3 distinct sites (site_01, site_02, site_03) with realistic age distribution (20-90).
    """
    output_path = Path(output_dir)
    sites = ["site_01", "site_02", "site_03"]

    for site in sites:
        os.makedirs(output_path / "physiological_data" / site, exist_ok=True)
        os.makedirs(output_path / "algorithmic_annotations" / site, exist_ok=True)

    demo_rows = []
    rng = np.random.default_rng(2026)

    for i in range(1, n_patients + 1):
        site_id = sites[(i - 1) % len(sites)]
        patient_id = f"test_pat_{i:03d}"
        session_id = "01"

        phys_dir = output_path / "physiological_data" / site_id
        algo_dir = output_path / "algorithmic_annotations" / site_id

        # 1. Create EDF signals
        signals = generate_synthetic_psg_signals(duration_seconds=30)
        edf = Edf(signals)

        phys_edf_path = phys_dir / f"{patient_id}_ses-{session_id}.edf"
        phys_edf_path_alt = phys_dir / f"{patient_id}_ses-1.edf"
        edf.write(phys_edf_path)
        edf.write(phys_edf_path_alt)

        # Also write a dummy algorithmic annotation edf file
        algo_edf_path = algo_dir / f"{patient_id}_ses-{session_id}_caisr_annotations.edf"
        algo_edf_path_alt = algo_dir / f"{patient_id}_ses-1_caisr_annotations.edf"
        edf.write(algo_edf_path)
        edf.write(algo_edf_path_alt)

        # Realistic age distribution between 20 and 90 years
        age = float(20 + (i % 70))
        # Label correlated with age + noise to test real ML metrics
        prob_impairment = 1.0 / (1.0 + np.exp(-((age - 60.0) / 10.0)))
        label = 1 if rng.uniform(0, 1) < prob_impairment else 0

        demo_rows.append({
            "SiteID": site_id,
            "BDSPPatientID": patient_id,
            "BidsFolder": patient_id,
            "SessionID": session_id,
            "Age": age,
            "Sex": "Male" if i % 2 == 0 else "Female",
            "Race": ["White", "Black", "Asian", "Hispanic"][i % 4],
            "Ethnicity": "Non-Hispanic" if i % 3 != 0 else "Hispanic",
            "BMI": 20.0 + (i % 15) * 1.2,
            "Cognitive_Impairment": label
        })

    demo_df = pd.DataFrame(demo_rows)
    demo_df.to_csv(output_path / "demographics.csv", index=False)

    print(f"[OK] Successfully created synthetic PSG dataset at: {output_path}")
    print(f"   Generated {n_patients} patients across 3 sites with EDF binary waveforms and demographics.csv.")
    return output_path


if __name__ == "__main__":
    create_synthetic_psg_dataset()
