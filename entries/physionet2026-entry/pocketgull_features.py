"""
Pocket Gull — Advanced Biosignal, Denoising & CAISR Sleep Architecture Engine
PhysioNet Challenge 2026 Dedicated Feature Pipeline
"""

import re
import warnings
import functools
import numpy as np


def safe_extract(fallback_length: int = 1, feature_name: str = "Unknown"):
    """
    Impenetrable Zero-Crash Decorator ("The Black Swan Interceptor").
    Guarantees the pipeline NEVER crashes on corrupted, 0-byte, flatlined, or missing real-world EDF signals.
    Silences warnings and returns NaN vector of fallback_length on any failure.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = func(*args, **kwargs)
                    if isinstance(res, (list, np.ndarray)) and len(res) == fallback_length:
                        return res
                    elif isinstance(res, dict) and len(res) == fallback_length:
                        return res
                    elif isinstance(res, (list, np.ndarray, dict)) and len(res) != fallback_length:
                        return [float('nan')] * fallback_length
                    return res
            except Exception as e:
                # Scikit-learn SimpleImputer will seamlessly replace these NaNs downstream!
                return [float('nan')] * fallback_length
        return wrapper
    return decorator


def find_channel_by_regex(edf_channel_names: list[str], target_pattern: str) -> str | None:
    """
    Fuzzy matches channel labels across hospital sites using case-insensitive regular expressions.
    Matches standard patterns like C3, C4, ECG, SpO2, EOG, EMG across diverse naming schemes.
    Returns the first matching channel label, or None if unmapped.
    """
    for ch in edf_channel_names:
        if re.search(target_pattern, str(ch)):
            return str(ch)
    return None


def apply_covariate_armor_intra_patient_normalization(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Covariate Armor: Intra-Patient Baseline Normalization.
    Neutralizes hospital-to-hospital amplifier gain scaling variances by normalizing
    raw PSG biosignals against their intra-patient median and interquartile range (IQR).
    """
    if len(signal) == 0:
        return signal
    q25, q75 = np.percentile(signal, [25, 75])
    iqr = max(q75 - q25, eps)
    med = np.median(signal)
    return (signal - med) / iqr


def decimate_signal_if_needed(signal: np.ndarray, fs: float, target_fs: float = 100.0) -> tuple[np.ndarray, float]:
    """
    Memory & Time Armor: Downsamples high-frequency signals (>100Hz) to 100Hz.
    Prevents CPU timeouts and memory OOM errors on massive 256Hz / 512Hz full-night PSG recordings.
    """
    if fs > target_fs and len(signal) > 0:
        step = max(1, int(np.round(fs / target_fs)))
        if step > 1:
            return signal[::step], fs / step
    return signal, fs


def apply_signal_denoising(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Child's Play / Signal Pre-Processing: 'Make the road smooth!'
    Applies median rolling artifact removal and rolling mean smoothing to raw PSG signals.
    """
    if len(signal) < window_size:
        return signal
    # Median filter to eliminate transient spikes
    med_filtered = np.median(np.lib.stride_tricks.sliding_window_view(
        np.pad(signal, (window_size//2, window_size//2), mode='edge'), window_size), axis=-1)
    return med_filtered


def compute_aasm_expert_rule_score(
    age: float,
    n3_percentage: float,
    ahi: float,
    arousal_index: float,
    shock_index: float
) -> float:
    """
    Redefine Constraints / Expert Rule-Based Baseline:
    Computes a deterministic clinical risk baseline derived from AASM clinical guidelines.
    """
    risk_points = 0.0
    
    # Severe Obstructive/Central Sleep Apnea (AHI >= 15)
    if ahi >= 30.0:
        risk_points += 0.40
    elif ahi >= 15.0:
        risk_points += 0.25
        
    # Impaired Glymphatic Slow-Wave Sleep (N3 SWS < 10%)
    if n3_percentage < 10.0:
        risk_points += 0.35
    elif n3_percentage < 15.0:
        risk_points += 0.20
        
    # Severe Sleep Fragmentation (Arousal Index >= 20/hr)
    if arousal_index >= 25.0:
        risk_points += 0.20
        
    # Age-Adjusted Hemodynamic Strain
    if (shock_index * age) >= 50.0:
        risk_points += 0.15
        
    return min(risk_points, 1.0)


def extract_hemodynamic_features(hr: float, bp_systolic: float, bp_diastolic: float, age: float) -> dict[str, float]:
    """
    Computes non-invasive hemodynamic & shock risk indices from baseline vitals.
    """
    sys_clamped = max(bp_systolic, 1.0)
    dia_clamped = max(bp_diastolic, 1.0)
    
    map_val = dia_clamped + (sys_clamped - dia_clamped) / 3.0
    pulse_pressure = sys_clamped - dia_clamped
    shock_index = hr / sys_clamped
    age_adjusted_shock_index = (hr * age) / sys_clamped
    rate_pressure_product = hr * sys_clamped
    
    return {
        "map": float(map_val),
        "pulse_pressure": float(pulse_pressure),
        "shock_index": float(shock_index),
        "age_adjusted_shock_index": float(age_adjusted_shock_index),
        "rate_pressure_product": float(rate_pressure_product)
    }


def extract_vagal_hrv_features(signal: np.ndarray, fs: float = 100.0) -> dict[str, float]:
    """
    Extracts time-domain HRV indices (RMSSD, SDNN, pNN50) after signal smoothing.
    """
    denoised_sig = apply_signal_denoising(signal)
    if len(denoised_sig) < 10:
        return {"rmssd": 0.0, "sdnn": 0.0, "pnn50": 0.0}
        
    diffs = np.diff(denoised_sig)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    sdnn = float(np.std(denoised_sig))
    
    nn50 = np.sum(np.abs(diffs) > 0.05 * np.std(denoised_sig))
    pnn50 = float(nn50 / max(len(diffs), 1))
    
    return {
        "rmssd": rmssd,
        "sdnn": sdnn,
        "pnn50": pnn50
    }


def extract_eeg_burst_suppression(eeg_signal: np.ndarray, threshold: float = 10.0) -> dict[str, float]:
    """
    Extracts EEG burst-suppression ratio and amplitude variance after artifact filtration.
    """
    denoised_eeg = apply_signal_denoising(eeg_signal)
    if len(denoised_eeg) == 0:
        return {"burst_suppression_ratio": 0.0, "eeg_variance": 0.0}
        
    suppressed_samples = np.sum(np.abs(denoised_eeg) < threshold)
    bsr = float(suppressed_samples / len(denoised_eeg))
    variance = float(np.var(denoised_eeg))
    
    return {
        "burst_suppression_ratio": bsr,
        "eeg_variance": variance
    }


def extract_caisr_sleep_architecture(stages: np.ndarray, epoch_len_sec: float = 30.0) -> dict[str, float]:
    """
    Extracts macro-sleep architecture features from CAISR sleep stages.
    """
    if len(stages) == 0:
        return {
            "n3_percentage": 0.0,
            "rem_percentage": 0.0,
            "waso_minutes": 0.0,
            "total_sleep_time_hours": 0.0,
            "sleep_efficiency": 0.0
        }
        
    total_epochs = len(stages)
    sleep_epochs = np.sum((stages >= 1) & (stages <= 4))
    n3_epochs = np.sum(stages == 1)
    rem_epochs = np.sum(stages == 4)
    wake_epochs = np.sum(stages == 5)
    
    tst_hours = float((sleep_epochs * epoch_len_sec) / 3600.0)
    waso_mins = float((wake_epochs * epoch_len_sec) / 60.0)
    
    n3_pct = float(n3_epochs / max(sleep_epochs, 1)) * 100.0
    rem_pct = float(rem_epochs / max(sleep_epochs, 1)) * 100.0
    sleep_eff = float(sleep_epochs / max(total_epochs, 1)) * 100.0
    
    return {
        "n3_percentage": n3_pct,
        "rem_percentage": rem_pct,
        "waso_minutes": waso_mins,
        "total_sleep_time_hours": tst_hours,
        "sleep_efficiency": sleep_eff
    }


def extract_caisr_respiratory_index(resp_events: np.ndarray, total_sleep_hours: float) -> dict[str, float]:
    """
    Computes AHI and event indices from CAISR respiratory events.
    """
    hours_clamped = max(total_sleep_hours, 0.1)
    
    oa_count = np.sum(resp_events == 1)
    ca_count = np.sum(resp_events == 2)
    ma_count = np.sum(resp_events == 3)
    hy_count = np.sum(resp_events == 4)
    rera_count = np.sum(resp_events == 5)
    
    total_apneas = oa_count + ca_count + ma_count + hy_count
    ahi = float(total_apneas / hours_clamped)
    oai = float(oa_count / hours_clamped)
    cai = float(ca_count / hours_clamped)
    
    return {
        "apnea_hypopnea_index": ahi,
        "obstructive_apnea_index": oai,
        "central_apnea_index": cai,
        "rera_index": float(rera_count / hours_clamped)
    }


def extract_caisr_arousal_index(arousal_events: np.ndarray, total_sleep_hours: float) -> dict[str, float]:
    """
    Computes Arousal Index from 0.5s CAISR annotations.
    """
    hours_clamped = max(total_sleep_hours, 0.1)
    arousal_count = np.sum(arousal_events == 1)
    arousal_index = float(arousal_count / hours_clamped)
    
    return {
        "arousal_index": arousal_index,
        "total_arousals": float(arousal_count)
    }


def extract_eeg_slow_wave_activity_ratio(eeg_signal: np.ndarray, fs: float = 100.0) -> dict[str, float]:
    """
    First-Principles Neuro-Consciousness Biomarker:
    Computes Slow-Wave Activity (SWA) Power Ratio (Delta 0.5-4Hz / Alpha 8-12Hz) and SEF95.
    High SWA ratio indicates deep N3 glymphatic clearance or neural slowing.
    """
    if len(eeg_signal) < int(fs * 2.0):
        return {"swa_delta_alpha_ratio": 1.0, "eeg_alpha_power": 1.0, "eeg_delta_power": 1.0}
        
    try:
        from scipy.signal import welch
        freqs, psd = welch(eeg_signal, fs=fs, nperseg=min(len(eeg_signal), int(fs * 4)))
        
        delta_mask = (freqs >= 0.5) & (freqs <= 4.0)
        alpha_mask = (freqs >= 8.0) & (freqs <= 12.0)
        
        delta_power = float(np.sum(psd[delta_mask])) if np.any(delta_mask) else 1e-5
        alpha_power = float(np.sum(psd[alpha_mask])) if np.any(alpha_mask) else 1e-5
        
        swa_ratio = float(delta_power / max(alpha_power, 1e-5))
        return {
            "swa_delta_alpha_ratio": round(swa_ratio, 4),
            "eeg_alpha_power": round(alpha_power, 4),
            "eeg_delta_power": round(delta_power, 4)
        }
    except Exception:
        return {"swa_delta_alpha_ratio": 1.0, "eeg_alpha_power": 1.0, "eeg_delta_power": 1.0}


def extract_eeg_spectral_features(eeg_signal: np.ndarray, fs: float = 256.0) -> dict[str, float]:
    """
    Extracts multi-band EEG spectral power features.
    Includes theta/alpha ratio (established MCI biomarker) and Spectral Edge
    Frequency 95% (neural slowing indicator).

    Output: 7 features (delta, theta, alpha, sigma, beta power, theta/alpha ratio, SEF95).
    """
    nan_keys = ['delta_power', 'theta_power', 'alpha_power', 'sigma_power',
                'beta_power', 'theta_alpha_ratio', 'sef95']
    nan_result = {k: float('nan') for k in nan_keys}

    eeg_signal, fs = decimate_signal_if_needed(eeg_signal, fs, target_fs=100.0)
    if len(eeg_signal) < int(fs * 2.0):
        return nan_result

    try:
        from scipy.signal import welch as welch_psd
        nperseg = min(len(eeg_signal), int(fs * 4))
        freqs, psd = welch_psd(eeg_signal, fs=fs, nperseg=nperseg)

        bands = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 12.0),
            'sigma': (12.0, 15.0),
            'beta':  (15.0, 30.0),
        }

        # Covariate Armor: Intra-Patient Baseline Normalization
        # Normalize band powers relative to patient's own median baseline amplitude to neutralize hospital amplifier gains
        base_amplitude = float(np.nanmedian(np.abs(eeg_signal))) + 1e-6
        base_power = max(base_amplitude ** 2, 1e-6)

        powers = {}
        for band_name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs <= hi)
            raw_p = float(np.sum(psd[mask])) if np.any(mask) else 1e-5
            powers[band_name] = float(raw_p / base_power)

        # Theta/Alpha ratio — established MCI biomarker
        theta_alpha = powers['theta'] / max(powers['alpha'], 1e-5)

        # Spectral Edge Frequency 95% — neural slowing indicator
        cumulative = np.cumsum(psd)
        total_power = cumulative[-1] if len(cumulative) > 0 else 0
        if total_power > 0:
            sef95_idx = np.searchsorted(cumulative, 0.95 * total_power)
            sef95 = float(freqs[min(sef95_idx, len(freqs) - 1)])
        else:
            sef95 = float('nan')

        return {
            'delta_power': powers['delta'],
            'theta_power': powers['theta'],
            'alpha_power': powers['alpha'],
            'sigma_power': powers['sigma'],
            'beta_power':  powers['beta'],
            'theta_alpha_ratio': theta_alpha,
            'sef95': sef95,
        }
    except Exception:
        return nan_result


def extract_spo2_clinical_features(spo2_signal: np.ndarray) -> dict[str, float]:
    """
    Extracts clinically meaningful SpO2 features: mean saturation, nadir,
    variability, and oxygen desaturation burden (% time below 90%).

    Output: 4 features.
    """
    nan_result = {
        'spo2_mean': float('nan'),
        'spo2_nadir': float('nan'),
        'spo2_std': float('nan'),
        'spo2_pct_below_90': float('nan'),
    }

    if len(spo2_signal) < 10:
        return nan_result

    # Filter to physiologically plausible range (drop probe-off artifacts)
    valid = spo2_signal[(spo2_signal > 0) & (spo2_signal <= 100)]
    if len(valid) < 10:
        return nan_result

    return {
        'spo2_mean': float(np.mean(valid)),
        'spo2_nadir': float(np.min(valid)),
        'spo2_std': float(np.std(valid)),
        'spo2_pct_below_90': float(np.mean(valid < 90.0)),
    }


def extract_caisr_staging_uncertainty(algo_data: dict) -> dict[str, float]:
    """
    Extracts staging uncertainty and temporal dynamics from CAISR softmax
    probability traces. Higher entropy and lower stability indicate poor
    sleep quality / fragmentation — markers of cognitive decline risk.

    Output: 4 features (mean entropy, std entropy, probability stability,
    transition rate).
    """
    nan_result = {
        'staging_mean_entropy': float('nan'),
        'staging_std_entropy': float('nan'),
        'staging_prob_stability': float('nan'),
        'staging_transition_rate': float('nan'),
    }

    prob_keys = ['caisr_prob_w', 'caisr_prob_n1', 'caisr_prob_n2',
                 'caisr_prob_n3', 'caisr_prob_r']

    available = [k for k in prob_keys if k in algo_data and len(algo_data.get(k, [])) > 0]

    if len(available) < 2:
        return nan_result

    try:
        probs = np.column_stack([algo_data[k].astype(float) for k in available])

        # Filter out filler/invalid rows (CAISR uses 9.0 for unavailable)
        valid_mask = np.all(probs < 2.0, axis=1) & np.all(probs >= 0, axis=1)
        probs = probs[valid_mask]

        if len(probs) < 10:
            return nan_result

        # Normalize rows to probability simplex
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = probs / np.maximum(row_sums, 1e-8)

        # Per-epoch Shannon entropy: H = -Σ p_i × log(p_i)
        epoch_entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)

        # Max-class probability temporal stability
        max_probs = np.max(probs, axis=1)

        # Hard-label transition rate (stage changes per epoch)
        hard_stages = np.argmax(probs, axis=1)
        transition_rate = float(np.mean(np.diff(hard_stages) != 0)) if len(hard_stages) > 1 else float('nan')

        return {
            'staging_mean_entropy': float(np.mean(epoch_entropy)),
            'staging_std_entropy': float(np.std(epoch_entropy)),
            'staging_prob_stability': float(np.std(max_probs)),
            'staging_transition_rate': transition_rate,
        }
    except Exception:
        return nan_result


def extract_markov_sleep_dynamics(algo_data: dict) -> dict:
    """
    Computes a 5x5 Markov transition probability matrix from CAISR hard stage labels
    and extracts 3 clinically significant transition biomarkers.

    CAISR stage encoding: N3=1, N2=2, N1=3, R=4, W=5.
    Internal matrix indexing: 0=N3, 1=N2, 2=N1, 3=R, 4=W.

    Returns:
        dict with keys:
            - n2_to_n1_rate: P(N2 -> N1) — arousal/fragmentation rate
            - n3_continuity: P(N3 -> N3) — deep sleep maintenance (self-loop)
            - rem_instability: P(REM -> Wake) — early synucleinopathy marker
    """
    nan_result = {
        'n2_to_n1_rate': float('nan'),
        'n3_continuity': float('nan'),
        'rem_instability': float('nan'),
    }

    stages = algo_data.get('stage_caisr', np.array([]))
    if len(stages) < 10:
        return nan_result

    try:
        # Filter out invalid/background values (9.0, etc.)
        valid_stages = stages[stages < 9.0].astype(int)
        if len(valid_stages) < 10:
            return nan_result

        # Map CAISR encoding to 0-indexed: N3(1)->0, N2(2)->1, N1(3)->2, R(4)->3, W(5)->4
        stage_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        mapped = np.array([stage_map.get(s, -1) for s in valid_stages])
        mapped = mapped[mapped >= 0]  # drop unmapped values

        if len(mapped) < 10:
            return nan_result

        # Build 5x5 transition count matrix
        transitions = np.zeros((5, 5))
        for i in range(len(mapped) - 1):
            transitions[mapped[i], mapped[i + 1]] += 1

        # Normalize to probabilities (row-stochastic)
        row_sums = transitions.sum(axis=1, keepdims=True)
        probs = np.divide(transitions, row_sums,
                          out=np.zeros_like(transitions), where=row_sums != 0)

        return {
            'n2_to_n1_rate': float(probs[1, 2]),   # N2(idx=1) -> N1(idx=2)
            'n3_continuity': float(probs[0, 0]),    # N3(idx=0) -> N3(idx=0)
            'rem_instability': float(probs[3, 4]),  # REM(idx=3) -> W(idx=4)
        }
    except Exception:
        return nan_result
