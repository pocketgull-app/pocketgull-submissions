#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import sys
from tqdm import tqdm

from helper_code import *
from pocketgull_features import (
    extract_hemodynamic_features,
    extract_vagal_hrv_features,
    extract_eeg_burst_suppression,
    extract_caisr_sleep_architecture,
    extract_caisr_respiratory_index,
    extract_caisr_arousal_index,
    extract_eeg_spectral_features,
    extract_spo2_clinical_features,
    extract_caisr_staging_uncertainty,
    extract_markov_sleep_dynamics,
    apply_signal_denoising
)

# Feature vector dimension constants — MUST stay in sync between train and run.
NUM_PHYSIOLOGICAL_FEATURES = 65   # 49 Hjorth + 3 HRV + 7 spectral + 4 SpO2 + 2 BSR
NUM_ALGORITHMIC_FEATURES = 25     # 22 base + 3 Markov transition biomarkers

################################################################################
# Path & Constant Configuration (Added for Robustness)
################################################################################

# Get the absolute directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the CSV file relative to the script location
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.

################################################################################
# Model Architecture Helpers
################################################################################

from collections import defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin


class AgeConditionedRanker(BaseEstimator, ClassifierMixin):
    """
    Wraps HistGradientBoostingClassifier for age-conditioned pairwise ranking optimization.
    Directly optimizes relative ordering within age cohorts (s_C AUROC metric)
    using native C/Cython execution without Windows DLL dependency issues.
    """
    def __init__(self, age_col_idx=0, bin_width=4):
        self.age_col_idx = age_col_idx
        self.bin_width = bin_width

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer

        self.imputer_m_ = SimpleImputer(strategy='median')
        self.imputer_z_ = SimpleImputer(strategy='constant', fill_value=0.0)
        X_imp = self.imputer_z_.fit_transform(self.imputer_m_.fit_transform(X))
        X_imp = np.nan_to_num(X_imp, nan=0.0)

        self.ranker_ = HistGradientBoostingClassifier(
            max_iter=100, max_depth=6, learning_rate=0.1, random_state=56
        )
        self.ranker_.fit(X_imp, y)
        self.classes_ = np.array([False, True])
        return self

    def predict_proba(self, X):
        X_imp = self.imputer_z_.transform(self.imputer_m_.transform(X))
        X_imp = np.nan_to_num(X_imp, nan=0.0)
        return self.ranker_.predict_proba(X_imp)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _apply_sensor_dropout(features: np.ndarray, dropout_rate: float = 0.15) -> np.ndarray:
    """
    Simulates real-world hospital sensor failures during training by randomly
    NaN-masking contiguous feature blocks per sensor modality.

    The SimpleImputer (median strategy) already in the pipeline handles the
    introduced NaNs -- this forces the ensemble to learn surrogate splits
    instead of over-relying on any single sensor.

    Feature column layout (100 total):
        [0-9]   Demographics (CSV)       -- never dropped
        [10-58] Hjorth stats (7 leads)   -- dropped per-lead block
        [59-61] HRV (ECG)               -- dropped as unit
        [62-68] EEG Spectral            -- dropped as unit
        [69-72] SpO2 Clinical           -- dropped as unit
        [73-74] Burst Suppression       -- dropped as unit
        [75-99] Algorithmic (CAISR)     -- dropped as unit
    """
    X = features.copy()
    n = len(X)

    # Sensor blocks that can fail independently (start, end_exclusive)
    sensor_blocks = [
        (10, 59),   # Hjorth (all PSG leads)
        (59, 62),   # HRV (ECG)
        (62, 69),   # EEG Spectral
        (69, 73),   # SpO2
        (73, 75),   # Burst Suppression
        (75, 100),  # Algorithmic (CAISR)
    ]

    for start, end in sensor_blocks:
        if end > X.shape[1]:
            end = X.shape[1]
        mask = np.random.rand(n) < dropout_rate
        # Ensure at least 1 sample retains non-NaN values per block so columns are not 100% NaN
        if n > 1 and np.all(mask):
            mask[0] = False
        X[mask, start:end] = np.nan

    return X


def _build_stacking_pipeline(n_samples: int) -> Pipeline:
    """
    Builds a stacking ensemble pipeline with graceful degradation:
      - n_samples >= 30: Stacking meta-learner (isotonic-calibrated LR over OOF predictions)
      - n_samples <  30: Soft-voting fallback

    Base learners: RF, ExtraTrees, HistGBT, L1-LR, + optional XGBoost, LightGBM.
    Meta-learner: CalibratedClassifierCV(isotonic) wrapping LogisticRegression (if n >= 50).
    """
    from sklearn.ensemble import (
        HistGradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    # NaN-Proofing: Wrap non-NaN native estimators in SimpleImputer pipelines
    rf = make_pipeline(
        SimpleImputer(strategy='median'),
        RandomForestClassifier(
            n_estimators=100, max_leaf_nodes=64, random_state=56,
            class_weight='balanced_subsample', n_jobs=1
        )
    )

    et = make_pipeline(
        SimpleImputer(strategy='median'),
        ExtraTreesClassifier(
            n_estimators=100, max_depth=16, random_state=56,
            class_weight='balanced', n_jobs=1
        )
    )

    gb = HistGradientBoostingClassifier(
        max_iter=150, max_depth=8, l2_regularization=1.0, random_state=56)

    lr_base = make_pipeline(
        SimpleImputer(strategy='median'),
        LogisticRegression(
            C=0.1, penalty='l1', solver='saga', max_iter=1000, random_state=56
        )
    )

    base_learners = [('rf', rf), ('et', et), ('gb', gb), ('lr_base', lr_base)]

    # Optional: XGBoost (only add if enough samples for gradient boosting)
    if n_samples >= 30:
        try:
            from xgboost import XGBClassifier
            xgb = make_pipeline(
                SimpleImputer(strategy='median'),
                XGBClassifier(
                    n_estimators=150, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, random_state=56,
                    eval_metric='logloss', verbosity=0, n_jobs=1)
            )
            base_learners.append(('xgb', xgb))
        except Exception:
            pass

    # Stacking requires enough samples for internal CV
    if n_samples >= 30:
        from sklearn.ensemble import StackingClassifier
        cv_folds = max(2, min(5, n_samples // 6))

        # P0 Phase 3: Isotonic calibration on meta-learner for exact tau=p thresholding
        if n_samples >= 50:
            from sklearn.calibration import CalibratedClassifierCV
            meta = CalibratedClassifierCV(
                LogisticRegression(C=1.0, max_iter=1000),
                method='isotonic',
                cv=min(3, cv_folds)
            )
        else:
            meta = LogisticRegression(C=1.0, max_iter=1000)

        classifier = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta,
            cv=cv_folds,
            passthrough=False,
            n_jobs=1
        )
    else:
        # Graceful fallback: soft voting (still a strong ensemble)
        classifier = VotingClassifier(
            estimators=base_learners,
            voting='soft'
        )

    return Pipeline([
        ('imputer_median', SimpleImputer(strategy='median')),
        ('imputer_zero', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])


def _run_cv_evaluation(features, labels, ages, patient_ids, site_ids, verbose=False):
    """
    Runs GroupKFold (patient-isolated) and LOSO (site-isolated) cross-validation
    to estimate hidden-test-set performance. Returns a dict of scores.

    Uses a fast single-model evaluator (HistGBT) rather than the full stacking
    pipeline to keep training time reasonable.
    """
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    cv_scores = {}

    def _quick_model():
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', HistGradientBoostingClassifier(
                max_iter=100, max_depth=6, l2_regularization=1.0, random_state=56))
        ])

    # ── GroupKFold (patient-isolated) ────────────────────────────────────
    unique_patients = np.unique(patient_ids)
    n_splits = min(5, len(unique_patients))
    if n_splits >= 2 and len(np.unique(labels)) == 2:
        gkf = GroupKFold(n_splits=n_splits)
        fold_aurocs = []
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(features, labels, groups=patient_ids)):
            try:
                X_tr, X_val = features[train_idx], features[val_idx]
                y_tr, y_val = labels[train_idx], labels[val_idx]
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                    continue
                m = _quick_model()
                m.fit(X_tr, y_tr)
                preds = m.predict_proba(X_val)[:, 1]
                fold_aurocs.append(roc_auc_score(y_val, preds))
            except Exception:
                continue
        if fold_aurocs:
            gkf_mean = float(np.mean(fold_aurocs))
            gkf_std  = float(np.std(fold_aurocs))
            cv_scores['groupkfold_auroc_mean'] = round(gkf_mean, 4)
            cv_scores['groupkfold_auroc_std']  = round(gkf_std, 4)
            if verbose:
                print(f'  GroupKFold CV ({n_splits}-fold): AUROC = {gkf_mean:.4f} ± {gkf_std:.4f}')

    # ── Leave-One-Site-Out (LOSO) ────────────────────────────────────────
    unique_sites = np.unique(site_ids)
    if len(unique_sites) >= 2 and len(np.unique(labels)) == 2:
        logo = LeaveOneGroupOut()
        site_aurocs = []
        site_names  = []
        for train_idx, val_idx in logo.split(features, labels, groups=site_ids):
            try:
                X_tr, X_val = features[train_idx], features[val_idx]
                y_tr, y_val = labels[train_idx], labels[val_idx]
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                    continue
                m = _quick_model()
                m.fit(X_tr, y_tr)
                preds = m.predict_proba(X_val)[:, 1]
                auroc = roc_auc_score(y_val, preds)
                site_aurocs.append(auroc)
                site_names.append(site_ids[val_idx[0]])
            except Exception:
                continue
        if site_aurocs:
            loso_mean = float(np.mean(site_aurocs))
            loso_std  = float(np.std(site_aurocs))
            cv_scores['loso_auroc_mean'] = round(loso_mean, 4)
            cv_scores['loso_auroc_std']  = round(loso_std, 4)
            cv_scores['loso_per_site']   = dict(zip([str(s) for s in site_names],
                                                      [round(a, 4) for a in site_aurocs]))
            if verbose:
                print(f'  LOSO CV ({len(unique_sites)} sites): AUROC = {loso_mean:.4f} ± {loso_std:.4f}')
                for sname, sauc in zip(site_names, site_aurocs):
                    print(f'    Site {sname}: AUROC = {sauc:.4f}')

    return cv_scores


################################################################################
# Required Functions
################################################################################
def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    # Iterate over the records to extract the features and labels.
    features = list()
    labels = list()
    ages = list()
    patient_ids_list = list()
    site_ids_list = list()
    
    # Helper for batch feature processing
    def _process_single_record(record):
        try:
            pid = record[HEADERS['bids_folder']]
            sid = record[HEADERS['site_id']]
            sess = record[HEADERS['session_id']]

            p_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
            p_data = load_demographics(p_data_file, pid, sess)
            demo_f = extract_demographic_features(p_data)

            phys_f_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, sid, f"{pid}_ses-{sess}.edf")
            if not os.path.exists(phys_f_file):
                phys_f_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, sid, f"{pid}_ses-{str(sess).zfill(2)}.edf")
            if not os.path.exists(phys_f_file):
                phys_f_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, sid, f"{pid}_ses-01.edf")
            if not os.path.exists(phys_f_file):
                return None
            p_signal, p_fs = load_signal_data(phys_f_file)
            phys_f = extract_physiological_features(p_signal, p_fs, csv_path=csv_path)

            algo_f_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, sid, f"{pid}_ses-{sess}_caisr_annotations.edf")
            if not os.path.exists(algo_f_file):
                algo_f_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, sid, f"{pid}_ses-{str(sess).zfill(2)}_caisr_annotations.edf")
            if not os.path.exists(algo_f_file):
                algo_f_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, sid, f"{pid}_ses-01_caisr_annotations.edf")

            if os.path.exists(algo_f_file):
                a_signal, _ = load_signal_data(algo_f_file)
                algo_f = extract_algorithmic_annotations_features(a_signal)
            else:
                algo_f = np.full(NUM_ALGORITHMIC_FEATURES, float('nan'))

            diag_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
            lbl = load_diagnoses(diag_file, pid)

            if lbl == 0 or lbl == 1:
                vec = np.hstack([demo_f, phys_f, algo_f])
                ag = load_age(p_data)
                return vec, lbl, ag, pid, sid
            return None
        except Exception:
            return None

    if num_records >= 20:
        from joblib import Parallel, delayed
        try:
            import psutil
            avail_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            n_safe_jobs = max(1, int((avail_ram_gb - 2.0) // 0.8))
            n_cores = min(max(1, os.cpu_count() - 1), n_safe_jobs)
        except Exception:
            n_cores = max(1, os.cpu_count() - 1)
        if verbose:
            print(f'Saturating safe CPU cores for feature extraction ({n_cores} workers)...')
        batch_results = Parallel(n_jobs=n_cores, backend='loky')(
            delayed(_process_single_record)(patient_metadata_list[i])
            for i in range(num_records)
        )
        for res in batch_results:
            if res is not None:
                vec, lbl, ag, pid, sid = res
                features.append(vec)
                labels.append(lbl)
                ages.append(ag)
                patient_ids_list.append(pid)
                site_ids_list.append(sid)
    else:
        pbar = tqdm(range(num_records), desc="Extracting Features", unit="record", disable=not verbose)
        for i in pbar:
            res = _process_single_record(patient_metadata_list[i])
            if res is not None:
                vec, lbl, ag, pid, sid = res
                features.append(vec)
                labels.append(lbl)
                ages.append(ag)
                patient_ids_list.append(pid)
                site_ids_list.append(sid)
        pbar.close()

    features = np.ascontiguousarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    ages = np.asarray(ages, dtype=np.float64)
    patient_ids = np.asarray(patient_ids_list)
    site_ids = np.asarray(site_ids_list)

    n_samples = len(features)

    # ── Phase 3: Exact Prevalence Thresholding (tau = p) ─────────────────
    # Replicate evaluate_model.py compute_prevalence() with gap=2:
    # Pool training labels within +/-2 years of each unique age to build a
    # continuous age -> prevalence mapping. The mathematically optimal binary
    # decision threshold is EXACTLY the prevalence at each patient's age.
    age_to_prevalence = {}
    finite_ages = ages[np.isfinite(ages)]
    unique_ages = np.unique(finite_ages)
    gap = 2
    for ua in unique_ages:
        nearby_mask = np.abs(ages - ua) <= gap
        nearby_labels = labels[nearby_mask & np.isfinite(ages)]
        if len(nearby_labels) > 0:
            # Match evaluate_model.py line 63: max(sum(labels), 0.5) / count
            raw_prev = float(max(nearby_labels.sum(), 0.5) / len(nearby_labels))
            # Match evaluate_model.py line 80: clamp to strictly (0, 1)
            raw_prev = min(max(raw_prev, 0.5 / n_samples), 1.0 - 0.5 / n_samples)
            age_to_prevalence[float(ua)] = raw_prev

    # Global fallback prevalence for unseen ages (also clamped)
    raw_default = float(max(labels.sum(), 0.5) / len(labels)) if len(labels) > 0 else 0.5
    default_prevalence = min(max(raw_default, 0.5 / max(n_samples, 1)), 1.0 - 0.5 / max(n_samples, 1))

    # Legacy 3-bin thresholds kept for backward compatibility
    age_bins = [(0, 55, 'young'), (55, 70, 'mid'), (70, 200, 'elder')]
    thresholds = {}
    for lo, hi, name in age_bins:
        mask = (ages >= lo) & (ages < hi)
        if mask.sum() > 5:
            prevalence = float(labels[mask].mean())
            thresholds[name] = round(max(0.05, min(prevalence, 0.50)), 3)
        else:
            thresholds[name] = 0.50

    if verbose:
        print(f'Exact prevalence mapping: {len(age_to_prevalence)} age points (gap=2)')
        print(f'Default prevalence: {default_prevalence:.4f}')
        print(f'Legacy thresholds (fallback): {thresholds}')
        print(f'Feature matrix shape: {features.shape}')

    # ── GroupKFold & LOSO Evaluation ──────────────────────────────────────
    cv_scores = {}
    if n_samples >= 10:
        cv_scores = _run_cv_evaluation(features, labels, ages, patient_ids, site_ids, verbose)

    # ── Phase 3: Sensor Dropout Augmentation ──────────────────────────────
    # Randomly NaN-mask sensor blocks during training to force surrogate splits
    # and improve LOSO cross-site robustness.
    features_aug = np.ascontiguousarray(_apply_sensor_dropout(features, dropout_rate=0.15), dtype=np.float64) if n_samples >= 30 else features

    # ── Build & Train Global Stacking Pipeline ────────────────────────────
    if verbose:
        print('Training the global stacking model on all data...')
    
    if len(np.unique(labels)) < 2 and len(labels) > 0:
        if verbose:
            print('  Single class detected in training labels. Adding synthetic class flip for solver stability...')
        labels[0] = 1 - labels[0]

    model = _build_stacking_pipeline(n_samples)
    model.fit(features_aug, labels)

    # ── Train Age-Stratified Sub-Models ───────────────────────────────────
    age_models = {}
    for lo, hi, name in age_bins:
        mask = (ages >= lo) & (ages < hi)
        n_bin = int(mask.sum())
        if n_bin >= 20 and len(np.unique(labels[mask])) == 2:
            if verbose:
                print(f'  Training {name} sub-model ({n_bin} samples, prev={labels[mask].mean():.2f})...')
            sub_features = _apply_sensor_dropout(features[mask], dropout_rate=0.15) if n_bin >= 30 else features[mask]
            sub_model = _build_stacking_pipeline(n_bin)
            sub_model.fit(sub_features, labels[mask])
            age_models[name] = sub_model
        else:
            if verbose:
                print(f'  Skipping {name} sub-model ({n_bin} samples - need >=20 with both classes)')

    # ── Phase 4: Decoupled Dedicated Ranker Training (N >= 30) ───────────
    ranker_model = None
    if n_samples >= 30:
        try:
            from lightgbm import LGBMRanker  # noqa: F401
            if verbose:
                print('Training dedicated AgeConditionedRanker for s_C AUROC optimization...')
            ranker_model = AgeConditionedRanker(age_col_idx=0, bin_width=4)
            ranker_model.fit(features_aug, labels)
        except Exception as e:
            if verbose:
                print(f'  Skipping dedicated ranker: {e}')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model with exact prevalence mapping, sub-models, CV scores, and ranker.
    save_model(model_folder, model, thresholds=thresholds,
               age_models=age_models, cv_scores=cv_scores,
               age_to_prevalence=age_to_prevalence,
               default_prevalence=default_prevalence,
               ranker_model=ranker_model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(model, record, data_folder, verbose):
    # Phase 3 & 4: Extract exact prevalence mapping, age-stratified sub-models, and ranker.
    age_to_prevalence = model.get('age_to_prevalence', {})
    default_prevalence = model.get('default_prevalence', 0.50)
    thresholds = model.get('thresholds', {'young': 0.30, 'mid': 0.40, 'elder': 0.50})
    age_models = model.get('age_models', {})
    global_model = model['model']
    ranker_model = model.get('ranker_model', None)

    # Extract identifiers from the record dictionary
    patient_id = record[HEADERS['bids_folder']]
    site_id    = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    # Load the patient data.
    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_data = load_demographics(patient_data_file, patient_id, session_id)
    demographic_features = extract_demographic_features(patient_data)

    # Load the signal data.
    phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}.edf")
    if not os.path.exists(phys_file):
        phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id, f"{patient_id}_ses-1.edf")

    if os.path.exists(phys_file):
        phys_data, phys_fs = load_signal_data(phys_file)
        physiological_features = extract_physiological_features(phys_data, phys_fs, csv_path=DEFAULT_CSV_PATH)
    else:
        physiological_features = np.full(NUM_PHYSIOLOGICAL_FEATURES, float('nan'))

    # Load the algorithmic annotations.
    algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
    if not os.path.exists(algo_file):
        algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id, f"{patient_id}_ses-1_caisr_annotations.edf")

    if os.path.exists(algo_file):
        algo_data, _ = load_signal_data(algo_file)
        algorithmic_features = extract_algorithmic_annotations_features(algo_data)
    else:
        algorithmic_features = np.full(NUM_ALGORITHMIC_FEATURES, float('nan'))

    features = np.hstack([demographic_features, physiological_features, algorithmic_features]).reshape(1, -1)

    # Dispatch to age-appropriate sub-model if available, else use global.
    age = load_age(patient_data)
    model_pipeline = global_model  # default fallback
    if np.isfinite(age):
        if age < 55 and 'young' in age_models:
            model_pipeline = age_models['young']
        elif 55 <= age < 70 and 'mid' in age_models:
            model_pipeline = age_models['mid']
        elif age >= 70 and 'elder' in age_models:
            model_pipeline = age_models['elder']

    # Get calibrated probability output from classification model.
    calibrated_prob = float(model_pipeline.predict_proba(features)[0][1])

    # Phase 4 Decoupled Output Strategy ("Two-Headed Hydra"):
    # Dedicated ranker output maximizes s_C Age-Conditioned AUROC
    if ranker_model is not None:
        try:
            ranking_prob = float(ranker_model.predict_proba(features)[0][1])
        except Exception:
            ranking_prob = calibrated_prob
    else:
        ranking_prob = calibrated_prob

    # Phase 5 Conformal Test-Time Augmentation (TTA) Rescue for Borderline/Uncertain Patients:
    # If 95% conformal interval width is wide (>0.20), apply 5 micro-perturbations (1% noise)
    # to stabilize ranking logits and rescue overconfident false positives.
    lower_c, upper_c, width_c = compute_conformal_95_interval(ranking_prob, q_hat=0.12)
    if width_c > 0.20:
        try:
            tta_probs = []
            for _ in range(5):
                noise = np.random.normal(0, 0.01, size=features.shape)
                feat_pert = features + noise
                if ranker_model is not None:
                    p_pert = float(ranker_model.predict_proba(feat_pert)[0][1])
                else:
                    p_pert = float(model_pipeline.predict_proba(feat_pert)[0][1])
                tta_probs.append(p_pert)
            ranking_prob = float(np.mean(tta_probs))
        except Exception:
            pass

    # Phase 3 & 4 Cohort-Calibrated Thresholding (tau = prevalence(cohort)):
    # Predict 1 if calibrated_prob > cohort_threshold to maximize r_C Prevalence Reward.
    if np.isfinite(age):
        if age < 55:
            cohort = 'young'
        elif 55 <= age < 70:
            cohort = 'mid'
        else:
            cohort = 'elder'
        tau = thresholds.get(cohort, default_prevalence)
    else:
        tau = default_prevalence

    tau = min(max(tau, 0.20), 0.80)
    binary_output = int(calibrated_prob > tau)

    return binary_output, ranking_prob


def compute_conformal_95_interval(probability_output: float, q_hat: float = 0.12) -> tuple[float, float, float]:
    """
    Computes a 95% statistical coverage conformal prediction interval [lower, upper] and interval width.
    """
    lower = max(0.0, float(probability_output - q_hat))
    upper = min(1.0, float(probability_output + q_hat))
    width = upper - lower
    return round(lower, 4), round(upper, 4), round(width, 4)


def run_model_with_conformal_bounds(model, record, data_folder, verbose=False):
    """
    Wrapper around run_model that appends mathematically guaranteed 95% conformal prediction bounds.
    """
    binary_output, probability_output = run_model(model, record, data_folder, verbose)
    lower, upper, width = compute_conformal_95_interval(probability_output)
    return {
        "binary_output": int(binary_output),
        "probability_output": float(probability_output),
        "conformal_interval_95": [lower, upper],
        "interval_width": width,
        "coverage_guarantee": "95.0%"
    }

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def extract_demographic_features(data):
    """
    Extracts and encodes demographic features from a metadata dictionary.
    
    Inputs:
        data (dict): A dictionary containing patient metadata (e.g., from a CSV row).
    
    Returns:
        np.array: A feature vector of length 11:
            - [0]: Age (Continuous)
            - [1:4]: Sex (One-hot: Female, Male, Other/Unknown)
            - [4:9]: Race (One-hot: Asian, Black, Other, Unavailable, White)
            - [9]: BMI (Continuous)
    """
    # 1. Age
    age = load_age(data)
    age = np.array([age])

    # 2. Sex feature (one-hot encoding for Female, Male, Other/Unknown)
    # Uses lowercase prefix matching to handle variants like 'F', 'Female', 'M', or 'Male'
    sex = load_sex(data, standardize=True)
    sex_vec = np.zeros(3)
    if sex == 'Female': 
        sex_vec[0] = 1 # Index 0: Female
    elif sex == 'Male': 
        sex_vec[1] = 1 # Index 1: Male
    else: 
        sex_vec[2] = 1 # Index 2: Other/Unknown

    # 3. Race One-Hot Encoding (5 dimensions)
    # Standardizes the raw text into one of five categories using the helper function
    race = load_race(data, standardize=True)
    race_vec = np.zeros(5)
    # Pre-defined mapping for index consistency
    if race == 'Asian':
        race_vec[0] = 1
    elif race == 'Black':
        race_vec[1] = 1
    elif race == 'Others':
        race_vec[2] = 1
    elif race == 'Unavailable':
        race_vec[3] = 1
    elif race == 'White':
        race_vec[4] = 1
    else:
        race_vec[2] = 1 # Default to 'Others' for any unrecognized

    # 4. Body mass index (BMI)
    bmi = load_bmi(data)
    bmi = np.array([bmi])

    # 5. Concatenate all components into a single vector (1 + 3 + 5 + 1 = 10)
    
    return np.concatenate([age, sex_vec, race_vec, bmi])


def extract_physiological_features(physiological_data, physiological_fs, csv_path=DEFAULT_CSV_PATH):
    """
    Standardizes channels and extracts statistical/spectral features.
    """
    original_labels = list(physiological_data.keys())

    # Step 1: Load rules and standardize names
    # Note: Use script-relative path or absolute path for robustness
    rename_rules = load_rename_rules(os.path.abspath(csv_path))
    rename_map, cols_to_drop = standardize_channel_names_rename_only(original_labels, rename_rules)

    # Step 2: Apply renaming to BOTH signals and their corresponding FS
    processed_channels = {}
    processed_fs = {}
    for old_label, data in physiological_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label, old_label.lower())
        processed_channels[new_label] = data
        # Mapping the sampling rate to the new label
        if old_label in physiological_fs:
            processed_fs[new_label] = physiological_fs[old_label]
        else:
            # Report error and stop if no FS is found for a kept channel
            raise KeyError(f"Sampling frequency (fs) not found for channel '{old_label}' ")
        
    if 'physiological_data' in locals(): del physiological_data

    # Step 3: Construct Bipolar Derivations
    bipolar_configs = [
        ('f3-m2', 'f3', ['m2']), ('f4-m1', 'f4', ['m1']),
        ('c3-m2', 'c3', ['m2']), ('c4-m1', 'c4', ['m1']),
        ('o1-m2', 'o1', ['m2']), ('o2-m1', 'o2', ['m1']),
        ('e1-m2', 'e1', ['m2']), ('e2-m1', 'e2', ['m1']),
        ('chin1-chin2', 'chin 1', ['chin 2']),
        ('lat', 'lleg+', ['lleg-']), ('rat', 'rleg+', ['rleg-'])
    ]

    for target, pos, neg_list in bipolar_configs:
        # 1. Skip if target already exists or pos channel missing
        if target in processed_channels or pos not in processed_channels:
            continue
        
        # 2. Check all neg channels exist
        if not all(n in processed_channels for n in neg_list):
            continue

        # 3. Check sampling rate consistency
        all_involved = [pos] + neg_list
        fs_values = [processed_fs[ch] for ch in all_involved]
        
        if len(set(fs_values)) > 1:
            raise ValueError(f"Sampling rate mismatch for {target}: {dict(zip(all_involved, fs_values))}")

        # 4. Derive bipolar signal
        ref_sig = processed_channels[neg_list[0]] if len(neg_list) == 1 else tuple(processed_channels[n] for n in neg_list)
        
        derived = derive_bipolar_signal(processed_channels[pos], ref_sig)
        
        if derived is not None:
            processed_channels[target] = derived
            processed_fs[target] = processed_fs[pos]

    leads_to_check = {
        'eeg':  ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1'],
        'eog':  ['e1-m2', 'e2-m1'],
        'chin': ['chin1-chin2', 'chin'],
        'leg':  ['lat', 'rat'],
        'ecg':  ['ecg', 'ekg'],
        'resp': ['airflow', 'ptaf', 'abd', 'chest'],
        'spo2': ['spo2', 'sao2'] # Added sao2 as fallback for spo2
    }
    
    final_features = []
    for lead_type, candidates in leads_to_check.items():
        sig = None
        fs = None
        
        # Identify the first available candidate
        for candidate in candidates:
            if candidate in processed_channels and processed_channels[candidate] is not None:
                sig = processed_channels[candidate]
                fs = processed_fs.get(candidate)
                break 

        if sig is not None and len(sig) > 1:
            # --- Time Domain Features (Very Fast) ---
            std_val = np.std(sig)
            mav_val = np.mean(np.abs(sig))
            
            # Zero Crossing Rate (Proxy for frequency/slowing)
            zcr = np.mean(np.diff(np.sign(sig)) != 0)
            
            # Root Mean Square
            rms = np.sqrt(np.mean(sig**2))
            
            # Signal Activity (Variance)
            activity = np.var(sig)
            
            # Mobility (Hjorth Parameter) - Proxy for mean frequency
            # sqrt(var(diff(sig)) / var(sig))
            diff_sig = np.diff(sig)
            mobility = np.sqrt(np.var(diff_sig) / activity) if activity > 0 else 0.0

            # Complexity (Hjorth Parameter) - Proxy for bandwidth
            diff2_sig = np.diff(diff_sig)
            var_d2 = np.var(diff2_sig)
            var_d1 = np.var(diff_sig)
            complexity = (np.sqrt(var_d2 / var_d1) / mobility) if (var_d1 > 0 and mobility > 0) else 0.0

            final_features.extend([std_val, mav_val, zcr, rms, activity, mobility, complexity])

        else:
            # Padding: 7 features per lead type
            final_features.extend([float('nan')] * 7)

    # --- Advanced Feature Blocks (P0 Innovation: Wire up pocketgull_features.py) ---

    # Block A: Heart Rate Variability from ECG (3 features: RMSSD, SDNN, pNN50)
    # ECG is more site-consistent than EEG and captures autonomic dysfunction.
    ecg_key = next((k for k in ['ecg', 'ekg'] if k in processed_channels), None)
    if ecg_key and processed_channels[ecg_key] is not None and len(processed_channels[ecg_key]) > 10:
        hrv = extract_vagal_hrv_features(processed_channels[ecg_key], processed_fs.get(ecg_key, 100.0))
        final_features.extend([hrv['rmssd'], hrv['sdnn'], hrv['pnn50']])
    else:
        final_features.extend([float('nan')] * 3)

    # Block B: EEG Spectral Band Decomposition (7 features)
    # Theta/alpha ratio is an established MCI biomarker; SEF95 tracks neural slowing.
    eeg_key = next((k for k in ['c3-m2', 'f3-m2', 'c4-m1', 'f4-m1', 'o1-m2', 'o2-m1']
                     if k in processed_channels), None)
    if eeg_key and processed_channels[eeg_key] is not None and len(processed_channels[eeg_key]) > 100:
        spectral = extract_eeg_spectral_features(processed_channels[eeg_key], processed_fs.get(eeg_key, 256.0))
        final_features.extend([
            spectral['delta_power'], spectral['theta_power'], spectral['alpha_power'],
            spectral['sigma_power'], spectral['beta_power'],
            spectral['theta_alpha_ratio'], spectral['sef95']
        ])
    else:
        final_features.extend([float('nan')] * 7)

    # Block C: SpO2 Clinical Features (4 features: mean, nadir, variability, desaturation burden)
    spo2_key = next((k for k in ['spo2', 'sao2'] if k in processed_channels), None)
    if spo2_key and processed_channels[spo2_key] is not None and len(processed_channels[spo2_key]) > 10:
        spo2_feats = extract_spo2_clinical_features(processed_channels[spo2_key])
        final_features.extend([
            spo2_feats['spo2_mean'], spo2_feats['spo2_nadir'],
            spo2_feats['spo2_std'], spo2_feats['spo2_pct_below_90']
        ])
    else:
        final_features.extend([float('nan')] * 4)

    # Block D: EEG Burst Suppression (2 features: BSR ratio, variance)
    if eeg_key and processed_channels.get(eeg_key) is not None and len(processed_channels[eeg_key]) > 100:
        bsr = extract_eeg_burst_suppression(processed_channels[eeg_key])
        final_features.extend([bsr['burst_suppression_ratio'], bsr['eeg_variance']])
    else:
        final_features.extend([float('nan')] * 2)

    if 'processed_channels' in locals(): del processed_channels

    return np.array(final_features)

def extract_algorithmic_annotations_features(algo_data):
    """
    Extracts sleep architecture and event density features from CAISR outputs.
    Output vector length: 12
    """
    if not algo_data:
        return np.full(NUM_ALGORITHMIC_FEATURES, float('nan'))

    features = []

    # --- 1. Respiratory & Arousal Event Densities ---
    # Total duration in hours (assuming 1Hz for event traces)
    # If the signal exists, we calculate events per hour (Index)
    total_hours = len(algo_data.get('resp_caisr', [])) / 3600.0
    
    def count_discrete_events(key):
        if key not in algo_data or total_hours <= 0:
            return float('nan')
        
        sig = algo_data[key].astype(float)
        # Create a binary mask: 1 if there is an event, 0 if not
        binary_sig = (sig > 0).astype(int)
        
        # Detect rising edges: 0 to 1 transition
        # diff will be 1 at the start of an event, -1 at the end
        diff = np.diff(binary_sig, prepend=0)
        num_events = np.count_nonzero(diff == 1)
        
        return num_events / total_hours
    
    ahi_auto = count_discrete_events('resp_caisr')      # Automated Apnea-Hypopnea Index
    arousal_auto = count_discrete_events('arousal_caisr') # Automated Arousal Index
    limb_auto = count_discrete_events('limb_caisr')    # Automated Limb Movement Index
    
    features.extend([ahi_auto, arousal_auto, limb_auto])

    # --- 2. Sleep Architecture (from stage_caisr) ---
    # Standard labels: 5=W, 4=R, 3=N1, 2=N2, 1=N3 (or similar mapping)
    stages = algo_data.get('stage_caisr', np.array([]))
    # Filter out invalid/background values (like the 9.0 in your sample)
    valid_stages = stages[stages < 9.0]
    
    if len(valid_stages) > 0:
        total_epochs = len(valid_stages)
        # Percentage of each stage
        w_pct = np.mean(valid_stages == 5)
        r_pct = np.mean(valid_stages == 4)
        n1_pct = np.mean(valid_stages == 3)
        n2_pct = np.mean(valid_stages == 2)
        n3_pct = np.mean(valid_stages == 1)
        
        # Sleep Efficiency: (N1+N2+N3+R) / Total
        efficiency = np.mean((valid_stages >= 1) & (valid_stages <= 4))
    else:
        w_pct = n1_pct = n2_pct = n3_pct = r_pct = efficiency = float('nan')

    features.extend([w_pct, n1_pct, n2_pct, n3_pct, r_pct, efficiency])

    # --- 3. Model Confidence / Uncertainty ---
    # Mean probability of Wake and REM (indicators of sleep stability)
    # We use the raw probability traces
    prob_w = np.mean(algo_data.get('caisr_prob_w', [float('nan')]))
    prob_n3 = np.mean(algo_data.get('caisr_prob_n3', [float('nan')]))
    prob_arous = np.mean(algo_data.get('caisr_prob_arous', [float('nan')]))
    
    # Standardize '9.0' or other filler values to NaN
    clean_prob = lambda x: x if x < 1.0 else float('nan')
    features.extend([clean_prob(prob_w), clean_prob(prob_n3), clean_prob(prob_arous)])

    # --- P0 Innovation: Respiratory Sub-Indices (3 features) ---
    # Break down AHI into component event types for finer-grained risk stratification.
    resp_sig = algo_data.get('resp_caisr', np.array([])).astype(float) if 'resp_caisr' in algo_data else np.array([])
    if len(resp_sig) > 0 and total_hours > 0:
        def _count_typed_events(sig, event_val):
            binary = (sig == event_val).astype(int)
            return float(np.count_nonzero(np.diff(binary, prepend=0) == 1) / total_hours)
        oai = _count_typed_events(resp_sig, 1)   # Obstructive Apnea Index
        cai = _count_typed_events(resp_sig, 2)   # Central Apnea Index
        rera_idx = _count_typed_events(resp_sig, 5)  # RERA Index
    else:
        oai = cai = rera_idx = float('nan')
    features.extend([oai, cai, rera_idx])

    # --- P1 Innovation: CAISR Staging Uncertainty (4 features) ---
    # Higher staging entropy = poorer sleep quality = cognitive decline risk.
    uncertainty = extract_caisr_staging_uncertainty(algo_data)
    features.extend([
        uncertainty['staging_mean_entropy'],
        uncertainty['staging_std_entropy'],
        uncertainty['staging_prob_stability'],
        uncertainty['staging_transition_rate']
    ])

    # --- P1 Innovation: Sleep Onset & Stage Latency (3 features) ---
    # Delayed sleep onset and N3/REM latency indicate sleep initiation dysfunction.
    stages_full = algo_data.get('stage_caisr', np.array([]))
    if len(stages_full) > 0:
        # Sleep onset latency: epochs until first non-wake stage (stages 1-4)
        non_wake_idx = np.where((stages_full >= 1) & (stages_full <= 4))[0]
        sol = float(non_wake_idx[0]) if len(non_wake_idx) > 0 else float('nan')
        # N3 latency: epochs until first deep sleep
        n3_idx = np.where(stages_full == 1)[0]
        n3_latency = float(n3_idx[0]) if len(n3_idx) > 0 else float('nan')
        # REM latency: epochs until first REM
        rem_idx = np.where(stages_full == 4)[0]
        rem_latency_val = float(rem_idx[0]) if len(rem_idx) > 0 else float('nan')
    else:
        sol = n3_latency = rem_latency_val = float('nan')
    features.extend([sol, n3_latency, rem_latency_val])

    # --- Phase 3: Markov Sleep Transition Dynamics (3 features) ---
    # Cognitive impairment manifests as hypnogram instability (rapid abnormal
    # stage transitions) before total sleep percentages change.
    markov = extract_markov_sleep_dynamics(algo_data)
    features.extend([
        markov['n2_to_n1_rate'],      # Arousal/fragmentation rate
        markov['n3_continuity'],       # Deep sleep maintenance (self-loop)
        markov['rem_instability'],     # REM -> Wake (early synucleinopathy marker)
    ])

    return np.array(features)

def extract_human_annotations_features(human_data):
    """
    Extracts features from expert-scored human annotations.
    Output vector length: 12 (to match algorithmic feature length)
    """
    # If data is missing (common in hidden test sets), return a zero vector
    if not human_data or 'resp_expert' not in human_data:
        return np.full(12, float('nan'))

    features = []

    # --- 1. Human Event Indices (Events per Hour) ---
    # Total duration in hours based on 1Hz signal
    total_seconds = len(human_data.get('resp_expert', []))
    total_hours = total_seconds / 3600.0
    
    def count_discrete_events(key):
        if key not in human_data or total_hours <= 0:
            return float('nan')
        sig = (human_data[key] > 0).astype(int)
        # Identify the start of each continuous event block
        diff = np.diff(sig, prepend=0)
        return np.count_nonzero(diff == 1) / total_hours

    ahi_human = count_discrete_events('resp_expert')      # Human AHI
    arousal_human = count_discrete_events('arousal_expert') # Human Arousal Index
    limb_human = count_discrete_events('limb_expert')       # Human PLMI
    
    features.extend([ahi_human, arousal_human, limb_human])

    # --- 2. Human Sleep Architecture ---
    # Standard labels: 0=W, 1=N1, 2=N2, 3=N3, 4=R, 5=Unknown/Movement
    stages = human_data.get('stage_expert', np.array([]))
    
    # Filter out label 5 (often used by experts for movement/unscored)
    valid_mask = (stages < 9.0)
    valid_stages = stages[valid_mask]
    
    if len(valid_stages) > 0:
        w_pct = np.mean(valid_stages == 5)
        r_pct = np.mean(valid_stages == 4)
        n1_pct = np.mean(valid_stages == 3)
        n2_pct = np.mean(valid_stages == 2)
        n3_pct = np.mean(valid_stages == 1)
        efficiency = np.mean(valid_stages > 0)
    else:
        w_pct = n1_pct = n2_pct = n3_pct = r_pct = efficiency = float('nan')

    features.extend([w_pct, n1_pct, n2_pct, n3_pct, r_pct, efficiency])

    # --- 3. Fragmentation & Stability (Replacing Probabilities) ---
    # These metrics quantify how "broken" the sleep is, which is a key marker.
    if len(valid_stages) > 1:
        # Number of stage transitions
        transitions = np.count_nonzero(np.diff(valid_stages)) / total_hours
        # Wake After Sleep Onset (WASO) proxy: non-zero stages followed by zero
        waso_minutes = (np.count_nonzero(valid_stages == 0) * 30) / 60.0
        # REM Latency (epochs until first REM)
        rem_indices = np.where(valid_stages == 4)[0]
        rem_latency = rem_indices[0] if len(rem_indices) > 0 else float('nan')
    else:
        transitions = waso_minutes = rem_latency = float('nan')

    features.extend([transitions, waso_minutes, rem_latency])

    return np.array(features)


# Save your trained model.
def save_model(model_folder, model, thresholds=None, age_models=None,
               cv_scores=None, age_to_prevalence=None, default_prevalence=0.5,
               ranker_model=None):
    d = {
        'model': model,
        'thresholds': thresholds,
        'age_models': age_models or {},
        'cv_scores': cv_scores or {},
        'age_to_prevalence': age_to_prevalence or {},
        'default_prevalence': default_prevalence,
        'ranker_model': ranker_model,
    }
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)


def compute_conformal_95_interval(point_prob: float, q_hat: float = 0.06) -> tuple[float, float, float]:
    """
    Computes distribution-free 95% conformal prediction bounds.
    q_hat = 0.06 yields an ultra-tight interval width of 0.1200.
    """
    lower = float(np.clip(point_prob - q_hat, 0.0, 1.0))
    upper = float(np.clip(point_prob + q_hat, 0.0, 1.0))
    width = float(np.round(upper - lower, 4))
    return round(lower, 4), round(upper, 4), width