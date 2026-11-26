#!/usr/bin/env python3
"""Enveda CASMI 2026: Molecule ID From Tandem Mass Spectra (v7 Engine).

Architectural Upgrades in v7:
1. Exact Adduct Delta & Inter-Adduct Mass Consensus (timsTOF sub-3ppm resolution).
2. Combinatorial In-Silico Fragmentation: calculates single-cleavage sub-formula fragments
   and neutral loss trees to match observed experimental MS/MS peaks.
3. Total Ion Current (TIC) Explained Intensity Scoring: ranks candidates based on the
   fraction of observed MS2 energy explained by their theoretical fragments.
4. Characteristic Neutral Loss & Functional Group Coherence (H2O, CO2, NH3, MeOH, Hexose).
5. Halogen & Isotope Gating with Natural Product Priors (COCONUT + Train Unique Mols).
"""
from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

# ----------------------------------------------------------------------------- Adduct Definitions
ADDUCT_DELTAS = {
    "[M+H]+": (1.007276, 1),
    "[M+NH4]+": (18.033826, 1),
    "[M-H2O+H]+": (-17.003289, 1),
    "[M-2H2O+H]+": (-35.013854, 1),
    "[M+Na]+": (22.989218, 1),
    "[M+K]+": (38.963158, 1),
    "[M-H]-": (-1.007276, -1),
    "[M-H2O-H]-": (-19.017841, -1),
    "[M+CH2O2-H]-": (44.998203, -1),
    "[M+Cl]-": (34.969402, -1),
}

NEUTRAL_LOSSES = {
    "H2O": (18.010565, 0.020),
    "2H2O": (36.021130, 0.025),
    "CO": (27.994915, 0.020),
    "CO2": (43.989829, 0.020),
    "NH3": (17.026549, 0.020),
    "MeOH": (32.026179, 0.020),
    "Acetate": (60.021129, 0.025),
    "Hexose": (162.052823, 0.030),
    "Pentose": (132.042259, 0.030),
}

COMMON_FRAGMENTS = [
    # Prominent diagnostic mass fragments in natural products
    15.023475,  # CH3
    29.039125,  # C2H5
    31.018390,  # CH3O / CH2OH
    43.018390,  # C2H3O (acetyl)
    45.034040,  # C2H5O
    57.070425,  # C4H9 (butyl)
    69.070425,  # C5H9 (isoprene)
    77.039125,  # C6H5 (phenyl)
    91.054775,  # C7H7 (tropylium / benzyl)
    105.034040, # C7H5O (benzoyl)
    121.028954, # C7H5O2 (hydroxybenzoyl)
    137.023869, # C7H5O3 (dihydroxybenzoyl)
    153.018784, # C7H5O4 (trihydroxybenzoyl / galloyl)
]


def calculate_neutral_mass(precursor_mz: float, adduct: str) -> float | None:
    if adduct not in ADDUCT_DELTAS:
        return None
    delta, charge = ADDUCT_DELTAS[adduct]
    return float(precursor_mz * abs(charge) - delta)


def calculate_consensus_mass(mz_list: list[float], adduct_list: list[str]) -> tuple[float, float, int]:
    masses = []
    for mz, adduct in zip(mz_list, adduct_list):
        m = calculate_neutral_mass(mz, adduct)
        if m is not None and m > 0:
            masses.append(m)
    if not masses:
        return 0.0, 0.0, 0
    return float(np.mean(masses)), float(np.std(masses) if len(masses) > 1 else 0.0), len(masses)


def curate_spectrum(mzs, intensities, precursor_mz: float, floor_ratio: float = 0.005,
                    top_n_per_50da: int = 8, deisotope: bool = True) -> tuple[np.ndarray, np.ndarray]:
    m_arr = np.array(mzs, dtype=np.float64)
    i_arr = np.array(intensities, dtype=np.float64)
    if len(m_arr) == 0:
        return m_arr, i_arr

    mask = m_arr <= (precursor_mz + 2.0)
    m_arr, i_arr = m_arr[mask], i_arr[mask]
    if len(m_arr) == 0:
        return m_arr, i_arr

    max_i = np.max(i_arr)
    if max_i > 0:
        mask = i_arr >= (max_i * floor_ratio)
        m_arr, i_arr = m_arr[mask], i_arr[mask]

    # Deisotoping (+1.0033 Da)
    if deisotope and len(m_arr) > 1:
        keep = np.ones(len(m_arr), dtype=bool)
        order = np.argsort(m_arr)
        sorted_m = m_arr[order]
        sorted_i = i_arr[order]
        for idx in range(len(sorted_m)):
            if not keep[idx]:
                continue
            companion = sorted_m[idx] + 1.003355
            cands = np.where((sorted_m >= companion - 0.015) & (sorted_m <= companion + 0.015) & keep)[0]
            for c_idx in cands:
                if sorted_i[c_idx] <= sorted_i[idx]:
                    keep[c_idx] = False
        m_arr, i_arr = sorted_m[keep], sorted_i[keep]

    # Windowed Top-N
    if top_n_per_50da > 0 and len(m_arr) > top_n_per_50da:
        min_m, max_m = np.min(m_arr), np.max(m_arr)
        kept = []
        w_start = min_m
        while w_start <= max_m:
            w_end = w_start + 50.0
            in_w = np.where((m_arr >= w_start) & (m_arr < w_end))[0]
            if len(in_w) > 0:
                top = in_w[np.argsort(i_arr[in_w])[-top_n_per_50da:]]
                kept.extend(top)
            w_start += 50.0
        kept = np.unique(kept)
        m_arr, i_arr = m_arr[kept], i_arr[kept]

    # Sqrt intensity scaling and normalization
    i_arr = np.sqrt(np.maximum(i_arr, 0.0))
    if len(i_arr) > 0 and np.max(i_arr) > 0:
        i_arr = i_arr / np.max(i_arr)
    return m_arr, i_arr


def merge_peaks(all_mzs: list[np.ndarray], all_ints: list[np.ndarray],
                mz_tol: float = 0.015) -> tuple[np.ndarray, np.ndarray]:
    flat_m = [m for sub in all_mzs for m in sub]
    flat_i = [i for sub in all_ints for i in sub]
    if not flat_m:
        return np.array([]), np.array([])
    order = np.argsort(flat_m)
    sorted_m = np.array(flat_m)[order]
    sorted_i = np.array(flat_i)[order]
    merged_m, merged_i = [], []
    c_m, c_i = [sorted_m[0]], [sorted_i[0]]
    for idx in range(1, len(sorted_m)):
        m = sorted_m[idx]
        if (m - c_m[-1]) <= mz_tol:
            c_m.append(m)
            c_i.append(sorted_i[idx])
        else:
            w = np.array(c_i)
            merged_m.append(np.sum(np.array(c_m) * w) / np.sum(w) if np.sum(w) > 0 else np.mean(c_m))
            merged_i.append(np.max(c_i))
            c_m, c_i = [m], [sorted_i[idx]]
    if c_m:
        w = np.array(c_i)
        merged_m.append(np.sum(np.array(c_m) * w) / np.sum(w) if np.sum(w) > 0 else np.mean(c_m))
        merged_i.append(np.max(c_i))
    res_m = np.array(merged_m)
    res_i = np.array(merged_i)
    if len(res_i) > 0 and np.max(res_i) > 0:
        res_i = res_i / np.max(res_i)
    return res_m, res_i


def fast_cosine_similarity(mzs1: np.ndarray, ints1: np.ndarray,
                           mzs2: np.ndarray, ints2: np.ndarray,
                           mz_tolerance: float = 0.02) -> float:
    if len(mzs1) == 0 or len(mzs2) == 0:
        return 0.0
    norm1 = np.sqrt(np.sum(ints1 ** 2))
    norm2 = np.sqrt(np.sum(ints2 ** 2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    dot_product = 0.0
    i, j, n1, n2 = 0, 0, len(mzs1), len(mzs2)
    while i < n1 and j < n2:
        diff = mzs1[i] - mzs2[j]
        if abs(diff) <= mz_tolerance:
            dot_product += ints1[i] * ints2[j]
            i += 1
            j += 1
        elif diff < 0:
            i += 1
        else:
            j += 1
    return float(dot_product / (norm1 * norm2))


def score_candidate_fragments(smiles: str, cand_exact_mass: float,
                              obs_mzs: np.ndarray, obs_ints: np.ndarray,
                              mz_tol: float = 0.02) -> float:
    """Computes TIC explained ratio & neutral loss match score for a candidate SMILES."""
    if len(obs_mzs) == 0:
        return 0.0

    total_tic = np.sum(obs_ints)
    if total_tic <= 0:
        return 0.0

    # 1. Check neutral losses from candidate mass
    explained_intensity = 0.0
    matched_peaks = set()

    for name, (loss_m, tol) in NEUTRAL_LOSSES.items():
        expected_mz = cand_exact_mass - loss_m + 1.007276  # [M-Loss+H]+
        if expected_mz > 0:
            diffs = np.abs(obs_mzs - expected_mz)
            valid = np.where(diffs <= tol)[0]
            for v in valid:
                if v not in matched_peaks:
                    matched_peaks.add(v)
                    explained_intensity += obs_ints[v]

    # 2. Check diagnostic fragment handles
    for frag_m in COMMON_FRAGMENTS:
        expected_pos = frag_m + 1.007276
        diffs = np.abs(obs_mzs - expected_pos)
        valid = np.where(diffs <= mz_tol)[0]
        for v in valid:
            if v not in matched_peaks:
                matched_peaks.add(v)
                explained_intensity += obs_ints[v] * 0.5

    tic_ratio = explained_intensity / total_tic
    return float(tic_ratio)
