"""
SEED per-trial preprocessing for Case Study 2 — Report 7 cross-dataset test.

WHAT THIS SCRIPT DOES:
- Reads raw SEED Preprocessed_EEG/*.mat files (15 subjects × 3 sessions = 45 files)
- Each session file contains 15 films/trials as separate variables (djc_eeg1 ... djc_eeg15)
- For each film:
    1. Select 32 DEAP-matching channels from SEED's 62
    2. Resample 200 Hz -> 128 Hz (match DEAP sampling rate)
    3. Band-pass filter 4-45 Hz (match DEAP filtering)
    4. Compute Differential Entropy per frequency band on the full film signal
    5. Result: 32 channels x 5 bands = 160-d feature vector per trial
- Per-subject z-score across trials
- Save three .npy files ready for upload to Google Drive

OUTPUT SHAPES:
- X_seed_DE_pertrial.npy       : (675, 160)   675 = 15 subjects x 45 trials
- y_seed_pertrial.npy          : (675,)       labels {0=Neg, 1=Neu, 2=Pos}
- subj_seed_pertrial.npy       : (675,)       subject IDs 1..15

USAGE:
- Edit SEED_ROOT below to point to your Preprocessed_EEG folder
- Run: python preprocess_seed_pertrial.py
- Upload the 3 output .npy files to Google Drive:
  /content/drive/MyDrive/case_study_2/processed/

REQUIRES:
- numpy, scipy  (pip install numpy scipy   OR   conda install numpy scipy)
"""

import os
import sys
import re
from pathlib import Path
import numpy as np
from scipy import signal as scisig
from scipy.io import loadmat

# ============================================================
# CONFIG - EDIT THESE PATHS TO MATCH YOUR SETUP
# ============================================================

# Path to the Preprocessed_EEG folder (contains 45 .mat files + label.mat + readme.txt)
SEED_ROOT   = r"V:\Data_Analytics\case study 2\seed dataset\SEED_EEG\Preprocessed_EEG"

# Where to write the 3 output .npy files (any local folder you want)
OUTPUT_DIR  = r"V:\Data_Analytics\case study 2\seed dataset\SEED_EEG\pertrial_output"

# Path to label.mat (usually inside Preprocessed_EEG itself)
LABEL_FILE  = os.path.join(SEED_ROOT, "label.mat")

# ============================================================
# CONSTANTS - matched to your DEAP preprocessing
# ============================================================

# SEED original sampling rate
FS_SEED_ORIG = 200

# Target sampling rate (matches your DEAP)
FS_TARGET    = 128

# Frequency bands (Hz) - same as your R1 preprocessing
BANDS = {
    'theta':     (4, 8),
    'alpha':     (8, 14),
    'beta_low':  (14, 20),
    'beta_high': (20, 30),
    'gamma':     (30, 45),
}

# SEED's 62 channels in file order (from channel-order.xlsx)
SEED_62_CHANNELS = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
    'O2', 'CB2',
]

# DEAP's 32 channels (standard 10-20 subset used in your R1 preprocessing)
DEAP_32_CHANNELS = [
    'FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
    'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'OZ', 'PZ',
    'FP2', 'AF4', 'FZ', 'F4', 'F8', 'FC6', 'FC2', 'CZ',
    'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2',
]

# Precompute indices of the 32 DEAP channels within SEED's 62-channel order
DEAP_CHANNEL_INDICES = [SEED_62_CHANNELS.index(ch) for ch in DEAP_32_CHANNELS]
print(f'DEAP-matching channels found in SEED: {len(DEAP_CHANNEL_INDICES)}/32')

# Label sequence per session (from readme.txt) — same for ALL 3 sessions
# 1=Pos, 0=Neu, -1=Neg
LABELS_PER_SESSION_RAW = np.array(
    [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1], dtype=np.int64
)
# Remap to {0=Neg, 1=Neu, 2=Pos} to match your R1 convention
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABELS_PER_SESSION = np.array([LABEL_MAP[v] for v in LABELS_PER_SESSION_RAW], dtype=np.int64)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def bandpass_filter(x, low, high, fs, order=5):
    """Zero-phase Butterworth band-pass filter along last axis."""
    nyq = fs / 2.0
    b, a = scisig.butter(order, [low / nyq, high / nyq], btype='band')
    return scisig.filtfilt(b, a, x, axis=-1)


def compute_de(signal_1d):
    """Differential Entropy of a Gaussian-approximated signal.
    DE = 0.5 * log(2 * pi * e * variance)"""
    var = np.var(signal_1d)
    if var <= 0:
        return 0.0
    return 0.5 * np.log(2 * np.pi * np.e * var)


def resample_signal(x, fs_orig, fs_target):
    """Resample along last axis using polyphase filtering."""
    if fs_orig == fs_target:
        return x
    return scisig.resample_poly(x, up=fs_target, down=fs_orig, axis=-1)


def parse_subject_id_from_filename(filename):
    """Extract subject ID (1..15) from filenames like '6_20131113.mat'."""
    m = re.match(r'^(\d+)_\d+\.mat$', filename)
    if m:
        return int(m.group(1))
    return None


def extract_trial_variables(mat_dict):
    """Find EEG trial variables inside a session .mat file.
    Names are typically like 'djc_eeg1', 'ww_eeg1' etc. (subject-initials + _eeg + trial_number).
    Returns list of (trial_number, eeg_array) sorted by trial_number.
    """
    trials = []
    for key in mat_dict.keys():
        if key.startswith('__'):
            continue
        m = re.match(r'^.*_eeg(\d+)$', key)
        if m:
            trial_num = int(m.group(1))
            trials.append((trial_num, mat_dict[key]))
    trials.sort(key=lambda t: t[0])
    return trials


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_one_film(eeg_film, fs_orig, channel_indices):
    """Process a single film's EEG signal.

    Args:
        eeg_film: (62, T) numpy array at fs_orig Hz
        fs_orig: original sampling rate (200 Hz for SEED)
        channel_indices: list of 32 indices into the 62 channels

    Returns:
        DE vector of shape (160,) = 32 channels x 5 bands, flattened as (channel x band)
    """
    # Step 1: Select 32 DEAP-matching channels
    eeg = eeg_film[channel_indices, :]      # (32, T)

    # Step 2: Resample 200 Hz -> 128 Hz
    eeg = resample_signal(eeg, fs_orig, FS_TARGET)   # (32, T')

    # Step 3: Compute DE per (channel, band) on the full film signal
    n_channels = eeg.shape[0]
    n_bands = len(BANDS)
    de_matrix = np.zeros((n_channels, n_bands), dtype=np.float32)

    for b_idx, (band_name, (low, high)) in enumerate(BANDS.items()):
        filtered = bandpass_filter(eeg, low, high, FS_TARGET, order=5)
        for c_idx in range(n_channels):
            de_matrix[c_idx, b_idx] = compute_de(filtered[c_idx, :])

    # Flatten to match DEAP feature ordering: 32 channels x 5 bands
    return de_matrix.reshape(-1).astype(np.float32)     # (160,)


def main():
    seed_root = Path(SEED_ROOT)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not seed_root.exists():
        print(f'ERROR: SEED root not found: {seed_root}')
        print('Edit SEED_ROOT at the top of this script.')
        sys.exit(1)

    # Find all session .mat files
    mat_files = sorted([f for f in seed_root.glob('*.mat') if f.name != 'label.mat'])
    print(f'\nFound {len(mat_files)} session files (expected 45)')

    # Group by subject (three sessions each)
    subject_sessions = {}
    for f in mat_files:
        sid = parse_subject_id_from_filename(f.name)
        if sid is None:
            print(f'WARNING: cannot parse subject id from {f.name}, skipping')
            continue
        subject_sessions.setdefault(sid, []).append(f)

    print(f'Subjects found: {sorted(subject_sessions.keys())}')

    # Sort each subject's sessions chronologically (by date in filename)
    for sid in subject_sessions:
        subject_sessions[sid].sort(key=lambda f: f.name)

    # Storage
    all_X = []
    all_y = []
    all_s = []

    # Process each subject
    for sid in sorted(subject_sessions.keys()):
        sessions = subject_sessions[sid]
        print(f'\n===== Subject {sid} — {len(sessions)} sessions =====')
        subject_X = []
        subject_y = []

        for sess_idx, sess_file in enumerate(sessions, start=1):
            mat = loadmat(str(sess_file))
            trials = extract_trial_variables(mat)
            print(f'  session {sess_idx} ({sess_file.name}): {len(trials)} trials')

            if len(trials) != 15:
                print(f'    WARNING: expected 15 trials, got {len(trials)}')

            for trial_num, eeg_film in trials:
                # eeg_film shape: (62, T)
                if eeg_film.shape[0] != 62:
                    print(f'    WARNING: expected 62 channels, got {eeg_film.shape[0]}')
                    continue

                de_vec = process_one_film(eeg_film, FS_SEED_ORIG, DEAP_CHANNEL_INDICES)

                # Label from the fixed sequence (index = trial_num - 1)
                label = LABELS_PER_SESSION[trial_num - 1]
                subject_X.append(de_vec)
                subject_y.append(label)

        subject_X = np.stack(subject_X, axis=0)   # (n_trials_this_subject, 160)
        subject_y = np.array(subject_y, dtype=np.int64)
        subject_s = np.full(len(subject_y), sid, dtype=np.int64)

        # Per-subject z-score across trials (match DEAP normalisation)
        mu  = subject_X.mean(axis=0, keepdims=True)
        std = subject_X.std(axis=0, keepdims=True) + 1e-8
        subject_X = ((subject_X - mu) / std).astype(np.float32)

        print(f'  final for subject {sid}: X {subject_X.shape}, y counts {np.bincount(subject_y).tolist()}')

        all_X.append(subject_X)
        all_y.append(subject_y)
        all_s.append(subject_s)

    # Concatenate all subjects
    X_final = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    s_final = np.concatenate(all_s, axis=0)

    print('\n===== FINAL =====')
    print(f'X: {X_final.shape}   dtype={X_final.dtype}')
    print(f'y: {y_final.shape}   dtype={y_final.dtype}   counts={np.bincount(y_final).tolist()}')
    print(f'subjects: {s_final.shape}   unique={np.unique(s_final).tolist()}')

    # Save outputs
    x_path = output_dir / 'X_seed_DE_pertrial.npy'
    y_path = output_dir / 'y_seed_pertrial.npy'
    s_path = output_dir / 'subj_seed_pertrial.npy'
    np.save(x_path, X_final)
    np.save(y_path, y_final)
    np.save(s_path, s_final)

    print(f'\nSaved to: {output_dir}')
    print(f'  - {x_path.name}   ({x_path.stat().st_size / 1024:.1f} KB)')
    print(f'  - {y_path.name}   ({y_path.stat().st_size / 1024:.1f} KB)')
    print(f'  - {s_path.name}   ({s_path.stat().st_size / 1024:.1f} KB)')
    print('\nDone. Upload these 3 files to Google Drive:')
    print('  /content/drive/MyDrive/case_study_2/processed/')


if __name__ == '__main__':
    main()
