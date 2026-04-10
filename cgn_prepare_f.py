#!/usr/bin/env python3
"""
cgn_prepare.py

Prepares SpeechBrain-compatible CSV files for the CGN (Corpus Gesproken
Nederlands) dataset as provided in the IST Labs shared data directory.

Structure expected in your working directory:
    CGN_train_for_course/     - ~50h Dutch read speech (train + validation)
    CGN_conversational_test/  - CTS test set (Conversational Telephone Speech)
    CGN_read_test/            - BN test set (Broadcast News / read speech)

Each folder contains kaldi-style metadata files:
    utt2spk, text, wav.scp, spk2gender

Since wav.scp paths are stale after copying, audio paths are rebuilt from
disk (same approach as jasmin_prepare.py).

Split strategy for training data:
  - Speaker-disjoint: 85% of speakers → train, 15% → validation
  - The two test sets are fixed (used as-is, no re-splitting)

Output CSVs (columns: ID, duration, wav, spk_id, wrd):
    csv/cgn/train.csv
    csv/cgn/valid.csv
    csv/cgn/test_cts.csv   (conversational test)
    csv/cgn/test_bn.csv    (broadcast news / read test)

Usage:
    python cgn_prepare.py

    Optionally override paths via environment variables:
        CGN_TRAIN_DIR - path to CGN_train_for_course/
        CGN_CTS_DIR   - path to CGN_conversational_test/
        CGN_BN_DIR    - path to CGN_read_test/
        CGN_OUT       - output directory for CSV files
"""

import os
import random
import soundfile as sf
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE = "/scratch/mziemlewski/speechbrain/recipes/final-report"

TRAIN_DIR  = os.environ.get("CGN_TRAIN_DIR", os.path.join(BASE, "CGN_train_for_course"))
CTS_DIR    = os.environ.get("CGN_CTS_DIR",   os.path.join(BASE, "CGN_conversational_test"))
BN_DIR     = os.environ.get("CGN_BN_DIR",    os.path.join(BASE, "CGN_read_test"))
OUTPUT_DIR = os.environ.get("CGN_OUT",        os.path.join(BASE, "csv/cgn"))

TRAIN_RATIO = 0.85   # of training speakers → train; remainder → valid
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_utt2spk(folder_path):
    """Return {utt_id: spk_id}."""
    mapping = {}
    fpath = os.path.join(folder_path, "utt2spk")
    with open(fpath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def load_text(folder_path):
    """Return {utt_id: transcription}."""
    text = {}
    fpath = os.path.join(folder_path, "text")
    with open(fpath, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            utt = parts[0]
            wrd = parts[1] if len(parts) == 2 else ""
            text[utt] = wrd
    return text


def build_wav_map(folder_path):
    """
    Rebuild {utt_id: absolute_wav_path} by scanning folder_path recursively.
    wav.scp is stale after copying, so we ignore it and find files on disk.
    """
    wav_map = {}
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.endswith(".wav"):
                utt_id = os.path.splitext(fname)[0]
                wav_map[utt_id] = os.path.join(root, fname)
    return wav_map


def get_duration(wav_path):
    """Return duration in seconds, or 0.0 on failure."""
    try:
        with sf.SoundFile(wav_path) as f:
            return len(f) / f.samplerate
    except Exception:
        return 0.0


def folder_to_df(folder_path, label=""):
    """
    Read a kaldi-style folder and return a DataFrame with all valid utterances.
    Skips utterances with missing audio or zero duration.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    utt2spk  = load_utt2spk(folder_path)
    text_map = load_text(folder_path)
    wav_map  = build_wav_map(folder_path)

    rows = []
    missing = 0
    zero_dur = 0

    for utt_id, spk_id in utt2spk.items():
        wav_path = wav_map.get(utt_id)
        if wav_path is None:
            missing += 1
            continue

        duration = get_duration(wav_path)
        if duration <= 0.0:
            zero_dur += 1
            continue

        wrd = text_map.get(utt_id, "").strip()
        rows.append({
            "ID":       utt_id,
            "duration": round(duration, 4),
            "wav":      wav_path,
            "spk_id":   spk_id,
            "wrd":      wrd,
        })

    df = pd.DataFrame(rows, columns=["ID", "duration", "wav", "spk_id", "wrd"])
    df = df[df["duration"] >= 0.5] # Filter out very short utterances (< 0.5s)
    tag = f"[{label}]" if label else ""
    print(f"[INFO] {tag} {len(df)} valid utterances | "
          f"{df['spk_id'].nunique()} speakers | "
          f"missing audio: {missing} | zero-duration: {zero_dur}")
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)

    # ------------------------------------------------------------------
    # Load training data
    # ------------------------------------------------------------------
    print("[INFO] Loading CGN training data...")
    train_full_df = folder_to_df(TRAIN_DIR, label="CGN train")

    # ------------------------------------------------------------------
    # Load CTS test set (needed early to remove overlap)
    # ------------------------------------------------------------------
    print("[INFO] Loading CGN CTS test set...")
    cts_df = folder_to_df(CTS_DIR, label="CGN CTS test")

    # ------------------------------------------------------------------
    # Load BN test set (unchanged)
    # ------------------------------------------------------------------
    print("[INFO] Loading CGN BN (read) test set...")
    bn_df = folder_to_df(BN_DIR, label="CGN BN test")

    # ------------------------------------------------------------------
    # remove overlapping speakers
    # ------------------------------------------------------------------
    cts_speakers = set(cts_df["spk_id"].unique())
    train_full_df = train_full_df[~train_full_df["spk_id"].isin(cts_speakers)]

    # ------------------------------------------------------------------
    # Training data: speaker-disjoint train / valid split
    # ------------------------------------------------------------------
    speakers = sorted(train_full_df["spk_id"].unique())
    random.shuffle(speakers)

    n_train    = int(len(speakers) * TRAIN_RATIO)
    train_spks = set(speakers[:n_train])
    valid_spks = set(speakers[n_train:])

    train_df = train_full_df[train_full_df["spk_id"].isin(train_spks)].reset_index(drop=True)
    valid_df = train_full_df[train_full_df["spk_id"].isin(valid_spks)].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Save all splits
    # ------------------------------------------------------------------
    splits = [
        ("train",    train_df, os.path.join(OUTPUT_DIR, "train.csv")),
        ("valid",    valid_df, os.path.join(OUTPUT_DIR, "valid.csv")),
        ("test_cts", cts_df,   os.path.join(OUTPUT_DIR, "test_cts.csv")),
        ("test_bn",  bn_df,    os.path.join(OUTPUT_DIR, "test_bn.csv")),
    ]

    print()
    for split_name, split_df, out_path in splits:
        split_df.to_csv(out_path, index=False)
        total_hours = split_df["duration"].sum() / 3600
        print(f"[INFO] {split_name:9s}: {len(split_df):5d} utterances | "
              f"{split_df['spk_id'].nunique():3d} speakers | "
              f"{total_hours:.2f} hours → {out_path}")

    print("\n[DONE] CGN CSV preparation complete.")


if __name__ == "__main__":
    main()
