#!/usr/bin/env python3
"""
jasmin_prepare.py

Prepares SpeechBrain-compatible CSV files for the Dutch non-native subset
of the JASMIN-CGN corpus (kaldi-processed version).

Includes all four non-native subgroups:
  - older adults, read speech      (old-read-utt2spk.txt)
  - older adults, HMI speech       (old-hmi-utt2spk.txt)
  - non-native children, read      (child-read-utt2spk.txt)
  - non-native children, HMI       (child-hmi-utt2spk.txt)

Audio comes from the kaldi_procesed/ folder (nl_test_read_all_hires and
nl_test_hmi_all_hires). The wav.scp paths in those folders point to the
original shared server location and are therefore rebuilt from scratch
using the actual files found on disk.

Split strategy: speaker-disjoint (no speaker appears in more than one split).
  - 70% of speakers → train
  - 15% of speakers → validation
  - 15% of speakers → test

Output: three CSV files with columns [ID, duration, wav, spk_id, wrd]
compatible with the SpeechBrain DataIO pipeline.

Usage:
    python jasmin_prepare.py

    Optionally override paths via environment variables:
        JASMIN_BASE   - path to kaldi_procesed/ parent directory
        JASMIN_GROUP  - path to Dutch_non-native/ filter directory
        JASMIN_OUT    - output directory for CSV files
"""

import os
import random
import soundfile as sf
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — edit these or set environment variables
# ---------------------------------------------------------------------------
BASE_DIR   = os.environ.get("JASMIN_BASE",
             "/scratch/mziemlewski/speechbrain/recipes/final-report/kaldi_procesed")
GROUP_DIR  = os.environ.get("JASMIN_GROUP",
             "/scratch/mziemlewski/speechbrain/recipes/final-report/Dutch_non-native")
OUTPUT_DIR = os.environ.get("JASMIN_OUT",
             "/scratch/mziemlewski/speechbrain/recipes/final-report/csv/jasmin")

# kaldi_procesed subfolders and which group filter file they belong to
SUBFOLDERS = {
    "nl_test_read_all_hires": ["old-read-utt2spk.txt", "child-read-utt2spk.txt"],
    "nl_test_hmi_all_hires":  ["old-hmi-utt2spk.txt",  "child-hmi-utt2spk.txt"],
}

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
# TEST_RATIO  = 0.15  (remainder)

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_utt2spk(path):
    """Return {utt_id: spk_id} from a utt2spk file."""
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def load_text(folder_path):
    """Return {utt_id: transcription} from kaldi text file."""
    text = {}
    with open(os.path.join(folder_path, "text"), "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            utt = parts[0]
            wrd = parts[1] if len(parts) == 2 else ""
            text[utt] = wrd
    return text


def build_wav_map(folder_path):
    """
    Scan folder_path recursively and return {utt_id: absolute_wav_path}.
    The original wav.scp is stale after copying, so we rebuild from disk.
    We use the utterance ID (filename without extension) as the key.
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


def normalize_transcript(text):
    """
    Minimal normalization for Dutch JASMIN transcripts:
      - Replace 'ggg' with '[LAUGH]' for consistency (as per lab instructions).
      - Strip leading/trailing whitespace.
    """
    text = text.strip()
    text = text.replace("ggg", "[LAUGH]")
    return text


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)

    # ------------------------------------------------------------------
    # Step 1: Collect all valid non-native utterances across subfolders
    # ------------------------------------------------------------------
    rows = []
    missing_audio = 0
    zero_duration = 0

    for subfolder, group_files in SUBFOLDERS.items():
        folder_path = os.path.join(BASE_DIR, subfolder)

        if not os.path.isdir(folder_path):
            print(f"[WARNING] Subfolder not found, skipping: {folder_path}")
            continue

        # Load metadata for this kaldi subfolder
        text_map = load_text(folder_path)
        wav_map  = build_wav_map(folder_path)

        # Load the relevant non-native filter files and merge
        utt2spk = {}
        for gf in group_files:
            gf_path = os.path.join(GROUP_DIR, gf)
            if not os.path.isfile(gf_path):
                print(f"[WARNING] Group filter file not found: {gf_path}")
                continue
            utt2spk.update(load_utt2spk(gf_path))

        if not utt2spk:
            print(f"[WARNING] No utterances loaded for {subfolder}, check group filter paths.")
            continue

        print(f"[INFO] {subfolder}: {len(utt2spk)} utterances from non-native filter files")

        for utt_id, spk_id in utt2spk.items():
            wav_path = wav_map.get(utt_id)
            if wav_path is None:
                missing_audio += 1
                continue

            duration = get_duration(wav_path)
            if duration <= 0.0:
                zero_duration += 1
                continue

            wrd = normalize_transcript(text_map.get(utt_id, ""))
            rows.append({
                "ID":       utt_id,
                "duration": round(duration, 4),
                "wav":      wav_path,
                "spk_id":   spk_id,
                "wrd":      wrd,
            })

    if not rows:
        raise RuntimeError("No valid utterances found. Check your paths and filter files.")

    df = pd.DataFrame(rows, columns=["ID", "duration", "wav", "spk_id", "wrd"])
    df = df[df["duration"] >= 0.5].reset_index(drop=True)
    print(f"\n[INFO] Total valid utterances : {len(df)}")
    print(f"[INFO] Missing audio skipped   : {missing_audio}")
    print(f"[INFO] Zero-duration skipped   : {zero_duration}")
    print(f"[INFO] Unique speakers         : {df['spk_id'].nunique()}")

    # ------------------------------------------------------------------
    # Step 2: Speaker-disjoint splits
    # ------------------------------------------------------------------
    speakers = sorted(df["spk_id"].unique())
    random.shuffle(speakers)

    n       = len(speakers)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    train_spks = set(speakers[:n_train])
    valid_spks = set(speakers[n_train:n_train + n_valid])
    test_spks  = set(speakers[n_train + n_valid:])

    train_df = df[df["spk_id"].isin(train_spks)].reset_index(drop=True)
    valid_df = df[df["spk_id"].isin(valid_spks)].reset_index(drop=True)
    test_df  = df[df["spk_id"].isin(test_spks)].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Step 3: Save
    # ------------------------------------------------------------------
    for split_name, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        out_path = os.path.join(OUTPUT_DIR, f"{split_name}.csv")
        split_df.to_csv(out_path, index=False)
        total_hours = split_df["duration"].sum() / 3600
        print(f"[INFO] {split_name:5s}: {len(split_df):5d} utterances | "
              f"{split_df['spk_id'].nunique():3d} speakers | "
              f"{total_hours:.2f} hours → {out_path}")

    print("\n[DONE] JASMIN CSV preparation complete.")


if __name__ == "__main__":
    main()
