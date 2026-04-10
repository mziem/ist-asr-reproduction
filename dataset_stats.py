#!/usr/bin/env python3
"""
dataset_stats.py

Computes and prints statistics for all CSV splits produced by
jasmin_prepare.py and cgn_prepare.py. Run this before training
to verify data integrity and collect numbers for the report.

Usage:
    python dataset_stats.py

Output:
    - Per-split: utterances, speakers, total hours, mean/std duration
    - Gender breakdown (from spk2gender files)
    - Empty transcript count
    - [UNK] and [LAUGH] token counts
    - Duration outliers (< 0.5s or > 20s)
    - Speaker overlap check across splits
"""

import os
import sys
from datetime import datetime
from contextlib import redirect_stdout
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE = "/scratch/mziemlewski/speechbrain/recipes/final-report"

SPLITS = {
    "CGN train":    os.path.join(BASE, "csv/cgn/train.csv"),
    "CGN valid":    os.path.join(BASE, "csv/cgn/valid.csv"),
    "CGN test-BN":  os.path.join(BASE, "csv/cgn/test_bn.csv"),
    "CGN test-CTS": os.path.join(BASE, "csv/cgn/test_cts.csv"),
    "JASMIN train": os.path.join(BASE, "csv/jasmin/train.csv"),
    "JASMIN valid": os.path.join(BASE, "csv/jasmin/valid.csv"),
    "JASMIN test":  os.path.join(BASE, "csv/jasmin/test.csv"),
}

# spk2gender files: map speaker ID prefix → gender file
# adjust paths if your folder names differ
GENDER_FILES = {
    "CGN":    [
        os.path.join(BASE, "CGN_train_for_course/spk2gender"),
        os.path.join(BASE, "CGN_conversational_test/spk2gender"),
        os.path.join(BASE, "CGN_read_test/spk2gender"),
    ],
    "JASMIN": [
        os.path.join(BASE, "kaldi_procesed/nl_test_read_all_hires/spk2gender"),
        os.path.join(BASE, "kaldi_procesed/nl_test_hmi_all_hires/spk2gender"),
    ],
}


class Tee:
    """Write output to multiple streams (console + file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_spk2gender(paths):
    """Return {spk_id: gender} merged from a list of spk2gender files."""
    mapping = {}
    for path in paths:
        if not os.path.isfile(path):
            print(f"  [WARNING] spk2gender not found: {path}")
            continue
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    mapping[parts[0]] = parts[1]
    return mapping


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def analyse_split(name, df, spk2gender):
    print(f"\n--- {name} ---")
    n_utt  = len(df)
    n_spk  = df["spk_id"].nunique()
    hours  = df["duration"].sum() / 3600
    mean_d = df["duration"].mean()
    std_d  = df["duration"].std()

    print(f"  Utterances : {n_utt}")
    print(f"  Speakers   : {n_spk}")
    print(f"  Duration   : {hours:.2f} h  (mean {mean_d:.2f}s ± {std_d:.2f}s)")

    # Gender
    if spk2gender:
        genders = df["spk_id"].map(spk2gender)
        counts  = genders.value_counts()
        unknown = genders.isna().sum()
        gender_str = "  | ".join(
            [f"{g}: {c} spk" for g, c in
             df.drop_duplicates("spk_id")["spk_id"].map(spk2gender)
             .value_counts().items()]
        )
        print(f"  Gender     : {gender_str}"
              + (f"  | unknown: {unknown} utt" if unknown else ""))

    # Transcript issues
    empty = (df["wrd"].isna() | (df["wrd"].str.strip() == "")).sum()
    unk   = df["wrd"].str.contains(r"\[UNK\]",   na=False).sum()
    laugh = df["wrd"].str.contains(r"\[LAUGH\]",  na=False).sum()
    print(f"  Empty wrd  : {empty}")
    print(f"  [UNK] utts : {unk}")
    print(f"  [LAUGH] utts: {laugh}")

    # Duration outliers
    short = (df["duration"] < 0.5).sum()
    long_ = (df["duration"] > 20.0).sum()
    if short or long_:
        print(f"  [WARN] Outliers: {short} < 0.5s | {long_} > 20s")
    else:
        print(f"  Outliers   : none")

    return set(df["spk_id"].unique())


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def generate_report():
    # Load gender mappings
    cgn_gender    = load_spk2gender(GENDER_FILES["CGN"])
    jasmin_gender = load_spk2gender(GENDER_FILES["JASMIN"])

    # Load all splits
    dfs = {}
    for name, path in SPLITS.items():
        if not os.path.isfile(path):
            print(f"[WARNING] CSV not found, skipping: {path}")
            continue
        dfs[name] = pd.read_csv(path)

    # ------------------------------------
    # Per-split analysis
    # ------------------------------------
    print_section("CGN SPLITS")
    cgn_speaker_sets = {}
    for name in ["CGN train", "CGN valid", "CGN test-BN", "CGN test-CTS"]:
        if name in dfs:
            cgn_speaker_sets[name] = analyse_split(name, dfs[name], cgn_gender)

    print_section("JASMIN SPLITS")
    jasmin_speaker_sets = {}
    for name in ["JASMIN train", "JASMIN valid", "JASMIN test"]:
        if name in dfs:
            jasmin_speaker_sets[name] = analyse_split(name, dfs[name], jasmin_gender)

    # ------------------------------------
    # Speaker overlap checks
    # ------------------------------------
    print_section("SPEAKER OVERLAP CHECKS")

    def overlap_check(sets, label):
        names = list(sets.keys())
        clean = True
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                overlap = sets[names[i]] & sets[names[j]]
                if overlap:
                    print(f"  [FAIL] {names[i]} ∩ {names[j]}: {len(overlap)} shared speakers")
                    clean = False
                else:
                    print(f"  [OK]   {names[i]} ∩ {names[j]}: no overlap")
        if clean:
            print(f"  All {label} splits are speaker-disjoint.")

    if len(cgn_speaker_sets) >= 2:
        print("\nCGN:")
        overlap_check(cgn_speaker_sets, "CGN")

    if len(jasmin_speaker_sets) >= 2:
        print("\nJASMIN:")
        overlap_check(jasmin_speaker_sets, "JASMIN")

    # ------------------------------------
    # Summary table (copy into report)
    # ------------------------------------
    print_section("SUMMARY TABLE (for report)")
    print(f"  {'Dataset':<12} {'Split':<10} {'Utts':>6} {'Spk':>5} {'Hours':>7}")
    print(f"  {'-'*44}")
    for name, df in dfs.items():
        dataset, split = name.split(" ", 1)
        print(f"  {dataset:<12} {split:<10} {len(df):>6} "
              f"{df['spk_id'].nunique():>5} {df['duration'].sum()/3600:>7.2f}")

    print("\n[DONE]")


def main():
    out_dir = os.environ.get("DATASET_STATS_OUT", os.path.join(BASE, "dataset_stats"))
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"dataset_stats_{timestamp}.txt")

    with open(out_path, "w", encoding="utf-8") as report_file:
        tee = Tee(sys.stdout, report_file)
        with redirect_stdout(tee):
            print(f"[INFO] Writing stats report to: {out_path}")
            print(f"[INFO] Generated at: {datetime.now().isoformat(timespec='seconds')}")
            generate_report()


if __name__ == "__main__":
    main()
