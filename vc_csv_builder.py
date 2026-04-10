#!/usr/bin/env python3
"""
vc_csv_builder.py — run on the LOGIN NODE after vc_convert.sh completes

Builds csv/vc/train.csv by combining:
  - csv/jasmin/train.csv  (all original JASMIN non-native utterances)
  - vc_audio/vc_*.wav     (Seed-VC converted CGN utterances)

Seed-VC output filename format (confirmed from source code):
    vc_{source_stem}_{target_stem}_{length_adjust}_{diffusion_steps}_{inference_cfg_rate}.wav
    e.g. vc_fn007290-0001234_N000055-fn000117_1.0_25_0.7.wav

The transcript for each VC file is recovered from csv/cgn/train.csv by
matching the source stem (the part before the first '_' boundary after 'vc_').

Output columns: ID, duration, wav, spk_id, wrd  (same as all project CSVs)

Usage:
    cd /scratch/mziemlewski/speechbrain/recipes/final-report
    python vc_csv_builder.py
"""

import pathlib
import re
import soundfile as sf
import pandas as pd

BASE = pathlib.Path("/scratch/mziemlewski/speechbrain/recipes/final-report")

JASMIN_TRAIN = BASE / "csv/jasmin/train.csv"
CGN_TRAIN    = BASE / "csv/cgn/train.csv"
VC_AUDIO_DIR = BASE / "vc_audio"
OUTPUT_DIR   = BASE / "csv/vc"

# Seed-VC suffix appended to every output file with our inference flags
VC_SUFFIX = "_1.0_25_0.7"

def get_duration(path: str) -> float:
    try:
        info = sf.info(path)
        return round(info.frames / info.samplerate, 4)
    except Exception:
        return 0.0

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jasmin = pd.read_csv(JASMIN_TRAIN)
    cgn    = pd.read_csv(CGN_TRAIN)

    # Build lookup: CGN source stem → transcript + speaker
    cgn_lookup = {pathlib.Path(r["wav"]).stem: r for _, r in cgn.iterrows()}

    # Find all converted files matching Seed-VC naming: vc_<src>_<tgt>_1.0_25_0.7.wav
    pattern = re.compile(r"^vc_(.+)" + re.escape(VC_SUFFIX) + r"$")
    vc_files = sorted(VC_AUDIO_DIR.glob(f"vc_*{VC_SUFFIX}.wav"))
    print(f"Found {len(vc_files)} converted VC files")

    vc_rows = []
    missing = 0

    for wav_path in vc_files:
        stem_no_suffix = wav_path.stem  # e.g. vc_<src>_<tgt>_1.0_25_0.7
        # Remove the VC suffix to get vc_<src>_<tgt>
        core = stem_no_suffix[: stem_no_suffix.index(VC_SUFFIX)]
        # core = vc_<src>_<tgt>; strip leading 'vc_'
        src_tgt = core[3:]  # remove 'vc_'

        # Recover source stem: try matching against known CGN stems
        # Source stem is everything up to the first '_' sequence that matches a CGN file.
        # We try progressively longer prefixes until we find a CGN match.
        src_stem = None
        parts = src_tgt.split("_")
        for end in range(1, len(parts)):
            candidate = "_".join(parts[:end])
            if candidate in cgn_lookup:
                src_stem = candidate
                break

        if src_stem is None:
            missing += 1
            continue

        duration = get_duration(str(wav_path))
        if duration < 0.5:
            continue

        cgn_row = cgn_lookup[src_stem]
        vc_rows.append({
            "ID":       wav_path.stem,
            "duration": duration,
            "wav":      str(wav_path),
            "spk_id":   f"vc_{cgn_row['spk_id']}",
            "wrd":      cgn_row["wrd"],
        })

    print(f"VC rows matched: {len(vc_rows)}  (unmatched: {missing})")

    vc_df   = pd.DataFrame(vc_rows, columns=["ID", "duration", "wav", "spk_id", "wrd"])
    combined = pd.concat([jasmin, vc_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    combined.to_csv(OUTPUT_DIR / "train.csv", index=False)
    print(f"\nWrote {len(combined)} rows → {OUTPUT_DIR / 'train.csv'}")
    print(f"  JASMIN original : {len(jasmin)}")
    print(f"  VC augmented    : {len(vc_rows)}")

    # Copy valid/test splits unchanged
    for split in ["valid.csv", "test.csv"]:
        src = BASE / "csv/jasmin" / split
        dst = OUTPUT_DIR / split
        if src.exists():
            pd.read_csv(src).to_csv(dst, index=False)
            print(f"  Copied {split}")

if __name__ == "__main__":
    main()
