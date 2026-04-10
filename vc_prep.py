#!/usr/bin/env python3
"""
vc_prep.py — run on the LOGIN NODE (no GPU, no SLURM needed)

Reads csv/cgn/train.csv and csv/jasmin/train.csv, then writes:
  vc_audio/cgn_sources.txt   — 250 sampled CGN wav paths (one per line)
  vc_audio/jasmin_targets.txt — one reference wav per JASMIN speaker (one per line)

These files are consumed by vc_convert.sh on the compute node.

Usage:
    cd /scratch/mziemlewski/speechbrain/recipes/final-report
    python vc_prep.py
"""

import pathlib
import random
import pandas as pd

BASE = pathlib.Path("/scratch/mziemlewski/speechbrain/recipes/final-report")
CGN_CSV    = BASE / "csv/cgn/train.csv"
JASMIN_CSV = BASE / "csv/jasmin/train.csv"
OUT_DIR    = BASE / "vc_audio"

N_SOURCES = 250
RANDOM_SEED = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_SEED)

# ── 250 CGN source utterances ─────────────────────────────────────────────────
cgn = pd.read_csv(CGN_CSV)
sources = cgn["wav"].sample(n=min(N_SOURCES, len(cgn)), random_state=RANDOM_SEED).tolist()
sources_file = OUT_DIR / "cgn_sources.txt"
sources_file.write_text("\n".join(sources) + "\n")
print(f"[sources] Wrote {len(sources)} CGN paths → {sources_file}")

# ── One reference per JASMIN speaker (longest utterance = best reference) ─────
jasmin = pd.read_csv(JASMIN_CSV)
refs = (
    jasmin
    .sort_values("duration", ascending=False)
    .groupby("spk_id", sort=False)
    .first()
    .reset_index()["wav"]
    .tolist()
)
targets_file = OUT_DIR / "jasmin_targets.txt"
targets_file.write_text("\n".join(refs) + "\n")
print(f"[targets] Wrote {len(refs)} JASMIN speaker references → {targets_file}")
print(f"\nNext step: sbatch jobs/vc_convert.sh")
