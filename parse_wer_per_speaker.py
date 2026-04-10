#!/usr/bin/env python3
"""
parse_wer_per_speaker.py

Parses SpeechBrain wer_*.txt files (alignment format) and produces:
  1. Per-speaker CSVs for the paired bootstrap significance test.
  2. A bias_metrics.csv summarising absolute and relative WER bias
     (JASMIN vs CGN-BN) per model.

SpeechBrain WER file format (alignment block):
    <utt_id>, %WER <wer> [ <errors> / <words>, <ins> ins, <del> del, <sub> sub ]

Run on DelftBlue login node (no GPU needed):
    python parse_wer_per_speaker.py

Outputs → results/bootstrap_input/
Copy locally:
    scp <netid>@login.delftblue.tudelft.nl:<BASE>/results/bootstrap_input/*.csv ./bootstrap_input/
"""

import csv
import os
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE = "/scratch/mziemlewski/speechbrain/recipes/final-report/results"

WER_DIRS = {
    "Baseline": os.path.join(BASE, "baseline/wer"),
    "CGN_FT":   os.path.join(BASE, "cgn_ft/wer"),
    "Atyp_FT":  os.path.join(BASE, "atyp_ft/wer"),
    "VC_FT":    os.path.join(BASE, "vc_ft/wer"),
    "DA_FT":    os.path.join(BASE, "da_ft/wer"),
}

TEST_SETS = ["test_bn", "test_cts", "test"]

OUTPUT_DIR = os.path.join(BASE, "bootstrap_input")

# ---------------------------------------------------------------------------
# Regex for utterance-level WER lines, e.g.:
#   N000225-fn000593-226.6766-227.1819, %WER 100.00 [ 1 / 1, 0 ins, 0 del, 1 sub ]
#   N02005-fn005921.6, %WER 75.00 [ 3 / 4, 0 ins, 0 del, 3 sub ]
# ---------------------------------------------------------------------------
UTT_PAT = re.compile(
    r"^(\S+),\s+%WER\s+[\d.]+\s+\[\s*(\d+)\s*/\s*(\d+)"
)


def extract_spk_id(utt_id):
    """
    Extract speaker ID from utterance ID.
    JASMIN:  N000225-fn000593-226.6766-227.1819  -> N000225
    CGN-BN:  N02005-fn005921.6                   -> N02005
    CGN-CTS: N08012-fn008183.126                 -> N08012
    Strategy: first hyphen-delimited token is always the speaker ID.
    """
    return utt_id.split("-")[0]


def parse_wer_file(wer_path):
    """
    Returns {spk_id: {"errors": int, "words": int}}.
    """
    spk_stats = defaultdict(lambda: {"errors": 0, "words": 0})
    if not os.path.isfile(wer_path):
        print(f"  [WARN] File not found: {wer_path}")
        return spk_stats

    with open(wer_path, encoding="utf-8") as f:
        for line in f:
            m = UTT_PAT.match(line.strip())
            if not m:
                continue
            utt_id = m.group(1)
            errors = int(m.group(2))
            words  = int(m.group(3))
            spk_id = extract_spk_id(utt_id)
            spk_stats[spk_id]["errors"] += errors
            spk_stats[spk_id]["words"]  += words

    return spk_stats


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bias_table = {m: {} for m in WER_DIRS}

    for model, wer_dir in WER_DIRS.items():
        for ts in TEST_SETS:
            wer_path = os.path.join(wer_dir, f"wer_{ts}.txt")
            print(f"[PARSE] {model} / {ts}", file=sys.stderr)
            spk_stats = parse_wer_file(wer_path)

            if not spk_stats:
                print(f"  [SKIP] No utterance lines found in {wer_path}", file=sys.stderr)
                continue

            total_words  = sum(s["words"]  for s in spk_stats.values())
            total_errors = sum(s["errors"] for s in spk_stats.values())
            wer = 100.0 * total_errors / total_words if total_words else float("nan")

            bias_table[model][ts] = (total_errors, total_words, wer)

            out_name = f"{model}_{ts}_per_speaker.csv"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["spk_id", "word_count", "errors"])
                for spk_id, s in sorted(spk_stats.items()):
                    writer.writerow([spk_id, s["words"], s["errors"]])

            print(f"  -> {len(spk_stats):3d} speakers | WER = {wer:6.2f}% | {out_path}", file=sys.stderr)

    # Bias metrics
    print("\n" + "=" * 60, file=sys.stderr)
    print("  BIAS METRICS  (JASMIN vs CGN-BN, read speech)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  {'Model':<12} {'WER BN':>8} {'WER JASMIN':>11} {'Abs bias':>10} {'Rel bias':>10}", file=sys.stderr)
    print(f"  {'-'*54}", file=sys.stderr)

    bias_rows = []
    for model in WER_DIRS:
        if "test_bn" not in bias_table[model] or "test" not in bias_table[model]:
            continue
        _, _, wer_bn     = bias_table[model]["test_bn"]
        _, _, wer_jasmin = bias_table[model]["test"]
        abs_bias = wer_jasmin - wer_bn
        rel_bias = 100.0 * abs_bias / wer_bn if wer_bn else float("nan")
        print(f"  {model:<12} {wer_bn:8.2f} {wer_jasmin:11.2f} {abs_bias:+10.2f} {rel_bias:+9.1f}%", file=sys.stderr)
        bias_rows.append({
            "model":        model,
            "wer_cgn_bn":   round(wer_bn, 2),
            "wer_jasmin":   round(wer_jasmin, 2),
            "abs_bias_pp":  round(abs_bias, 2),
            "rel_bias_pct": round(rel_bias, 1),
        })

    bias_path = os.path.join(OUTPUT_DIR, "bias_metrics.csv")
    if bias_rows:
        with open(bias_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=bias_rows[0].keys())
            writer.writeheader()
            writer.writerows(bias_rows)
        print(f"\n  Bias table saved -> {bias_path}", file=sys.stderr)

    print("\n[DONE]", file=sys.stderr)
    print(f"Copy CSVs locally:", file=sys.stderr)
    print(f"  scp <netid>@login.delftblue.tudelft.nl:{OUTPUT_DIR}/*.csv ./bootstrap_input/", file=sys.stderr)


if __name__ == "__main__":
    main()
