#!/usr/bin/env python3
"""
parse_and_plot.py  (updated — all 5 models + bias chart)

Two modes:
  1. PARSE  — run on the cluster to extract training curves and test results
               into clean CSVs. No display needed, outputs files only.
  2. PLOT   — run locally (or on cluster) with the exported CSVs to produce
               publication-quality figures for the report.

Usage on cluster (parse only):
    python parse_and_plot.py --mode parse

Usage (plot from CSVs, all figures):
    python parse_and_plot.py --mode plot --csv_dir ./results_csv --out_dir ./figures

Usage (parse + plot in one go):
    python parse_and_plot.py --mode both

Dependencies:
    pip install matplotlib pandas seaborn
"""

import argparse
import os
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# CONFIG — all five models
# ---------------------------------------------------------------------------
BASE = "/scratch/mziemlewski/speechbrain/recipes/final-report/results"

EXPERIMENTS = {
    "Baseline": {
        "log":       os.path.join(BASE, "baseline/train_log.txt"),
        "wer_dir":   os.path.join(BASE, "baseline/wer"),
        "test_sets": ["test_bn", "test_cts", "test"],
    },
    "CGN_FT": {
        "log":       os.path.join(BASE, "cgn_ft/train_log.txt"),
        "wer_dir":   os.path.join(BASE, "cgn_ft/wer"),
        "test_sets": ["test_bn", "test_cts", "test"],
    },
    "Atyp_FT": {
        "log":       os.path.join(BASE, "atyp_ft/train_log.txt"),
        "wer_dir":   os.path.join(BASE, "atyp_ft/wer"),
        "test_sets": ["test_bn", "test_cts", "test"],
    },
    "VC": {
        "log":       os.path.join(BASE, "vc_ft/train_log.txt"),
        "wer_dir":   os.path.join(BASE, "vc_ft/wer"),
        "test_sets": ["test_bn", "test_cts", "test"],
    },
    "DA": {
        "log":       os.path.join(BASE, "da_ft/train_log.txt"),
        "wer_dir":   os.path.join(BASE, "da_ft/wer"),
        "test_sets": ["test_bn", "test_cts", "test"],
    },
}

TEST_SET_LABELS = {
    "test_bn":  "CGN-BN",
    "test_cts": "CGN-CTS",
    "test":     "JASMIN",
}

# Baseline is zero-shot — no training log
TRAINED_MODELS = ["CGN_FT", "Atyp_FT", "VC", "DA"]

MODEL_ORDER = ["Baseline", "CGN_FT", "Atyp_FT", "VC", "DA"]

# Colourblind-safe palette
MODEL_COLORS = {
    "Baseline": "#888780",
    "CGN_FT":   "#0072B2",
    "Atyp_FT":  "#D55E00",
    "VC":       "#009E73",
    "DA":       "#CC79A7",
}

MODEL_LABELS = {
    "Baseline": "Baseline",
    "CGN_FT":   "CGN_FT",
    "Atyp_FT":  "Atyp_FT",
    "VC":       "VC",
    "DA":       "DA",
}

# ---------------------------------------------------------------------------
# PARSING
# ---------------------------------------------------------------------------

def parse_train_log(log_path):
    """
    Parse SpeechBrain train_log.txt.
    Returns DataFrame: epoch, train_loss, valid_loss, valid_cer, valid_wer
    """
    rows = []
    epoch_pat = re.compile(
        r"epoch:\s*(\d+).*?train loss:\s*([\d.e+-]+).*?"
        r"valid loss:\s*([\d.e+-]+).*?valid CER:\s*([\d.e+-]+).*?valid WER:\s*([\d.e+-]+)",
        re.IGNORECASE,
    )
    if not os.path.isfile(log_path):
        print(f"  [WARN] Log not found: {log_path}")
        return pd.DataFrame()

    with open(log_path) as f:
        for line in f:
            m = epoch_pat.search(line)
            if m:
                rows.append({
                    "epoch":      int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "valid_loss": float(m.group(3)),
                    "valid_cer":  float(m.group(4)),
                    "valid_wer":  float(m.group(5)),
                })
    return pd.DataFrame(rows)


def parse_wer_file(wer_path):
    """Extract overall WER from a SpeechBrain wer_*.txt file (first %WER line)."""
    wer_pat = re.compile(r"^%WER\s+([\d.]+)")
    if not os.path.isfile(wer_path):
        return None
    with open(wer_path) as f:
        for line in f:
            m = wer_pat.match(line.strip())
            if m:
                return float(m.group(1))
    return None


def parse_all(csv_dir):
    os.makedirs(csv_dir, exist_ok=True)

    # ── Training curves ───────────────────────────────────────────────────────
    curve_rows = []
    for model in TRAINED_MODELS:
        cfg = EXPERIMENTS.get(model, {})
        log = cfg.get("log", "")
        df = parse_train_log(log)
        if df.empty:
            continue
        df["model"] = model
        curve_rows.append(df)
        out = os.path.join(csv_dir, f"train_curve_{model.lower()}.csv")
        df.to_csv(out, index=False)
        print(f"[PARSE] {model} training curve → {out}")

    if curve_rows:
        combined = pd.concat(curve_rows, ignore_index=True)
        combined.to_csv(os.path.join(csv_dir, "all_train_curves.csv"), index=False)
        print(f"[PARSE] All curves combined → {csv_dir}/all_train_curves.csv")

    # ── Test WER results ──────────────────────────────────────────────────────
    result_rows = []
    for model, cfg in EXPERIMENTS.items():
        wer_dir = cfg.get("wer_dir", "")
        for ts in cfg["test_sets"]:
            wer_path = os.path.join(wer_dir, f"wer_{ts}.txt")
            wer = parse_wer_file(wer_path)
            label = TEST_SET_LABELS.get(ts, ts)
            result_rows.append({"model": model, "test_set": label, "wer": wer})
            status = f"{wer:.2f}" if wer is not None else "not found"
            print(f"[PARSE] {model:10s} / {label:8s}: WER={status}")

    results_df = pd.DataFrame(result_rows)
    out = os.path.join(csv_dir, "test_results.csv")
    results_df.to_csv(out, index=False)
    print(f"[PARSE] Test results → {out}")

    # ── Bias metrics (derived from test results) ──────────────────────────────
    bias_rows = []
    for model in MODEL_ORDER:
        sub = results_df[results_df["model"] == model].set_index("test_set")
        wer_bn  = sub.loc["CGN-BN",  "wer"] if "CGN-BN"  in sub.index else None
        wer_jas = sub.loc["JASMIN",  "wer"] if "JASMIN"  in sub.index else None
        if wer_bn is not None and wer_jas is not None:
            abs_bias = round(wer_jas - wer_bn, 2)
            rel_bias = round(abs_bias / wer_bn * 100, 1)
            bias_rows.append({
                "model":      model,
                "wer_bn":     wer_bn,
                "wer_jasmin": wer_jas,
                "abs_bias":   abs_bias,
                "rel_bias":   rel_bias,
            })
            print(f"[PARSE] {model:10s} bias: abs={abs_bias:+.2f}pp  rel={rel_bias:+.1f}%")

    bias_df = pd.DataFrame(bias_rows)
    out = os.path.join(csv_dir, "bias_metrics.csv")
    bias_df.to_csv(out, index=False)
    print(f"[PARSE] Bias metrics → {out}")

    return results_df


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def set_style():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "font.family":       "sans-serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "grid.linewidth":    0.5,
        "grid.alpha":        0.45,
        "legend.frameon":    False,
    })


def plot_training_curves(csv_dir, out_dir):
    """Validation loss + WER per epoch for all fine-tuned models."""
    curve_path = os.path.join(csv_dir, "all_train_curves.csv")
    if not os.path.isfile(curve_path):
        print("[PLOT] No training curve CSV found, skipping.")
        return

    df = pd.read_csv(curve_path)
    models = [m for m in TRAINED_MODELS if m in df["model"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for model in models:
        sub = df[df["model"] == model].sort_values("epoch")
        color = MODEL_COLORS.get(model, "#333")
        label = MODEL_LABELS.get(model, model)
        axes[0].plot(sub["epoch"], sub["valid_loss"],
                     marker="o", markersize=4, label=label,
                     color=color, linewidth=1.8)
        axes[1].plot(sub["epoch"], sub["valid_wer"],
                     marker="o", markersize=4, label=label,
                     color=color, linewidth=1.8)

    for ax, ylabel, title in zip(
        axes,
        ["Validation loss",    "Validation WER (%)"],
        ["Validation loss per epoch", "Validation WER per epoch"],
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, fontweight="normal", pad=8)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(fontsize=9)

    fig.tight_layout()
    for ext in (".pdf", ".png"):
        out = os.path.join(out_dir, f"training_curves{ext}")
        fig.savefig(out)
        print(f"[PLOT] Training curves → {out}")
    plt.close(fig)


def plot_wer_grouped_bar(csv_dir, out_dir):
    """Grouped bar chart: WER per model × test set, with value labels."""
    results_path = os.path.join(csv_dir, "test_results.csv")
    if not os.path.isfile(results_path):
        print("[PLOT] No test results CSV found, skipping.")
        return

    df = pd.read_csv(results_path).dropna(subset=["wer"])
    models_present = [m for m in MODEL_ORDER if m in df["model"].values]
    test_sets = ["CGN-BN", "CGN-CTS", "JASMIN"]

    fig, ax = plt.subplots(figsize=(9, 5))
    n_models = len(models_present)
    width = 0.8 / n_models
    x = range(len(test_sets))

    for i, model in enumerate(models_present):
        sub = df[df["model"] == model].set_index("test_set")
        vals = [sub.loc[ts, "wer"] if ts in sub.index else 0 for ts in test_sets]
        offsets = [xi + (i - n_models / 2 + 0.5) * width for xi in x]
        bars = ax.bar(offsets, vals, width=width * 0.9,
                      color=MODEL_COLORS.get(model, "#aaa"),
                      label=MODEL_LABELS.get(model, model), zorder=3)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(list(x))
    ax.set_xticklabels(test_sets, fontsize=11)
    ax.set_ylabel("WER (%)")
    ax.set_title("Word Error Rate by model and test set",
                 fontsize=11, fontweight="normal", pad=10)
    ax.set_ylim(0, df["wer"].max() * 1.18)
    ax.legend(fontsize=9, loc="upper left")
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    for ext in (".pdf", ".png"):
        out = os.path.join(out_dir, f"wer_grouped_bar{ext}")
        fig.savefig(out)
        print(f"[PLOT] WER grouped bar → {out}")
    plt.close(fig)


def plot_wer_heatmap(csv_dir, out_dir):
    """Heatmap: WER for all models × all test sets."""
    results_path = os.path.join(csv_dir, "test_results.csv")
    if not os.path.isfile(results_path):
        return

    df = pd.read_csv(results_path).dropna(subset=["wer"])
    models_present = [m for m in MODEL_ORDER if m in df["model"].values]
    pivot = df.pivot_table(index="model", columns="test_set", values="wer")
    pivot = pivot.reindex(models_present)[["CGN-BN", "CGN-CTS", "JASMIN"]]

    fig, ax = plt.subplots(figsize=(6, 0.75 * len(models_present) + 1.5))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "WER (%)", "shrink": 0.8},
        ax=ax,
    )
    ax.set_title("WER (%) — all models × all test sets",
                 fontsize=11, fontweight="normal", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    for ext in (".pdf", ".png"):
        out = os.path.join(out_dir, f"wer_heatmap{ext}")
        fig.savefig(out)
        print(f"[PLOT] WER heatmap → {out}")
    plt.close(fig)


def plot_bias(csv_dir, out_dir):
    """
    Dual-axis bar chart: absolute bias (pp) and relative bias (%)
    for all models, parsed from bias_metrics.csv.
    """
    bias_path = os.path.join(csv_dir, "bias_metrics.csv")
    if not os.path.isfile(bias_path):
        print("[PLOT] No bias CSV found, skipping.")
        return

    df = pd.read_csv(bias_path)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values("model").dropna(subset=["abs_bias", "rel_bias"])

    labels = [MODEL_LABELS.get(m, m) for m in df["model"]]
    x = range(len(df))
    w = 0.38

    fig, ax1 = plt.subplots(figsize=(6, 3.8))
    ax2 = ax1.twinx()

    bars1 = ax1.bar([xi - w / 2 for xi in x], df["abs_bias"], w,
                    color="#0072B2", alpha=0.85, label="Abs. bias (pp)", zorder=3)
    bars2 = ax2.bar([xi + w / 2 for xi in x], df["rel_bias"], w,
                    color="#D55E00", alpha=0.85, label="Rel. bias (%)", zorder=3)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.4,
                 f"{bar.get_height():.1f}", ha="center", va="bottom",
                 fontsize=8, color="#0072B2")
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.5,
                 f"{bar.get_height():.1f}", ha="center", va="bottom",
                 fontsize=8, color="#D55E00")

    ax1.set_ylabel("Absolute bias (pp)", color="#0072B2")
    ax2.set_ylabel("Relative bias (%)",  color="#D55E00")
    ax1.tick_params(axis="y", labelcolor="#0072B2")
    ax2.tick_params(axis="y", labelcolor="#D55E00")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_title("ASR bias: JASMIN vs CGN-BN",
                  fontsize=11, fontweight="normal", pad=8)
    ax1.yaxis.grid(True, linewidth=0.5, alpha=0.45, zorder=0)
    ax1.set_axisbelow(True)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    fig.tight_layout()
    for ext in (".pdf", ".png"):
        out = os.path.join(out_dir, f"bias_chart{ext}")
        fig.savefig(out)
        print(f"[PLOT] Bias chart → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["parse", "plot", "both"], default="both")
    parser.add_argument("--csv_dir", default="./results_csv",
                        help="Directory for CSV outputs (parse) or inputs (plot)")
    parser.add_argument("--out_dir", default="./figures",
                        help="Directory for figure outputs")
    args = parser.parse_args()

    if args.mode in ("parse", "both"):
        print(f"\n[MODE] Parsing → {args.csv_dir}")
        parse_all(args.csv_dir)

    if args.mode in ("plot", "both"):
        os.makedirs(args.out_dir, exist_ok=True)
        set_style()
        print(f"\n[MODE] Plotting → {args.out_dir}")
        plot_training_curves(args.csv_dir, args.out_dir)
        plot_wer_grouped_bar(csv_dir=args.csv_dir, out_dir=args.out_dir)
        plot_wer_heatmap(csv_dir=args.csv_dir, out_dir=args.out_dir)
        plot_bias(csv_dir=args.csv_dir, out_dir=args.out_dir)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
