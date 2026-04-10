#!/bin/bash
#SBATCH --job-name=ist_mbnet2
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-EEMCS-Courses-DSAIT4095
#SBATCH --mail-type=START,END

# ============================================================
# mbnet_eval.sh
#
# Evaluates VC-generated speech quality using MBNet MOS prediction.
#
# One-time setup on the LOGIN NODE before submitting this job:
#   cd /scratch/mziemlewski/
#   git clone https://github.com/sky1456723/Pytorch-MBNet.git
#   module load 2025 && module load miniconda3
#   conda create -n mbnet python=3.10 -y
#   conda activate mbnet
#   pip install numpy pandas scipy torch tqdm tensorboard librosa soundfile
#
# The pre-trained checkpoint (model-50000.pt) is downloaded automatically
# by MBNet on first run if not present, OR copy it manually to:
#   /scratch/mziemlewski/Pytorch-MBNet/pre_trained/model-50000.pt
# ============================================================

NETID=mziemlewski
MBNET_DIR=/scratch/${NETID}/Pytorch-MBNet
VC_AUDIO=/scratch/${NETID}/speechbrain/recipes/final-report/vc_audio
SAMPLE_DIR=${MBNET_DIR}/data/wav
RESULTS_DIR=${MBNET_DIR}/results

module load 2025
module load miniconda3
conda activate mbnet
module load cuda/12.9

mkdir -p ${SAMPLE_DIR} ${RESULTS_DIR}

# ── Copy a random 100-utterance sample of VC files for MOS evaluation ────────
python3 - <<'EOF'
import pathlib, random, shutil

random.seed(42)
src = pathlib.Path("/scratch/mziemlewski/speechbrain/recipes/final-report/vc_audio")
dst = pathlib.Path("/scratch/mziemlewski/Pytorch-MBNet/data/wav")
dst.mkdir(parents=True, exist_ok=True)

vc_files = list(src.glob("vc_*.wav"))
sample = random.sample(vc_files, min(100, len(vc_files)))

for f in sample:
    target = dst / f.name
    if not target.exists():
        shutil.copy2(f, target)

print(f"Copied {len(sample)} VC wav files to {dst}")
EOF

# ── Run MBNet inference ───────────────────────────────────────────────────────
python ${MBNET_DIR}/inference.py \
    --data_path  ${SAMPLE_DIR} \
    --model_path ${MBNET_DIR}/pre_trained/model-50000.pt \
    --save_dir   ${RESULTS_DIR}

echo "MBNet done. Check ${RESULTS_DIR} for predict_mean.csv"

# ── Print summary statistics ──────────────────────────────────────────────────
python3 - <<'EOF'
import pathlib, glob, pandas as pd

results_dir = pathlib.Path("/scratch/mziemlewski/Pytorch-MBNet/results")
# Find the most recent timestamped subfolder
folders = sorted(results_dir.glob("*/"))
if not folders:
    print("No result folders found.")
else:
    latest = folders[-1]
    csv_path = latest / "predict_mean.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Column names vary by MBNet version — try common ones
        score_col = next((c for c in df.columns if "mean" in c.lower() or "mos" in c.lower()), df.columns[-1])
        scores = df[score_col].dropna()
        print(f"\n=== MBNet MOS Summary (n={len(scores)}) ===")
        print(f"  Mean  : {scores.mean():.3f}")
        print(f"  Std   : {scores.std():.3f}")
        print(f"  Min   : {scores.min():.3f}")
        print(f"  Max   : {scores.max():.3f}")
        print(f"\nFull results: {csv_path}")
    else:
        print(f"predict_mean.csv not found in {latest}")
EOF
