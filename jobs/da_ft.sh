#!/bin/bash
#SBATCH --job-name=ist_da_ft
#SBATCH --partition=gpu-a100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-EEMCS-Courses-DSAIT4095
#SBATCH --mail-type=START,END

# DA training is slightly longer than Atyp_FT because concat_original=True
# effectively 4x the batch per step (original + 3 augmented copies).
# 8h wall time is conservative; adjust if needed.

module load 2025
module load miniconda3
conda activate IST-ASR
module load cuda/12.9

cd /scratch/mziemlewski/speechbrain/recipes/final-report

python train_with_whisper.py hparams/da_ft.yaml
