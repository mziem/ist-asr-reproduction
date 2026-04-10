#!/bin/bash
#SBATCH --job-name=ist_cgn_ft
#SBATCH --partition=gpu-a100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-EEMCS-Courses-DSAIT4095
#SBATCH --mail-type=START,END     # Email notifications for job status

mkdir -p /scratch/mziemlewski/speechbrain/recipes/final-report/logs

module load 2025
module load miniconda3
conda activate IST-ASR
module load cuda/12.9

cd /scratch/mziemlewski/speechbrain/recipes/final-report

python train_with_whisper.py hparams/cgn_ft.yaml
