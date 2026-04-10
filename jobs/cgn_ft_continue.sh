#!/bin/bash
#SBATCH --job-name=ist_cgn_ft_continue
#SBATCH --partition=gpu-a100
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000MB
#SBATCH --account=Education-EEMCS-Courses-DSAIT4095
#SBATCH --mail-type=START,END     # Email notifications for job status

module load 2025
module load miniconda3
conda activate IST-ASR
module load cuda/12.9

cd /scratch/mziemlewski/speechbrain/recipes/final-report

# SpeechBrain reads save_folder from the YAML and automatically
# finds + loads the latest checkpoint. No extra flags needed.
python train_with_whisper.py hparams/cgn_ft_continue.yaml
