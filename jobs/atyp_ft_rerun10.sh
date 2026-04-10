#!/bin/bash
#SBATCH --job-name=ist_atyp_ft_rerun10
#SBATCH --partition=gpu-a100
#SBATCH --time=08:00:00
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

python train_with_whisper.py hparams/atyp_ft_rerun10.yaml
