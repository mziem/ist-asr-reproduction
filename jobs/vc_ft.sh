#!/bin/bash
#SBATCH --job-name=ist_vc_ft
#SBATCH --partition=gpu-a100
#SBATCH --time=07:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-EEMCS-Courses-DSAIT4095
#SBATCH --mail-type=START,END


module load 2025
module load miniconda3
conda activate IST-ASR
module load cuda/12.9

cd /scratch/mziemlewski/speechbrain/recipes/final-report

python train_with_whisper.py hparams/vc_ft.yaml
