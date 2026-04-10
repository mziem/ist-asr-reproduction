#!/bin/bash
#SBATCH --job-name=ist_vc_convert
#SBATCH --partition=gpu-a100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-EEMCS-Courses-DSAIT4095
#SBATCH --mail-type=START,END

# ============================================================
# vc_convert.sh
#
# Converts 250 CGN source utterances into JASMIN non-native
# speaker style using Seed-VC (inference.py), exactly as
# specified in the Week 7 lab instructions.
#
# Prerequisites (done once on the login node):
#   git clone https://github.com/Plachtaa/seed-vc /scratch/mziemlewski/seed-vc
#   conda create -n seedvc python=3.10 -y
#   conda activate seedvc && pip install -r requirements.txt
#   python vc_prep.py   # writes cgn_sources.txt + jasmin_targets.txt
#
# Seed-VC output naming (confirmed from source):
#   vc_{source_stem}_{target_stem}_{length_adjust}_{diffusion_steps}_{inference_cfg_rate}.wav
# With our flags that is:
#   vc_{source_stem}_{target_stem}_1.0_25_0.7.wav
# ============================================================

BASE=/scratch/mziemlewski/speechbrain/recipes/final-report
SEEDVC_DIR=/scratch/mziemlewski/seed-vc
OUT_DIR=${BASE}/vc_audio
SOURCES=${OUT_DIR}/cgn_sources.txt
TARGETS=${OUT_DIR}/jasmin_targets.txt

module load 2025
module load miniconda3
conda activate seedvc
module load cuda/12.9

mkdir -p ${OUT_DIR}

# Sanity check
if [[ ! -f ${SOURCES} || ! -f ${TARGETS} ]]; then
    echo "ERROR: Run vc_prep.py on the login node first to generate source/target lists."
    exit 1
fi

mapfile -t SOURCE_PATHS < "${SOURCES}"
mapfile -t TARGET_PATHS < "${TARGETS}"

N_SOURCES=${#SOURCE_PATHS[@]}
N_TARGETS=${#TARGET_PATHS[@]}
echo "Sources : ${N_SOURCES}"
echo "Targets : ${N_TARGETS}"

# Assign targets to sources by cycling (modulo)
CONVERTED=0
SKIPPED=0
FAILED=0

for i in "${!SOURCE_PATHS[@]}"; do
    SRC="${SOURCE_PATHS[$i]}"
    TGT="${TARGET_PATHS[$((i % N_TARGETS))]}"

    SRC_STEM=$(basename "${SRC}" .wav)
    TGT_STEM=$(basename "${TGT}" .wav)

    # Seed-VC output filename (confirmed naming convention)
    EXPECTED_OUT="${OUT_DIR}/vc_${SRC_STEM}_${TGT_STEM}_1.0_25_0.7.wav"

    if [[ -f "${EXPECTED_OUT}" ]]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Exact call as per Week 7 lab instructions
    cd ${SEEDVC_DIR}
    python inference.py \
        --source "${SRC}" \
        --target "${TGT}" \
        --output "${OUT_DIR}" \
        --diffusion-steps 25 \
        --length-adjust 1.0 \
        --inference-cfg-rate 0.7 \
        --f0-condition False \
        --auto-f0-adjust False \
        --semi-tone-shift 0 \
        --fp16 True

    if [[ $? -ne 0 ]]; then
        echo "WARN: inference.py failed for source=${SRC_STEM}"
        FAILED=$((FAILED + 1))
        continue
    fi

    CONVERTED=$((CONVERTED + 1))
    if (( CONVERTED % 25 == 0 )); then
        echo "[${CONVERTED}/${N_SOURCES}] converted (skipped=${SKIPPED}, failed=${FAILED})"
    fi
done

echo ""
echo "=== Done ==="
echo "Converted : ${CONVERTED}"
echo "Skipped   : ${SKIPPED}  (already existed)"
echo "Failed    : ${FAILED}"
echo "Output    : ${OUT_DIR}"
