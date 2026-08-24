#!/bin/bash
#SBATCH -J eval_wav2bound_scoco
#SBATCH -o logs/%j_eval_wav2bound_scoco.out
#SBATCH -e logs/%j_eval_wav2bound_scoco.err
#SBATCH -p a6
#SBATCH --qos=regular
#SBATCH --gres=gpu:1
#SBATCH -n 1

export HOME=/data/sls/scratch/limingw
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate wav2boundaries
ROOT=$HOME/workplace/VGUnsupASR

N=1
s=1
e=1
prom=0.2
height=0.22
if [ $s -le 0 ] && [ $e -ge 0 ]; then
    SAVE_DIR=$ROOT/wav2boundaries/dataset/spokencoco/train_wav2boundaries_sylber_bnd_2048_oov_e2e_refinement/checkpoints_xlsr_retry_loss_quant_0.3/
    cd ../../wav2boundaries

    VAD_FILE=$ROOT/wav2boundaries/miscellaneous/vads/spokencoco_val2
    OUTPUT_DIR=$ROOT/wav2boundaries/dataset/spokencoco/train_wav2boundaries_sylber_bnd_2048_oov_e2e_refinement/val2_prom_${prom}_${height}_1_real/

    mkdir -p $OUTPUT_DIR

    python infere_tunable.py $N ${VAD_FILE} ${SAVE_DIR}/checkpoint_best.pt ${OUTPUT_DIR}/ $prom $height 
    cat $OUTPUT_DIR/* > $OUTPUT_DIR/all_word_boundaries.txt

    VAD_FILE=$ROOT/wav2boundaries/miscellaneous/vads/spokencoco
    OUTPUT_DIR=$ROOT/wav2boundaries/dataset/spokencoco/train_wav2boundaries_sylber_bnd_2048_oov_e2e_refinement/train_prom_${prom}_${height}_1_real/

    mkdir -p $OUTPUT_DIR

    python infere_tunable.py $N ${VAD_FILE} ${SAVE_DIR}/checkpoint_best.pt ${OUTPUT_DIR}/ $prom $height
    cat $OUTPUT_DIR/* > $OUTPUT_DIR/all_word_boundaries.txt

    mkdir -p $ROOT/wav2boundaries/dataset/spokencoco/train_wav2boundaries_sylber_bnd_2048_oov_e2e_refinement/missing_prom_${prom}_${height}_1_real/
    touch $ROOT/wav2boundaries/dataset/spokencoco/train_wav2boundaries_sylber_bnd_2048_oov_e2e_refinement/missing_prom_${prom}_${height}_1_real/all_word_boundaries.txt
fi

if [ $s -le 1 ] && [ $e -ge 1 ]; then
    python extract_features/convert_wavbnd_outs_to_gradseg_style.py \
        --word_boundaries_txt $ROOT/wav2boundaries/dataset/spokencoco/train_wav2boundaries_sylber_bnd_2048_oov_e2e_refinement/{}_prom_0.2_0.33_1_real/all_word_boundaries.txt \
        --backoff_dir $ROOT/manifest/spokencoco/without_silence/sylber_bnd \
        --length_dir $ROOT/manifest/spokencoco/without_silence/feat/hubert_large_ll60k_sylber_bnd/top_2048/km_dir_2048/discrete_speech_{}_clsattnbndkmeans \
        --output_dir $ROOT/wav2boundaries/dataset/spokencoco/train_wav2boundaries_sylber_bnd_2048_oov_e2e_refinement \
        --splits train
#        --splits val2
#        --splits val2,train
fi
