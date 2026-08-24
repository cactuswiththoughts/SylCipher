#!/bin/bash
#SBATCH -J train_pusm_libri_clean_vad
#SBATCH -o logs/%j_train_pusm_libri_clean_vad.out
#SBATCH -e logs/%j_train_pusm_libri_clean_vad.err
#SBATCH -p a6
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --qos=regular

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../path.sh"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

tgt_dir=${MANIFEST_ROOT}/librispeech_clean_vad
nclus=2048
CONFIG_NAME=l1_w2vu_${nclus}clus_interleave
topk=2048

seg_name=sylber_bnd
seed=2
TASK_DATA=$tgt_dir/feat_hubert_large_L21_sylber_bnd/top_$topk/km_dir_$topk
TEXT_DATA=$tgt_dir/hierber_xeus_no_noise/top_$topk/km_dir_$topk/text_train
ckpt_dir=$tgt_dir/hierber_xeus_no_noise/multirun/${CONFIG_NAME}_top${topk}_${nclus}clus_seed${seed}
#set -euo pipefail

stage=1
stop_stage=1
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then 
    bash create_data_dir_sylber.sh $topk
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then 
    fairseq-hydra-train \
        -m --config-dir ${SYLCIPHER_ROOT}/train_pusm_libri_clean_vad/config/gan \
        --config-name $CONFIG_NAME \
        dataset.valid_subset=test \
        task.data=$TASK_DATA \
        task.text_data=$TEXT_DATA \
        common.user_dir=${SYLCIPHER_ROOT}/train_pusm_libri_clean_vad \
        model.code_penalty=0 model.gradient_penalty=0.0 \
        model.smoothness_weight=0.0 'common.seed=range(2,3)' \
        checkpoint.save_dir='./' \
        hydra.run.dir=$ckpt_dir \
        hydra.sweep.dir=$ckpt_dir
fi

if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]; then
    DATA_DIR=$TASK_DATA
    CKPT_PATH=$ckpt_dir/0
    if [ ! -f $DATA_DIR/dict.txt ]; then
        ln -s $DATA_DIR/../dict.txt $DATA_DIR/dict.txt
    fi
    echo $CKPT_PATH

    TEXT_PATH=$tgt_dir/train_gradseg_style_oov_1024.json
    python dump_prediction_multilevel_pusm.py $DATA_DIR --ckpt-path $CKPT_PATH/checkpoint_best.pt --user-dir $PWD --text-path $TEXT_PATH --max-text-positions 100 --split "train|train" --postfix "train_layer_0" --use-discrete-labels 0 --tgt-enc-layer 0

    TEXT_PATH=$tgt_dir/test_gradseg_style_oov_1024.json
    python dump_prediction_multilevel_pusm.py $DATA_DIR --ckpt-path $CKPT_PATH/checkpoint_best.pt --user-dir $PWD --text-path $TEXT_PATH --max-text-positions 100 --split "test|test" --postfix "test_layer_0" --use-discrete-labels 0 --tgt-enc-layer 0
fi
