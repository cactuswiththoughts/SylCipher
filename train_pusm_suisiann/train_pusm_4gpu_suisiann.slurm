#!/bin/bash
#SBATCH -J train_pusm_suisiann_vad
#SBATCH -o logs/%j_train_pusm_suisiann_vad.out
#SBATCH -e logs/%j_train_pusm_suisiann_vad.err
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --qos=regular

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../path.sh"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

tgt_dir=${MANIFEST_ROOT}/suisiann_vad
CONFIG_NAME=l1_w2vu_1024clus
feat_name=xeus_L18
topk=1024
seg_name=sylber_ft_bnd_aishell3
TASK_DATA=$tgt_dir/feat_${feat_name}_${seg_name}/top_$topk/km_dir_$topk
TEXT_DATA=$tgt_dir/feat_${feat_name}_${seg_name}/top_$topk/km_dir_$topk/text_train
ckpt_dir=$tgt_dir/feat_${feat_name}_${seg_name}/multirun/${CONFIG_NAME}_no_tone

stage=10
stop_stage=10
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    python ${SYLCIPHER_ROOT}/train_sylcipher_mdcc_char/utils/prepare_suisiann.py \
        --audio-dir ${tgt_dir}/wavs \
        --text-path ${DATA_ROOT}/SuiSiann/0.2.1/SuiSiann.csv \
        --tsv-path ${MANIFEST_ROOT}/suisiann/train.tsv \
        --out-path ${MANIFEST_ROOT}/suisiann/train_gradseg_style.json \
        --bad-path ${MANIFEST_ROOT}/suisiann/train_bad_files.txt \
        --no-tone False
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    bash create_data_dir_sylber.sh $seg_name $topk
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    fairseq-hydra-train \
        -m --config-dir ${SYLCIPHER_ROOT}/train_pusm_suisiann/config/gan \
        --config-name $CONFIG_NAME \
        task.data=$TASK_DATA \
        task.text_data=$TEXT_DATA \
        common.user_dir=${SYLCIPHER_ROOT}/train_pusm_suisiann \
        model.code_penalty=0 model.gradient_penalty=0.0 \
        model.smoothness_weight=0.0 'common.seed=range(0,1)' \
        checkpoint.save_dir='./' \
        hydra.run.dir=$ckpt_dir \
        hydra.sweep.dir=$ckpt_dir
fi

if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]; then
    DATA_DIR=$TASK_DATA
    CKPT_PATH=$ckpt_dir/0
    if [ ! -f $DATA_DIR/dict.txt ]; then
        ln -sf $DATA_DIR/../dict.txt $DATA_DIR/dict.txt
    fi
    echo $CKPT_PATH

    TEXT_PATH=$tgt_dir/train_gradseg_style_oov_1024.json
    python dump_prediction_multilevel_pusm.py $DATA_DIR --ckpt-path $CKPT_PATH/checkpoint_best.pt --user-dir $PWD --text-path $TEXT_PATH --max-text-positions 100 --split "train|train" --postfix "train_layer_0" --use-discrete-labels 0 --tgt-enc-layer 0

    TEXT_PATH=$tgt_dir/test_gradseg_style_oov_1024.json
    python dump_prediction_multilevel_pusm.py $DATA_DIR --ckpt-path $CKPT_PATH/checkpoint_best.pt --user-dir $PWD --text-path $TEXT_PATH --max-text-positions 100 --split "test|test" --postfix "test_layer_0" --use-discrete-labels 0 --tgt-enc-layer 0
fi
