#!/bin/bash
#SBATCH -J train_sylcipher_aishell3_mdcc_suisiann_transfer
#SBATCH -o logs/%j_train_sylcipher_aishell3_mdcc_suisiann_transfer.out
#SBATCH -e logs/%j_train_sylcipher_aishell3_mdcc_suisiann_transfer.err
##SBATCH -p 2080
#SBATCH -p a5
#SBATCH --gres=gpu:1
##SBATCH -p a6
##SBATCH --gres=gpu:4
#SBATCH -n 1
#SBATCH --qos=regular
##SBATCH --qos=priority
#SBATCH --requeue

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../path.sh"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE

####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

topk=2063
DATA_ROOT="${MANIFEST_ROOT}/aishell3_mdcc_suisiann/feat_xeus_L18_sylber_bnd/top_2048/km_dir_2048"
SAVE_DIR="${MANIFEST_ROOT}/aishell3_mdcc_suisiann/feat_xeus_L18_sylber_bnd/top_2048/km_dir_2048/sylcipher_transfer_aishell3_mdcc_suisiann_finetuned"
TRAIN_SET="discrete_speech_train_clsattnbndkmeans|text_train"
VALID_SET="discrete_speech_test_clsattnbndkmeans|text_test"
INIT_CLUS_DIR="${MANIFEST_ROOT}/aishell3_mdcc_suisiann/feat_xeus_L18_sylber_bnd/top_2048/km_dir_2048/seg_feats/km_model.npy"
INIT_MODEL_DIR="${MANIFEST_ROOT}/aishell3_mdcc_suisiann/feat_xeus_L18_sylber_bnd/top_2048/km_dir_2048/sylcipher_transfer_aishell3_mdcc_suisiann/checkpoint_best.pt"
USER_DIR="${SYLCIPHER_TL_USER_DIR}"
DUMP_DIR="${SYLCIPHER_ROOT}/SpeechT5/SpeechSyllableT5_transfer_learning"
lr=2e-4

stage=1
stop_stage=1
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]
then
cwd=$PWD
python "${SCRIPT_DIR}/transfer_weights.py" \
    --tgt_dir ${MANIFEST_ROOT}/suisiann_vad/feat_xeus_L18_sylber_ft_bnd/top_1024/km_dir_1024 \
    --src_dir ${MANIFEST_ROOT}/aishell_3_vad_pinyin_no_tone/feat_xeus_L18_sylber_ft_stage2_bnd/top_2048/km_dir_2048 \
    --ckpt_path ${MANIFEST_ROOT}/aishell_3_vad_pinyin_no_tone/feat_xeus_L18_sylber_ft_stage2_bnd/top_2048/km_dir_2048/model_codebook_0.3_encoder_only_sylber_bnd_diff_bnd_with_uniskip_v2/checkpoint_best.pt \
    --out_ckpt_path ${MANIFEST_ROOT}/suisiann_vad/feat_xeus_L18_sylber_ft_bnd/top_1024/km_dir_1024/sylcipher_transfer_aishell_3/checkpoint_best.pt
cd $cwd
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]
then
cwd=$PWD
python "${SCRIPT_DIR}/combine_weights.py"
cd $cwd
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]
then
text_path=$DATA_ROOT/..
target_dir=$DATA_ROOT
cp $DATA_ROOT/dict.txt $DATA_ROOT/..
rm -r $target_dir/text_train
mkdir -p $target_dir/text_train
python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $text_path/train_words.txt --only-source --destdir $target_dir/text_train --srcdict $text_path/dict.txt
mv $target_dir/text_train/train.bin $target_dir/text_train/text_train.bin
mv $target_dir/text_train/train.idx $target_dir/text_train/text_train.idx

for i in {0..2}
do
rm -r $target_dir/text_test$i
mkdir -p $target_dir/text_test${i}
python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $text_path/test${i}_words.txt --only-source --destdir $target_dir/text_test${i} --srcdict $text_path/dict.txt
mv $target_dir/text_test${i}/train.bin $target_dir/text_test${i}/text_test${i}.bin
mv $target_dir/text_test${i}/train.idx $target_dir/text_test${i}/text_test${i}.idx
done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]
then
cwd=$PWD
cd $DUMP_DIR
DATA_DIR=$DATA_ROOT
CKPT_PATH=$(dirname $INIT_MODEL_DIR)
TEXT_PATH=${DATA_ROOT}/discrete_speech_test0_clsattnbndkmeans/meta_data.json

echo $CKPT_PATH
echo $TEXT_PATH
python dump_prediction_multilevel.py $DATA_DIR --ckpt-path $CKPT_PATH/checkpoint_best.pt --user-dir $USER_DIR --text-path $TEXT_PATH --max-text-positions 100 --split "discrete_speech_test0_clsattnbndkmeans|text_test0" --postfix "test0_layer_0" --use-discrete-labels 1 --tgt-enc-layer 0 
cd $cwd
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]
then
cwd=$PWD
cd $DUMP_DIR
DATA_DIR=$DATA_ROOT
CKPT_PATH=$(dirname $INIT_MODEL_DIR)
TEXT_PATH=${DATA_ROOT}/discrete_speech_test1_clsattnbndkmeans/meta_data.json

echo $CKPT_PATH
echo $TEXT_PATH
python dump_prediction_multilevel.py $DATA_DIR --ckpt-path $CKPT_PATH/checkpoint_best.pt --user-dir $USER_DIR --text-path $TEXT_PATH --max-text-positions 100 --split "discrete_speech_test1_clsattnbndkmeans|text_test1" --postfix "test1_layer_0" --use-discrete-labels 1 --tgt-enc-layer 0 
cd $cwd
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]
then
srun fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --distributed-world-size 4 \
  --distributed-port 12345 \
  --no-epoch-checkpoints \
  --ddp-backend legacy_ddp \
  --user-dir $USER_DIR \
  --log-format json \
  --seed 0 \
  \
  --task speecht5 \
  --t5-task discrete_pretrain_gan \
  --label-rates 1 \
  --sample-rate 1 \
  \
  --num-workers 0 \
  --max-tokens 1000 \
  --encoder-layers 2 \
  --max-speech-sample-size 100 \
  --update-freq 16 \
  --batch-ratio "[1,1]" \
  \
  --criterion speecht5 \
  --gan-loss-weight 0 \
  --gradient-penalty 0 \
  --entropy-penalty 0 \
  --entropy-threshold 1 \
  --optimizer adam \
  --adam-betas "(0.9, 0.98)" \
  --adam-eps 1e-06 \
  --weight-decay 0 \
  --power 1 \
  --clip-norm 20.0 \
  --lr $lr \
  --lr-scheduler polynomial_decay \
  --mask 0.3 \
  --mask-random 0.1 \
  \
  --max-update 1000000 \
  --warmup-updates 150 \
  --total-num-update 1000000 \
  --skip-invalid-size-inputs-valid-test \
  --required-batch-size-multiple 1 \
  \
  --arch t5_transformer_base_1_layer \
  --share-input-output-embed \
  --find-unused-parameters \
  --bert-init \
  --relative-position-embedding \
  --loss-weights="[10,0.1]" \
  --bart-weight 1.0 \
  --masked-lm-weight 0.0 \
  --unigram-weight 0.0 \
  --max-text-positions 100 \
  --discriminator-depth 3 \
  --discriminator-kernel 8 \
  --discriminator-causal \
  --start-gan-updates 1000000000000 \
  --policy-pretrain-weight 1 \
  --policy-loss-weight 0 \
  --init-clus-dir $INIT_CLUS_DIR \
  --init-model-dir $INIT_MODEL_DIR \
  --use-valid-s2t \
  --post-process "none" \
  --report-similarity \
  --similarity-meanpool \
  --ce-weight 0 \
  --ctc-weight 1 \
  --report-accuracy \
  --sampling-nums 1 \
  --km-size $topk \
  --frame-target-classes 100 \
  --latent-vars 100 \
  --latent-groups 3 \
  --word-freq 16 \
  --word-freq-weight 500 \
  --speech-encoder-prenet-mask 0.3 \
  --speech-encoder-prenet-mask-random  0.1 \
  --speech-encoder-prenet-poisson-lambda 3.5 \
  --speech-encoder-prenet-mask-length "span-poisson" \
  --speech-encoder-prenet-replace-length -1 \
  --replace-length -1 \
  --code-loss-weight 1.0 \
  --word-count-loss-weight 500000 \
  --text-loss-ratio 1 \
  --frame-target-loss-weight 1.0 \
  --policy-pretrain-pos-weight-bce 1.5 \
  --use-codebook \
  --codebook-prob 0.3 \
  --boundary-sum-multiply 1 \
  --policy-feat-dim 1024 \
  --mlm-unmasked-weight 0.5 \
  --no-load-policy \
  --save-interval 50 \
  --reset-optimizer \
  --reset-lr-scheduler
  # --no-load-quantizer \
  # --no-load-policy \
fi
