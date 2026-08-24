#!/bin/bash
#SBATCH -J train_sylcipher_mdcc_transfer_aux_tgt
#SBATCH -o logs/%j_train_sylcipher_mdcc_transfer_aux_tgt.out
#SBATCH -e logs/%j_train_sylcipher_mdcc_transfer_aux_tgt.err
##SBATCH -p a5
##SBATCH --gres=gpu:1
#SBATCH -p a6
#SBATCH --gres=gpu:4
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

topk=1680
#topk=1024
DATA_ROOT="${MANIFEST_ROOT}/mdcc_vad/feat_xeus_L18_sylber_bnd/top_$topk/km_dir_$topk"
SAVE_DIR="${MANIFEST_ROOT}/mdcc_vad/feat_xeus_L18_sylber_bnd/top_$topk/km_dir_$topk/sylcipher_transfer_interleaved_pusm"
TRAIN_SET="discrete_speech_train_clsattnbndkmeans|text_train"
VALID_SET="discrete_speech_test_clsattnbndkmeans|text_test"
INIT_CLUS_DIR="${MANIFEST_ROOT}/mdcc_vad/feat_xeus_L18_sylber_bnd/top_$topk/km_dir_$topk/seg_feats/km_model.npy"
USER_DIR=${SYLCIPHER_TL_USER_DIR}
lr=2e-4

stage=10
stop_stage=10
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]
then
    # TARGET_PATH=/data/sls/scratch/limingw/junruin2_satori/data/hubert/aishell_3_vad_pinyin_no_tone/feat_xeus_L18_sylber_ft_stage2_bnd/top_2048/km_dir_2048/model_codebook_0.3_encoder_only_sylber_bnd_diff_bnd_with_uniskip/predictions_with_fn
    # TARGET_PATH=${MANIFEST_ROOT}/mdcc_vad/feat_xeus_L18_sylber_bnd/multirun/l1_w2vu_${topk}clus_no_tone_xeus_L18_top${topk}_${topk}clus_v2/0/predictions_with_fn
    TARGET_PATH=${MANIFEST_ROOT}/mdcc_vad/feat_xeus_L18_sylber_bnd/multirun/l1_w2vu_1680clus_with_tone_xeus_L18_top1680_1680clus/0/predictions_with_fn
    ln -sf $TARGET_PATH/train_layer_0_preds.txt $DATA_ROOT/discrete_speech_train_clsattnbndkmeans/data.txt
    ln -sf $TARGET_PATH/test_layer_0_preds.txt $DATA_ROOT/discrete_speech_test_clsattnbndkmeans/data.txt
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]
then
CUDA_LAUNCH_BLOCKING=1 srun fairseq-train ${DATA_ROOT} \
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
  # --reset-optimizer \
  # --reset-lr-scheduler
  # --no-load-quantizer \
  # --no-load-policy \
fi
