#!/bin/bash
#SBATCH -J train_libri_clean_top_2048_jstti_enc_sylber_bnd_with_uniskip
#SBATCH -o logs/%j_train_libri_clean_top_2048_jstti_enc_sylber_bnd_with_uniskip.out
#SBATCH -e logs/%j_train_libri_clean_top_2048_jstti_enc_sylber_bnd_with_uniskip.err
##SBATCH -p mit_normal_gpu
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=8
##SBATCH --mail-user=junruin2@mit.edu
##SBATCH --mail-type=ALL
##SBATCH --time=24:00:00
##SBATCH --exclusive
##SBATCH --exclude=node0060,node0063,node0021,node0054,node0043

source /orcd/home/002/limingw/miniconda3/etc/profile.d/conda.sh 
conda activate fairseq

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

DATA_ROOT="/orcd/pool/006/limingw/data/hubert/libri-train-clean-vad-pyphen_plus/feat_hubert_large_l21_ori_feats_sylber_bnd/top_2048/km_dir_2048"
SAVE_DIR="/orcd/pool/006/limingw/data/hubert/libri-train-clean-vad-pyphen_plus/feat_hubert_large_l21_ori_feats_sylber_bnd/top_2048/km_dir_2048/model_codebook_0.3_encoder_only_sylber_bnd_diff_bnd_with_uniskip"
TRAIN_SET="discrete_speech_train_clsattnbndkmeans|text_train"
VALID_SET="discrete_speech_valid_clsattnbndkmeans|text_valid"

INIT_CLUS_DIR="/orcd/pool/006/limingw/data/hubert/libri-train-clean-vad-pyphen_plus/feat_hubert_large_l21_ori_feats_sylber_bnd/top_2048/km_dir_2048/km_model.npy"
INIT_MODEL_DIR="/orcd/pool/006/limingw/data/hubert/libri-train-clean-vad-pyphen_plus/feat_hubert_large_l21_ori_feats_sylber_bnd/top_2048/km_dir_2048/model_codebook_0.3_clsattn_bnd_kmeans_encoder_only_diff_bnd/checkpoint_best.pt"

s=10
e=10
if [ $s -le 0 ] && [ $e -ge 0 ]; then
srun fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --distributed-world-size 8 \
  --distributed-port 12345 \
  --no-epoch-checkpoints \
  --ddp-backend legacy_ddp \
  --user-dir /nobackup/users/junruin2/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024_uniskip/speecht5 \
  --log-format json \
  --seed 0 \
  \
  --task speecht5 \
  --t5-task discrete_pretrain_gan \
  --label-rates 1 \
  --sample-rate 1 \
  \
  --num-workers 0 \
  --max-tokens 500 \
  --encoder-layers 2 \
  --max-speech-sample-size 100 \
  --update-freq 1800 \
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
  --lr 1e-3 \
  --lr-scheduler polynomial_decay \
  --mask 0.3 \
  --mask-random 0.1 \
  \
  --max-update 1000000 \
  --warmup-updates 0 \
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
  --bart-weight 0.0 \
  --masked-lm-weight 0.0 \
  --unigram-weight 0.0 \
  --max-text-positions 100 \
  --discriminator-depth 3 \
  --discriminator-kernel 8 \
  --discriminator-causal \
  --start-gan-updates 1000000000000 \
  --policy-pretrain-weight 0 \
  --policy-loss-weight 1 \
  --init-clus-dir $INIT_CLUS_DIR \
  --use-valid-s2t \
  --post-process "none" \
  --report-similarity \
  --similarity-meanpool \
  --ce-weight 0 \
  --ctc-weight 1 \
  --report-accuracy \
  --sampling-nums 1 \
  --km-size 2048 \
  --frame-target-classes 100 \
  --latent-vars 100 \
  --latent-groups 3 \
  --word-freq 12 \
  --word-freq-weight 50000 \
  --speech-encoder-prenet-mask 0.3 \
  --speech-encoder-prenet-mask-random  0.1 \
  --speech-encoder-prenet-poisson-lambda 3.5 \
  --speech-encoder-prenet-mask-length "span-poisson" \
  --speech-encoder-prenet-replace-length -1 \
  --replace-length -1 \
  --code-loss-weight 1.0 \
  --word-count-loss-weight 50000000 \
  --text-loss-ratio 10 \
  --frame-target-loss-weight 1.0 \
  --policy-pretrain-pos-weight-bce 1.5 \
  --use-codebook \
  --codebook-prob 0.0 \
  --boundary-sum-multiply 1 \
  --policy-feat-dim 1024 \
  --mlm-unmasked-weight 0.5 \
  --init-model-dir $INIT_MODEL_DIR \
  --uni-loss-weight 1 \
  --skip-loss-weight 1 \
  --no-load-quantizer \
  --reset-optimizer \
  --reset-lr-scheduler \
  # --use-masked \
  # --use-transformer-policy
  # --turn-off-finetune-all-enc-dec \
  # --turn-off-finetune-all-enc-dec \
  # --no-load-speech-prepost
  # --reset-optimizer \
  # --reset-lr-scheduler \
  # --reset-optimizer \
  # --reset-lr-scheduler \
fi

if [ $s -le 10 ] && [ $e -ge 10 ]; then
  DATA_DIR=$DATA_ROOT
  USER_DIR=/orcd/pool/006/limingw/workplace/VGUnsupASR/SpeechT5/SpeechSyllableT5/speecht5
  CKPT_PATH=$SAVE_DIR
  #TEXT_PATH=/orcd/pool/006/limingw/data/LibriSpeech_vad/ls_train_clean_vad_syllable_pyphen_plus_valid_true.json
  TEXT_PATH=/orcd/pool/006/limingw/data/LibriSpeech_vad/ls_train_clean_vad_syllable_pyphen_plus_valid_true_oov_2048.json
  echo $CKPT_PATH
  cwd=$PWD
  cd $USER_DIR/..
  #python dump_prediction.py $DATA_DIR --ckpt-path $CKPT_PATH/checkpoint_best.pt --user-dir $USER_DIR --max-text-positions 100 --split "discrete_speech_valid_true_clsattnbndkmeans|text_valid_true" --postfix "valid_true_layer_0" --use-discrete-labels 0 --tgt-enc-layer 0
  python dump_prediction_and_boundary_multilevel.py $DATA_DIR --ckpt-path $CKPT_PATH/checkpoint_best.pt --user-dir $USER_DIR --text-path $TEXT_PATH --max-text-positions 100 --split "discrete_speech_valid_true_clsattnbndkmeans|text_valid_true" --postfix "valid_true_layer_0" --use-discrete-labels 0 --tgt-enc-layer 0
fi
