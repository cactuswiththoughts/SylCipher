#!/bin/bash
#SBATCH -J train_libri_clean_vad
#SBATCH -o logs/%j_train_libri_clean_vad.out
#SBATCH -e logs/%j_train_libri_clean_vad.err
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --qos=regular


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

topk=2048
DATA_ROOT="${MANIFEST_ROOT}/librispeech_clean_vad/feat_hubert_large_L21_sylber_bnd/top_${topk}/km_dir_${topk}/"
SAVE_DIR="${MANIFEST_ROOT}/librispeech_clean_vad/feat_hubert_large_L21_sylber_bnd/top_${topk}/km_dir_${topk}/model_codebook_0.3_clsattn_bnd_kmeans_encoder_only"
TRAIN_SET="discrete_speech_train_clsattnbndkmeans|text_train"
VALID_SET="discrete_speech_valid_clsattnbndkmeans|text_valid"
INIT_CLUS_DIR="${MANIFEST_ROOT}/librispeech_clean_vad/feat_hubert_large_L21_sylber_bnd/top_${topk}/km_dir_${topk}/seg_feats/km_model.npy"
USER_DIR="${SYLCIPHER_USER_DIR}"

s=0
e=2
cwd=$PWD
if [ $s -le 0 ] && [ $e -ge 0 ]; then
    bash create_data_dir_sylber.sh $topk
fi

if [ $s -le 1 ] && [ $e -ge 1 ]; then
    bash prepare_sylber_features.sh $topk
fi

if [ $s -le 2 ] && [ $e -ge 2 ]; then
    bash prepare_text_sylber.sh    
fi

if [ $s -le 10 ] && [ $e -ge 10 ]; then
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
  --max-tokens 500 \
  --encoder-layers 2 \
  --max-speech-sample-size 100 \
  --update-freq 32 \
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
  --lr 2e-4 \
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
  --save-interval 50 #\
  # --reset-optimizer \
  # --reset-lr-scheduler
  # --no-load-quantizer \
  # --no-load-policy \
fi

if [ $s -le 20 ] && [ $e -ge 20 ]; then
    DATA_DIR=$DATA_ROOT
    CKPT_PATH=$SAVE_DIR
    TEXT_PATH=${MANIFEST_ROOT}/librispeech_clean_vad/ls_train_clean_vad_syllable_pyphen_plus_valid_true.json
    echo $CKPT_PATH
    cd "${SYLCIPHER_ROOT}/SpeechT5/SpeechSyllableT5"
    python dump_prediction_multilevel.py $DATA_DIR --ckpt-path $CKPT_PATH/checkpoint_best.pt --user-dir $USER_DIR --max-text-positions 100 --split "discrete_speech_valid_true_clsattnbndkmeans|text_valid_true" --postfix "valid_true_layer_0" --use-discrete-labels 0 --tgt-enc-layer 0
fi
