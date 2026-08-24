#!/bin/bash
#SBATCH -J train_aishell_vad_pinyin_with_tone_top_1024_jstti_enc_sylber_ft_stage2_bnd
#SBATCH -o logs/%j_train_aishell_vad_pinyin_with_tone_top_1024_jstti_enc_sylber_ft_stage2_bnd.out
#SBATCH -e logs/%j_train_aishell_vad_pinyin_with_tone_top_1024_jstti_enc_sylber_ft_stage2_bnd.err
#SBATCH --mail-user=junruin2@mit.edu
#SBATCH --qos=sched_level_2 
#SBATCH --exclude=node0060,node0063,node0021,node0054,node0042,node0029,node0014
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
##SBATCH --nodes=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --exclusive


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


DATA_ROOT="/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024/"
SAVE_DIR="/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024/model_codebook_0.3_kmeans_encoder_only_retry_6"
TRAIN_SET="discrete_speech_train_clsattnbndkmeans|text_train"
VALID_SET="discrete_speech_test_clsattnbndkmeans|text_test"
#VALID_SET="discrete_speech_valid_clsattnbndkmeans|text_valid"
INIT_CLUS_DIR="/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024/km_model.npy"
INIT_MODEL_DIR="/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024/model_codebook_0.3_kmeans_encoder_only_retry_5/checkpoint_best.pt"

s=50
e=50
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
  --user-dir ${SYLCIPHER_USER_DIR} \
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
  --km-size 431 \
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
  --init-model-dir $INIT_MODEL_DIR \
  --reset-optimizer \
  --reset-lr-scheduler
  # --no-load-quantizer \
  # --no-load-policy \
fi

if [ $s -le 10 ] && [ $e -ge 10 ]; then
    cwd=$PWD
    cd /nobackup/users/limingw/workplace/VGUnsupASR/SpeechT5/SpeechSyllableT5

    DATA_DIR=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024
    USER_DIR=${SYLCIPHER_USER_DIR}
    CKPT_PATH=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024/model_codebook_0.3_kmeans_encoder_only_retry_6
    TEXT_PATH=/nobackup/users/junruin2/data/hubert/aishell_3_vad/test_gradseg_style_with_tone.json
    echo $CKPT_PATH
    python dump_prediction_multilevel.py $DATA_DIR --ckpt-path $CKPT_PATH/checkpoint_best.pt --user-dir $USER_DIR --text-path $TEXT_PATH --max-text-positions 100 --split "discrete_speech_test_clsattnbndkmeans|text_test" --postfix "test_layer_0" --use-discrete-labels 0 --tgt-enc-layer 0
    
    cd $cwd
fi

if [ $s -le 20 ] && [ $e -ge 20 ]; then
    # USER_DIR=${SYLCIPHER_USER_DIR}
    USER_DIR=${SYLCIPHER_USER_DIR}
    DATA_ROOT="/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024/"
    SAVE_DIR="/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024/model_codebook_0.3_kmeans_encoder_only_diff_bnd" 
    TRAIN_SET="discrete_speech_train_clsattnbndkmeans|text_train"
    VALID_SET="discrete_speech_test_clsattnbndkmeans|text_test"
    INIT_CLUS_DIR="/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024/km_model.npy"
    INIT_MODEL_DIR="/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024/model_codebook_0.3_kmeans_encoder_only_retry_6/checkpoint_best.pt"
 
    srun fairseq-train ${DATA_ROOT} \
      --save-dir ${SAVE_DIR} \
      --tensorboard-logdir ${SAVE_DIR} \
      --train-subset ${TRAIN_SET} \
      --valid-subset ${VALID_SET} \
      --distributed-world-size 8 \
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
      --update-freq 1300 \
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
      --max-update 400 \
      --warmup-updates 0 \
      --total-num-update 400 \
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
      --km-size 1024 \
      --frame-target-classes 100 \
      --latent-vars 100 \
      --latent-groups 3 \
      --word-freq 20 \
      --word-freq-weight 0 \
      --speech-encoder-prenet-mask 0.3 \
      --speech-encoder-prenet-mask-random  0.1 \
      --speech-encoder-prenet-poisson-lambda 3.5 \
      --speech-encoder-prenet-mask-length "span-poisson" \
      --speech-encoder-prenet-replace-length -1 \
      --replace-length -1 \
      --code-loss-weight 1.0 \
      --word-count-loss-weight 1.0 \
      --text-loss-ratio 10 \
      --frame-target-loss-weight 0 \
      --policy-pretrain-pos-weight-bce 1.5 \
      --use-codebook \
      --codebook-prob 0.0 \
      --boundary-sum-multiply 1 \
      --policy-feat-dim 1024 \
      --mlm-unmasked-weight 0.5 \
      --init-model-dir $INIT_MODEL_DIR \
      --no-load-quantizer \
      --reset-optimizer \
      --reset-lr-scheduler \
      # --no-load-policy \
      # --word-count-loss-weight 500000
fi

if [ $s -le 30 ] && [ $e -ge 30 ]; then
    cwd=$PWD   
    cd /nobackup/users/limingw/workplace/VGUnsupASR/SpeechT5/SpeechSyllableT5

    DATA_DIR=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024
    USER_DIR=${SYLCIPHER_USER_DIR}
    CKPT_PATH=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024/model_codebook_0.3_kmeans_encoder_only_retry_6
    TEXT_PATH=/nobackup/users/junruin2/data/hubert/aishell_3_vad/test_gradseg_style_with_tone.json
    echo $CKPT_PATH
    python dump_prediction_multilevel.py $DATA_DIR --ckpt-path $CKPT_PATH/checkpoint_best.pt --user-dir $USER_DIR --text-path $TEXT_PATH --max-text-positions 100 --split "discrete_speech_test_clsattnbndkmeans|text_test" --postfix "test_layer_0" --use-discrete-labels 0 --tgt-enc-layer 0
    
    DATA_DIR=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024
    USER_DIR=${SYLCIPHER_USER_DIR}
    CKPT_PATH=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024/model_codebook_0.3_kmeans_encoder_only_diff_bnd
    TEXT_PATH=/nobackup/users/junruin2/data/hubert/aishell_3_vad/test_gradseg_style_with_tone.json 
    echo $CKPT_PATH
    python dump_prediction_multilevel.py $DATA_DIR --ckpt-path $CKPT_PATH/checkpoint_best.pt --user-dir $USER_DIR --text-path $TEXT_PATH --max-text-positions 100 --split "discrete_speech_test_clsattnbndkmeans|text_test" --postfix "test_layer_0" --use-discrete-labels 0 --tgt-enc-layer 0

    cd $cwd
fi

if [ $s -le 40 ] && [ $e -ge 40 ]; then
    USER_DIR=${SYLCIPHER_ROOT}/SpeechT5/SpeechSyllableT5_uniskip/speecht5
    DATA_ROOT="/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024"
    SAVE_DIR="/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024/model_codebook_0.3_encoder_only_sylber_bnd_diff_bnd_with_uniskip"
    TRAIN_SET="discrete_speech_train_clsattnbndkmeans|text_train"
    VALID_SET="discrete_speech_test_clsattnbndkmeans|text_test"
    INIT_CLUS_DIR="/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024/km_model.npy"
    INIT_MODEL_DIR="/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024/model_codebook_0.3_kmeans_encoder_only_diff_bnd/checkpoint_best.pt"

    srun fairseq-train ${DATA_ROOT} \
      --save-dir ${SAVE_DIR} \
      --tensorboard-logdir ${SAVE_DIR} \
      --train-subset ${TRAIN_SET} \
      --valid-subset ${VALID_SET} \
      --distributed-world-size 8 \
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
      --max-update 2000 \
      --warmup-updates 0 \
      --total-num-update 2000 \
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
      --km-size 1024 \
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

if [ $s -le 50 ] && [ $e -ge 50 ]; then
    cd /nobackup/users/limingw/workplace/VGUnsupASR/SpeechT5/SpeechSyllableT5
    cwd=$PWD
    DATA_DIR=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024
    USER_DIR=${SYLCIPHER_USER_DIR}
    CKPT_PATH=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024/model_codebook_0.3_kmeans_encoder_only_diff_bnd
    TEXT_PATH=/nobackup/users/junruin2/data/hubert/aishell_3_vad/test_gradseg_style_with_tone.json 
#    echo $CKPT_PATH
#    python dump_prediction_multilevel.py $DATA_DIR --ckpt-path $CKPT_PATH/checkpoint_best.pt --user-dir $USER_DIR --text-path $TEXT_PATH --max-text-positions 100 --split "discrete_speech_test_clsattnbndkmeans|text_test" --postfix "test_layer_0" --use-discrete-labels 0 --tgt-enc-layer 0

    USER_DIR=${SYLCIPHER_ROOT}/SpeechT5/SpeechSyllableT5_uniskip/speecht5
    DATA_DIR=$DATA_ROOT
    CKPT_PATH=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_xeus_L18_sylber_ft_stage2_bnd/top_1024/km_dir_1024/model_codebook_0.3_encoder_only_sylber_bnd_diff_bnd_with_uniskip
    TEXT_PATH=/nobackup/users/junruin2/data/hubert/aishell_3_vad/test_gradseg_style_with_tone.json
    echo $CKPT_PATH
    python dump_prediction_multilevel.py $DATA_DIR --ckpt-path $CKPT_PATH/checkpoint_best.pt --user-dir $USER_DIR --text-path $TEXT_PATH --max-text-positions 100 --split "discrete_speech_test_clsattnbndkmeans|text_test" --postfix "test_layer_0" --use-discrete-labels 1 --tgt-enc-layer 0
    cd $cwd
fi
