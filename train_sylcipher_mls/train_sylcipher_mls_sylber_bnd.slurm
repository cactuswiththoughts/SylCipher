#!/bin/bash
#SBATCH -J train_sylcipher_mls
#SBATCH -o logs/%j_train_sylcipher_mls.out
#SBATCH -e logs/%j_train_sylcipher_mls.err
##SBATCH -p a6
#SBATCH -p a5
##SBATCH --gres=gpu:4
#SBATCH --gres=gpu:1
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

topk=4096
DATA_ROOT="${MANIFEST_ROOT}/mls/feat_xeus_L18_sylber_bnd/top_4096/km_dir_4096"
SAVE_DIR="${MANIFEST_ROOT}/mls/feat_xeus_L18_sylber_bnd/top_4096/km_dir_4096/sylcipher_mls_shared"
TRAIN_SET="discrete_speech_train-shared_clsattnbndkmeans|text_train"
VALID_SET="discrete_speech_test0-shared_clsattnbndkmeans|text_test0"
INIT_CLUS_DIR="${MANIFEST_ROOT}/mls/feat_xeus_L18_sylber_bnd/top_4096/km_dir_4096/seg_feats/km_model.npy"
USER_DIR="${SYLCIPHER_MULTI_USER_DIR}"
lr=2e-4
world_size=4

stage=20
stop_stage=20
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]
then
cwd=$PWD
cd $USER_DIR/..
for lang in dutch french german italian; do
ln -sf /data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_${lang}_vad/feat_xeus_L18_sylber_bnd/multirun/l1_w2vu_1024clus_no_tone_xeus_L18_top1024_1024clus/0/predictions_with_fn/train_layer_0_preds.txt /data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_${lang}_vad/feat_xeus_L18_sylber_bnd/top_1024/km_dir_1024/discrete_speech_train_clsattnbndkmeans/data.txt
ln -sf /data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_${lang}_vad/feat_xeus_L18_sylber_bnd/multirun/l1_w2vu_1024clus_no_tone_xeus_L18_top1024_1024clus/0/predictions_with_fn/test_layer_0_preds.txt /data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_${lang}_vad/feat_xeus_L18_sylber_bnd/top_1024/km_dir_1024/discrete_speech_test_clsattnbndkmeans/data.txt
done

src_dirs=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_dutch_vad/feat_xeus_L18_sylber_bnd/top_1024/km_dir_1024/,/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_french_vad/feat_xeus_L18_sylber_bnd/top_1024/km_dir_1024/,/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_german_vad/feat_xeus_L18_sylber_bnd/top_1024/km_dir_1024/,/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_italian_vad/feat_xeus_L18_sylber_bnd/top_1024/km_dir_1024/

#python combine_weights.py --src-dirs $src_dirs --lang-ids "<nl>,<fr>,<de>,<it>" --out-dir $DATA_ROOT
python combine_weights_shared.py --src-dirs $src_dirs --lang-ids "<nl>,<fr>,<de>,<it>" --out-dir $DATA_ROOT
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

for i in {0..3}
do
rm -r $target_dir/text_test$i
mkdir -p $target_dir/text_test${i}
python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $text_path/test${i}_words.txt --only-source --destdir $target_dir/text_test${i} --srcdict $text_path/dict.txt
mv $target_dir/text_test${i}/train.bin $target_dir/text_test${i}/text_test${i}.bin
mv $target_dir/text_test${i}/train.idx $target_dir/text_test${i}/text_test${i}.idx
done
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]
then
for pair in 0,dutch 1,french 2,german 3,italian; do
i=${pair%,*}
lang=${pair#*,}
echo $i: $lang
ln -sf /data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_${lang}_vad/feat_xeus_L18_sylber_bnd/top_1024/km_dir_1024/discrete_speech_train_clsattnbndkmeans/data.npy /data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/feat_xeus_L18_sylber_bnd/top_4096/km_dir_4096/discrete_speech_train-shared_clsattnbndkmeans/data${i}.npy
ln -sf /data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_${lang}_vad/feat_xeus_L18_sylber_bnd/top_1024/km_dir_1024/discrete_speech_test_clsattnbndkmeans/data.npy /data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/feat_xeus_L18_sylber_bnd/top_4096/km_dir_4096/discrete_speech_test${i}-shared_clsattnbndkmeans/data.npy
done

srun fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --distributed-world-size $world_size \
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
  --frame-target-classes 400 \
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
  --save-interval 50
fi

if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]
then
  DATA_DIR=$DATA_ROOT
  CKPT_PATH=$SAVE_DIR
  for i in {1..3}
  do
  TEXT_PATH=$DATA_ROOT/discrete_speech_test${i}-shared_clsattnbndkmeans/meta_data.json
  echo $TEXT_PATH

  cwd=$PWD
  cd $USER_DIR/../../SpeechSyllableT5_transfer_learning_multilingual
#  python dump_prediction_multilevel.py $DATA_DIR --ckpt-path $CKPT_PATH/checkpoint_best.pt --user-dir $USER_DIR --text-path $TEXT_PATH --max-text-positions 100 --split "discrete_speech_test_clsattnbndkmeans|text_test" --postfix "test_layer_0" --use-discrete-labels 1 --tgt-enc-layer 0
  python dump_prediction_multilevel.py $DATA_DIR --ckpt-path $CKPT_PATH/checkpoint_best.pt --user-dir $USER_DIR --text-path $TEXT_PATH --max-text-positions 100 --split "discrete_speech_test${i}-shared_clsattnbndkmeans|text_test${i}" --postfix "test${i}_layer_0" --use-discrete-labels 0 --tgt-enc-layer 0 --num-lang 4
  cd $cwd
  done
fi
