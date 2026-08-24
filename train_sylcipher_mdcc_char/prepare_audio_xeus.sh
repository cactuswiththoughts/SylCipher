#!/usr/bin/env zsh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
set -e
source_dir=$1
tgt_dir=$2
model=$3

echo stage 1
if [ -z "$4" ]
  then
    dim=512
  else
    dim=$4
fi

echo "using $dim dim for PCA"

if [ -z "$5" ]
  then
    layer=14
  else
    layer=$5
fi

if [ -z "$6" ]
  then
    orig_n_clus=128
  else
    orig_n_clus=$6
fi
n_clus=1024
#n_clus=2048

echo "extracting from layer $layer"
train_split=train
valid_split=valid
test_split=valid_true

all_splits=($train_split)
echo $source_dir
if [[ -f "$source_dir/valid.tsv" ]]; then
    all_splits+=('valid')
fi

if [[ -f "$source_dir/valid_true.tsv" ]]; then
    all_splits+=('valid_true')
fi

if [[ -f "$source_dir/dev.tsv" ]]; then
    all_splits+=('dev')
fi

if [[ -f "$source_dir/test.tsv" ]]; then
    all_splits+=('test')
fi

echo "processing splits: $all_splits"

stage=3
stop_stage=6

mkdir -p $tgt_dir
echo stage 2
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo $source_dir,$tgt_dir
    cp $source_dir/*.tsv $tgt_dir || true
fi

setopt shwordsplit
echo stage 3
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    cwd=$PWD
    cd ../../XEUS
    for split in $all_splits; do
        echo stage 3 $split
        for x in test train; do
            data=$source_dir/$x.tsv
            save_prefix=$tgt_dir/$x
            python xeus_extract_features.py \
                --data $data --save-prefix $save_prefix
        done
    done
    cd $cwd
fi

echo stage 5
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    CONDA_ROOT=$HOME/miniconda3
    source $CONDA_ROOT/etc/profile.d/conda.sh
    conda activate fairseq
    for split in $all_splits; do
        ln -s $tgt_dir/${split}.npy $tgt_dir/${split}_0_1.npy
        ln -s $tgt_dir/${split}.lengths $tgt_dir/${split}_0_1.len
    done
    python extract_features/simple_kmeans/learn_kmeans_faiss.py \
        $tgt_dir ${train_split} 1 $tgt_dir/km_model $orig_n_clus --percent -1 
#    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_cluster_faiss.py $tgt_dir/${train_split}.tsv \
#    --checkpoint $model --save-dir $tgt_dir -f "CLUS$orig_n_clus"
fi

echo stage 6
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    CONDA_ROOT=$HOME/miniconda3
    source $CONDA_ROOT/etc/profile.d/conda.sh
    conda activate fairseq
    for split in $all_splits; do
#      python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_apply_cluster_faiss.py $tgt_dir \
#      --checkpoint $model --path $tgt_dir/CLUS$orig_n_clus --split $split
      python extract_features/simple_kmeans/dump_km_label_faiss.py $tgt_dir $split $tgt_dir/km_model 1 0 $tgt_dir/km_idx_clsattnbndkmeans
    done
fi
