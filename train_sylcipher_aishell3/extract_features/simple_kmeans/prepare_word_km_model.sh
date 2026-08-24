#!/bin/bash

set -e

topk=1024
feature_type=$1 # be consistent with the naming of the features
wrd_bnd_type=$2 # be consistent with the naming of the word boundaries
# point to the directory structure in step 8
feat_dir=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_${feature_type}_${wrd_bnd_type}/top_${topk}/km_dir_${topk}/seg_feats

km_dir=$feat_dir/km_model
nclus=${topk}

nshard=1

mkdir -p -m 777 $(dirname $km_dir)
python learn_kmeans_faiss.py $feat_dir "train" ${nshard} "${km_dir}" ${nclus} --percent -1
# XXX python learn_kmeans_faiss.py $feat_dir "test" ${nshard} "${km_dir}" ${nclus} --percent -1
