#!/bin/bash

set -e

topk=2048
feature_type=hubert_large_L21 # be consistent with the naming of the features
wrd_bnd_type=sylber_bnd # be consistent with the naming of the word boundaries
# point to the directory structure in step 8
feat_dir=$1
km_dir=$feat_dir/km_model
nclus=$2
percent=$3

nshard=1

mkdir -p -m 777 $(dirname $km_dir)
python learn_kmeans_faiss.py $feat_dir "train" ${nshard} "${km_dir}" ${nclus} --percent $percent #-1
