#!/bin/bash

set -e

# point to the directory structure in step 8
feat_dir=$1
topk=$2
km_dir=$feat_dir/km_model
nclus=${topk}

nshard=1

mkdir -p -m 777 $(dirname $km_dir)
python learn_kmeans_faiss.py $feat_dir "train" ${nshard} "${km_dir}" ${nclus} --percent -1
