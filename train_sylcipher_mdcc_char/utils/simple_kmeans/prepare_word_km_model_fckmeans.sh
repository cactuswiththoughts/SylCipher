#!/bin/bash

set -e

topk=4096
small_size=1024
feature_type=xxx # be consistent with the naming of the features
wrd_bnd_type=your_flag_for_vghubert_bnds # be consistent with the naming of the word boundaries
# point to the directory structure in step 8
feat_dir=/path/to/data/libri-train-clean-top-${topk}/feat_${feature_type}_no_sil_${wrd_bnd_type}/top_${topk}/km_dir_${topk}/seg_feats

# where to save the newly trained k-means model
km_dir=$feats_dir/km_model_fckmeans_from_${small_size}
nclus=${topk}

# this should point to your k-means model trained on the 1024-word corpus
km_path_fixed=/path/to/data/libri-train-clean-top-${small_size}/feat_${feature_type}_no_sil_${wrd_bnd_type}/top_${small_size}/km_dir_${small_size}/seg_feats/km_model

# this should point to your k-means model trained from scratch on the 4096-word corpus
km_path_phase_1=$feats_dir/km_model

nshard=1

mkdir -p -m 777 $(dirname $km_dir)
python learn_FC_kmeans_1.py $feat_dir "train" ${nshard} "${km_dir}" "${km_path_phase_1}" "${km_path_fixed}" ${nclus} --percent -1
