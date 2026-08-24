#!/bin/bash

set -e

topk=1024
feature_type=$1 # be consistent with the naming of the features
wrd_bnd_type=$2 # be consistent with the naming of the word boundaries
# point to the directory structure in step 8 + /seg_feats
feat_dir=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_${feature_type}_${wrd_bnd_type}/top_${topk}/km_dir_${topk}/seg_feats

lab_dir=$feat_dir/km_idx_clsattnbndkmeans
wrd_bnd_type_for_vghubert_bnds=$2
# note that km_dir points to the km model obtained from VG-HuBERT boundaries
km_dir=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_${feature_type}_${wrd_bnd_type_for_vghubert_bnds}/top_${topk}/km_dir_${topk}/seg_feats/km_model

nshard=1

mkdir -p -m 777 $lab_dir

python dump_km_label_faiss.py $feat_dir "train" "${km_dir}" 1 0 $lab_dir
python dump_km_label_faiss.py $feat_dir "test" "${km_dir}" 1 0 $lab_dir


for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/train_${rank}_${nshard}.km
done > $lab_dir/train.km

mv $lab_dir/test_0_1.km $lab_dir/test.km
