#!/bin/bash

set -e

topk=$2
feature_type=hubert_large_L21 # be consistent with the naming of the features
wrd_bnd_type=sylber_bnd # be consistent with the naming of the word boundaries
# point to the directory structure in step 8 + /seg_feats
feat_dir=$1

lab_dir=$feat_dir/km_idx_clsattnbndkmeans
wrd_bnd_type_for_vghubert_bnds=sylber_bnd
# note that km_dir points to the km model obtained from VG-HuBERT boundaries
km_dir=$feat_dir/km_model

nshard=1

mkdir -p -m 777 $lab_dir


python dump_km_label_faiss.py $feat_dir "train" "${km_dir}" 1 0 $lab_dir
python dump_km_label_faiss.py $feat_dir "valid" "${km_dir}" 1 0 $lab_dir
python dump_km_label_faiss.py $feat_dir "valid_true" "${km_dir}" 1 0 $lab_dir


for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/train_${rank}_${nshard}.km
done > $lab_dir/train.km

mv $lab_dir/valid_0_1.km $lab_dir/valid.km
mv $lab_dir/valid_true_0_1.km $lab_dir/valid_true.km

#for rank in $(seq 0 $((nshard - 1))); do
#  rm $lab_dir/train_${rank}_${nshard}.km
#done

