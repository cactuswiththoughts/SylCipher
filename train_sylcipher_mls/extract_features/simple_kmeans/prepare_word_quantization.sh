#!/bin/bash

set -e

feat_dir=$1

lab_dir=$feat_dir/km_idx_clsattnbndkmeans
# note that km_dir points to the km model obtained from VG-HuBERT boundaries
km_dir=$feat_dir/km_model

nshard=1

mkdir -p -m 777 $lab_dir


python dump_km_label_faiss.py $feat_dir "train" "${km_dir}" 1 0 $lab_dir
python dump_km_label_faiss.py $feat_dir "test" "${km_dir}" 1 0 $lab_dir

for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/train_${rank}_${nshard}.km
done > $lab_dir/train.km

mv $lab_dir/test_0_1.km $lab_dir/test.km

#for rank in $(seq 0 $((nshard - 1))); do
#  rm $lab_dir/train_${rank}_${nshard}.km
#done
