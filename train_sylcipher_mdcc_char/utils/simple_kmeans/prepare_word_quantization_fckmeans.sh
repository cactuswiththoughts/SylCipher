#!/bin/bash

set -e

topk=4096
small_size=1024
feature_type=xxx # be consistent with the naming of the features
wrd_bnd_type=your_flag_for_word_seg_bnds_on_wav2bnd_large_1024 # be consistent with the naming of the word boundaries
# point to the directory structure in step 8 + /seg_feats
feat_dir=/path/to/data/libri-train-clean-top-${topk}/feat_${feature_type}_no_sil_${wrd_bnd_type}/top_${topk}/km_dir_${topk}/seg_feats

lab_dir=$feat_dir/km_idx_clsattnbndkmeans

km_dir=$feats_dir/km_model_fckmeans_from_${small_size}

nshard=1


mkdir -p -m 777 $lab_dir

python dump_km_label.py $feat_dir "train" "${km_dir}" 1 0 $lab_dir
python dump_km_label.py $feat_dir "valid" "${km_dir}" 1 0 $lab_dir
python dump_km_label.py $feat_dir "valid_true" "${km_dir}" 1 0 $lab_dir


for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/train_${rank}_${nshard}.km
done > $lab_dir/train.km

mv $lab_dir/valid_0_1.km $lab_dir/valid.km
mv $lab_dir/valid_true_0_1.km $lab_dir/valid_true.km

for rank in $(seq 0 $((nshard - 1))); do
  rm $lab_dir/train_${rank}_${nshard}.km
done

