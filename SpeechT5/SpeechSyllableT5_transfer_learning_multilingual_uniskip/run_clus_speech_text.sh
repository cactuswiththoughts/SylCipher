#!/bin/bash

set -e
nshard=$1
layer=$2
feat_dir=$3
lab_dir=$4
km_dir=$5
nclus=$6



cd /nobackup/users/junruin2/fairseq_kq/examples/hubert/simple_kmeans_20220121

ngpus=$(nvidia-smi -L | wc -l)
rrr=$(bc <<< "${nshard}%${ngpus}")
if [ "$rrr" -ne "0" ]; then
   echo "Invalid number of shards!";
   exit;
fi
ntasks=$(bc <<< "${nshard}/${ngpus}")
if [ "$ntasks" -gt "8" ]; then
   echo "Number of tasks too large!";
   exit;
fi


mkdir -p -m 777 $(dirname "${km_dir}_speech")
python learn_kmeans_faiss.py $feat_dir "train_speech_layer_${layer}" ${nshard} "${km_dir}_speech" ${nclus} --percent -1


mkdir -p -m 777 $lab_dir


python dump_km_label_faiss.py $feat_dir "train_speech_layer_${layer}" "${km_dir}_speech" 1 0 "${lab_dir}_from_speech_km"
python dump_km_label_faiss.py $feat_dir "valid_true_speech_layer_${layer}" "${km_dir}_speech" 1 0 "${lab_dir}_from_speech_km"

python dump_km_label_faiss.py $feat_dir "train_text_layer_${layer}" "${km_dir}_speech" 1 0 "${lab_dir}_from_speech_km"
python dump_km_label_faiss.py $feat_dir "valid_true_text_layer_${layer}" "${km_dir}_speech" 1 0 "${lab_dir}_from_speech_km"


mkdir -p -m 777 $(dirname "${km_dir}_text")
python learn_kmeans_faiss.py $feat_dir "train_text_layer_${layer}" ${nshard} "${km_dir}_text" ${nclus} --percent -1


mkdir -p -m 777 $lab_dir


python dump_km_label_faiss.py $feat_dir "train_speech_layer_${layer}" "${km_dir}_text" 1 0 "${lab_dir}_from_text_km"
python dump_km_label_faiss.py $feat_dir "valid_true_speech_layer_${layer}" "${km_dir}_text" 1 0 "${lab_dir}_from_text_km"

python dump_km_label_faiss.py $feat_dir "train_text_layer_${layer}" "${km_dir}_text" 1 0 "${lab_dir}_from_text_km"
python dump_km_label_faiss.py $feat_dir "valid_true_text_layer_${layer}" "${km_dir}_text" 1 0 "${lab_dir}_from_text_km"