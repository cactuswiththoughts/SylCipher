#!/bin/bash

echo Convert a fairseq manifest directory to JSTTI format

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../path.sh"

feat_name=$1
topk=$2
seg_name=$3

data_dir=${MANIFEST_ROOT}/mdcc_vad
src_dir=${MANIFEST_ROOT}/mdcc_vad
manifest_dir=${MANIFEST_ROOT}/mdcc_vad_raw_character/feat_${feat_name}_${seg_name}/top_${topk}
echo $manifest_dir

for x in train test; do
    mkdir -p $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans
    mkdir -p $manifest_dir/km_dir_${topk}/text_${x}
done

echo Step 1: Create file_list.txt
for x in train test; do
    ln -sf $src_dir/${x}_file_list.txt $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/file_list.txt
    rm -f $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/bnd_errors.txt
    touch $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/bnd_errors.txt
done

echo Step 2: Create data.{npy,lengths} and data_dict_bnd.pkl
for x in train test; do
    ln -sf $data_dir/feat_$feat_name/${x}.npy $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/data.npy
    ln -sf $data_dir/feat_$feat_name/${x}.lengths $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/data.lengths
    ln -sf $data_dir/$seg_name/${x}_data_bnd_dict.pkl $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/data_dict_bnd.pkl
done
