#!/bin/bash

echo Convert a fairseq manifest directory to JSTTI format
feat_name=hubert_large_L21
seg_name=sylber_bnd
topk=$1
topk_orig=$1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../path.sh"

data_dir=${MANIFEST_ROOT}/librispeech_clean_vad
manifest_dir=${MANIFEST_ROOT}/librispeech_clean_vad/feat_${feat_name}_${seg_name}/top_${topk}/km_dir_${topk}
boundary_path=${MANIFEST_ROOT}/librispeech_clean_vad/$seg_name

for x in train valid valid_true; do
    if [ ! -d $manifest_dir/discrete_speech_${x}_clsattnbndkmeans ]; then
        mkdir -p $manifest_dir/discrete_speech_${x}_clsattnbndkmeans
    fi

    if [ ! -d $manifest_dir/text_${x} ]; then
        mkdir -p $manifest_dir/text_${x}
    fi
done

echo Step 1: Create file_list.txt
for x in train valid valid_true; do
    echo $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/file_list.txt
    sed '1d' $data_dir/tsvs/${x}.tsv > $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/file_list.txt
    cut -f1 $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/file_list.txt > $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/new_file_list.txt
    mv $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/new_file_list.txt $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/file_list.txt
done

echo Step 2: Create data.{npy,lengths}
for x in train valid valid_true; do
    ln -s $data_dir/feat_$feat_name/${x}_0_1.npy $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data.npy
    ln -s $data_dir/feat_$feat_name/${x}_0_1.len $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data.lengths
    ln -s $data_dir/label_hubert_base_L6/$x.km $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data_frame_clus.km
done
