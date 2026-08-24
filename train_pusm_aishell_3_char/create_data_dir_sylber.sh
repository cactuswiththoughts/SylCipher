#!/bin/bash

echo Convert a fairseq manifest directory to JSTTI format

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../path.sh"

feat_name=xeus_L18
label_feat_name=xeus_L18
#seg_name=sylber_bnd
seg_name=$1

#n_clus=512
#topk=1024
#topk_orig=1024
#topk=2048
#topk_orig=2048
topk=$2
topk_orig=$2
src_dir=${MANIFEST_ROOT}/aishell_3_vad_raw_character/feat_${feat_name}_$seg_name/top_${topk}
manifest_dir=${MANIFEST_ROOT}/aishell_3_vad_raw_character/feat_${feat_name}_$seg_name/top_${topk}/km_dir_${topk}
if [ ! -d $manifest_dir/text_train ]; then
    mkdir -p $manifest_dir/text_train
fi

echo Step 1: Create file_list.txt
for x in train test; do
    ln -s $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/file_list.txt $manifest_dir/$x.files
    ln -s $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/bnd_errors.txt $manifest_dir/${x}_bnd_errors.txt
done

echo Step 2: Create data.{npy,lengths}
for x in train test; do
    rm $manifest_dir/${x}.lengths
    ln -s $src_dir/km_dir_${topk}/seg_feats/${x}_0_1.len $manifest_dir/${x}.lengths
    python convert_km_to_npy.py \
        --in_path $src_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/data.km \
        --out_path $manifest_dir/${x}.npy
done

echo Step 2: Create data.{wrd}
for x in train test; do
    ln -s $src_dir/${x}_words.txt $manifest_dir/${x}.wrd
done

echo Step 3: Link unpaired text data
ln -s $src_dir/km_dir_${topk}/text_train/text_train.idx $manifest_dir/text_train/train.idx
ln -s $src_dir/km_dir_${topk}/text_train/text_train.bin $manifest_dir/text_train/train.bin
ln -s $src_dir/km_dir_${topk}/dict.txt $manifest_dir/text_train/dict.txt
