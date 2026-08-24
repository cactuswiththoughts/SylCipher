#!/bin/bash

echo Convert a fairseq manifest directory to JSTTI format
#feat_name=wavlm_L21  
feat_name=$1 
#feat_name=xeus_L18
#label_feat_name=wavlm_L21
label_feat_name=$1
#label_feat_name=xeus_L18
topk=$2
n_clus=$2
#topk=1024
#n_clus=1024
#seg_name=sylber_bnd
seg_name=$3

data_dir=$4
src_dir=$4
manifest_dir=$4/feat_${feat_name}_${seg_name}/top_${topk}
echo $manifest_dir

for x in train test; do
    if [ ! -d $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans ]; then
        mkdir -p $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans
    fi

    if [ ! -d $manifest_dir/km_dir_${topk}/text_${x} ]; then
        mkdir -p $manifest_dir/km_dir_${topk}/text_${x}
    fi
done

echo Step 1: Create file_list.txt
for x in train test; do
    echo $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/file_list.txt
    ln -sf $src_dir/${x}_file_list.txt $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/file_list.txt
    rm $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/bnd_errors.txt
    touch $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/bnd_errors.txt
#    ln -s $src_dir/${x}_bad_files.txt $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/bnd_errors.txt
done

echo Step 2: Create data.{npy,lengths} and data_dict_bnd.pkl
for x in train test; do
    ln -sf $data_dir/feat_$feat_name/${x}.npy $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/data.npy
    ln -sf $data_dir/feat_$feat_name/${x}.lengths $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/data.lengths
    ln -sf $data_dir/$seg_name/${x}_data_bnd_dict.pkl $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/data_dict_bnd.pkl
done
