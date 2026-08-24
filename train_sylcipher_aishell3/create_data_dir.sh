#!/bin/bash

echo Convert a fairseq manifest directory to JSTTI format
feat_name=xeus_L18
topk=1024
topk_orig=1024

src_dir=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin_no_tone/feat_${feat_name}_sylber_ft_stage2_bnd/top_2048
feat_dir=/nobackup/users/junruin2/data/hubert/aishell_3_vad/feat_${feat_name}
manifest_dir=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_${feat_name}_sylber_ft_stage2_bnd/top_1024

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
    rm $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/{data.npy,data.lengths,file_list.txt,bnd_errors.txt}
    ln -s $feat_dir/$x.npy $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/data.npy
    ln -s $feat_dir/$x.lengths $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/data.lengths
    ln -s $src_dir/km_dir_2048/discrete_speech_${x}_clsattnbndkmeans/file_list.txt $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/file_list.txt
    ln -s $src_dir/km_dir_2048/discrete_speech_${x}_clsattnbndkmeans/bnd_errors.txt $manifest_dir/km_dir_${topk}/discrete_speech_${x}_clsattnbndkmeans/bnd_errors.txt
done
