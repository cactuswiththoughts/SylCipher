#!/bin/bash
#SBATCH -J prepare_mls
#SBATCH -o logs/%j_prepare_mls.out
#SBATCH -e logs/%j_prepare_mls.err
##SBATCH -p 2080
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --qos=regular

export HOME=/data/sls/scratch/limingw
export FAIRSEQ_ROOT=$HOME/fairseq-0.12.1
export KENLM_ROOT=/usr/users/limingw/softwares/kenlm/build/bin
source /data/sls/scratch/limingw/miniconda3/etc/profile.d/conda.sh
conda activate fairseq

part=$1
lang=$2
#n_parts=20
n_parts=1

feat_name=xeus
seg_name=sylber_bnd
l=18
topk=1024
model=/data/sls/scratch/limingw/models/xeus/xeus.pth
wav_root=/data/sls/scratch/limingw/data/MLS

stage=90
stop_stage=90
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    tgt_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls
    python utils/prepare_mls.py --mls_root $wav_root --out_dir $tgt_dir --hours 100
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    root_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/
    for tgt_dir in $root_dir/*; do
        echo $tgt_dir
        for x in test train; do
            python utils/rvads.py -r /data/sls/scratch/limingw/fairseq-0.12.1/examples/wav2vec/unsupervised/rVADfast/src/rVADfast/legacy_files --part $part --n-parts $n_parts < $tgt_dir/$x.tsv > $tgt_dir/${x}_${part}.vads
        done
    done
fi

if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]; then
    root_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls
    for tgt_dir in $root_dir/*; do
        python utils/combine_vad.py --in_path "$tgt_dir/{}_{}.vads" --out_path "$tgt_dir/{}.vads" --n_parts $n_parts
        for x in train test; do
            python utils/remove_silence.py --tsv $tgt_dir/$x.tsv --vads $tgt_dir/${x}.vads --out ${tgt_dir}_vad/wavs
        done
    done
fi

if [ $stage -le 30 ] && [ $stop_stage -ge 30 ]; then
    #root_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls
    #for tgt_dir in $root_dir/*_vad; do
    tgt_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_${lang}_vad
    for x in train test; do
        python utils/convert_tsv_to_filelist.py --tsv $tgt_dir/$x.tsv --filelist $tgt_dir/${x}_file_list.txt
    done
    #done
fi

if [ $stage -le 40 ] && [ $stop_stage -ge 40 ]; then
    #root_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls
    #for tgt_dir in $root_dir/*_vad; do
    tgt_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_${lang}_vad
    bash create_data_dir_sylber.sh ${feat_name}_L$l $topk $seg_name $tgt_dir
    #done
fi

if [ $stage -le 50 ] && [ $stop_stage -ge 50 ]; then
    #root_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls
    conda activate espnet2
    #for tgt_dir in $root_dir/*_vad; do
    tgt_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_${lang}_vad
    output_dir=$tgt_dir/feat_${feat_name}_L$l
    zsh prepare_audio_xeus.sh $tgt_dir $output_dir $model 512 $l 100
    #done
    conda deactivate
fi

if [ $stage -le 60 ] && [ $stop_stage -ge 60 ]; then
    # root_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls
    cwd=$PWD
    cd ../../sylber
    conda activate sylber
    # for tgt_dir in $root_dir/*_vad; do 
    tgt_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_${lang}_vad
    for x in test train; do
        python extract_boundary.py --wav-dir $tgt_dir/wavs \
            --file-list $tgt_dir/${x}_file_list.txt \
            --out-path $tgt_dir/sylber_bnd/${x}_data_bnd_dict_${part}.pkl \
            --part $part --n-parts $n_parts
    done
    # done 
    conda deactivate
    cd $cwd 
fi

if [ $stage -le 70 ] && [ $stop_stage -ge 70 ]; then
#    root_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls
    cwd=$PWD
    cd ../../sylber
#    for tgt_dir in $root_dir/*; do
    tgt_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_${lang}_vad
    for x in test train; do
        conda activate sylber
        python combine_pkl.py --in_path $tgt_dir/$seg_name/${x}_data_bnd_dict_{}.pkl --out_path $tgt_dir/$seg_name/${x}_data_bnd_dict.json --n_parts $n_parts
        conda deactivate

        conda activate fairseq
        python json_to_pkl.py --in_path $tgt_dir/$seg_name/${x}_data_bnd_dict.json --out_path $tgt_dir/$seg_name/${x}_data_bnd_dict.pkl
        conda deactivate
    done 
    cd $cwd
fi

if [ $stage -le 80 ] && [ $stop_stage -ge 80 ]; then
    tgt_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_${lang}_vad
    bash prepare_sylber_features_sls.sh $tgt_dir/feat_${feat_name}_L${l} $tgt_dir/feat_${feat_name}_L${l}_$seg_name $topk
fi

if [ $stage -le 90 ] && [ $stop_stage -ge 90 ]; then
    tgt_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_${lang}_vad
    for x in train test; do
        ln -sf /data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_${lang}/${x}_gradseg_style.json /data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/mls/mls_${lang}_vad/${x}_gradseg_style.json
    done
    bash prepare_text_sls.sh $tgt_dir/feat_${feat_name}_L${l}_$seg_name $topk True
fi
