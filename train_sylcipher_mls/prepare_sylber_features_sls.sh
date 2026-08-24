#!/bin/bash
#SBATCH -J prepare_sylber_feat_fleurs
#SBATCH -o logs/%j_prepare_sylber_feat_fleurs.out
#SBATCH -e logs/%j_prepare_sylber_feat_fleurs.err
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

data_dir=$1
manifest_dir=$2
topk=$3
nclus=$topk
manifest_dir=$manifest_dir/top_$topk/km_dir_$topk

s=0
e=30
echo stage 0: pool word features
if [ $s -le 0 ] && [ $e -ge 0 ]; then
    rm $manifest_dir/km_model.npy
    for x in test train; do
        python extract_features/pool_word_feats_wav2bnd.py --data_path $manifest_dir --split $x
    done
fi

if [ $s -le 10 ] && [ $e -ge 10 ]; then
    cwd=$PWD
    cd extract_features/simple_kmeans
    bash prepare_word_km_model.sh $manifest_dir/seg_feats $nclus #$topk 
    cd $cwd
fi

echo stage 20: extract k-means units
if [ $s -le 20 ] && [ $e -ge 20 ]; then
    cwd=$PWD
    cd extract_features/simple_kmeans
    bash prepare_word_quantization.sh $manifest_dir/seg_feats
    cd $cwd
fi

echo stage 30: create soft links
if [ $s -le 30 ] && [ $e -ge 30 ]; then
    bash extract_features/gen_clus.sh $topk $manifest_dir
    for x in test train; do
        ln -sf $manifest_dir/seg_feats/km_idx_clsattnbndkmeans/$x.km $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data.km
        ln -sf $data_dir/km_idx_clsattnbndkmeans/${x}_0_1.km $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data_frame_clus.km
    done
fi
