#!/bin/bash
#SBATCH -J prepare_sylber_feat_scoco_clean_vad
#SBATCH -o logs/%j_prepare_sylber_feat_scoco_clean_vad.out
#SBATCH -e logs/%j_prepare_sylber_feat_scoco_clean_vad.err
#SBATCH --mail-user=junruin2@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --exclude=node0060,node0063,node0021,node0054,node0042,node0029,node0014,node0048,node0049,node0008
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --exclusive

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../path.sh"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

feat_name=hubert_large_L21
seg_name=sylber_bnd
topk=$1
data_dir=${DATA_ROOT}/LibriSpeech_vad
manifest_dir=${MANIFEST_ROOT}/librispeech_clean_vad/feat_${feat_name}_${seg_name}/top_${topk}/km_dir_${topk}
boundary_path=$data_dir/$seg_name

s=0
e=2
echo stage 0: pool word features
if [ $s -le 0 ] && [ $e -ge 0 ]; then
    for x in valid valid_true train; do
        ln -s $data_dir/$seg_name/${x}_data_bnd_dict.pkl $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data_dict_bnd.pkl
        echo $boundary_path/${x}_data_dict_bnd.pkl
        python extract_features/pool_word_feats_wav2bnd.py --data_path $manifest_dir --split $x
    done
fi

echo stage 1: k-means clustering on the pooled word features
if [ $s -le 1 ] && [ $e -ge 1 ]; then
    cwd=$PWD
    cd extract_features/simple_kmeans
    # kmeans clustering
    bash prepare_word_km_model.sh $manifest_dir/seg_feats $topk
    bash prepare_word_quantization.sh $manifest_dir/seg_feats $topk
    cd $cwd
fi

echo stage 2: create soft links
if [ $s -le 2 ] && [ $e -ge 2 ]; then
    bash extract_features/gen_clus.sh $topk $manifest_dir
    ln -s $manifest_dir/seg_feats/km_model.npy $manifest_dir/km_model.npy
    for x in train valid valid_true; do
        ln -s $manifest_dir/seg_feats/km_idx_clsattnbndkmeans/${x}.km $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data.km
    done
fi
