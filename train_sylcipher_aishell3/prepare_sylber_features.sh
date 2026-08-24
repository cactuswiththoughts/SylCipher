#!/bin/bash
#SBATCH -J prepare_sylber_aishell_3_vad
#SBATCH -o logs/%j_prepare_sylber_aishell_3_vad.out
#SBATCH -e logs/%j_prepare_sylber_aishell_3_vad.err
#SBATCH --qos=sched_level_2
#SBATCH --mail-user=junruin2@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --exclude=node0060,node0063,node0021,node0054,node0042,node0029,node0014,node0048,node0049,node0008
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --exclusive

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../path.sh"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate exp_spring2022

topk=1024
n_clus=1024
feat_name=xeus_L18
seg_name=sylber_ft_stage2_bnd
data_dir=/nobackup/users/junruin2/data/hubert/aishell_3_vad
src_dir=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin_no_tone/feat_${feat_name}_${seg_name}/top_2048/km_dir_2048
manifest_dir=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_${feat_name}_$seg_name/top_$topk/km_dir_$topk

s=0
e=2
echo stage 0: pool word features
if [ $s -le 0 ] && [ $e -ge 0 ]; then
    for x in test train; do
        rm $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data_dict_bnd.pkl 
        ln -s $src_dir/discrete_speech_${x}_clsattnbndkmeans/data_dict_bnd.pkl $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data_dict_bnd.pkl
        if [ ! -d $manifest_dir/seg_feats ]; then
            mkdir $manifest_dir/seg_feats
        fi
        ln -s $src_dir/seg_feats/${x}_0_1.npy $manifest_dir/seg_feats/${x}_0_1.npy
        ln -s $src_dir/seg_feats/${x}_0_1.len $manifest_dir/seg_feats/${x}_0_1.len
    done
fi

echo stage 1: k-means clustering
if [ $s -le 1 ] && [ $e -ge 1 ]; then
    cwd=$PWD
    cd extract_features/simple_kmeans
    # kmeans clustering
    bash prepare_word_km_model.sh $feat_name $seg_name
    bash prepare_word_quantization.sh $feat_name $seg_name
    cd $cwd
fi

echo stage 2: create soft links
if [ $s -le 2 ] && [ $e -ge 2 ]; then
    bash extract_features/gen_clus.sh $topk $manifest_dir
    ln -s $manifest_dir/seg_feats/km_model.npy $manifest_dir/km_model.npy
    for x in train test; do
        rm $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/meta_data.json
        rm $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data_frame_clus.km
        rm $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data.km
        ln -s $data_dir/${x}_gradseg_style_with_tone_oov_$topk.json $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/meta_data.json
        ln -s $src_dir/discrete_speech_${x}_clsattnbndkmeans/data_frame_clus.km $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data_frame_clus.km
        ln -s $manifest_dir/seg_feats/km_idx_clsattnbndkmeans/$x.km $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data.km 
    done
fi
