#!/bin/bash
#SBATCH -J prepare_forced_align_aishell_vad
#SBATCH -o logs/%j_prepare_forced_align_feat_aishell_vad.out
#SBATCH -e logs/%j_prepare_forced_align_feat_aishell_vad.err
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

topk=2048
#feat_name=hubert_large_ll60k_L21
#feat_name=hubert_large_ll60k_L14
#feat_name=hubert_large_ll60k_L18
#feat_name=xlsr2_960m_1000k_L42
#feat_name=xlsr2_960m_1000k_L28
feat_name=wavlm-large_L21
#feat_name=xlsr_53_56k_new_L21
#feat_name=wav2vec2_zh_L9
data_dir=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin
#src_dir=/nobackup/users/junruin2/data/hubert/aishell_vad/feat_${feat_name}
src_dir=/nobackup/users/junruin2/data/hubert/aishell_3_vad/feat_wavlm-large_L21
manifest_dir=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_${feat_name}_forced_align_bnd/top_2048/km_dir_2048
boundary_path=/nobackup/users/limingw/data/hubert/aishell_3_vad/forced_align_bnd

s=0
e=1
echo stage 0: pool word features
if [ $s -le 0 ] && [ $e -ge 0 ]; then
#    python extract_features/create_gt_unsup_seg.py
    for x in test train; do
        rm $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data_dict_bnd.pkl 
        ln -s $boundary_path/${x}_data_bnd_dict.pkl $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data_dict_bnd.pkl
        ln -s $data_dir/${x}_gradseg_style.json $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/meta_data.json
        echo $boundary_path/${x}_data_bnd_dict.pkl
        python extract_features/pool_word_feats_wav2bnd.py \
            --data_path $manifest_dir --split $x
    done
fi

echo stage 1: k-means clustering
if [ $s -le 1 ] && [ $e -ge 1 ]; then
    cwd=$PWD
    cd extract_features/simple_kmeans
    # kmeans clustering
    bash prepare_word_km_model.sh $feat_name
    bash prepare_word_quantization.sh $feat_name
    cd $cwd
fi

echo stage 2: create soft links
if [ $s -le 2 ] && [ $e -ge 2 ]; then
    bash extract_features/gen_clus.sh 431 $manifest_dir
    ln -s $manifest_dir/seg_feats/km_model.npy $manifest_dir/km_model.npy
    for x in train test; do
        rm $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/meta_data.json
        rm $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data_frame_clus.km
        rm $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data.km
        ln -s $data_dir/${x}_gradseg_style_oov_2048.json $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/meta_data.json
        ln -s $src_dir/km_idx_clsattnbndkmeans/${x}_0_1.km $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data_frame_clus.km
        ln -s $manifest_dir/seg_feats/km_idx_clsattnbndkmeans/$x.km $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data.km 
    done
fi
