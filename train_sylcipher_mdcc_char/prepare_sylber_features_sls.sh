#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../path.sh"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# $1 unused (legacy); character jobs reuse phonetic MDCC features under mdcc_vad.
feat_dir=$1
manifest_prefix=$2
topk=$3
phonetic_dir=${MANIFEST_ROOT}/mdcc_vad
manifest_dir=${manifest_prefix}/top_$topk/km_dir_$topk

s=${s:-30}
e=${e:-30}
echo "Link pooled Sylber features into $manifest_dir"
if [ $s -le 30 ] && [ $e -ge 30 ]; then
    bash extract_features/gen_clus.sh $topk $manifest_dir
    for x in test train; do
        ln -sf ${phonetic_dir}/feat_xeus_L18_sylber_bnd/top_$topk/km_dir_$topk/seg_feats $manifest_dir/seg_feats
        ln -sf ${phonetic_dir}/feat_xeus_L18_sylber_bnd/top_$topk/km_dir_$topk/seg_feats/km_idx_clsattnbndkmeans/$x.km \
            $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data.km
        ln -sf ${phonetic_dir}/feat_xeus_L18/km_idx_clsattnbndkmeans/${x}_0_1.km \
            $manifest_dir/discrete_speech_${x}_clsattnbndkmeans/data_frame_clus.km
    done
fi
