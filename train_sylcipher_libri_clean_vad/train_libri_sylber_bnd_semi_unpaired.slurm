#!/bin/bash
#SBATCH -J train_libri_clean_vad_semi_unpaired
#SBATCH -o logs/%j_train_libri_clean_vad_semi_unpaired.out
#SBATCH -e logs/%j_train_libri_clean_vad_semi_unpaired.err
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --qos=regular

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../path.sh"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

ulimit -s unlimited

s=10
e=10
topk=2048
DATA_ROOT="${MANIFEST_ROOT}/librispeech_clean_vad/feat_hubert_large_L21_sylber_bnd/top_${topk}/km_dir_${topk}"
WIKI_ROOT="${MANIFEST_ROOT}/wiki/"
if [ $s -le 0 ] && [ $e -ge 0 ]; then 
    python data_curation/prepare_domain_text.py \
        --out-dir /data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/newscrawl_text_train \
        --domains newscrawl \
        --max-lines 300000 \
        --raw-dir 2014
fi

if [ $s -le 1 ] && [ $e -ge 1 ]; then
    pip install wikiextractor

    python -m wikiextractor.WikiExtractor \
        -o $WIKI_ROOT/extracted \
        --json \
        --processes 16 \
        $WIKI_ROOT/enwiki-latest-pages-articles-multistream1.xml-p1p41242.bz2

    python data_curation/prepare_domain_text.py \
        --out-dir $WIKI_ROOT \
        --domains wiki \
        --wiki-json-dir $WIKI_ROOT/extracted \
        --max-lines 300000
fi

if [ $s -le 3 ] && [ $e -ge 3 ]; then
    python data_curation/extract_libri_gt_syllable_boundary_pyphen_plus_unpaired.py \
        --input_file /data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/newscrawl_text_train/newscrawl.txt \
        --output_prefix /data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/newscrawl_text_train/train_newscrawl_pyphen_plus
fi

if [ $s -le 4 ] && [ $e -ge 4 ]; then
    python data_curation/extract_libri_gt_syllable_boundary_pyphen_plus_unpaired.py \
        --input_file /data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/wiki/wiki.txt \
        --output_prefix /data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/wiki/train_wiki_pyphen_plus
fi

if [ $s -le 10 ] && [ $e -ge 10 ]; then
    #ratio=0.25
    #for ratio in 0.25 0.5 0.75
    for ratio in 0.25
    do
    bash data_curation/prepare_mixed_text.sh $ratio
    done
fi
