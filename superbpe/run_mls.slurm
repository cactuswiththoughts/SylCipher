#!/bin/bash
#SBATCH -J bpe_mls
#SBATCH -o logs/%j_bpe_mls.out
#SBATCH -e logs/%j_bpe_mls.err
#SBATCH --qos=regular
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=2080

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../path.sh"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${SUPERBPE_ENV}"

lang=$1
if [ -z "$lang" ]; then
    echo "Usage: sbatch run_mls.slurm <dutch|french|german|italian>"
    exit 1
fi
if [ $lang = german ]; then
    vocab_size=9500
elif [ $lang = dutch ]; then
    vocab_size=10350
elif [ $lang = french ]; then
    vocab_size=12000
elif [ $lang = italian ]; then
    vocab_size=5000
else
    echo "Unknown language: $lang"
    exit 1
fi

mls_root=${DATA_ROOT}/MLS/mls_${lang}
mls_man=${MANIFEST_ROOT}/mls/mls_${lang}_vad
corpus_dir=${SCRIPT_DIR}/mls_${lang}_vad
tok_dir=tokenizers/olmo2_mls_${lang}_vad_bpe
matched_dir=tokenizers/olmo2_mls_${lang}_vad_bpe_matched

s=${s:-20}
e=${e:-40}

if [ $s -le 0 ] && [ $e -ge 0 ]; then
    python prepare_mls_word.py \
        --mls_root $mls_root \
        --filelist ${mls_man}/train_file_list.txt ${mls_man}/test_file_list.txt \
        --out_dir $corpus_dir
fi

if [ $s -le 10 ] && [ $e -ge 10 ]; then
python -m train_tokenizer \
    --output_dir $tok_dir \
    --corpus_dir $corpus_dir \
    --num_bytes $((10**10)) \
    --vocab_size ${vocab_size} \
    --do_whitespace_pretokenization true
fi

if [ $s -le 20 ] && [ $e -ge 20 ]; then
    python match_tokens.py \
        --in_dir $tok_dir \
        --dict_path ${mls_man}/feat_xeus_L18_sylber_bnd/syllable_1024.txt \
        --out_dir $matched_dir \
        --lowercase
fi

if [ $s -le 30 ] && [ $e -ge 30 ]; then
    for x in train test; do
        python tokenize_superbpe.py \
            --metadata_path ${corpus_dir}/${x}_gradseg_style.json \
            --tokenizer_path $tok_dir/tokenizer.json \
            --output_prefix ${mls_man}/${x}_bpe
    done
fi

if [ $s -le 40 ] && [ $e -ge 40 ]; then
    for x in train test; do
        python greedy_tokenize.py \
            --metadata_path ${corpus_dir}/${x}_gradseg_style.json \
            --tokenizer_path ${matched_dir}/syllable_bpe.txt \
            --output_prefix ${mls_man}/${x}_bpe_matched
    done
fi
