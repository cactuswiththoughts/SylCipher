#!/bin/bash
#SBATCH -J bpe_libri
#SBATCH -o logs/%j_bpe_libri.out
#SBATCH -e logs/%j_bpe_libri.err
#SBATCH --qos=regular
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=2080

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../path.sh"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${SUPERBPE_ENV}"

libri_txt=${SUPERBPE_CORPUS:-${SCRIPT_DIR}/librispeech_clean_vad}
libri_man=${MANIFEST_ROOT}/librispeech_clean_vad
tok_dir=tokenizers/olmo2_libri_clean_vad_bpe
tok_dir_v2=tokenizers/olmo2_libri_clean_vad_bpe_v2
matched_dir=tokenizers/olmo2_libri_clean_vad_bpe_matched

s=${s:-1}
e=${e:-1}
if [ $s -le 0 ] && [ $e -ge 0 ]; then
python -m train_tokenizer \
    --output_dir $tok_dir \
    --corpus_dir $libri_txt \
    --num_bytes $((10**10)) \
    --vocab_size 17000 \
    --do_whitespace_pretokenization true
fi

if [ $s -le 1 ] && [ $e -ge 1 ]; then
python -m train_tokenizer \
    --output_dir $tok_dir_v2 \
    --corpus_dir $libri_txt \
    --num_bytes $((10**10)) \
    --vocab_size 17000 \
    --do_whitespace_pretokenization true
fi

if [ $s -le 10 ] && [ $e -ge 10 ]; then
    python match_tokens.py \
        --in_dir $tok_dir \
        --dict_path ${libri_man}/syllable_pyphen_plus_dict_2048.txt \
        --out_dir $matched_dir
fi

if [ $s -le 20 ] && [ $e -ge 20 ]; then
    for x in valid valid_true train; do
        python tokenize_superbpe.py \
            --metadata_path ${libri_man}/ls_train_clean_vad_gradseg_style_${x}.json \
            --tokenizer_path $tok_dir/tokenizer.json \
            --output_prefix ${libri_man}/${x}_bpe
    done
fi

if [ $s -le 30 ] && [ $e -ge 30 ]; then
    for x in valid valid_true train; do
        python greedy_tokenize.py \
            --metadata_path ${libri_man}/ls_train_clean_vad_gradseg_style_${x}.json \
            --tokenizer_path $matched_dir/syllable_bpe.txt \
            --output_prefix ${libri_man}/${x}_bpe_matched
    done
fi
