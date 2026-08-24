#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../path.sh"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

tgt_dir=$1
topk=$2
with_tone=$3
text_path=$tgt_dir/top_$topk
target_dir=$tgt_dir/top_$topk/km_dir_$topk
dict_path=$tgt_dir/syllable_$topk.txt
rm -f $dict_path

if [ "$with_tone" = "False" ]; then
    python data_curation/extract_text_no_prune.py --write_dir $text_path --topk $topk --dict_path $dict_path \
        --meta_template $text_path/../../{}_gradseg_style.json
    for x in train test; do
        ln -sf $text_path/../../${x}_gradseg_style_oov_${topk}.json $target_dir/discrete_speech_${x}_clsattnbndkmeans/meta_data.json
    done
else
    python data_curation/extract_text_no_prune.py --write_dir $text_path --topk $topk --dict_path $dict_path \
        --meta_template $text_path/../../{}_gradseg_style_with_tone.json
    for x in train test; do
        ln -sf $text_path/../../${x}_gradseg_style_with_tone_oov_${topk}.json $target_dir/discrete_speech_${x}_clsattnbndkmeans/meta_data.json
    done
fi
cp $text_path/dict.txt $target_dir

python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $text_path/train_words.txt --only-source --destdir $target_dir/text_train --srcdict $text_path/dict.txt
mv $target_dir/text_train/train.bin $target_dir/text_train/text_train.bin
mv $target_dir/text_train/train.idx $target_dir/text_train/text_train.idx

python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $text_path/test_words.txt --only-source --destdir $target_dir/text_test --srcdict $text_path/dict.txt
mv $target_dir/text_test/train.bin $target_dir/text_test/text_test.bin
mv $target_dir/text_test/train.idx $target_dir/text_test/text_test.idx
