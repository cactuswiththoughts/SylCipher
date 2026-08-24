#!/bin/bash
#SBATCH -J prepare_text_fleurs
#SBATCH -o logs/%j_prepare_text_fleurs.out
#SBATCH -e logs/%j_prepare_text_fleurs.err
#SBATCH -p 2080
##SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --qos=regular

export HOME=/data/sls/scratch/limingw
export FAIRSEQ_ROOT=$HOME/fairseq-0.12.1
export KENLM_ROOT=/usr/users/limingw/softwares/kenlm/build/bin
source /data/sls/scratch/limingw/miniconda3/etc/profile.d/conda.sh
conda activate fairseq

tgt_dir=$1
topk=$2
with_tone=$3
text_path=$tgt_dir/top_$topk
target_dir=$tgt_dir/top_$topk/km_dir_$topk
dict_path=$tgt_dir/syllable_$topk.txt
if [ -f $dict_path ]; then
    rm $dict_path
fi 

python data_curation/extract_text_no_prune.py --write_dir $text_path --topk $topk --dict_path $dict_path --meta_template $text_path/../../{}_gradseg_style.json 
for x in train test; do
ln -sf $text_path/../../${x}_gradseg_style_oov_${topk}.json $target_dir/discrete_speech_${x}_clsattnbndkmeans/meta_data.json
done

cp $text_path/dict.txt $target_dir

python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $text_path/train_words.txt --only-source --destdir $target_dir/text_train --srcdict $text_path/dict.txt
mv $target_dir/text_train/train.bin $target_dir/text_train/text_train.bin
mv $target_dir/text_train/train.idx $target_dir/text_train/text_train.idx

python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $text_path/test_words.txt --only-source --destdir $target_dir/text_test --srcdict $text_path/dict.txt
mv $target_dir/text_test/train.bin $target_dir/text_test/text_test.bin
mv $target_dir/text_test/train.idx $target_dir/text_test/text_test.idx
