#!/bin/bash

export HOME=/data/sls/scratch/limingw
export FAIRSEQ_ROOT=/data/sls/scratch/limingw/fairseq-0.12.1
source /data/sls/scratch/limingw/miniconda3/etc/profile.d/conda.sh
conda activate fairseq
ratio=$1

# Mix two train_words.txt sources and binarize them into fairseq text_train format.
# Both sources should use the same tokenization as the provided dictionary.
source1=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/feat_hubert_large_L21_sylber_bnd/top_2048/train_words.txt
#source2=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/spokencoco/without_silence/train_syllable_pyphen_plus.txt
#source2=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/wiki/train_wiki_pyphen_plus.txt
source2=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/newscrawl_text_train/train_newscrawl_pyphen_plus.txt
dict_path=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/feat_hubert_large_L21_sylber_bnd/top_2048/dict.txt
#text_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/feat_hubert_large_L21_sylber_bnd/top_2048/km_dir_2048/wiki_mixed${ratio}_text_train
#target_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/feat_hubert_large_L21_sylber_bnd/top_2048/km_dir_2048/wiki_mixed${ratio}_text_train
text_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/feat_hubert_large_L21_sylber_bnd/top_2048/km_dir_2048/newscrawl_mixed${ratio}_text_train
target_dir=/data/sls/scratch/limingw/workplace/VGUnsupASR/manifest/librispeech_clean_vad/feat_hubert_large_L21_sylber_bnd/top_2048/km_dir_2048/newscrawl_mixed${ratio}_text_train
seed=0
if [ ! -d $text_dir ]
then
mkdir -p $text_dir
fi

mixed_train_words=${text_dir}/mixed${ratio}_train_words.txt
mixed_dict=${text_dir}/dict.txt
python data_curation/mix_text_data.py \
  --source1 ${source1} \
  --source2 ${source2} \
  --dict_path ${dict_path} \
  --output ${mixed_train_words} \
  --output_dict ${mixed_dict} \
  --ratio ${ratio} \
  --use_all_source1 \
  --seed ${seed}

python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py \
  --dataset-impl mmap \
  --trainpref ${mixed_train_words} \
  --only-source \
  --destdir ${target_dir} \
  --srcdict ${mixed_dict} \
  2>&1 | tee ${target_dir}/preprocess.log

cp ${target_dir}/train.bin ${target_dir}/mixed${ratio}_text_train.bin
cp ${target_dir}/train.idx ${target_dir}/mixed${ratio}_text_train.idx
