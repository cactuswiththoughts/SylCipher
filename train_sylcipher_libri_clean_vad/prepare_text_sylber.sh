#!/bin/bash
#SBATCH -J prepare_text
#SBATCH -o logs/%j_prepare_text.out
#SBATCH -e logs/%j_prepare_text.err
#SBATCH --mail-user=junruin2@mit.edu
#SBATCH --mail-type=ALL
##SBATCH --exclude=node0060,node0063,node0021,node0054,node0042,node0029,node0014,node0048,node0049,node0008
##SBATCH --gres=gpu:1
##SBATCH --gpus-per-node=4
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --exclusive

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../path.sh"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

feat_name=hubert_large_L21
seg_name=sylber_bnd
topk=$1
data_dir=${MANIFEST_ROOT}/librispeech_clean_vad
text_dir=${MANIFEST_ROOT}/librispeech_clean_vad/feat_${feat_name}_${seg_name}/top_${topk}/
target_dir=${MANIFEST_ROOT}/librispeech_clean_vad/feat_${feat_name}_${seg_name}/top_$topk/km_dir_$topk

python data_curation/extract_text_no_prune.py --topk $topk --write_dir $text_dir --dict_path $text_dir/syllable_pyphen_plus_dict_${topk}.txt --meta_template $data_dir/ls_train_clean_vad_syllable_pyphen_plus_{}.json \
        --out_template $data_dir/{}_syllable_pyphen_plus_oov_${topk}.json

for x in train valid valid_true; do
    rm $target_dir/discrete_speech_${x}_clsattnbndkmeans/meta_data.json
    ln -s $data_dir/${x}_syllable_pyphen_plus_oov_${topk}.json $target_dir/discrete_speech_${x}_clsattnbndkmeans/meta_data.json
    ln -s $text_dir/${x}_bnd_errors.txt $target_dir/discrete_speech_${x}_clsattnbndkmeans/bnd_errors.txt

    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $text_dir/${x}_words.txt --only-source --destdir $target_dir/text_$x --srcdict $text_dir/dict.txt
    mv $target_dir/text_$x/train.bin $target_dir/text_$x/text_$x.bin
    mv $target_dir/text_$x/train.idx $target_dir/text_$x/text_$x.idx
done
