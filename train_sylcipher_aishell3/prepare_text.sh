#!/bin/bash
#SBATCH -J prepare_text
#SBATCH -o logs/%j_prepare_text.out
#SBATCH -e logs/%j_prepare_text.err
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
FAIRSEQ_ROOT=/nobackup/users/junruin2/fairseq_expspring2022
#feat_name=hubert_large_ll60k_L21
feat_name=xeus_L18
seg_name=sylber_ft_stage2_bnd
text_path=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_${feat_name}_${seg_name}/top_1024/
target_dir=/nobackup/users/junruin2/data/hubert/aishell_3_vad_pinyin/feat_${feat_name}_${seg_name}/top_1024/km_dir_1024

#python data_curation/extract_text_no_prune.py

python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $text_path/train_words.txt --only-source --destdir $target_dir/text_train --srcdict $text_path/dict.txt
mv $target_dir/text_train/train.bin $target_dir/text_train/text_train.bin
mv $target_dir/text_train/train.idx $target_dir/text_train/text_train.idx

python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $text_path/test_words.txt --only-source --destdir $target_dir/text_test --srcdict $text_path/dict.txt
mv $target_dir/text_test/train.bin $target_dir/text_test/text_test.bin
mv $target_dir/text_test/train.idx $target_dir/text_test/text_test.idx
