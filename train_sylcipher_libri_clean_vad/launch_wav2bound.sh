#!/bin/bash
#SBATCH -J launch_wav2bound_je2e
#SBATCH -o logs/%j_launch_wav2bound_je2e.out
#SBATCH -e logs/%j_launch_wav2bound_je2e.err
#SBATCH -p a6
##SBATCH --gres=gpu:4
#SBATCH --qos=regular
#SBATCH --gres=gpu:1
#SBATCH -n 1

export HOME=/data/sls/scratch/limingw
source /data/sls/scratch/limingw/miniconda3/etc/profile.d/conda.sh 
conda activate wav2boundaries

config_file='../../wav2boundaries/config/spokencoco-2048-diff-bnd.yaml'
fairseq_dir='/data/sls/scratch/limingw/fairseq_fork/'

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PWD

root_home=$PWD
dataset=$(cat $config_file | grep data: | head -n 1 | cut -d ' ' -f 4)
config_dir=$root_home/$(dirname $config_file)
config_name=$(echo $(basename $config_file) | cut -d '.' -f 1)

s=1
e=1
if [ $s -le 0 ] && [ $e -ge 0 ]; then
    for x in train val2; do
        python extract_features/prepare_initial_wav2boundaries_files.py \
            --split $x --write_dir $dataset
    done

    for x in train val2; do
        bound_file=$dataset/$x
        python extract_features/format_boundary.py $bound_file $dataset
    done
fi

if [ $s -le 1 ] && [ $e -ge 1 ]; then
    cd ../../wav2boundaries
    PYTHONPATH=$PYTHONPATH:$fairseq_dir python $fairseq_dir/fairseq_cli/hydra_train.py -m --config-dir $config_dir --config-name $config_name task.data=${dataset}
fi
