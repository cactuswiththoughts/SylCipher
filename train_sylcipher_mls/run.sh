#!/bin/bash

#for part in {0..19}; do
#    sbatch prepare_mls.slurm $part
#done

# portuguese polish spanish
for lang in dutch french german italian; do
    sbatch prepare_mls.slurm 0 $lang
done   
