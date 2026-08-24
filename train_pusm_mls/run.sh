#!/bin/bash

#for lang in dutch french german italian; do
#    sbatch train_pusm_4gpu_mls.slurm $lang 
#done
sbatch train_pusm_4gpu_mls.slurm german 2 3
sbatch train_pusm_4gpu_mls.slurm german 3 4
sbatch train_pusm_4gpu_mls.slurm german 4 5
