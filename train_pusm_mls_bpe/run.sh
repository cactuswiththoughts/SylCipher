#!/bin/bash

#for lang in dutch french german italian; do
#for lang in german; do
for lang in dutch french german italian; do
    sbatch train_pusm_4gpu_mls.slurm $lang 
done 
