#!/bin/bash

for lang in dutch french italian; do
    sbatch run_mls.slurm $lang
done
