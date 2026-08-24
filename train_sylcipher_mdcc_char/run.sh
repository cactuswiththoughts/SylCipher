#!/bin/bash

for part in {0..9}; do
    sbatch prepare_mdcc.slurm $part
done
