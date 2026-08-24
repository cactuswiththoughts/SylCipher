#!/bin/bash

n_clusters=$1
exp_dir=$2

for x in $(seq 0 $((n_clusters - 1))); do
  echo "$x 1"
done > ${exp_dir}/dict.audio.txt

cp ${exp_dir}/dict.audio.txt ${exp_dir}/dict.audio0.txt
